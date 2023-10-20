import os
import time
import clr
import System
import pandas as pd
import signal
import openpyxl
import datetime
import concurrent.futures
import ctypes
import threading
import select
import timeit
import numpy as np
import csv

#os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '10'
os.environ['PYDEVD_INTERRUPT_THREAD_TIMEOUT'] = '3000'  #

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\Kinesis\\ThorLabs.MotionControl.KCube.DCServoCLI.dll")
#clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.Settings.dll")
#clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor.dll")
#clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.ControlParameters.dll")
from Thorlabs.MotionControl.GenericMotorCLI.Settings import *
from Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor import *
from Thorlabs.MotionControl.GenericMotorCLI.ControlParameters import *
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from System import Decimal


class ThorlabsDevices:
    def __init__(self):
        """
        Initializes a collection of Thorlabs devices.
        """
        self.devices = {}

    def connect_device(self, serial_number):
        """
        Connects a Thorlabs device with the given serial number.

        Args:
            serial_number (str): The serial number of the device to connect.

        Raises:
            Exception: If the connection or initialization fails.
        """
        if serial_number not in self.devices:
            try:
                # Attempt to create and connect the KCubeDCServo device
                device = KCubeDCServo.CreateKCubeDCServo(serial_number)
                device.Connect(serial_number)
                time.sleep(0.25)
                device.StartPolling(250)
                time.sleep(0.25)  # Wait statements are important to allow settings to be sent to the device
                device.EnableDevice()

                # Wait for Settings to Initialize
                if not device.IsSettingsInitialized():
                    device.WaitForSettingsInitialized(10000)  # 10-second timeout
                    assert device.IsSettingsInitialized() is True
                
                
                device_name = device.DeviceName # Get device name
                device_info = device.GetDeviceInfo() # Get Device information

                # Before homing or moving the device, ensure the motor's configuration is loaded
                motor_config = device.LoadMotorConfiguration(
                    serial_number, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)

                if motor_config is not None:
                    actuator = motor_config.DeviceSettingsName
                    print(f"Device connected to the driver {actuator}")
                self.devices[serial_number] = device
                print(f"Device name: {device_name}")
                print(device_info.Description)

                print(f"Connected to the device with serial number {serial_number}\n")
            except Exception as e:
                print(f"Failed to connect the device: {str(e)}")

    def disconnect_device(self, serial_number):
        """
        Disconnects a Thorlabs device with the given serial number.

        Args:
            serial_number (str): The serial number of the device to disconnect.

        Raises:
            Exception: If the disconnection fails.
        """
        if serial_number in self.devices:
            try:
                self.devices[serial_number].Disconnect(True)
                del self.devices[serial_number]
                print(f"Device disconnected: {serial_number}")
            except Exception as e:
                print(f"Failed to disconnect the device: {str(e)}")

# Define a function to home a single device
def home_device_and_set_velocity(device, initial_position, velocity):
    """
    Documentar
    """
    
    try:    
            print(f"Homing device with serial number {device.DeviceID}")
            device.Home(60000)  
            device.SetVelocityParams(Decimal(2), Decimal(1.5))
            print(f"Device with serial number {device.DeviceID} has completed homing.")
            print(f'Moving to initial position {initial_position}')
            device.MoveTo(Decimal(initial_position), 60000) 
            device.SetVelocityParams(Decimal(velocity), Decimal(3))
            velocity_parameters = device.GetVelocityParams()
            max_velocity = velocity_parameters.MaxVelocity
            acceleration = velocity_parameters.Acceleration
            print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')
    except Exception as error:
        print(error)

def shif_device(device, final_position,waitTimeout):
    """
    Move a device to a specified final position.

    This function moves the specified 'device' to the given 'final_position'.

    Parameters:
    - device: The device object to be moved.
    - final_position (Decimal): The final position to which the device should be moved (Decimal).

    Returns:
    None

    Example:
    To move 'my_device' to a final position of Decimal('20.0'):
    >>> shift_device(my_device, Decimal('20.0'))
    This will move 'my_device' to the position Decimal('20.0').
    """
    try:
        device.MoveTo(Decimal(final_position),waitTimeout)
    except Exception as e:
        print(e)

def hysteresis(device,initial_position, final_position, cycles,waitTimeout):
    """
    Perform a hysteresis test on a given device.

    This function moves the device back and forth between the specified
    initial_position and final_position for the specified number of cycles.

    Parameters:
    - device: The device object to be tested.
    - initial_position (Decimal): The initial position where the hysteresis test starts.
    - final_position (Decimal): The maximum shift to be achieved during each cycle.
    - cycles (int): The number of cycles to perform.

    Returns:
    None

    Example:
    To perform a hysteresis test on a device:
    >>> hysteresis(my_device, Decimal('0.0'), Decimal('10.0'), 5)
    This will move 'my_device' back and forth between positions 10.0 and 0.0
    for 5 cycles.
    """
    try:
        for _ in range(cycles):
            while (device.Status.IsMoving):
                pass
            device.MoveTo(Decimal(final_position),waitTimeout)
            while (device.Status.IsMoving):
                pass
            device.MoveTo(Decimal(initial_position),waitTimeout)
    except Exception as e:
        print(e)

def stress_relaxation(device,foward_position, waiting_time, cycles):
    for _ in range(cycles):
        device.MoveBy(-foward_position)
        time.sleep(waiting_time)



def set_parameters(case,velocity,initial_position,final_position,cycles=None,forward_position=None, waiting_time=None):
    '''
    DOCUMENTACntar 
    
    '''
 
    initial_position=initial_position+25 #Initial position must be
    final_position=25-final_position
    if case==1:
        execution_time=(initial_position-final_position)/velocity
        waitTimeout=execution_time*1000+2000
        waitTimeout+=waitTimeout*0.1
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        return velocity,initial_position,final_position,execution_time,waitTimeout
    if case==2:
        execution_time= (initial_position-final_position)*2*cycles/velocity
        waitTimeout=(initial_position-final_position)*1000/velocity+2000
        waitTimeout+=waitTimeout*0.1
        waitTimeout=System.Convert.ToUInt64(waitTimeout)#Duration to wait for command to execute
        return velocity,initial_position,final_position,cycles,execution_time,waitTimeout
    if case==3:
        if forward_position*cycles>20:
            cycles-=1
        execution_time=(forward_position/velocity+waiting_time)*cycles
        print(execution_time)
        waitTimeout=forward_position/velocity*1000
        waitTimeout+=waitTimeout*0.1
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        return velocity,forward_position, waiting_time,cycles,execution_time,waitTimeout
    
def get_position(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    devices_list=list(devices_dictionary.values())
    initial_positions=list(map(lambda device: device.Position,devices_list))
    # Create an empty DataFrame within capture_values
    columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
    data = pd.DataFrame(columns=columns)
    sample_interval = 1.0 / sample_frequency
    acumulated_time=0   
    time_start=datetime.datetime.now()
    while acumulated_time<=execution_time: 
        # Execute obtener_posicion_y_hora for each device
        current_time = timeit.default_timer()
        data.loc[data.shape[0]] = [datetime.datetime.now()]+ list(map(lambda device: device.Position,devices_list))
        line_time=timeit.default_timer()
        sleep_time=sample_interval-(line_time-current_time)
        # Sleep for one second before the next capture
        time.sleep(sleep_time)
        acumulated_time+=sleep_time
    
    data["time"]=data['timestamp']-time_start
    data["seconds"] = data["time"].dt.total_seconds()
    data["minutes"] = data["seconds"] / 60
    data["hours"] = data["seconds"] / 3600
    data["milliseconds"] = data["time"].dt.total_seconds() * 1000
    data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=initial_positions-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
    data["time"]=data["time"].apply(lambda x: str(x).split()[2])
    data=data.sort_index(axis=1, ascending=False)
    print(1)
    print('muestras',execution_time/sample_interval, len(data))
    print(data["seconds"].diff().describe())
    #print(data)# Cargar el archivo de Excel existente
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
    
        # Escribir el DataFrame en la hoja seleccionada
        data.to_excel(writer, sheet_name=sheet_name, index=False)

def get_position2(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    'No usar'
    try:
        devices_list=list(devices_dictionary.values())
        initial_positions=list(map(lambda device: device.Position,devices_list))
        # Create an empty DataFrame within capture_values
        columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        data = pd.DataFrame(columns=columns)
        sample_interval = 1.0 / sample_frequency
        start_time = datetime.datetime.now()
        def sample_function():
            timestamp = datetime.datetime.now()
            positions = list(map(lambda device: device.Position, devices_list))
            info = [timestamp] + positions
            data.loc[data.shape[0]] = info
        
        end_time = start_time + datetime.timedelta(seconds=execution_time)

        def sample_periodically():
            while datetime.datetime.now() < end_time:
                sample_function()
                time.sleep(sample_interval)

        # Start sampling in a separate thread
        sampling_thread1 = threading.Thread(target=sample_periodically)
        sampling_thread1.start()
        # Wait for the specified execution_time
        sampling_thread1.join()
        data["time"]=data['timestamp']-start_time
        data["seconds"] = data["time"].dt.total_seconds()
        data["minutes"] = data["seconds"] / 60
        data["hours"] = data["seconds"] / 3600
        data["milliseconds"] = data["time"].dt.total_seconds() * 1000
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=initial_positions-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        data["time"]=data["time"].apply(lambda x: str(x).split()[2])
        data=data.sort_index(axis=1, ascending=False)
        print('2')
        print(data)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
    
        # Escribir el DataFrame en la hoja seleccionada
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    # Guardar el archivo de Excel
        writer.save()
       
    except Exception as e:
        print(e)

def get_position33(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    try:

        devices_list=list(devices_dictionary.values())
        initial_positions=list(map(lambda device: device.Position,devices_list))
        # Create an empty DataFrame within capture_values
        columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        data = pd.DataFrame(columns=columns)
        sample_interval = 1.0 / sample_frequency
        print('33')
        print(execution_time/sample_interval)
        def sample_function():
            data.loc[data.shape[0]] = [timeit.default_timer()] +list(map(lambda device: device.Position, devices_list))

        def sample_periodically():
            nonlocal start_time
            while (timeit.default_timer() - start_time) <= execution_time:
                current_time = timeit.default_timer()
                sample_function()
                line_time=timeit.default_timer()
                sleep_time=sample_interval-(line_time-current_time)
                time.sleep(sleep_time)  # Use time.sleep() to control the sampling rate

        start_time = timeit.default_timer()
        data.loc[data.shape[0]] = [start_time] +initial_positions
        # Start sampling in a separate thread
        sampling_thread = threading.Thread(target=sample_periodically)
        sampling_thread.start()
        # Wait for the specified execution_time
        sampling_thread.join()
        print('3')
        
        data["seconds"]=data['timestamp']-start_time
        data['timestamp'] = data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f'))
       # data["seconds"] = data["time"].dt.total_seconds()
        data["minutes"] = data["seconds"] / 60
        data["hours"] = data["seconds"] / 3600
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=initial_positions-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        #data["time"]=data["time"].apply(lambda x: str(x).split()[2])
        data=data.sort_index(axis=1, ascending=False)
        print(data)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
               
        # Escribir el DataFrame en la hoja seleccionada
            data.to_excel(writer, sheet_name=sheet_name, index=False)

# Guardar el archivo de Excel
        writer.save()
    except Exception as e:
        print(e)

def custom_timestamp_to_datetime(timestamp):
    # Replace this with the actual conversion logic based on the format
    # For example, if the timestamp represents milliseconds since some point in time:
    seconds = timestamp / 1000.0  # Convert milliseconds to seconds
    return datetime.datetime.fromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f')

def get_position4(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    try:
    # Create a list to collect data
        data_list = []
        devices_list = list(devices_dictionary.values())
        initial_positions = list(map(lambda device: device.Position, devices_list))
        sample_interval = 1.0 / sample_frequency
        sleep_time=sample_interval
        start_time = timeit.default_timer()
        # Mide el tiempo de ejecución
        tiempo_extra = timeit.timeit(lambda: (timeit.default_timer() - start_time) <= execution_time or len(data_list) < execution_time / sample_interval, number=1000) # Ejecuta la función 1000 veces
        data_list.append([start_time] + list(map(lambda device: device.Position, devices_list)))           
        while (timeit.default_timer() - start_time) <= execution_time or len(data_list)<execution_time/sample_interval:
            time.sleep(max(0, sleep_time-tiempo_extra))  # Ensure sleep_time is non-negative
            current_time = timeit.default_timer()
            data_list.append([timeit.default_timer()] + list(map(lambda device: device.Position, devices_list)))
            line_time = timeit.default_timer()
            sleep_time = sample_interval - (line_time - current_time)
        # Create the DataFrame from the collected data
        columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        data = pd.DataFrame(data_list, columns=columns)
        # Calculate additional columns
        data["seconds"] = data['timestamp'] - start_time
        data['timestamp'] = data['timestamp'].apply(custom_timestamp_to_datetime)
        data["minutes"] = data["seconds"] / 60
        data["hours"] = data["seconds"] / 3600
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]] = initial_positions - data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        data=data.sort_index(axis=1, ascending=False)
        print(4)
        print('muestras',execution_time/sample_interval, len(data))
        print(data["seconds"].diff().describe())
        # Save to Excel
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            
    except Exception as e:
        print(e)

def get_position5(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    #Mejor funcion por ahora 
    try:
    # Create a list to collect data
        data_list = []
        devices_list = list(devices_dictionary.values())
        initial_positions = list(map(lambda device: device.Position, devices_list))
        sample_interval = 1.0 / sample_frequency
        sleep_time=sample_interval
        start_time = timeit.default_timer()
        # Mide el tiempo de ejecución
        tiempo_extra = timeit.timeit(lambda: (timeit.default_timer() - start_time) <= execution_time or len(data_list) < execution_time / sample_interval, number=1000) # Ejecuta la función 1000 veces
        data_list.append([datetime.datetime.now()] + initial_positions)           
        while (timeit.default_timer() - start_time) <= execution_time or len(data_list)<execution_time/sample_interval:
            time.sleep(max(0, sleep_time-tiempo_extra))  # Ensure sleep_time is non-negative
            current_time = timeit.default_timer()
            data_list.append([datetime.datetime.now()] + list(map(lambda device: device.Position, devices_list)))
            line_time = timeit.default_timer()
            sleep_time = sample_interval - (line_time - current_time)
        # Create the DataFrame from the collected data
        columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        data = pd.DataFrame(data_list, columns=columns)
        # Calculate additional columns
        data["time"]=data['timestamp']-data_list[0][0]
        data['timestamp']=data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        data["seconds"] = data["time"].dt.total_seconds()
        #data["minutes"] = data["seconds"] / 60
        #data["hours"] = data["seconds"] / 3600
        #data["milliseconds"] = data["time"].dt.total_seconds() * 1000
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=initial_positions-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        data["time"]=data["time"].apply(lambda x: str(x).split()[2])
        data = data.drop("time", axis=1)
        data = data[['timestamp',"seconds"] + [col for col in data.columns if col not in ['timestamp',"seconds"]]]
        print(5)
        print('muestras',execution_time/sample_interval, len(data))
        print(data["seconds"].diff().describe())
        data=data.sort_index(axis=1, ascending=False)
        # Save to Excel
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
        
    except Exception as e:
        print(e)

def get_position6(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    #Mejor funcion por ahora 
    try:
    # Create a list to collect data
        data_list = []
        devices_list = list(devices_dictionary.values())
        initial_positions = list(map(lambda device: device.Position, devices_list))
        sample_interval = 1.0 / sample_frequency
        sleep_time=sample_interval
        start_time = timeit.default_timer()
        new_time=start_time.copy()
        # Mide el tiempo de ejecución
        tiempo_extra = timeit.timeit(lambda: (timeit.default_timer() - start_time) <= execution_time or len(data_list) < execution_time / sample_interval, number=1000) # Ejecuta la función 1000 veces
        data_list.append([datetime.datetime.now()] + initial_positions)           
        while (timeit.default_timer() - start_time) <= execution_time or len(data_list)<execution_time/sample_interval:
            while timeit.default_timer()<(new_time+(sleep_time-tiempo_extra)):
                pass                       
            current_time = timeit.default_timer()
            new_time=timeit.default_timer()
            data_list.append([datetime.datetime.now()] + list(map(lambda device: device.Position, devices_list)))
            line_time = timeit.default_timer()
            sleep_time = sample_interval - (line_time - current_time)
        # Create the DataFrame from the collected data
        columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        data = pd.DataFrame(data_list, columns=columns)
        # Calculate additional columns
        data["time"]=data['timestamp']-data_list[0][0]
        data['timestamp']=data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        data["seconds"] = data["time"].dt.total_seconds()
        #data["minutes"] = data["seconds"] / 60
        #data["hours"] = data["seconds"] / 3600
        #data["milliseconds"] = data["time"].dt.total_seconds() * 1000
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=initial_positions-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        data["time"]=data["time"].apply(lambda x: str(x).split()[2])
        data = data.drop("time", axis=1)
        data = data[['timestamp',"seconds"] + [col for col in data.columns if col not in ['timestamp',"seconds"]]]
        print(5)
        print('muestras',execution_time/sample_interval, len(data))
        print(data["seconds"].diff().describe())
        data=data.sort_index(axis=1, ascending=False)
        # Save to Excel
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
        
    except Exception as e:
        print(e)

def get_position7(devices_dictionary, execution_time, sample_frequency, path,name, sheet_name='1'):
    #Mejor funcion por ahora 
    try:
    # Create a list to collect data
        devices_list = list(devices_dictionary.values())
        sample_interval = 1.0 / sample_frequency
        start_time = time.perf_counter()
        # Encabezados de las columnas
        columnas = ['timestamp','secs'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        # Abre el archivo CSV en modo de escritura
        with open(f'{path}\{name}.csv', mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            # Escribe la primera fila con los encabezados
            escritor_csv.writerow(columnas)
            # Ejecuta el bucle para escribir datos en el archivo CSV
            while (time.perf_counter() - start_time) <= execution_time * 1.1:   
                data_line = [datetime.datetime.now(),time.perf_counter()] + list(map(lambda device: device.Position, devices_list))
                escritor_csv.writerow(data_line)
        # Lee el archivo CSV como un DataFrame de Pandas
        data = pd.read_csv(f'{path}\{name}.csv')
        print(data)
        
        # Calculate additional columns
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data["time"]=data['timestamp']-data['timestamp'].iloc[0]
        data['Seconds_counter']=data['secs'] - data['secs'].iloc[0] ##segundos contados con time.perf_counter
        data["seconds"] = data["time"].dt.total_seconds()
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]].iloc[0]-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        data["time"]=data["time"].apply(lambda x: str(x).split()[2])
        data = data.drop("time", axis=1)
        data = data[['timestamp',"seconds"] + [col for col in data.columns if col not in ['timestamp',"seconds"]]]
        print(5)
        print('muestras',execution_time/sample_interval, len(data))
        print(data["seconds"].diff().describe())
        data=data.sort_index(axis=1, ascending=False)
        diferencia_de_tiempo = data['timestamp'].diff()
        freq=0.01
        umbral_tiempo = pd.Timedelta(seconds=freq)
        data_muestreo=data.copy()
        data_muestreo=data_muestreo[diferencia_de_tiempo >= umbral_tiempo].reset_index(inplace=True)

        data['timestamp']=data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        data_muestreo['timestamp']=data_muestreo['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        # Save to Excel

        with pd.ExcelWriter(f'{path}\{name}.xlsx', engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            data_muestreo.to_excel(writer, sheet_name="muestreo_{freq}", index=False)
        
    except Exception as e:
        print(e)

def get_position1(devices_dictionary, execution_time, sample_frequency, path,name, sheet_name='1'):
    #Mejor funcion por ahora 
    try:
    # Create a list to collect data
        devices_list = list(devices_dictionary.values())
        sample_interval = 1.0 / sample_frequency
        start_time = time.perf_counter()
        # Encabezados de las columnas
        columnas = ['timestamp','secs'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        # Abre el archivo CSV en modo de escritura
        with open(f'{path}\{name}.csv', mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            # Escribe la primera fila con los encabezados
            escritor_csv.writerow(columnas)
            # Ejecuta el bucle para escribir datos en el archivo CSV
            while (time.perf_counter() - start_time) <= execution_time * 1.1:   
                data_line = [datetime.datetime.now(),time.perf_counter()] + list(map(lambda device: device.Position, devices_list))
                escritor_csv.writerow(data_line)
        # Lee el archivo CSV como un DataFrame de Pandas
        data = pd.read_csv(f'{path}\{name}.csv')
        print(data)
        
        # Calculate additional columns
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data["time"]=data['timestamp']-data['timestamp'].iloc[0]
        data['Seconds_counter']=data['secs'] - data['secs'].iloc[0] ##segundos contados con time.perf_counter
        data["seconds"] = data["time"].dt.total_seconds()
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]].iloc[0]-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        data["time"]=data["time"].apply(lambda x: str(x).split()[2])
        data = data.drop("time", axis=1)
        data = data[['timestamp',"seconds"] + [col for col in data.columns if col not in ['timestamp',"seconds"]]]
        print(5)
        print('muestras',execution_time/sample_interval, len(data))
        print(data["seconds"].diff().describe())
        data=data.sort_index(axis=1, ascending=False)
        diferencia_de_tiempo = data['timestamp'].diff()
        freq=0.01
        umbral_tiempo = pd.Timedelta(seconds=freq)
        data_muestreo=data.copy()
        data_muestreo=data_muestreo[diferencia_de_tiempo >= umbral_tiempo].reset_index(inplace=True)


        data['timestamp']=data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        data_muestreo['timestamp']=data_muestreo['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        # Save to Excel

        with pd.ExcelWriter(f'{path}\{name}.xlsx', engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            data_muestreo.to_excel(writer, sheet_name="muestreo_{freq}", index=False)
        
    except Exception as e:
        print(e)

def get_position8(devices_dictionary, execution_time, sample_frequency,final_position, path,name, sheet_name='1'):
    try:
    # Create a list to collect data
        devices_list = list(devices_dictionary.values())
        start_time = time.perf_counter()
        # Encabezados de las columnas
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        # Abre el archivo CSV en modo de escritura
        with open(f'{path}\{name}.csv', mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            # Escribe la primera fila con los encabezados
            escritor_csv.writerow(columnas)
            # Ejecuta el bucle para escribir datos en el archivo CSV
            while True:   
                data_line = [time.perf_counter()] + list(map(lambda device: device.Position, devices_list))
                escritor_csv.writerow(data_line)
                if (time.perf_counter() - start_time) > execution_time*1.1:
                    if (all(Decimal(final_position-0.0008)<valor<=Decimal(final_position)  for valor in map(lambda device: device.Position, devices_list))):
                        break

        # Lee el archivo CSV como un DataFrame de Pandas
        data = pd.read_csv(f'{path}\{name}.csv')
        print(data)
        # Calculate additional columns
        data['seconds']=data['seconds'] - data['seconds'].iloc[0] ##segundos contados con time.perf_counter
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]].iloc[0]-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        print(5)
        print(data["seconds"].diff().describe())
        # Save to Excel
        with pd.ExcelWriter(f'{path}\{name}.xlsx', engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
        
    except Exception as e:
        print(e)

def get_position(devices_dictionary, execution_time, sample_frequency,final_position, path,name, sheet_name='1'):
    try:
        devices_list = list(devices_dictionary.values())
        start_time = time.perf_counter()
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        with open(f'{path}\{name}.csv', mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(columnas)
            while True:   
                data_line = [time.perf_counter()] + list(map(lambda device: device.Position, devices_list))
                escritor_csv.writerow(data_line)
                if (time.perf_counter() - start_time) > execution_time*1.1:
                    if (all(Decimal(final_position-0.0008)<valor<=Decimal(final_position)  for valor in map(lambda device: device.Position, devices_list))):
                        break
        data = pd.read_csv(f'{path}\{name}.csv')
        print(data)
        data['seconds']=data['seconds'] - data['seconds'].iloc[0] #time difference 
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]].iloc[0]-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        print(data["seconds"].diff().describe())
        print(data)
        #data.to_excel(f'{path}\{name}.xlsx')
        writer = openpyxl.Workbook()
        data.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save(f'{path}\{name}.xlsx')

    except Exception as e:
        print(e)

def get_table_position2(devices_dictionary, execution_time,final_position, path,name, sheet_name='1'):
    #Takes 2 of the positions with different methods
    try:
        devices_list = list(devices_dictionary.values())
        start_time = time.perf_counter()
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]+['real_position_D' + elemento for elemento in list(devices_dictionary.keys())]
        take_data=True
        with open(f'{path}\{name}.csv', mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(columnas)
            while take_data:   
                data_line = [time.perf_counter()] + list(map(lambda device: device.Position, devices_list))+ list(map(lambda device: device.DevicePosition, devices_list))
                escritor_csv.writerow(data_line)
                if (time.perf_counter() - start_time) > execution_time*1.1:
                    if (all(Decimal(final_position-0.0008)<valor<=Decimal(final_position)  for valor in map(lambda device: device.Position, devices_list))):
                        take_data=False
        data = pd.read_csv(f'{path}\{name}.csv')
        print(data)
        data['seconds']=data['seconds'] - data['seconds'].iloc[0] #time difference 
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]].iloc[0]-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        print(data["seconds"].diff().describe())
        print(data)
        with pd.ExcelWriter(f'{path}\{name}.xlsx', engine='openpyxl') as writer:
        # Escribir el DataFrame en la hoja seleccionada
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(e)
 

def get_table_position9(devices_dictionary, execution_time,final_position, path,name, sheet_name='1'):
    # this code has problems saving the data in excel
    try:
        data_list = []
        devices_list = list(devices_dictionary.values())
        start_time = time.perf_counter()
        columns = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        take_data=True
        while take_data:   
            data_list.append([time.perf_counter()] + list(map(lambda device: device.Position, devices_list)))
            if (time.perf_counter() - start_time) > execution_time*1.1:
                if (all(Decimal(final_position-0.0008)<valor<=Decimal(final_position)  for valor in map(lambda device: device.Position, devices_list))):
                    take_data=False

        data = pd.DataFrame(data_list, columns=columns)
        data['seconds']=data['seconds'] - data['seconds'].iloc[0] #time difference 
        data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]].iloc[0]-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
        print(data)
        print(data["seconds"].diff().describe())
        
        data.to_csv(f'{path}\{name}.csv', index=False)
        # with pd.ExcelWriter('vale.xlsx', engine='openpyxl') as writer:
        # # Escribir el DataFrame en la hoja seleccionada
        #     data.to_excel(writer, sheet_name=sheet_name, index=False)

    except Exception as e:
        print(e)


def get_table_position(devices_dictionary, execution_time,final_position, path,name, sheet_name='1'):
    try:
        devices_list = list(devices_dictionary.values())
        start_time = time.perf_counter()
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
        take_data=True
        with open(f'{path}\{name}.csv', mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(columnas)
            while take_data:   
                data_line = [time.perf_counter()] + list(map(lambda device: device.Position, devices_list))
                escritor_csv.writerow(data_line)
                if (time.perf_counter() - start_time) > execution_time*1.1:
                    if (all(Decimal(final_position-0.0008)<valor<=Decimal(final_position)  for valor in map(lambda device: device.Position, devices_list))):
                        take_data=False
    except Exception as e:
        print(e)

def read_and_modify_csv_data(path,name):
    try:
        data = pd.read_csv(f'{path}\{name}.csv')
        data['seconds']=data['seconds'] - data['seconds'].iloc[0] #time difference 
        columns_real=[columna for columna in data.columns if 'real_position_' in columna]
        columns_relative = [column.replace('real_position_', 'relative_position_') for column in columns_real]
        data[columns_relative]=data[columns_real].iloc[0]-data[columns_real]
        with pd.ExcelWriter('vale.xlsx', engine='openpyxl') as writer:
        # Escribir el DataFrame en la hoja seleccionada
            data.to_excel(writer, index=False)
    except Exception as e:
        print(e)

def main():
    """The main entry point for the application"""
    try:
        DeviceManagerCLI.BuildDeviceList()
        available_devices = DeviceManagerCLI.GetDeviceList()  # List of available devices
        # Create an instance of ThorlabsDevices
        thorlabs_devices = ThorlabsDevices()
        # Iterate through the list of serial numbers and assign devices
        for serial_number in available_devices:
            thorlabs_devices.connect_device(serial_number)
        
        parameters=[1,0,15] #velocity,initial position,final position
        path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento"

        #CASE 1

        #Step 1: get the parameters
        #velocity,initial_position,final_position,execution_time,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],cycles=None,forward_position=None, waiting_time=None)

      
        # Do the homing and set the velocity
        # Perform homing and place the device in the initial position
        # Initialize tasks in parallel for all the devices
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Execute home_device in parallel for all devices
        #     for device in thorlabs_devices.devices.values():
        #         executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        # print('valelinda2')
        # name=f"{parameters[0]}_{parameters[1]}_{parameters[2]}_solo_csvdata"
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     p3=executor.submit(get_table_position, thorlabs_devices.devices, execution_time,final_position=final_position,path=path,name=name, sheet_name='1')  
        #         # Start the tasks in futures
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
        #     # Wait for all of the tasks to complete
        #     concurrent.futures.wait([p3] + futures)
        # read_and_modify_csv_data(path,name)

        # # Do the desirable movement
        # devices_list=list(thorlabs_devices.devices.values())
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     p1=executor.submit(get_position4, thorlabs_devices.devices, execution_time, sample_frequency=10,path='Ensayo4.xlsx', sheet_name='intento')
        #     p2=executor.submit(get_position, thorlabs_devices.devices, execution_time, sample_frequency=10,path='Ensayo1.xlsx', sheet_name='intento')
        #     p3=executor.submit(shif_device, devices_list[0], final_position,execution_time)
        #     concurrent.futures.wait([p1,p2,p3])

        #Case 2
        velocity,initial_position,final_position,cycles,execution_time,waitTimeout=set_parameters(case=2,velocity=2.4,initial_position=0,final_position=15,cycles=3,forward_position=None, waiting_time=None)
        print(velocity,initial_position,final_position,cycles,execution_time,waitTimeout)
        # Create a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit your tasks to the executor
            futures = [executor.submit(home_device_and_set_velocity, device, initial_position, velocity) for device in thorlabs_devices.devices.values()]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            p3=executor.submit(get_position5, thorlabs_devices.devices, execution_time, sample_frequency=10,path='Ensayohysteresis4.xlsx', sheet_name='intento')
             
                # Start the tasks in futures
            futures = []
            for device in thorlabs_devices.devices.values():
                futures.append(executor.submit(hysteresis, device, initial_position, final_position, cycles,waitTimeout))
            # Wait for all of the tasks to complete
            concurrent.futures.wait([p3] + futures)

        
            

        print('valelinda3')

        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()