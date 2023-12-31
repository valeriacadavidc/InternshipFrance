
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
import schedule

import sched

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


def set_parameters(case,velocity,initial_position,final_position,frequency,cycles=None,forward_position=None, waiting_time=None):
    '''
    DOCUMENTACntar 
    
    '''
 
    initial_position=initial_position+25 #Initial position must be
    final_position=25-final_position
    if case==1:
        execution_time=(initial_position-final_position)/velocity
        num_samples = int((execution_time * 1.1 + 16) * frequency)
        print(num_samples)
        sample_time = 1 / frequency
        waitTimeout=execution_time*1000+2000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        return velocity,initial_position,final_position,sample_time,num_samples,execution_time,waitTimeout
    if case==2:
        execution_time= (initial_position-final_position)*2*cycles/velocity
        waitTimeout=(initial_position-final_position)*1000/velocity+2000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)#Duration to wait for command to execute
        return velocity,initial_position,final_position,cycles,execution_time,waitTimeout
    if case==3:
        if forward_position*cycles>20:
            cycles-=1
        execution_time=(forward_position/velocity+waiting_time)*cycles
        print(execution_time)
        waitTimeout=forward_position/velocity*1000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        return velocity,forward_position, waiting_time,cycles,execution_time,waitTimeout
    
def get_table_position(devices_dictionary, execution_time,final_position, path,name):
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


def collect_data(devices_dictionary,frequency,execution_time,path,name):
    # Get the list of devices from the dictionary
    devices_list = list(devices_dictionary.values())
    data_list = []
    num_samples= int((execution_time*1.11+4)*frequency)+100 #Use esta formula haciendo una interpolacion entre dos retrasos
    sample_time=1/frequency
    def write_timestamp():
        """This function writes a timestamp and device positions to the data list."""
        data_line = [time.perf_counter()] + list(map(lambda device: device.Position, devices_list))
        data_list.append(data_line)
    # Create a scheduler object.
    scheduler = schedule.Scheduler()
    # Schedule the function `write_timestamp()` to run every 0.01 seconds for the specified duration.
    scheduler.every(sample_time).seconds.do(write_timestamp)
    # Start the scheduler.
    scheduler.run_pending()
    while len(data_list) < num_samples + 1 :
        scheduler.run_pending()
    # Stop the scheduler.
    scheduler.clear()
    # Create column names for the DataFrame
    columnas = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]

    # Convert the data list to a DataFrame
    data = pd.DataFrame(data_list, columns=columnas)
    data['seconds']=data['seconds'] - data['seconds'].iloc[0] #time difference 
    columns_real=[columna for columna in data.columns if 'real_position_' in columna]
    columns_relative = [column.replace('real_position_', 'relative_position_') for column in columns_real]
    data[columns_relative]=data[columns_real].iloc[0]-data[columns_real]
    data.to_csv('{path}\{name}_all.csv'.format(path=path,name=name), index=False)

def collect_data1(devices_dictionary,frequency,execution_time,path,name):
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses schedule.Scheduler()
    # Get the list of devices from the dictionary
    # Get the list of devices from the dictionary
    devices_list = list(devices_dictionary.values())
    sample = 0
    num_samples = int((execution_time * 1.1 + 16) * frequency) + 30
    print(num_samples)
    sample_time = 1 / frequency
    columnas = ['seconds'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
    # Create a CSV file to save the data
    csv_file_path = f'{path}\\{name}.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header to the CSV file
        csv_writer.writerow(columnas)
        def write_timestamp():
            """This function writes a timestamp and device positions to the CSV file."""
            nonlocal sample
            data_line = [time.perf_counter()] + [device.Position for device in devices_list]
            sample += 1
            csv_writer.writerow(data_line)
        # Create a scheduler object.
        scheduler = schedule.Scheduler()
        # Schedule the function `write_timestamp()` to run every 0.01 seconds for the specified duration.
        scheduler.every(sample_time).seconds.do(write_timestamp)
        # Start the scheduler.
        scheduler.run_pending()
        while sample < num_samples:
            scheduler.run_pending()
        # Stop the scheduler.
        scheduler.clear()
    data = pd.read_csv(f'{path}\{name}.csv')
    data['seconds']=data['seconds'] - data['seconds'].iloc[0] 
    data.to_csv('{path}\{name}_all.csv'.format(path=path,name=name), index=False)

def collect_data2(devices_dictionary, sample_time, num_samples, path, name):
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses sched
    try:
        # Get the list of devices from the dictionary
        devices_list = list(devices_dictionary.values())
        sample = 0
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in devices_dictionary.keys()]
        # Create a CSV file to save the data
        csv_file_path = f'{path}\\{name}.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header to the CSV file
            csv_writer.writerow(columnas)
            def write_timestamp(sc):
                """This function writes a timestamp and device positions to the CSV file."""
                nonlocal sample
                data_line = [time.perf_counter()] + [device.Position for device in devices_list]
                sample += 1
                csv_writer.writerow(data_line)
                if sample < num_samples:
                    sc.enter(sample_time, 1, write_timestamp, (sc,))

            # Create a scheduler object
            s = sched.scheduler(time.perf_counter, time.sleep)
            # Schedule the function `write_timestamp()` to run immediately and then repeatedly every sample_time seconds
            s.enter(0, 1, write_timestamp, (s,))
            s.run()
        print('valelinda')
        data = pd.read_csv(f'{path}\\{name}.csv')
        data['seconds'] = data['seconds'] - data['seconds'].iloc[0]
        data.to_csv(f'{path}\\{name}_all.csv', index=False)
    except Exception as e:
        # Handle the exception here, you can print an error message or log it
        print(f"An exception occurred: {str(e)}")
        data = pd.read_csv(f'{path}\\{name}.csv')
        data['seconds'] = data['seconds'] - data['seconds'].iloc[0]
        data.to_csv(f'{path}\\{name}_all.csv', index=False)

def read_and_modify_csv_data(path,name):
    try:
        data = pd.read_csv(f'{path}\{name}.csv')
        data['seconds']=data['seconds'] - data['seconds'].iloc[0] #time difference 
        columns_real=[columna for columna in data.columns if 'real_position_' in columna]
        columns_relative = [column.replace('real_position_', 'relative_position_') for column in columns_real]
        data[columns_relative]=data[columns_real].iloc[0]-data[columns_real]
        data.to_csv('{path}\{name}_all.csv'.format(path=path,name=name), index=False)
        # with pd.ExcelWriter('{path}\{name}.xlsx'.format(path=path,name=name), engine='openpyxl') as writer:
        # # Escribir el DataFrame en la hoja seleccionada
        #     data.to_excel(writer, index=False)
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
        
        # parameters=[2,0,15] #velocity,initial position,final position
        # path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento"

        # #CASE 1

        # #Step 1: get the parameters
        # velocity,initial_position,final_position,execution_time,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],cycles=None,forward_position=None, waiting_time=None)
      
        # # Do the homing and set the velocity
        # # Perform homing and place the device in the initial position
        # # Initialize tasks in parallel for all the devices
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Execute home_device in parallel for all devices
        #     for device in thorlabs_devices.devices.values():
        #         executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        # print('valelinda2')
        # name=f"v_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_intentoguardarcsv"
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     p1=executor.submit(get_table_position, thorlabs_devices.devices, execution_time,final_position=final_position,path=path,name=name, sheet_name='1')  
        #         # Start the tasks in futures
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
        #     # Wait for all of the tasks to complete
        #     concurrent.futures.wait([p1] + futures)

        # read_and_modify_csv_data(path,name)

        #Codigo repetibilidad con cliclo

        # parameters=[1,0,6.25] #velocity,initial position,final position
        # path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541"
        # for i in range(20):
        #     velocity,initial_position,final_position,execution_time,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],cycles=None,forward_position=None, waiting_time=None)
        #     # Do the homing and set the velocity
        #     # Perform homing and place the device in the initial position
        #     # Initialize tasks in parallel for all the devices
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         # Execute home_device in parallel for all devices
        #         for device in thorlabs_devices.devices.values():
        #             executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        #     name=f"v_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_rep_{i}"
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #         p1=executor.submit(get_table_position, thorlabs_devices.devices, execution_time,final_position=final_position,path=path,name=name)  
        #             # Start the tasks in futures
        #         futures = []
        #         for device in thorlabs_devices.devices.values():
        #             futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
        #         # Wait for all of the tasks to complete
        #         concurrent.futures.wait([p1] + futures)
        #     read_and_modify_csv_data(path,name)
        #     print(f'Fin ciclo {i}')

        #codigo para hacer repetibilidad iniciando bajo las mismas condiciones (reiniciando computador y apagando equipos) con la funcion que usa un scheduler para tomar los datos

        parameters1=[0.01,0,0.25,10] #velocity,initial position,final position, frecuency
        parameters2=[0.01,0,6.25,50] #velocity,initial position,final position, frecuency
        parameters3=[0.01,0,12.5,100] #velocity,initial position,final position, frecuency
        parameters4=[1,0,0.25,50] #velocity,initial position,final position, frecuency
        parameters5=[1,0,6.25,100] #velocity,initial position,final position, frecuency
        parameters6=[1,0,12.5,10] #velocity,initial position,final position, frecuency
        parameters7=[2,0,0.25,100] #velocity,initial position,final position, frecuency
        parameters8=[2,0,6.25,10] #velocity,initial position,final position, frecuency
        parameters9=[2,0,12.5,200] #velocity,initial position,final position, frecuency

        parameters=parameters9

        #path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541_iguales_condiciones"
        path=r"C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Taguchi_funcion_schedule\\3"
        #for i in range(20): if I want to run
        velocity,initial_position,final_position,sample_time,num_samples,execution_time,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],frequency=parameters[3],cycles=None,forward_position=None, waiting_time=None)
        # Do the homing and set the velocity
        # Perform homing and place the device in the initial position
        # Initialize tasks in parallel for all the devices
        i=13
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute home_device in parallel for all devices
            for device in thorlabs_devices.devices.values():
                executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        name=f"v_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_freq_{parameters[3]}_rep_{i}_sched_25minutes"
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            #p1=executor.submit(collect_data1,thorlabs_devices.devices,frequency=parameters[3],execution_time=execution_time,path=path,name=name)
            p1=executor.submit(collect_data2,thorlabs_devices.devices,sample_time=sample_time, num_samples=num_samples, path=path, name=name)
                # Start the tasks in futures
            futures = []
            for device in thorlabs_devices.devices.values():
                futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
            # Wait for all of the tasks to complete
            concurrent.futures.wait([p1] + futures)
        print(f'Fin ciclo {i}')


        print('valelinda3')
        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()