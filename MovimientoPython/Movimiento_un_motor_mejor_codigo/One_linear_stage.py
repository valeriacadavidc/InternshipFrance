
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
        num_samples = int((execution_time * 1.1 + 16) * frequency)
        sample_time = 1 / frequency
        return velocity,initial_position,final_position,cycles,sample_time,num_samples,execution_time,waitTimeout
    if case==3:
        if forward_position*cycles>20:
            cycles-=1
        execution_time=(forward_position/velocity+waiting_time)*cycles
        print(execution_time)
        waitTimeout=forward_position/velocity*1000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        return velocity,forward_position, waiting_time,cycles,execution_time,waitTimeout

def collect_data(devices_dictionary, sample_time, num_samples, path, name):
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
                initial_time=time.perf_counter()
                nonlocal sample
                data_line = [time.perf_counter()] + [device.Position for device in devices_list]
                sample += 1
                csv_writer.writerow(data_line)
                if sample < num_samples:
                    delay=time.perf_counter()-initial_time
                    sc.enter(sample_time-delay, 1, write_timestamp, (sc,))
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
        
        #Caso 1 shift device
        #codigo para hacer repetibilidad iniciando bajo las mismas condiciones (reiniciando computador y apagando equipos) con la funcion que usa un scheduler para tomar los datos

        # parameters1=[0.01,0,0.25,10] #velocity,initial position,final position, frecuency
        # parameters2=[0.01,0,6.25,50] #velocity,initial position,final position, frecuency
        # parameters3=[0.01,0,12.5,100] #velocity,initial position,final position, frecuency
        # parameters4=[1,0,0.25,50] #velocity,initial position,final position, frecuency
        # parameters5=[1,0,6.25,100] #velocity,initial position,final position, frecuency
        # parameters6=[1,0,12.5,10] #velocity,initial position,final position, frecuency
        # parameters7=[2,0,0.25,100] #velocity,initial position,final position, frecuency
        # parameters8=[2,0,6.25,10] #velocity,initial position,final position, frecuency
        # parameters9=[2,0,12.5,50] #velocity,initial position,final position, frecuency
        # parameters=parameters9
        # #path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541_iguales_condiciones"
        # path=r"C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Taguchi_funcion_schedule\\9"
        # #for i in range(20): if I want to run
        # velocity,initial_position,final_position,sample_time,num_samples,execution_time,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],frequency=parameters[3],cycles=None,forward_position=None, waiting_time=None)
        # # Do the homing and set the velocity
        # # Perform homing and place the device in the initial position
        # # Initialize tasks in parallel for all the devices
        # i=5
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Execute home_device in parallel for all devices
        #     for device in thorlabs_devices.devices.values():
        #         executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        # name=f"v_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_freq_{parameters[3]}_rep_{i}"
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     p1=executor.submit(collect_data,thorlabs_devices.devices,sample_time=sample_time, num_samples=num_samples, path=path, name=name)
        #     # Start the tasks in futures
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
        #     # Wait for all of the tasks to complete
        #     concurrent.futures.wait([p1] + futures)
        # print(f'Fin ciclo shif device {i}')
        # print('valelinda3')

        #Caso 2 histeresis

        parameters=[0.25,0,2.5,20,10] #velocity,initial position,final position, frecuency, ciclos
    
        #path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541_iguales_condiciones"
        path=r"C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Intentos_precarga"
        #for i in range(20): if I want to run
        velocity,initial_position,final_position,cycles,sample_time,num_samples,execution_time,waitTimeout=set_parameters(case=2,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],frequency=parameters[3],cycles=parameters[4],forward_position=None, waiting_time=None)
        # Do the homing and set the velocity
        # Perform homing and place the device in the initial position
        # Initialize tasks in parallel for all the devices
        i=1
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute home_device in parallel for all devices
            for device in thorlabs_devices.devices.values():
                executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        name=f"histeresis_v_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_freq_{parameters[3]}_ciclos_{parameters[4]}_intento_{i}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            p1=executor.submit(collect_data,thorlabs_devices.devices,sample_time=sample_time, num_samples=num_samples, path=path, name=name)
            # Start the tasks in futures
            futures = []
            for device in thorlabs_devices.devices.values():
                futures.append(executor.submit(hysteresis,device,initial_position, final_position, cycles,waitTimeout))
            # Wait for all of the tasks to complete
            concurrent.futures.wait([p1] + futures)
        print(f'Fin ciclo {i}')
        print('valelinda3')


        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    #cProfile.run("main()", sort='cumulative')
    main()