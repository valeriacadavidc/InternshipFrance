
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

        #Codigo repetibilidad

        parameters=[1,0,6.25] #velocity,initial position,final position
        path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541"
        for i in range(20):
            velocity,initial_position,final_position,execution_time,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],cycles=None,forward_position=None, waiting_time=None)
            # Do the homing and set the velocity
            # Perform homing and place the device in the initial position
            # Initialize tasks in parallel for all the devices
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Execute home_device in parallel for all devices
                for device in thorlabs_devices.devices.values():
                    executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
            name=f"v_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_rep_{i}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                p1=executor.submit(get_table_position, thorlabs_devices.devices, execution_time,final_position=final_position,path=path,name=name)  
                    # Start the tasks in futures
                futures = []
                for device in thorlabs_devices.devices.values():
                    futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
                # Wait for all of the tasks to complete
                concurrent.futures.wait([p1] + futures)
            read_and_modify_csv_data(path,name)
            print(f'Fin ciclo {i}')



        print('valelinda3')
        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()