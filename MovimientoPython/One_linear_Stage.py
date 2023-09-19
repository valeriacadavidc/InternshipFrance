import os
import time
import clr
import System
import pandas as pd
import openpyxl
import datetime
import concurrent.futures
import ctypes
import threading


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
            print(f"Device with serial number {device.DeviceID} has completed homing.")
            print(f'Moving to initial position {initial_position}')
            device.MoveTo(Decimal(initial_position), 60000) 
            device.SetVelocityParams(Decimal(velocity), Decimal(2))
            velocity_parameters = device.GetVelocityParams()
            max_velocity = velocity_parameters.MaxVelocity
            acceleration = velocity_parameters.Acceleration
            print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')
    except Exception as error:
        print(error)

def shif_device(device, final_position,execution_time):
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
        device.MoveTo(Decimal(final_position),100000)
    except Exception as e:
        print(e)

def hysteresis(device,initial_position, final_position, cycles):
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
    for _ in range(cycles):
        device.MoveTo(Decimal(final_position)) 
        device.MoveTo(Decimal(initial_position))

def stress_relaxation(device,foward_position, waiting_time, cycles):
    for _ in range(cycles):
        device.MoveBy(-foward_position)
        time.sleep(waiting_time)



def set_parameters(case,velocity,initial_position,final_position,cycles=None,forward_position=None, waiting_time=None):
    '''
    DOCUMENTACntar 
    
    '''
 
    initial_position=initial_position+20 #Initial position must be
    final_position=20-final_position
    if case==1:
        execution_time=(initial_position-final_position)/velocity
        print(execution_time)
        return velocity,initial_position,final_position,execution_time
    if case==2:
        execution_time= (initial_position-final_position)*2*cycles/velocity
        print(execution_time)
        return velocity,initial_position,final_position,cycles,execution_time
    if case==3:
        if forward_position*cycles>20:
            cycles-=1
        execution_time=(forward_position/velocity+waiting_time)*cycles
        print(execution_time)
        return velocity,forward_position, waiting_time,cycles,execution_time


def get_position(devices_dictionary, execution_time, sample_frequency, path, sheet_name):
    devices_list=list(devices_dictionary.values())
    initial_positions=list(map(lambda device: device.Position,devices_list))
    # Create an empty DataFrame within capture_values
    columns = ['timestamp'] + ['real_position_' + elemento for elemento in list(devices_dictionary.keys())]
    data = pd.DataFrame(columns=columns)
    sample_interval = 1.0 / sample_frequency
    acumulated_time=0   
    time_start=datetime.datetime.now()
    while acumulated_time<execution_time+3: 
        # Execute obtener_posicion_y_hora for each device
        data.loc[data.shape[0]] = [datetime.datetime.now()]+ list(map(lambda device: device.Position,devices_list))
        # Sleep for one second before the next capture
        time.sleep(sample_interval)
        acumulated_time+=sample_interval
    
    data["time"]=data['timestamp']-time_start
    data["seconds"] = data["time"].dt.total_seconds()
    data["minutes"] = data["seconds"] / 60
    data["hours"] = data["seconds"] / 3600
    data["milliseconds"] = data["time"].dt.total_seconds() * 1000
    data[['relative_position_' + elemento for elemento in list(devices_dictionary.keys())]]=initial_positions-data[['real_position_' + elemento for elemento in list(devices_dictionary.keys())]]
    data["time"]=data["time"].apply(lambda x: str(x).split()[2])
    data=data.sort_index(axis=1, ascending=False)
    print(data)# Cargar el archivo de Excel existente
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
    
        # Escribir el DataFrame en la hoja seleccionada
        data.to_excel(writer, sheet_name=sheet_name, index=False)

# Guardar el archivo de Excel
    writer.save()


    # Return the DataFrame once capture_values has finished

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
        
        parameters=[1,0,10] #velocity,initial position,final position

        #CASE 1

        #Step 1: get the parameters
        velocity,initial_position,final_position,execution_time=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],cycles=None,forward_position=None, waiting_time=None)

      
        #Do the homing and set the velocity
        #Perform homing and place the device in the initial position
        # Initialize tasks in parallel for all the devices
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute home_device in parallel for all devices
            for device in thorlabs_devices.devices.values():
                executor.submit(home_device_and_set_velocity, device, initial_position, velocity)
        print('valelinda2')
       
        #Do the desirable movement
        devices_list=list(thorlabs_devices.devices.values())
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(get_position, thorlabs_devices.devices, execution_time, sample_frequency=10,path='ensayo1.xlsx', sheet_name='intento1')
            executor.submit(shif_device, devices_list[0], final_position,execution_time)

        print('valelinda3')

        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()