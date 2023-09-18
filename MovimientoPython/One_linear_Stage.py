import os
import time
import clr
import System
import pandas as pd
import datetime
import concurrent.futures
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
def home_device(device, initial_position):
    """
    Home a device and move it to an initial position.

    This function homes the specified 'device' and then moves it to the specified 'initial_position'.

    Parameters:
    - device: The device object to be homed and moved.
    - initial_position (Decimal): The initial position to which the device should be moved (Decimal).

    Returns:
    None

    Example:
    To home 'my_device' and move it to an initial position of Decimal('10.0'):
    >>> home_device(my_device, Decimal('10.0'))
    This will home 'my_device' and then move it to the position Decimal('10.0').
    """
    print(f"Homing device with serial number {device.DeviceID}")
    device.Home(60000)  
    print(f"Device with serial number {device.DeviceID} has completed homing.")
    print(f'Moving to initial position {initial_position}')
    device.MoveTo(Decimal(initial_position), 60000) 

def shif_device(device, final_position):
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
    device.MoveTo(final_position)

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
        device.MoveTo(final_position) 
        device.MoveTo(initial_position)



def validate_parameters(case,devices_dictionary,velocity,initial_position,final_position,cycles=None,travel_time=None, waiting_time=None):
    '''
    Validates parameters and performs necessary operations based on the specified case.

    Parameters:
    - case (int): The type of case (1 for shift, 2 for hysteresis, 3 for stress relaxation).
    - devices_dictionary (dict): A dictionary containing device information.
    - velocity (float): Velocity value to set for devices (maximum 2.4 mm/s).
    - initial_position (float): Initial position with a correction of +20 mm.
    - final_position (float): Final position with a correction of -20 mm.
    - cycles (int, optional): Number of cycles (only applicable for case 2).
    - travel_time (float, optional): Travel time (only applicable for case 3).
    - waiting_time (float, optional): Waiting time (only applicable for case 3).

    Returns:
    - Tuple of values based on the specified case:
      - Case 1: initial_position, final_position,execution_time
      - Case 2: (initial_position, final_position, cycles)
      - Case 3: (forward_position, waiting_time)
    
    If there are validation errors, a list of error messages is returned.
    
    '''
 
    initial_position=initial_position+20 #Initial position must be
    final_position=final_position-20

    #Set the velocity to all the devices 
    try:
        for device in devices_dictionary.values():
            device.SetVelocityParams(Decimal(velocity), Decimal(2))
            velocity_parameters = device.GetVelocityParams()
            max_velocity = velocity_parameters.MaxVelocity
            acceleration = velocity_parameters.Acceleration
            print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')
    except Exception as error:
        print(error)

    if case==1:
        execution_time=(initial_position-final_position)/velocity
        return Decimal(initial_position),Decimal(final_position),execution_time
    if case==2:
        return Decimal(initial_position),Decimal(final_position),cycles
    if case==3:
        forward_position=Decimal(travel_time*velocity) #The distance to be moved is calculated according to the time to be moved.
        return forward_position, waiting_time


def get_position(devices_dictionary, execution_time, sample_frequency):
    devices_list=list(devices_dictionary.values())
    # Create an empty DataFrame within capture_values
    columns = ['Timestamp'] + list(devices_dictionary.keys())
    data = pd.DataFrame(columns=columns)
    sample_interval = 1.0 / sample_frequency
    acumulated_time=0
    initial_positions=list(map(lambda device: device.Position,devices_list))
    time_start=datetime.datetime.now()
    
    while acumulated_time<time<execution_time+3: 
        # Execute obtener_posicion_y_hora for each device
        data.loc[len(data)] = [datetime.datetime.now()]+ list(map(lambda device: device.Position,devices_list))
        # Sleep for one second before the next capture
        time.sleep(sample_interval)
        acumulated_time+=sample_interval
        
    data["time"]=data['Timestamp']-time_start
    data["seconds"]= data["time"].hour* 3600 + data["time"].minute * 60 + data["time"].second
    data["minutes"]=data["seconds"]/ 3600
    data["hours"]=data["seconds"]/60
    print(data)

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
        
        parameters=[1,0,20] #velocity,initial position, final position

        #CASE 1
        initial_position,final_position,execution_time=validate_parameters(case=1,devices_dictionary=thorlabs_devices.devices,velocity=parameters[0],initial_position=parameters[0],final_position=parameters[0],cycles=None,travel_time=None, waiting_time=None)

        #Perform homing and place the device in the initial position
        # Initialize tasks in parallel for all the devices
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute home_device in parallel for all devices
            for device in thorlabs_devices.devices.values():
                executor.submit(home_device, device, initial_position)

        #Toma
        devices_list=list(thorlabs_devices.devices.values())
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(shif_device, devices_list, [final_position] * len(devices_list))
            executor.submit(get_position, devices_list, execution_time)
 


        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()