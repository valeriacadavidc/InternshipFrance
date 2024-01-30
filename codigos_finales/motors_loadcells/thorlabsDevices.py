"""
    This code is responsible for creating a class that facilitates the control of motorized linear stages, specifically the MTS25-Z8, 
    controlled by the Thorlabs KDC101 DC Servo Motor Driver. It manages the connection and disconnection to the equipment
    as well as the execution of desired movements.

    Author: Valeria Cadavid Castro
    Last Modified: January 18, 2024
"""

# Importing necessary libraries and modules
import os
import time
import clr
from System import Decimal

# Retrieving the script directory to load the DLL files for handling the motorized stages
script_dir = os.path.dirname(os.path.realpath(__file__))

# Adding references to Thorlabs DLLs
clr.AddReference(os.path.join(script_dir, "DLL_files\\DLL_THORLABS\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll"))
clr.AddReference(os.path.join(script_dir, "DLL_files\\DLL_THORLABS\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll"))
clr.AddReference(os.path.join(script_dir, "DLL_files\\DLL_THORLABS\\Kinesis\\ThorLabs.MotionControl.KCube.DCServoCLI.dll"))

# Importing specific classes and modules from Thorlabs DLLs
from Thorlabs.MotionControl.GenericMotorCLI.Settings import *
from Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor import *
from Thorlabs.MotionControl.GenericMotorCLI.ControlParameters import *
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *

# Importing shared variables from the 'motors_loadcells' module
import process.motors_loadcells.shared_variables as shared_variables


class ThorlabsDevices:
    def __init__(self):
        """
        Initializes a collection of Thorlabs devices.
        """
        self.devices = {} # Dictionary where the devices that are connected are stored

    def detect_devices(self):
        """
        Detects Thorlabs devices connected to the system.

        Returns:
            available_devices: A list of available Thorlabs devices.
        """
        DeviceManagerCLI.BuildDeviceList()
        available_devices = DeviceManagerCLI.GetDeviceList() 
        return available_devices

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
                device.StartPolling(1) #The polling rate is set to 1 ms
                time.sleep(0.25)  # Wait statements are important to allow settings to be sent to the device
                device.EnableDevice()  # Enable the device to receive commands

                # Wait for Settings to Initialize
                if not device.IsSettingsInitialized():
                    device.WaitForSettingsInitialized(10000)  # 10-second timeout
                    assert device.IsSettingsInitialized() is True

                actuator = device.DeviceName # Get device name
                device_info = device.GetDeviceInfo() # Get Device information

                # Before homing or moving the device, ensure the motor's configuration is loaded
                motor_config = device.LoadMotorConfiguration(
                    serial_number, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
                
                if motor_config is not None:
                    device_name = motor_config.DeviceSettingsName
                    print(f"Device connected to the driver {device_name}")

                self.devices[serial_number] = device
                print(f"Device name: {actuator}")
                print(device_info.Description)
                print(f"Connected to the device with serial number {serial_number}\n")
                time.sleep(0.25)
                device.StartPolling(1) #The polling rate is set to 1 ms
                time.sleep(0.25) 
                device.SetBacklash(Decimal(0)) #The backlash value of the device is set to zero
            except Exception as e:
            
                print(f"Failed to connect the device: {str(e)}")
                # Attempt to create a new instance of the KCubeDCServo device
                device = KCubeDCServo.CreateKCubeDCServo(serial_number)
                # Disconnect any existing device connection
                device.Disconnect(True)

    def disconnect_device(self, serial_number):
        """
        Disconnects a Thorlabs device with the given serial number.
        Checks if the specified serial number corresponds to a connected device.
        If the specified serial number is associated with a connected device, it attempts to disconnect the device.
        If the disconnection is successful, the device is removed from the collection of connected devices.

        Args:
            serial_number (str): The serial number of the device to disconnect.

        Raises:
            Exception: If the disconnection fails.
        """
        #Checks if the specified serial number corresponds to a connected device.
        if serial_number in self.devices:
            try:
                # Attempt to disconnect the device associated with the specified serial number
                self.devices[serial_number].Disconnect(True)
                # Remove the disconnected device from the collection
                del self.devices[serial_number]

                print(f"Device disconnected: {serial_number}")
            except Exception as e:
                print(f"Failed to disconnect the device: {str(e)}")

    def homming(self,device):
        """
        Initiates homing for the specified Thorlabs device.

        Args:
            device: The Thorlabs device to perform homing.

        Raises:
            Exception: If homing fails.
        
        Executing this function is important to trust in the movement of the motor, this action is
        recommended by the manufacturers (Thorlabs).
        """
        try:    
            device.Home(60000) 
        except Exception as error:
            print(error)

    def move_initial_position(self,device, initial_position):
        """
        Moves the specified Thorlabs device to the initial position. This posicion is where the experiment starts.

        Args:
            device: The Thorlabs device to be moved.
            initial_position (Decimal): The initial position to which the device should be moved.

        Raises:
            Exception: If the movement fails.
        """
        try:    
            #A speed of 2 mm/s is set to move to the initial position in case the stored position of the device is too low.
            device.SetVelocityParams(Decimal(2), Decimal(1.5))
            print(f'Moving to initial position {initial_position}')
            device.MoveTo(Decimal(initial_position), 60000) 

        except Exception as error:
            print(error)

    def set_velocity(self,device, velocity):
        """
        Sets the velocity parameters for the specified Thorlabs device.

        Args:
            device: The Thorlabs device for which velocity parameters are set.
            velocity: The desired velocity.

        Raises:
            Exception: If setting the velocity parameters fails.
        """
        try:    
            # The desired velocity for the device is set and 4.5 mm/s**2 is used as the acceleration.
            device.SetVelocityParams(Decimal(velocity), Decimal(4.5))

            # This line of code retrieves the current velocity parameters of the device after setting new values. 
            # It allows checking the actual parameters applied to the device.
            velocity_parameters = device.GetVelocityParams() 
            max_velocity = velocity_parameters.MaxVelocity
            acceleration = velocity_parameters.Acceleration
            print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')

        except Exception as error:
            print(error)

    def home_device_and_set_velocity(self,device, initial_position, velocity,polling_rate):
        """
        Performs homing, sets velocity parameters, and moves the Thorlabs device to the initial position.

        Args:
            device: The Thorlabs device to be configured.
            initial_position (Decimal): The initial position to which the device should be moved.
            velocity (Decimal): The desired velocity.
            polling_rate (int): The polling rate in milliseconds.

        Raises:
            Exception: If any of the operations fail.
        """
        try:  
            time.sleep(0.25)
            device.StartPolling(polling_rate)
            time.sleep(0.25) 
            print(f'Polling rate of {polling_rate} ms for de device {device.DeviceID} established')
            print(f"Homing device with serial number {device.DeviceID}")
            device.Home(60000)  
            device.SetVelocityParams(Decimal(2), Decimal(1.5))
            print(f"Device with serial number {device.DeviceID} has completed homing.")
            device.SetBacklash(Decimal(0))
            print(f'Backlash 0 mm established')
            print(f'Moving to initial position {initial_position}')
            device.MoveTo(Decimal(initial_position), 60000) 
            device.SetVelocityParams(Decimal(velocity), Decimal(4.5))
            velocity_parameters = device.GetVelocityParams()
            max_velocity = velocity_parameters.MaxVelocity
            acceleration = velocity_parameters.Acceleration
            print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')
            
        except Exception as error:
            print(error)

    def shif_device(self,device, final_position,waitTimeout):
        """
        Moves the specified Thorlabs device to a specified final position.

        Args:
            device: The Thorlabs device to be moved.
            final_position: The final position to which the device should be moved.
            wait_timeout (int): The wait timeout in milliseconds. The function will wait until the move completes or the timeout elapses, whichever comes first. 

        Raises:
            Exception: If the movement fails.

        """
        try:
            device.MoveTo(Decimal(final_position),waitTimeout)
        except Exception as e:
            print(e)

    def hysteresis_fp(self,device,initial_position, final_position, cycles,waitTimeout):
        """
        This function moves the device back and forth between the specified
        initial_position and final_position for the specified number of cycles.
        The key distinction from the function 'hysteresis' is that this function
        does not return the linear stage to the initial position in the last cycle; it concludes
        at the final position.

        Args:
            device: The Thorlabs device to be tested.
            initial_position: The initial position where the hysteresis test starts.
            final_position: The maximum shift to be achieved during each cycle.
            cycles (int): The number of cycles to perform.
            wait_timeout (int): The maximum time to wait for the device to reach the target position in miliseconds.

        Raises:
            Exception: If the hysteresis test fails.
        """
        try:
            #The loop iterates through the specified number of cycles
            for i in range(cycles):
                # Wait until the device is no longer moving
                if i < cycles - 1:
                    # Wait until the device is no longer moving
                    while (device.Status.IsMoving):
                        pass
                    # Move the device to the specified final position
                    device.MoveTo(Decimal(final_position),waitTimeout)
                    # Wait until the device completes the movement
                    while (device.Status.IsMoving):
                        pass
                    # Move the device back to the initial position
                    device.MoveTo(Decimal(initial_position),waitTimeout)
                else:
                    # In the last cycle, move the device to the specified final position and end the movement
                    while (device.Status.IsMoving):
                        pass
                    # Move the device back to the initial position
                    device.MoveTo(Decimal(final_position),waitTimeout)

        except Exception as e:
            print(e)
    
    def hysteresis(self,device,initial_position, final_position, cycles,waitTimeout):
        """
        Performs a hysteresis test on the specified Thorlabs device.
        This function moves the device back and forth between the specified
        initial_position and final_position for the specified number of cycles.

        Args:
            device: The Thorlabs device to be moved.
            initial_position: The initial position where the hysteresis test starts.
            final_position: The shift to be achieved during each cycle.
            cycles (int): The number of cycles to perform.
            wait_timeout (int): The maximum time to wait for the device to reach the target position.

        Raises:
            Exception: If the hysteresis test fails.
        """
        try:
            #The loop iterates through the specified number of cycles
            for _ in range(cycles):
                # Wait until the device is no longer moving
                while (device.Status.IsMoving):
                    pass
                # Move the device to the specified final position
                device.MoveTo(Decimal(final_position),waitTimeout)
                # Wait until the device completes the movement
                while (device.Status.IsMoving):
                    pass
                # Move the device back to the initial position
                device.MoveTo(Decimal(initial_position),waitTimeout)

        except Exception as e:
            print(e)

    def stress_relaxation(self,device,initial_position,forward_position, waiting_time, cycles,waitTimeout):
        """
        Perform a stress relaxation test on the specified Thorlabs device.

        Args:
            device: The device object to be tested.
            initial_position: The initial position for the stress relaxation test.
            forward_position: The shift to be achieved during each cycle.
            waiting_time (float): The time to wait after each movement.
            cycles (int): The number of cycles to perform.
            waitTimeout (int): The maximum time to wait for the device to reach the target position.

        Raises:
            None

        This function performs a stress relaxation test on the specified device, moving it back and forth between the
        initial position and positions obtained by subtracting multiples of 'forward_position'. The test is performed for
        the specified number of cycles, and a waiting time is introduced after each movement.

        """
        try:
            for cycle in range(1,cycles+1,1):
                 # Move the device to the adjusted position for the current cycle
                device.MoveTo(Decimal(initial_position-forward_position*cycle),waitTimeout)

                # Wait until the device completes the movement
                while (device.Status.IsMoving):
                    pass

                # Record the start time to measure waiting time
                start_time = time.perf_counter()
                # Wait for the specified waiting time or until a signal is received to stop the test
                while  time.perf_counter()-start_time < waiting_time:
                    save_data=shared_variables.get_movements_motors_completed()
                    if save_data:
                        # If save data is True stops the execution of the function
                        return
                    else:
                        pass
                
        except Exception as e:
            print(e)
    
    def stress_relaxation_time_sleep(self,device,initial_position,forward_position, waiting_time, cycles,waitTimeout):
        """
        Perform a stress relaxation test on the specified Thorlabs device.

        Args:
            device: The device object to be tested.
            initial_position: The initial position for the stress relaxation test.
            forward_position: The maximum shift to be achieved during each cycle.
            waiting_time (float): The time to wait after each movement.
            cycles (int): The number of cycles to perform.
            waitTimeout (int): The maximum time to wait for the device to reach the target position.

        Raises:
            None

        This function performs a stress relaxation test on the specified device, moving it back and forth between the
        initial position and positions obtained by subtracting multiples of 'forward_position'. The test is performed for
        the specified number of cycles, and a waiting time is introduced after each movement.

        """
        try:
            for cycle in range(1,cycles+1,1):
                # Move the device to the adjusted position for the current cycle
                device.MoveTo(Decimal(initial_position-forward_position*cycle),waitTimeout)
                # Introduce a waiting time using time.sleep
                time.sleep(waiting_time)
        except Exception as e:
            print(e)

