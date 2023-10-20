"""An example that uses the .NET Kinesis Libraries to connect to a KDC."""
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
                
                # Get device name
                device_name = device.DeviceName
                print(f"Device name: {device_name}")
            
                # Get Device information
                device_info = device.GetDeviceInfo()
                print(device_info.Description)

                # Before homing or moving the device, ensure the motor's configuration is loaded
                motor_config = device.LoadMotorConfiguration(
                    serial_number, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)

                if motor_config is not None:
                    actuator = motor_config.DeviceSettingsName
                    print(f"Device connected to the driver {actuator}")
                self.devices[serial_number] = device

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
def home_device(device):
    print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    print(f"Homing device with serial number {device.DeviceID}")
    device.Home(60000)  # 6s timeout, blocking call
    print(f"Device with serial number {device.DeviceID} has completed homing.")
    position = 25
    d = Decimal(position)
    print(f'Device Homed. Moving to position {position}')
    device.MoveTo(d, 60000)  # 6s timeout again
    print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))



#TO do a cycle for moving the device
def movement(device, step_size, wait_time, cycles):
    print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    cycle = 0
    while cycle < cycles:
        device.MoveTo(device.Position - Decimal(step_size), 60000)
        time.sleep(wait_time)
        cycle += 1
    print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))



def capture_values(devices, final_time):
    # Create an empty DataFrame within capture_values
    columns = ['Timestamp'] + list(devices.devices.keys())
    data = pd.DataFrame(columns=columns)
    accumulated_time = 0
    time_start=datetime.datetime.now()
    while accumulated_time < final_time + 3: 
        # Execute obtener_posicion_y_hora for each device
        results = [datetime.datetime.now()-time_start]+ list(map(lambda device: device.Position, list(devices.devices.values())))
        data.loc[len(data)] = results
        # Sleep for one second before the next capture
        accumulated_time += 0.004
        time.sleep(0.004)

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
   
        # Initialize tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute home_device in parallel for all devices
            for device in thorlabs_devices.devices.values():
                executor.submit(home_device, device)

        with concurrent.futures.ThreadPoolExecutor() as executor_1:
            executor_1.submit(capture_values, devices=thorlabs_devices, final_time=20)
            for device in thorlabs_devices.devices.values():
                executor_1.submit(movement, device, step_size=3, wait_time=3, cycles=3)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for device in thorlabs_devices.devices.values():
        #         executor.submit(movement, device, step_size=3, wait_time=3, cycles=3)

        # for device in thorlabs_devices.devices.values():
        #     device.SetVelocityParams(Decimal(0.008), Decimal(4.5))
        #     velocity_parameters = device.GetVelocityParams()
        #     max_velocity = velocity_parameters.MaxVelocity
        #     acceleration = velocity_parameters.Acceleration
        #     print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')
        
        # with concurrent.futures.ThreadPoolExecutor() as executor_1:
        #     executor_1.submit(capture_values, devices=thorlabs_devices, final_time=20)
        #     for device in thorlabs_devices.devices.values():
        #         executor_1.submit(movement, device, step_size=0.2, wait_time=3, cycles=3)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for device in thorlabs_devices.devices.values():
        #         executor.submit(movement, device, step_size=1, wait_time=3, cycles=3)
      

        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()


# class ThorlabsDevices:
#     def __init__(self):
#         """
#         Initializes a collection of Thorlabs devices.
#         """
#         self.devices = {}

#     def connect_device(self, serial_number):
#         """
#         Connects a Thorlabs device with the given serial number.

#         Args:
#             serial_number (str): The serial number of the device to connect.

#         Raises:
#             Exception: If the connection or initialization fails.
#         """
#         if serial_number not in self.devices:
#             try:
#                 # Attempt to create and connect the KCubeDCServo device
#                 device = KCubeDCServo.CreateKCubeDCServo(serial_number)
#                 device.Connect(serial_number)
#                 time.sleep(0.25)
#                 device.StartPolling(250)
#                 time.sleep(0.25)  # Wait statements are important to allow settings to be sent to the device
#                 device.EnableDevice()

#                 # Wait for Settings to Initialize
#                 if not device.IsSettingsInitialized():
#                     device.WaitForSettingsInitialized(10000)  # 10-second timeout
#                     assert device.IsSettingsInitialized() is True
                
#                 # Get device name
#                 device_name = device.DeviceName
#                 print(f"Device name: {device_name}")
            
#                 # Get Device information
#                 device_info = device.GetDeviceInfo()
#                 print(device_info.Description)

#                 # Before homing or moving the device, ensure the motor's configuration is loaded
#                 motor_config = device.LoadMotorConfiguration(
#                     serial_number, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)

#                 if motor_config is not None:
#                     actuator = motor_config.DeviceSettingsName
#                     print(f"Device connected to the driver {actuator}")
#                 self.devices[serial_number] = device

#                 print(f"Connected to the device with serial number {serial_number}\n")
#             except Exception as e:
#                 print(f"Failed to connect the device: {str(e)}")

#     def disconnect_device(self, serial_number):
#         """
#         Disconnects a Thorlabs device with the given serial number.

#         Args:
#             serial_number (str): The serial number of the device to disconnect.

#         Raises:
#             Exception: If the disconnection fails.
#         """
#         if serial_number in self.devices:
#             try:
#                 self.devices[serial_number].Disconnect(True)
#                 del self.devices[serial_number]
#                 print(f"Device disconnected: {serial_number}")
#             except Exception as e:
#                 print(f"Failed to disconnect the device: {str(e)}")

# # Define a function to home a single device
# def home_device(device):
#     print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
#     print(f"Homing device with serial number {device.DeviceID}")
#     device.Home(60000)  # 6s timeout, blocking call
#     print(f"Device with serial number {device.DeviceID} has completed homing.")
#     position = 25
#     d = Decimal(position)
#     print(f'Device Homed. Moving to position {position}')
#     device.MoveTo(d, 60000)  # 6s timeout again
#     print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

# def movement(device, step_size, wait_time, cycles):
#     print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
#     cycle = 0
#     while cycle < cycles:
#         device.MoveTo(device.Position - Decimal(step_size), 60000)
#         time.sleep(wait_time)
#         cycle += 1
#     print('\n', f'{device.DeviceID}: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

# def get_position():
#     pass

# def capture_values(devices, final_time):
#     # Crear un DataFrame vacío dentro de capture_values
#     columnas=['Timestamp']+list(devices.devices.keys())
#     data = pd.DataFrame(columns=columnas)
#     acumulated_time=0
#     while acumulated_time<final_time+2: 
#         # Ejecutar obtener_posicion_y_hora para cada dispositivo
#         resultados = [pd.Timestamp.now()]+ list(map(lambda device: device.Position,list(devices.devices.values())))
#         data.loc[len(data)] = resultados
#         # Dormir durante un segundo antes de la próxima captura
#         acumulated_time+=1
#         time.sleep(1)

#     print(data)

#     # Retornar el DataFrame una vez que capture_values haya terminado
 

# def main():
#     """The main entry point for the application"""
#     try:
#         DeviceManagerCLI.BuildDeviceList()
#         available_devices = DeviceManagerCLI.GetDeviceList()  # List of available devices

#         # Create an instance of ThorlabsDevices
#         thorlabs_devices = ThorlabsDevices()

#         # Iterate through the list of serial numbers and assign devices
#         for serial_number in available_devices:
#             thorlabs_devices.connect_device(serial_number)
   
#         # Initialize tasks in parallel
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # Execute home_device in parallel for all devices
#             for device in thorlabs_devices.devices.values():
#                 executor.submit(home_device, device)



#         with concurrent.futures.ThreadPoolExecutor() as executor_1:
#             executor_1.submit(capture_values,devices=thorlabs_devices, final_time=20)
#             for device in thorlabs_devices.devices.values():
#                 executor_1.submit(movement, device, step_size=3, wait_time=3, cycles=3)


#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             for device in thorlabs_devices.devices.values():
#                 executor.submit(movement, device, step_size=3, wait_time=3, cycles=3)

#         for device in thorlabs_devices.devices.values():
#             device.SetVelocityParams(Decimal(0.1), Decimal(4.5))
#             velocity_parameters = device.GetVelocityParams()
#             max_velocity = velocity_parameters.MaxVelocity
#             acceleration = velocity_parameters.Acceleration
#             print(f'\n{device.DeviceID}\nMaximum Velocity: {max_velocity}, Acceleration: {acceleration}')

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             for device in thorlabs_devices.devices.values():
#                 executor.submit(movement, device, step_size=1, wait_time=3, cycles=3)

#         for device in list(thorlabs_devices.devices.values()):
#             thorlabs_devices.disconnect_device(device.DeviceID)

#     except Exception as e:
#         print(e)

# if __name__ == "__main__":
#     main()