
import os
import time
import clr

import System
from System import Decimal
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
import re
from datetime import datetime

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



clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK_USB_DLL.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK.Devices.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FTD2XX_NET.dll")
import FUTEK_USB_DLL
import FUTEK.Devices


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
                device.StartPolling(1)
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


def home_device_and_set_velocity(device, initial_position, velocity,polling_rate):
    """
    Documentar
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

def stress_relaxation(device,initial_position,forward_position, waiting_time, cycles,waitTimeout):
    try:
        for cycle in range(1,cycles+1,1):
            device.MoveTo(Decimal(initial_position-forward_position*cycle),waitTimeout)
            time.sleep(waiting_time)
    except Exception as e:
        print(e)



def set_parameters(case,velocity,initial_position,final_position=None,polling_rate=1,frequency=100,cycles=None,forward_position=None, waiting_time=None):
    '''
    DOCUMENTACntar 
    
    '''
 
    initial_position=initial_position+25 #Initial position must be
    if case==1:
        
        final_position=25-final_position
        execution_time=(initial_position-final_position)/velocity
        num_samples = int((execution_time * 1.1 + 16) * frequency)
        print(num_samples)
        sample_time = 1 / frequency
        waitTimeout=execution_time*1000+2000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        return velocity,initial_position,final_position,polling_rate,sample_time,num_samples,waitTimeout
    if case==2:
        
        final_position=25-final_position
        execution_time= (initial_position-final_position)*2*cycles/velocity
        waitTimeout=(initial_position-final_position)*1000/velocity+2000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)#Duration to wait for command to execute
        num_samples = int((execution_time * 1.1 + 16) * frequency)
        sample_time = 1 / frequency
        return velocity,initial_position,final_position,cycles,polling_rate,sample_time,num_samples,waitTimeout
    if case==3:
        while (True):
            if forward_position*cycles>20:
                cycles-=1
            if forward_position*cycles<=20:
                break
            
        execution_time=(forward_position/velocity+waiting_time)*cycles
        waitTimeout=forward_position/velocity*1000
        waitTimeout+=waitTimeout*0.7
        waitTimeout=System.Convert.ToUInt64(waitTimeout)
        sample_time = 1 / frequency
        num_samples = int((execution_time * 1.1 + 16) * frequency)
        return velocity,initial_position,forward_position,waiting_time,cycles,polling_rate,sample_time,num_samples,waitTimeout
    
    
def get_position(device):
    return device.Position

def get_loadcell_reading(loadcell):
    return loadcell.GetChannelXReading(0)

def force_units_and_conversion_factor(from_unit='lb', to_unit='N'):
    data_units_force = [
        ('Force',90,'dyne','dyn','dyne','dyn',1,0),
('Force',91,'dyne','dyn','kilodyne','kdyn',0.001,0),
('Force',92,'dyne','dyn','megadyne','Mdyn',0.000001,0),
('Force',93,'dyne','dyn','newton','N',0.00001,0),
('Force',94,'dyne','dyn','kilonewton','kN',0.00000001,0),
('Force',95,'dyne','dyn','microgram','µg',1019.716,0),
('Force',96,'dyne','dyn','milligram','mg',1.019716,0),
('Force',97,'dyne','dyn','gram','g',0.001019716,0),
('Force',98,'dyne','dyn','kilogram','kg',0.000001019716,0),
('Force',99,'dyne','dyn','metric ton','MT',0.000000001019716,0),
('Force',100,'dyne','dyn','ounce','oz',0.00003596942,0),
('Force',101,'dyne','dyn','pound','lb',0.000002248089,0),
('Force',102,'dyne','dyn','kilopound','klb',0.000000002248089,0),
('Force',103,'dyne','dyn','short ton (US)','ST',0.000000001124044,0),
('Force',104,'dyne','dyn','long ton (UK)','LT',0.000000001003611,0),
('Force',105,'kilodyne','kdyn','dyne','dyn',1000,0),
('Force',106,'kilodyne','kdyn','kilodyne','kdyn',1,0),
('Force',107,'kilodyne','kdyn','megadyne','Mdyn',0.001,0),
('Force',108,'kilodyne','kdyn','newton','N',0.01,0),
('Force',109,'kilodyne','kdyn','kilonewton','kN',0.00001,0),
('Force',110,'kilodyne','kdyn','microgram','µg',1019716,0),
('Force',111,'kilodyne','kdyn','milligram','mg',1019.716,0),
('Force',112,'kilodyne','kdyn','gram','g',1.019716,0),
('Force',113,'kilodyne','kdyn','kilogram','kg',0.001019716,0),
('Force',114,'kilodyne','kdyn','metric ton','MT',0.000001019716,0),
('Force',115,'kilodyne','kdyn','ounce','oz',0.03596942,0),
('Force',116,'kilodyne','kdyn','pound','lb',0.002248089,0),
('Force',117,'kilodyne','kdyn','kilopound','klb',0.000002248089,0),
('Force',118,'kilodyne','kdyn','short ton (US)','ST',0.000001124044,0),
('Force',119,'kilodyne','kdyn','long ton (UK)','LT',0.000001003611,0),
('Force',120,'megadyne','Mdyn','dyne','dyn',1000000,0),
('Force',121,'megadyne','Mdyn','kilodyne','kdyn',1000,0),
('Force',122,'megadyne','Mdyn','megadyne','Mdyn',1,0),
('Force',123,'megadyne','Mdyn','newton','N',10,0),
('Force',124,'megadyne','Mdyn','kilonewton','kN',0.01,0),
('Force',125,'megadyne','Mdyn','microgram','µg',1019716000,0),
('Force',126,'megadyne','Mdyn','milligram','mg',1019716,0),
('Force',127,'megadyne','Mdyn','gram','g',1019.716,0),
('Force',128,'megadyne','Mdyn','kilogram','kg',1.019716,0),
('Force',129,'megadyne','Mdyn','metric ton','MT',0.001019716,0),
('Force',130,'megadyne','Mdyn','ounce','oz',35.96942,0),
('Force',131,'megadyne','Mdyn','pound','lb',2.248089,0),
('Force',132,'megadyne','Mdyn','kilopound','klb',0.002248089,0),
('Force',133,'megadyne','Mdyn','short ton (US)','ST',0.001124044,0),
('Force',134,'megadyne','Mdyn','long ton (UK)','LT',0.001003611,0),
('Force',135,'newton','N','dyne','dyn',100000,0),
('Force',136,'newton','N','kilodyne','kdyn',100,0),
('Force',137,'newton','N','megadyne','Mdyn',0.1,0),
('Force',138,'newton','N','newton','N',1,0),
('Force',139,'newton','N','kilonewton','kN',0.001,0),
('Force',140,'newton','N','microgram','µg',101971600,0),
('Force',141,'newton','N','milligram','mg',101971.6,0),
('Force',142,'newton','N','gram','g',101.9716,0),
('Force',143,'newton','N','kilogram','kg',0.1019716,0),
('Force',144,'newton','N','metric ton','MT',0.0001019716,0),
('Force',145,'newton','N','ounce','oz',3.596942,0),
('Force',146,'newton','N','pound','lb',0.2248089,0),
('Force',147,'newton','N','kilopound','klb',0.0002248089,0),
('Force',148,'newton','N','short ton (US)','ST',0.0001124044,0),
('Force',149,'newton','N','long ton (UK)','LT',0.0001003611,0),
('Force',150,'kilonewton','kN','dyne','dyn',100000000,0),
('Force',151,'kilonewton','kN','kilodyne','kdyn',100000,0),
('Force',152,'kilonewton','kN','megadyne','Mdyn',100,0),
('Force',153,'kilonewton','kN','newton','N',1000,0),
('Force',154,'kilonewton','kN','kilonewton','kN',1,0),
('Force',155,'kilonewton','kN','microgram','µg',101971600000,0),
('Force',156,'kilonewton','kN','milligram','mg',101971600,0),
('Force',157,'kilonewton','kN','gram','g',101971.6,0),
('Force',158,'kilonewton','kN','kilogram','kg',101.9716,0),
('Force',159,'kilonewton','kN','metric ton','MT',0.1019716,0),
('Force',160,'kilonewton','kN','ounce','oz',3596.942,0),
('Force',161,'kilonewton','kN','pound','lb',224.8089,0),
('Force',162,'kilonewton','kN','kilopound','klb',0.2248089,0),
('Force',163,'kilonewton','kN','short ton (US)','ST',0.1124044,0),
('Force',164,'kilonewton','kN','long ton (UK)','LT',0.1003611,0),
('Force',165,'microgram','µg','dyne','dyn',0.0009806652,0),
('Force',166,'microgram','µg','kilodyne','kdyn',0.0000009806652,0),
('Force',167,'microgram','µg','megadyne','Mdyn',0.0000000009806652,0),
('Force',168,'microgram','µg','newton','N',0.000000009806652,0),
('Force',169,'microgram','µg','kilonewton','kN',0.000000000009806652,0),
('Force',170,'microgram','µg','microgram','µg',1,0),
('Force',171,'microgram','µg','milligram','mg',0.001,0),
('Force',172,'microgram','µg','gram','g',0.000001,0),
('Force',173,'microgram','µg','kilogram','kg',0.000000001,0),
('Force',174,'microgram','µg','metric ton','MT',0.000000000001,0),
('Force',175,'microgram','µg','ounce','oz',0.00000003527396,0),
('Force',176,'microgram','µg','pound','lb',0.000000002204623,0),
('Force',177,'microgram','µg','kilopound','klb',0.000000000002204623,0),
('Force',178,'microgram','µg','short ton (US)','ST',0.000000000001102311,0),
('Force',179,'microgram','µg','long ton (UK)','LT',9.842065E-13,0),
('Force',180,'milligram','mg','dyne','dyn',0.9806652,0),
('Force',181,'milligram','mg','kilodyne','kdyn',0.0009806652,0),
('Force',182,'milligram','mg','megadyne','Mdyn',0.0000009806652,0),
('Force',183,'milligram','mg','newton','N',0.000009806652,0),
('Force',184,'milligram','mg','kilonewton','kN',0.000000009806652,0),
('Force',185,'milligram','mg','microgram','µg',1000,0),
('Force',186,'milligram','mg','milligram','mg',1,0),
('Force',187,'milligram','mg','gram','g',0.001,0),
('Force',188,'milligram','mg','kilogram','kg',0.000001,0),
('Force',189,'milligram','mg','metric ton','MT',0.000000001,0),
('Force',190,'milligram','mg','ounce','oz',0.00003527396,0),
('Force',191,'milligram','mg','pound','lb',0.000002204623,0),
('Force',192,'milligram','mg','kilopound','klb',0.000000002204623,0),
('Force',193,'milligram','mg','short ton (US)','ST',0.000000001102311,0),
('Force',194,'milligram','mg','long ton (UK)','LT',0.0000000009842065,0),
('Force',195,'gram','g','dyne','dyn',980.6652,0),
('Force',196,'gram','g','kilodyne','kdyn',0.9806652,0),
('Force',197,'gram','g','megadyne','Mdyn',0.0009806652,0),
('Force',198,'gram','g','newton','N',0.009806652,0),
('Force',199,'gram','g','kilonewton','kN',0.000009806652,0),
('Force',200,'gram','g','microgram','µg',1000000,0),
('Force',201,'gram','g','milligram','mg',1000,0),
('Force',202,'gram','g','gram','g',1,0),
('Force',203,'gram','g','kilogram','kg',0.001,0),
('Force',204,'gram','g','metric ton','MT',0.000001,0),
('Force',205,'gram','g','ounce','oz',0.03527396,0),
('Force',206,'gram','g','pound','lb',0.002204623,0),
('Force',207,'gram','g','kilopound','klb',0.000002204623,0),
('Force',208,'gram','g','short ton (US)','ST',0.000001102311,0),
('Force',209,'gram','g','long ton (UK)','LT',0.0000009842065,0),
('Force',210,'kilogram','kg','dyne','dyn',980665.2,0),
('Force',211,'kilogram','kg','kilodyne','kdyn',980.6652,0),
('Force',212,'kilogram','kg','megadyne','Mdyn',0.9806652,0),
('Force',213,'kilogram','kg','newton','N',9.806652,0),
('Force',214,'kilogram','kg','kilonewton','kN',0.009806652,0),
('Force',215,'kilogram','kg','microgram','µg',1000000000,0),
('Force',216,'kilogram','kg','milligram','mg',1000000,0),
('Force',217,'kilogram','kg','gram','g',1000,0),
('Force',218,'kilogram','kg','kilogram','kg',1,0),
('Force',219,'kilogram','kg','metric ton','MT',0.001,0),
('Force',220,'kilogram','kg','ounce','oz',35.27396,0),
('Force',221,'kilogram','kg','pound','lb',2.204623,0),
('Force',222,'kilogram','kg','kilopound','klb',0.002204623,0),
('Force',223,'kilogram','kg','short ton (US)','ST',0.001102311,0),
('Force',224,'kilogram','kg','long ton (UK)','LT',0.0009842065,0),
('Force',225,'metric ton','MT','dyne','dyn',980665200,0),
('Force',226,'metric ton','MT','kilodyne','kdyn',980665.2,0),
('Force',227,'metric ton','MT','megadyne','Mdyn',980.6652,0),
('Force',228,'metric ton','MT','newton','N',9806.652,0),
('Force',229,'metric ton','MT','kilonewton','kN',9.806652,0),
('Force',230,'metric ton','MT','microgram','µg',1000000000000,0),
('Force',231,'metric ton','MT','milligram','mg',1000000000,0),
('Force',232,'metric ton','MT','gram','g',1000000,0),
('Force',233,'metric ton','MT','kilogram','kg',1000,0),
('Force',234,'metric ton','MT','metric ton','MT',1,0),
('Force',235,'metric ton','MT','ounce','oz',35273.96,0),
('Force',236,'metric ton','MT','pound','lb',2204.623,0),
('Force',237,'metric ton','MT','kilopound','klb',2.204623,0),
('Force',238,'metric ton','MT','short ton (US)','ST',1.102311,0),
('Force',239,'metric ton','MT','long ton (UK)','LT',0.9842065,0),
('Force',240,'ounce','oz','dyne','dyn',27801.39,0),
('Force',241,'ounce','oz','kilodyne','kdyn',27.80139,0),
('Force',242,'ounce','oz','megadyne','Mdyn',0.02780139,0),
('Force',243,'ounce','oz','newton','N',0.2780139,0),
('Force',244,'ounce','oz','kilonewton','kN',0.0002780139,0),
('Force',245,'ounce','oz','microgram','µg',28349520,0),
('Force',246,'ounce','oz','milligram','mg',28349.52,0),
('Force',247,'ounce','oz','gram','g',28.34952,0),
('Force',248,'ounce','oz','kilogram','kg',0.02834952,0),
('Force',249,'ounce','oz','metric ton','MT',0.00002834952,0),
('Force',250,'ounce','oz','ounce','oz',1,0),
('Force',251,'ounce','oz','pound','lb',0.0625,0),
('Force',252,'ounce','oz','kilopound','klb',0.0000625,0),
('Force',253,'ounce','oz','short ton (US)','ST',0.00003125,0),
('Force',254,'ounce','oz','long ton (UK)','LT',2.79017857142857E-05,0),
('Force',255,'pound','lb','dyne','dyn',444822.2,0),
('Force',256,'pound','lb','kilodyne','kdyn',444.8222,0),
('Force',257,'pound','lb','megadyne','Mdyn',0.4448222,0),
('Force',258,'pound','lb','newton','N',4.448222,0),
('Force',259,'pound','lb','kilonewton','kN',0.004448222,0),
('Force',260,'pound','lb','microgram','µg',453592400,0),
('Force',261,'pound','lb','milligram','mg',453592.4,0),
('Force',262,'pound','lb','gram','g',453.5924,0),
('Force',263,'pound','lb','kilogram','kg',0.4535924,0),
('Force',264,'pound','lb','metric ton','MT',0.0004535924,0),
('Force',265,'pound','lb','ounce','oz',16,0),
('Force',266,'pound','lb','pound','lb',1,0),
('Force',267,'pound','lb','kilopound','klb',0.001,0),
('Force',268,'pound','lb','short ton (US)','ST',0.0005,0),
('Force',269,'pound','lb','long ton (UK)','LT',0.000446428571428571,0),
('Force',270,'kilopound','klb','dyne','dyn',444822200,0),
('Force',271,'kilopound','klb','kilodyne','kdyn',444822.2,0),
('Force',272,'kilopound','klb','megadyne','Mdyn',444.8222,0),
('Force',273,'kilopound','klb','newton','N',4448.222,0),
('Force',274,'kilopound','klb','kilonewton','kN',4.448222,0),
('Force',275,'kilopound','klb','microgram','µg',453592400000,0),
('Force',276,'kilopound','klb','milligram','mg',453592400,0),
('Force',277,'kilopound','klb','gram','g',453592.4,0),
('Force',278,'kilopound','klb','kilogram','kg',453.5924,0),
('Force',279,'kilopound','klb','metric ton','MT',0.4535924,0),
('Force',280,'kilopound','klb','ounce','oz',16000,0),
('Force',281,'kilopound','klb','pound','lb',1000,0),
('Force',282,'kilopound','klb','kilopound','klb',1,0),
('Force',283,'kilopound','klb','short ton (US)','ST',0.5,0),
('Force',284,'kilopound','klb','long ton (UK)','LT',0.446428571428571,0),
('Force',285,'short ton (US)','ST','dyne','dyn',889644500,0),
('Force',286,'short ton (US)','ST','kilodyne','kdyn',889644.5,0),
('Force',287,'short ton (US)','ST','megadyne','Mdyn',889.6445,0),
('Force',288,'short ton (US)','ST','newton','N',8896.445,0),
('Force',289,'short ton (US)','ST','kilonewton','kN',8.896445,0),
('Force',290,'short ton (US)','ST','microgram','µg',907184700000,0),
('Force',291,'short ton (US)','ST','milligram','mg',907184700,0),
('Force',292,'short ton (US)','ST','gram','g',907184.7,0),
('Force',293,'short ton (US)','ST','kilogram','kg',907.1847,0),
('Force',294,'short ton (US)','ST','metric ton','MT',0.9071847,0),
('Force',295,'short ton (US)','ST','ounce','oz',32000,0),
('Force',296,'short ton (US)','ST','pound','lb',2000,0),
('Force',297,'short ton (US)','ST','kilopound','klb',2,0),
('Force',298,'short ton (US)','ST','short ton (US)','ST',1,0),
('Force',299,'short ton (US)','ST','long ton (UK)','LT',0.892857142857142,0),
('Force',300,'long ton (UK)','LT','dyne','dyn',996401900,0),
('Force',301,'long ton (UK)','LT','kilodyne','kdyn',996401.9,0),
('Force',302,'long ton (UK)','LT','megadyne','Mdyn',996.4019,0),
('Force',303,'long ton (UK)','LT','newton','N',9964.019,0),
('Force',304,'long ton (UK)','LT','kilonewton','kN',9.964019,0),
('Force',305,'long ton (UK)','LT','microgram','µg',1016047000000,0),
('Force',306,'long ton (UK)','LT','milligram','mg',1016047000,0),
('Force',307,'long ton (UK)','LT','gram','g',1016047,0),
('Force',308,'long ton (UK)','LT','kilogram','kg',1016.047,0),
('Force',309,'long ton (UK)','LT','metric ton','MT',1.016047,0),
('Force',310,'long ton (UK)','LT','ounce','oz',35840,0),
('Force',311,'long ton (UK)','LT','pound','lb',2240,0),
('Force',312,'long ton (UK)','LT','kilopound','klb',2.24,0),
('Force',313,'long ton (UK)','LT','short ton (US)','ST',1.12,0),
('Force',314,'long ton (UK)','LT','long ton (UK)','LT',1,0),
    ]
    columns = ["Unit Type", "Conversion", "Convert From", "From Unit", "Convert To", "To Unit", "Scale Factor", "Offset"]

    df_units_force = pd.DataFrame(data_units_force,columns=columns)
    conversion_factor = df_units_force[(df_units_force['From Unit'] == from_unit) & (df_units_force['To Unit'] == to_unit)]['Scale Factor'].values[0]
    force_units=list(df_units_force["From Unit"].unique())
    return force_units,conversion_factor


def collect_data(motor_devices_dictionary,load_cells_list, sample_time, num_samples, desired_uni_force,path, name):
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses sched
    try:
        # Get the list of devices from the dictionary
        devices_list = list(motor_devices_dictionary.values())
        sample = 0
        columnas = ['seconds'] + ['real_position_' + elemento +'_mm' for elemento in motor_devices_dictionary.keys()]+['load_cell_'+loadcell.GetInstrumentSerialNumber()+ '_'+ str(loadcell.GetChannelXUnitOfMeasure(0)) for loadcell in load_cells_list]

        # Obtiene la fecha y hora actual
        fecha_hora_inicial = datetime.now()
        # Create a CSV file to save the data
        csv_file_path = f'{path}\\{name}.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(columnas) # Write the header to the CSV file
            def write_timestamp(sc):
                """This function writes a timestamp and device positions to the CSV file."""
                #initial_time=time.perf_counter()
                nonlocal sample, initial_time
                # Combine the results into a single array at the same time     
                data_line = [time.perf_counter()-initial_time] + [device.Position for device in devices_list]+[loadcell.GetChannelXReading(0) for loadcell in load_cells_list]
                #data_line = [time.perf_counter()-initial_time] + [device.Position for device in devices_list]+list(map(lambda loadcell: loadcell.GetChannelXReading(0), load_cells_list))
                #data_line = [time.perf_counter()-initial_time] + [device.Position for device in devices_list]
                
                sample += 1
                csv_writer.writerow(data_line)
                if sample < num_samples:
                    sc.enter(sample_time, 1, write_timestamp, (sc,))
            # Create a scheduler object
            s = sched.scheduler(time.perf_counter, time.sleep)
            # Schedule the function `write_timestamp()` to run immediately and then repeatedly every sample_time seconds
            initial_time = time.perf_counter()
            s.enter(0, 1, write_timestamp, (s,))
            s.run()

        #Linear Stage Position (mm)-Serial Number
        #

        data_position_force = pd.read_csv(f'{path}\{name}.csv')
        data_position_force['Sample Number']=np.arange(0,len(data_position_force))
        data_position_force['Date']=fecha_hora_inicial +pd.to_timedelta(data_position_force['seconds'], unit='s')
        data_position_force['Time Elapsed']=pd.to_datetime(data_position_force['seconds'], unit='s').dt.time

        for column in data_position_force.columns:
            # Check if the column name contains 'load_cell'
            if 'load_cell' in column:
                # Use regular expressions to extract unit and serial number
                match = re.match(r'load_cell_(\d+)_(\w+)', column)
                if match:
                    serial_number = match.group(1)
                    unit = match.group(2)
                    
                    _,conversion_factor=force_units_and_conversion_factor(from_unit=unit, to_unit=desired_uni_force)
                    
                    # Create the new column name
                    new_column_name = f'Tracking Value ({desired_uni_force})-{serial_number}'
                    data_position_force[new_column_name]=data_position_force[column]*conversion_factor

                    #Peak Values
                    current_peak_index = 0
                    current_peak_value = data_position_force.loc[current_peak_index, new_column_name]

                    # Iterate over the DataFrame
                    for index, value in enumerate(data_position_force[new_column_name]):
                        # Check if the current value is greater than the current peak value
                        if value > current_peak_value:
                            current_peak_index = index
                            current_peak_value = value

                        # Update the DataFrame with the current peak value
                        data_position_force.loc[index, f'Peak Value ({desired_uni_force})-{serial_number}'] = current_peak_value

                    #Valley Value
                    current_valley_index = 0
                    current_valley_value = data_position_force.loc[current_valley_index, new_column_name]

                    # Iterate over the DataFrame
                    for index, value in enumerate(data_position_force[new_column_name]):
                        # Check if the current value is smaller than the current valley value
                        if value < current_valley_value:
                            current_valley_index = index
                            current_valley_value = value

                        # Update the DataFrame with the current valley value
                        data_position_force.loc[index, f'Valley Value ({desired_uni_force})-{serial_number}'] = current_valley_value


                    # Rename the column
                    #data_position_force.rename(columns={column: new_column_name}, inplace=True)


        # data_position_force['seconds']=data_position_force['seconds'] - data_position_force['seconds'].iloc[0] #time difference 
        # columns_real=[columna for columna in data_position_force.columns if 'real_position_' in columna]
        # columns_relative = [column.replace('real_position_', 'relative_position_') for column in columns_real]
        # data_position_force[columns_relative]=data_position_force[columns_real].iloc[0]-data_position_force[columns_real]
    
        writer = pd.ExcelWriter(f'{path}\\{name}.xlsx', engine='openpyxl')
        data_position_force.to_excel(writer, sheet_name='Data')
        writer.close()
    except Exception as e:
        # Handle the exception here, you can print an error message or log it
        print(f"An exception occurred: {str(e)}")

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
        
             #force_units,conversion_factor=force_units_and_conversion_factor()
        #Conexion celdas de carga
        repository_FUTEK = FUTEK.Devices.DeviceRepository()
        devices_FUTEK = list(repository_FUTEK.DetectDevices())

  
        #CASO 1: shift
  
        # parameters=[0.5,0,5,1,100] #velocity mm/s,initial position mm,final position mm, polling rate ms y entero, frecuency Hz
        # path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\celdadecargaymotor"
        # #for i in range(20): if I want to run
        # velocity,initial_position,final_position,polling_rate,sample_time,num_samples,waitTimeout=set_parameters(case=1,velocity=parameters[0],initial_position=parameters[1],final_position=parameters[2],polling_rate=parameters[3],frequency=parameters[4],cycles=None,forward_position=None, waiting_time=None)
        # # Do the homing and set the velocity
        # # Perform homing and place the device in the initial position
        # # Initialize tasks in parallel for all the devices
        # i=25
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Execute home_device in parallel for all devices
         
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(home_device_and_set_velocity, device, initial_position, velocity,polling_rate))
        #         # Wait for all of the tasks to complete
        #     concurrent.futures.wait(futures)
        # name=f"Shift_vel_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_pollrate_{parameters[3]}_samplefreq_{parameters[4]}_exp_{i}_all"

        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     p1=executor.submit(collect_data,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,desired_uni_force='N', path=path, name=name)
        #     # Start the tasks in futures
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(shif_device, device, final_position,waitTimeout))
        #     # Wait for all of the tasks to complete
        #     concurrent.futures.wait([p1] + futures)
        # print(f'Fin ciclo shif device {i}')
        # print('valelinda')

        #CASO 2 histeresis

        parameters=[1.5,0,15,1,50,10] #velocity,initial position,final position, polling rate, frecuency, ciclos
    
        path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\celdadecargaymotor"
        #for i in range(20): if I want to run
        velocity,initial_position,final_position,cycles,polling_rate,sample_time,num_samples,waitTimeout=set_parameters(case=2,velocity=parameters[0],
        initial_position=parameters[1],final_position=parameters[2],polling_rate=parameters[3],frequency=parameters[4],cycles=parameters[5],forward_position=None,
        waiting_time=None)
        # Do the homing and set the velocity
        # Perform homing and place the device in the initial position
        # Initialize tasks in parallel for all the devices
        i=1
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute home_device in parallel for all devices
            for device in thorlabs_devices.devices.values():
                executor.submit(home_device_and_set_velocity, device, initial_position, velocity,polling_rate)
        name=f"histeresis_vel_{parameters[0]}_pi_{parameters[1]}_pf_{parameters[2]}_pollrate_{parameters[3]}_samplefreq_{parameters[4]}_ciclos_{parameters[5]}_exp_{i}_all"
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            p1=executor.submit(collect_data,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,desired_uni_force='N', path=path, name=name)
            # Start the tasks in futures
            futures = []
            for device in thorlabs_devices.devices.values():
                futures.append(executor.submit(hysteresis,device,initial_position, final_position, cycles,waitTimeout))
            # Wait for all of the tasks to complete
            concurrent.futures.wait([p1] + futures)
        print(f'Fin ciclo {i}')
        print('valelinda3')

        #CASO 3 stress relaxation

        # parameters=[2,0,1,50,5,5,5] #velocity mm/s,initial position mm, polling rate ms y entero, frecuency Hz,cycles,foward position mm,waiting_time
        # path=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\celdadecargaymotor"
        # #for i in range(20): if I want to run
        # velocity,initial_position,forward_position,waiting_time,cycles,polling_rate,sample_time,num_samples,waitTimeout=set_parameters(case=3,velocity=parameters[0],
        # initial_position=parameters[1],final_position=None,polling_rate=parameters[2],frequency=parameters[3],cycles=parameters[4],forward_position=parameters[5],
        # waiting_time=parameters[6])
        # print("velocity:", velocity)
        # print("initial position:", initial_position)
        # print("forward_position:", forward_position)
        # print("waiting_time:", waiting_time)
        # print("cycles:", cycles)
        # print("polling_rate:", polling_rate)
        # print("sample_time:", sample_time)
        # print("num_samples:", num_samples)
        # print("waitTimeout:", waitTimeout)
        # # Do the homing and set the velocity
        # # Perform homing and place the device in the initial position
        # # Initialize tasks in parallel for all the devices
        # i=1
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Execute home_device in parallel for all devices
         
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(home_device_and_set_velocity, device, initial_position, velocity,polling_rate))
        #         # Wait for all of the tasks to complete
        #     concurrent.futures.wait(futures)
        # name=f"stress_relaxation_vel_{parameters[0]}_pi_{parameters[1]}_pollrate_{parameters[2]}_samplefreq_{parameters[3]}_cycles_{cycles}_fp_{parameters[5]}_waittime_{parameters[6]}_exp_{i}_all"

        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     p1=executor.submit(collect_data,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,desired_uni_force='N', path=path, name=name)
        #     # Start the tasks in futures
        #     futures = []
        #     for device in thorlabs_devices.devices.values():
        #         futures.append(executor.submit(stress_relaxation,device,initial_position,forward_position, waiting_time, cycles,waitTimeout))
        #     # Wait for all of the tasks to complete
        #     concurrent.futures.wait([p1] + futures)
        # print(f'Fin ciclo shif device {i}')
        # print('valelinda')


        for loadcell in devices_FUTEK:
            repository_FUTEK.DisconnectDevice( loadcell.GetModelNumber(),loadcell.GetInstrumentSerialNumber())
            print(f"Load cell disconnected: {loadcell.GetInstrumentSerialNumber()}")

        for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    #cProfile.run("main()", sort='cumulative')
    main()