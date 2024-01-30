"""
    This code provides the functionality to the user interface created for the movement of Thorlabs motorized linear
    stages and acquisition of force data from the USB225 load cells. This is done by making use of the files and code
    in the python folder python\src\process\motors_loadcells.

    Author: Valeria Cadavid Castro
    Last Modified: January 22, 2024
"""

# Importing necessary libraries and modules
from electron.electronSocket import ElectronSocket
import utils.imageUtils as imageUtils
from process.motors_loadcells.thorlabsDevices import ThorlabsDevices
import process.motors_loadcells.motors_loadcells as motors_loadcells
import concurrent.futures
import clr
import os

# Importing shared variables from the 'motors_loadcells' module
import process.motors_loadcells.shared_variables as shared_variables

def get_path_until_directory(path, target_directory):
    """
    Get the path until a specified target directory.

    Parameters:
    - path: The initial path.
    - target_directory: The target directory name.

    Returns:
    - The path until the specified target directory.
    """
    while os.path.basename(path) != target_directory:
        path = os.path.dirname(path)
    return path

# Get the script directory and navigate to the "src" directory
script_dir = os.path.dirname(os.path.realpath(__file__))
script_dir= get_path_until_directory(script_dir,"src")

# Add references to required DLL files using the obtained path, for the access to the files for the management of the load cells of FUTEK
clr.AddReference(os.path.join(script_dir, "process\\motors_loadcells\\DLL_files\\DLL_FUTEK\\FUTEK_USB_DLL.dll"))
clr.AddReference(os.path.join(script_dir, "process\\motors_loadcells\\DLL_files\\DLL_FUTEK\\FUTEK.Devices.dll"))
clr.AddReference(os.path.join(script_dir, "process\\motors_loadcells\\DLL_files\\DLL_FUTEK\\FTD2XX_NET.dll"))

# Import the necessary modules after setting the references
import FUTEK_USB_DLL
import FUTEK.Devices

# Provides access to FUTEK Devices and their operations
repository_FUTEK = FUTEK.Devices.DeviceRepository()

def connectMotors():
    """
    Connect Thorlabs motors and FUTEK load cells.

    Returns:
    - result: A dictionary containing the connection status.
        - "isConnected": True if all devices are successfully connected, False otherwise.
    """
    # Create an instance of ThorlabsDevices
    global thorlabs_devices
    thorlabs_devices = ThorlabsDevices()
    
    # Detect available Thorlabs devices
    available_devices=thorlabs_devices.detect_devices()

    # Connect to detected Thorlabs devices
    for serial_number in available_devices:
        thorlabs_devices.connect_device(serial_number)

     # Detect available FUTEK devices
    global devices_FUTEK
    devices_FUTEK = list(repository_FUTEK.DetectDevices())
    print("Load cells detected")

     # Check if Thorlabs and FUTEK devices are both available and connected
    if len(available_devices)>0 and len(available_devices)==len(thorlabs_devices.devices.keys()) and len(devices_FUTEK)>0:
        result = {"isConnected"  : True,}
        
    else:
        result = {"isConnected"  : False,}
    
    return result

def disconnectMotors():
    """
    Disconnect Thorlabs motors and FUTEK load cells.

    Returns:
    - result: A dictionary indicating the disconnection status.
        - "isConnected": False, indicating that devices are disconnected.
    """
    # Disconnect Thorlabs devices
    for device in list(thorlabs_devices.devices.values()):
            thorlabs_devices.disconnect_device(device.DeviceID)

    # Disconnect FUTEK devices
    for loadcell in devices_FUTEK:
        repository_FUTEK.DisconnectDevice( loadcell.GetModelNumber(),loadcell.GetInstrumentSerialNumber())
        print(f"Load cell disconnected: {loadcell.GetInstrumentSerialNumber()}")
    
    # Update result indicating that devices are disconnected
    result = {
        "isConnected"  : False
    }
    
    return result

def get_linearstages_and_loadcells():
    """
    Get information about connected linear stages and load cells.

    Returns:
    - resultLS: A dictionary containing information about connected linear stages.
        - "linearStages": A list of dictionaries, where each dictionary represents a connected linear stage.
            - "name": Serial number of the linear stage.
            - "unit": Unit of measure for the linear stage (e.g., "mm").

    - resultLC: A dictionary containing information about connected load cells.
        - "loadcells": A list of dictionaries, where each dictionary represents a connected load cell.
            - "name": Serial number of the load cell.
            - "unit": Unit of measure for the load cell.
    """
    # Linear Stages Information
    resultLS = {"linearStages": []}
    for serial_number, motor in thorlabs_devices.devices.items():
        # Create a dictionary for each device and append it to the result list
        device_info = {"name": f"{serial_number}", "unit": "mm"}
        resultLS["linearStages"].append(device_info)

    # Load Cells Information
    resultLC = {"loadcells": []}
    for device in devices_FUTEK:
        # Get the serial number and unit of measure
        serial_number = device.GetInstrumentSerialNumber()
        unit_of_measure = device.GetChannelXUnitOfMeasure(0)  # Assuming channel 0

        # Create a dictionary for each device and append it to the result list
        device_info = {"name": f"{serial_number}", "unit": f"{unit_of_measure}"}
        resultLC["loadcells"].append(device_info)

    return resultLS, resultLC

def motorMove(data):
    """
    Move Thorlabs linear stages to the specified initial position, that is the initial position experimental.

    Args:
    - data: A dictionary containing motor position information.
        - "position": The target position for the motors.

    Returns:
    - result: A dictionary indicating the success of the motor movement.
        - "isMotorsSetup": True, indicating that the motors are set up at the specified position.
    """
    # Calculate the initial position based on the target position
    if data['position']<0: #To avoid errors in the given parameter
        initial_position=25
    else:
        initial_position = 25 - data['position'] 

    # Move Thorlabs motors to the specified initial position
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        # Start displacement to the initial position for each Thorlabs device
        for device in thorlabs_devices.devices.values():
            futures.append(executor.submit(thorlabs_devices.move_initial_position, device, initial_position))

        # Wait for all of the tasks to complete
        concurrent.futures.wait(futures)

    # Wait until all motors have stopped moving
    while all(device.Status.IsMoving == True for device in thorlabs_devices.devices.values()):
        pass

    # Update result indicating that motors are set up at the specified position
    result = {
        "isMotorsSetup": True
    }

    return result

def motorsNormaldisplacements(data):
    """
    Perform synchronized displacement of Thorlabs motors to a specified final position acquire position and force data.

    Args:
    - data: A dictionary containing parameters for motor displacement.
        - "velocity": The velocity at which the motors should move.
        - "finalPosition": The final position to which the motors should be displaced or the position that the motors should move from the actual position.
        - "samplingFrequency": The frequency at which data should be sampled during the displacement.
        - "forceUnit": The unit of force measurement.
        - "file": The file name for saving the data.

    Returns:
    - result: A dictionary containing information about the process.
        - "ended": True, indicating the completion of the displacement process.
        - "feedback": None, as there is no specific feedback provided in this function.
        - "plots": A list of temporary file paths for the generated position and force plots.

    """
    # Extract parameters from the data dictionary
    velocity = data['velocity']
    final_position = data['finalPosition']
    frequency = data['samplingFrequency']
    force_unit = data['forceUnit']
    path_name = data['file']
    path, _ = os.path.splitext((path_name))

    # Set the variable indicating that motor movements are not yet completed
    shared_variables.set_movements_motors_completed(False)

    # Perform synchronized displacement of Thorlabs motors
    data = motors_loadcells.synchronized_displacement(path, thorlabs_devices, devices_FUTEK, velocity, final_position, frequency, force_unit)
    print(path)

    # Generate position and force plots in both English and Spanish
    _, fig_pos = motors_loadcells.plot_graph_positions(data, path=path, language='english')
    _, fig_force = motors_loadcells.plot_graph_forces(data, path=path, language='english')
    _, _ = motors_loadcells.plot_graph_positions(data, path=path, language='spanish')
    _, _ = motors_loadcells.plot_graph_forces(data, path=path, language='spanish')

    # Prepare result dictionary with plot information
    result = {
        "ended": True,
        "feedback": None,
        "plots": [
            imageUtils.pltToTmpFile(fig_pos),
            imageUtils.pltToTmpFile(fig_force)
        ]
    }

    return result

def motorsHysteresis(data):
    """
    Perform synchronized hysteresis displacement of Thorlabs motors and acquire position and force data.

    Args:
    - data: A dictionary containing parameters for hysteresis displacement.
        - "velocity": The velocity at which the motors should move.
        - "finalPosition": The final position to which the motors should be displaced or the position that the motors should move from the actual position.
        - "samplingFrequency": The frequency at which data should be sampled during the displacement.
        - "cycles": The number of cycles to perform during hysteresis.
        - "forceUnit": The unit of force measurement.
        - "file": The file name for saving the data.

    Returns:
    - result: A dictionary containing information about the process.
        - "ended": True, indicating the completion of the hysteresis displacement process.
        - "feedback": None, as there is no specific feedback provided in this function.
        - "plots": A list of temporary file paths for the generated position and force plots.
    """
    print(data)

    # Extract parameters from the data dictionary
    velocity = data['velocity']
    final_position = data['finalPosition']
    frequency = data['samplingFrequency']
    cycles = data["cycles"]
    force_unit = data['forceUnit']
    path_name = data['file']
    path, _ = os.path.splitext((path_name))

    # Set the variable indicating that motor movements are not yet completed
    shared_variables.set_movements_motors_completed(False)

    # Perform synchronized hysteresis displacement of Thorlabs motors
    data = motors_loadcells.synchronized_hysteresis(path, thorlabs_devices, devices_FUTEK, velocity, final_position, cycles, frequency, force_unit)

    # Generate position and force plots in both English and Spanish
    _, fig_pos = motors_loadcells.plot_graph_positions(data, path=path, language='english')
    _, fig_force = motors_loadcells.plot_graph_forces(data, path=path, language='english')
    _, _ = motors_loadcells.plot_graph_positions(data, path=path, language='spanish')
    _, _ = motors_loadcells.plot_graph_forces(data, path=path, language='spanish')

    # Prepare result dictionary with plot information
    result = {
        "ended": True,
        "feedback": None,
        "plots": [
            imageUtils.pltToTmpFile(fig_pos),
            imageUtils.pltToTmpFile(fig_force)
        ]
    }

    return result

def motorsHysteresis_fp(data):
    """
    Perform synchronized hysteresis displacement of Thorlabs motors and acquire position and force data.
    Perform a synchronized hysteresis experiment with Thorlabs devices and FUTEK load cells.
    The key distinction from the function 'motorsHysteresis' is that this function
    does not return the linear stage to the initial position in the last cycle; it concludes
    at the final position.

    Args:
    - data: A dictionary containing parameters for hysteresis displacement.
        - "velocity": The velocity at which the motors should move.
        - "finalPosition": The final position to which the motors should be displaced or the position that the motors should move from the actual position.
        - "samplingFrequency": The frequency at which data should be sampled during the displacement.
        - "cycles": The number of cycles to perform during hysteresis.
        - "forceUnit": The unit of force measurement.
        - "file": The file name for saving the data.

    Returns:
    - result: A dictionary containing information about the process.
        - "ended": True, indicating the completion of the hysteresis displacement process.
        - "feedback": None, as there is no specific feedback provided in this function.
        - "plots": A list of temporary file paths for the generated position and force plots.
    """
    print(data)

    # Extract parameters from the data dictionary
    velocity = data['velocity']
    final_position = data['finalPosition']
    frequency = data['samplingFrequency']
    cycles = data["cycles"]
    force_unit = data['forceUnit']
    path_name = data['file']
    path, _ = os.path.splitext((path_name))

    # Set the variable indicating that motor movements are not yet completed
    shared_variables.set_movements_motors_completed(False)

    # Perform synchronized hysteresis displacement of Thorlabs motors
    data = motors_loadcells.synchronized_hysteresis_fp(path, thorlabs_devices, devices_FUTEK, velocity, final_position, cycles, frequency, force_unit)

    # Generate position and force plots in both English and Spanish
    _, fig_pos = motors_loadcells.plot_graph_positions(data, path=path, language='english')
    _, fig_force = motors_loadcells.plot_graph_forces(data, path=path, language='english')
    _, _ = motors_loadcells.plot_graph_positions(data, path=path, language='spanish')
    _, _ = motors_loadcells.plot_graph_forces(data, path=path, language='spanish')

    # Prepare result dictionary with plot information
    result = {
        "ended": True,
        "feedback": None,
        "plots": [
            imageUtils.pltToTmpFile(fig_pos),
            imageUtils.pltToTmpFile(fig_force)
        ]
    }

    return result

def motorsStressRelaxation(data):
    """
    Perform synchronized stress relaxation displacement of Thorlabs motors and acquire position and force data.

    Args:
    - data: A dictionary containing parameters for stress relaxation displacement.
        - "velocity": The velocity at which the motors should move.
        - "forwardPosition": The forward position or step size to which the motors should be displaced each cycle from the actual position.
        - "samplingFrequency": The frequency at which data should be sampled during the displacement.
        - "cycles": The number of cycles to perform stress relaxation.
        - "forceUnit": The unit of force measurement.
        - "file": The file name for saving the data.
        - "waitingTimes": The waiting time for stress relaxation cycles.

    Returns:
    - result: A dictionary containing information about the process.
        - "ended": True, indicating the completion of the stress relaxation displacement process.
        - "feedback": None, as there is no specific feedback provided in this function.
        - "plots": A list of temporary file paths for the generated position and force plots.
    """
    print(data)

    # Extract parameters from the data dictionary
    velocity = data['velocity']
    forward_position = data['forwardPosition']
    frequency = data['samplingFrequency']
    cycles = data["cycles"]
    force_unit = data['forceUnit']
    path_name = data['file']
    waiting_time = data["waitingTimes"]
    path, _ = os.path.splitext((path_name))

    # Set the variable indicating that motor movements are not yet completed
    shared_variables.set_movements_motors_completed(False)

    # Perform synchronized stress relaxation displacement of Thorlabs motors
    data = motors_loadcells.synchronized_stress_relaxation(path, thorlabs_devices, devices_FUTEK, velocity, forward_position, waiting_time, cycles, frequency, force_unit)

    # Generate position and force plots in both English and Spanish
    _, fig_pos = motors_loadcells.plot_graph_positions(data, path=path, language='english')
    _, fig_force = motors_loadcells.plot_graph_forces(data, path=path, language='english')
    _, _ = motors_loadcells.plot_graph_positions(data, path=path, language='spanish')
    _, _ = motors_loadcells.plot_graph_forces(data, path=path, language='spanish')

    # Prepare result dictionary with plot information
    result = {
        "ended": True,
        "feedback": None,
        "plots": [
            imageUtils.pltToTmpFile(fig_pos),
            imageUtils.pltToTmpFile(fig_force)
        ]
    }

    return result

def motorsStop():
    """
    Stop immediate motion of Thorlabs motors.

    This function stops immediate motion of Thorlabs motors using a thread pool.

    Returns:
    - result: An empty dictionary indicating the completion of the stop process.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Initialize an empty list to store the future objects for each motor stop task
        futures_stop = []
        # Iterate over Thorlabs devices
        for device in thorlabs_devices.devices.values():
            # Submit a motor stop task to the executor and append the returned Future object to the list
            futures_stop.append(executor.submit(device.StopImmediate))
        # Wait for all submitted tasks (motor stops) to complete
        concurrent.futures.wait(futures_stop)

    # Set the variable indicating that motor movements are completed
    shared_variables.set_movements_motors_completed(True)

    # Prepare result dictionary (empty, as no specific information is returned)
    result = {}

    return result

def listForceUnits():
    """
    List available force units.

    This function retrieves the available force units from the `motors_loadcells` module
    and returns a sorted list of unit names.

    Returns:
    - result: A dictionary containing the list of force units under the key "units".
    """
    # Retrieve the available force units and conversion factors
    units_list, _ = motors_loadcells.force_units_and_conversion_factor()
    # Sort the list of units
    units_list = sorted(units_list)
    
    # Prepare the result dictionary
    result = {
        "units": units_list
    }

    return result

def home():
    """
    Home all Thorlabs linear stages.

    This function initiates homing for all connected Thorlabs linear stages using
    the `thorlabs_devices` module. It submits homing tasks to a ThreadPoolExecutor
    and waits for the tasks to complete.

    Executing this function is important to trust in the movement of the motor, this action is
    recommended by the manufacturers (Thorlabs).

    Returns:
    - result: A dictionary indicating the completion of homing with the key "home_ended".
    """
    
    # Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:            
        # Initialize an empty list to store the future objects for each device homing task         
        futures = []
        # Iterate over Thorlabs devices
        for device in thorlabs_devices.devices.values():
            # Submit a homing task to the executor and append the returned Future object to the list
            futures.append(executor.submit(thorlabs_devices.homming, device))
            print('enviados a homming')
        # Wait for all submitted tasks (homing) to complete
        concurrent.futures.wait(futures)

    # Wait until all devices finish homing
    while all(device.Status.IsHoming == True for device in thorlabs_devices.devices.values()):
        pass

    # Prepare the result dictionary indicating the completion of homing
    result = {
        "home_ended": True
    }

    return result

# Handling different commands using ElectronSocket

# Handle the command "linearstages_and_loadcells" with the function get_linearstages_and_loadcells
ElectronSocket.getInstance().handle("linearstages_and_loadcells", get_linearstages_and_loadcells)

# Handle the command "connect-motors" with the function connectMotors
ElectronSocket.getInstance().handle("connect-motors", connectMotors)

# Handle the command "motors-home" with the function home
ElectronSocket.getInstance().handle("motors-home", home)

# Handle the command "disconnect-motors" with the function disconnectMotors
ElectronSocket.getInstance().handle("disconnect-motors", disconnectMotors)

# Handle the command "motors-move" with the function motorMove
ElectronSocket.getInstance().handle("motors-move", motorMove)

# Handle the command "force-units" with the function listForceUnits
ElectronSocket.getInstance().handle("force-units", listForceUnits)

# Handle the command "motors-normal-displacement" with the function motorsNormaldisplacements
ElectronSocket.getInstance().handle("motors-normal-displacement", motorsNormaldisplacements)

# Handle the command "motors-hysteresis" with the function motorsHysteresis
ElectronSocket.getInstance().handle("motors-hysteresis", motorsHysteresis)

# Handle the command "motors-hysteresis_fp" with the function motorsHysteresis
ElectronSocket.getInstance().handle("motors-hysteresis_fp", motorsHysteresis_fp)

# Handle the command "motors-stress-relaxation" with the function motorsStressRelaxation
ElectronSocket.getInstance().handle("motors-stress-relaxation", motorsStressRelaxation)

# Handle the command "motors-stop" with the function motorsStop
ElectronSocket.getInstance().handle("motors-stop", motorsStop)

