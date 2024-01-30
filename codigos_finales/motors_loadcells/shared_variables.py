"""
    This code is utilized to manage a shared variable across different files, enabling changes made to the variable in one file to be 
    recognized by code executing in another file. It is primarily used to track the completion status of desired motor movements, 
    allowing code in different modules to know when to cease recording position and force data. For example, it is employed to determine
    when the motor has finished executing a desired movement or when the stop button has been pressed, indicating that data recording 
    should be halted.
    
    Author: Valeria Cadavid Castro
    Last Modified: January 18, 2024
"""

# shared state variable indicating whether motor movements are completed
movements_motors_completed = False

def set_movements_motors_completed(value):
    """
    Set the shared state variable to indicate the completion status of motor movements.

    Args:
        value (bool): The value to set for the completion status.

    Returns:
        None

    This function sets the shared state variable 'movements_motors_completed' to the specified boolean value,
    indicating whether motor movements are completed.

    """
    global movements_motors_completed
    movements_motors_completed = value

def get_movements_motors_completed():
    """
    Retrieve the current value of the shared state variable indicating motor movements completion.

    Returns:
        bool: The current value of 'movements_motors_completed'.

    This function returns the current value of the shared state variable 'movements_motors_completed',
    which indicates whether motor movements are completed.

    """
    return movements_motors_completed