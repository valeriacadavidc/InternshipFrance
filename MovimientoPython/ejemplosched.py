import datetime
import sched
import time

# Initialize variables
sample_interval = 1.0 / 20  # 20 samples per second
sample_count = 20
data_list = []

# # Create a scheduler
# s = sched.scheduler(time.time, time.sleep)

# # Define a function to perform the sampling
# def sample_data(sc, sample_num):
#     # Record the start time of the sample
#     sample_start_time = time.time()

#     # Perform your data collection here, replace this line with your data collection logic
#     data = datetime.datetime.now() # Initialize with the current time

#     # Append the data to your data list
#     data_list.append(data)

#     # Calculate the time elapsed for this sample
#     sample_elapsed_time = time.time() - sample_start_time

#     # Calculate the time to sleep until the next sample
#     sleep_time = sample_interval - sample_elapsed_time

#     # If sleep_time is negative, continue immediately to the next sample
#     if sleep_time < 0:
#         sleep_time = 0

#     # Schedule the next sample
#     if sample_num < sample_count:
#         s.enter(sleep_time, 1, sample_data, (sc, sample_num + 1))

# # Schedule the initial sample
# s.enter(0, 1, sample_data, (s, 1))

# # Run the scheduler
# s.run()

# time_diff_list = [data_list[i] - data_list[i - 1] for i in range(1, len(data_list))]
# for i, data in enumerate(time_diff_list):
#     print(f"Sample {i+1}: {data[0]}")
# # Display the collected data
# for i, data in enumerate(data_list):
#     print(f"Sample {i+1}: {data[0]}")


import signal
import datetime

# Define the signal handler function
def signal_handler(signum, frame):
    current_time = datetime.datetime.now()
    time_samples.append(current_time)

# Set the desired frequency (in Hz)
desired_frequency = 10
num_samples = 10

# Calculate the time interval in seconds for the desired frequency
time_interval = 1.0 / desired_frequency

# Initialize a list to store time samples
time_samples = []

# Register the signal handler for the SIGALRM signal
signal.signal(signal.SIGALRM, signal_handler)

# Set up a recurring timer to trigger the SIGALRM signal
signal.setitimer(signal.ITIMER_REAL, time_interval, time_interval)

# Wait until the desired number of samples is captured
while len(time_samples) < num_samples:
    pass

# Stop the timer
signal.setitimer(signal.ITIMER_REAL, 0)

# Print the captured time samples
for sample in time_samples:
    print(sample)