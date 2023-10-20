from apscheduler.schedulers.background import BackgroundScheduler
import csv
import time
from datetime import datetime, timedelta
import pandas as pd

name = 'valelinda'
path = '.'  # Replace with your desired path
duration_seconds = 10  # Specify the duration in seconds

# # Create a list to store the data
# data_list = []

# def write_timestamp():
#     """This function writes a timestamp to the data list."""
#     timestamp = time.perf_counter()
#     data_list.append([timestamp])

# # Create a scheduler
# scheduler = BackgroundScheduler()

# # Calculate the time to stop the scheduler
# current_time = datetime.now()
# time_end = current_time + timedelta(seconds=duration_seconds)

# # Schedule the job to run every 0.001 seconds until time_end
# scheduler.add_job(write_timestamp, 'interval', seconds=0.001, end_date=time_end)

# # Start the scheduler
# scheduler.start()

# try:
#     # Run the scheduler until time_end is reached
#     while datetime.now() < time_end:
#         pass
# except (KeyboardInterrupt, SystemExit):
#     pass

# # Shut down the scheduler gracefully (optional)
# scheduler.shutdown()

# # Convert the data list to a DataFrame
# data = pd.DataFrame(data_list, columns=["seconds"])

# # Calculate the time differences
# data['seconds'] = data['seconds'] - data['seconds'].iloc[0]

# # Save the data to a CSV file
# data.to_csv(f'{name}.csv', index=False)
# data['seconds'] = data['seconds'].round(2)
# # Print the data and its statistics
# print(data)
# print(data.diff())
# print(data.diff().describe())
# print(data)

import time
import schedule
import pandas as pd

data_list = []

def write_timestamp():
    """This function writes a timestamp to the data list."""
    timestamp = time.perf_counter()
    data_list.append([timestamp])

# Create a scheduler object.
scheduler = schedule.Scheduler()

# Schedule the function `write_timestamp()` to run every 0.01 seconds for 10 seconds.
scheduler.every(0.01).seconds.do(write_timestamp)

# Start the scheduler.
scheduler.run_pending()

# Wait for 10 seconds.
start_time = time.perf_counter()
# while time.perf_counter() - start_time < 10 and len(data_list)<:
#     scheduler.run_pending()
while len(data_list)<1000+1:
    scheduler.run_pending()
# Stop the scheduler.
scheduler.clear()

# Convert the data list to a DataFrame
data = pd.DataFrame(data_list, columns=["seconds"])

# Calculate the time differences
data['seconds'] = data['seconds'] - data['seconds'].iloc[0]

# Save the data to a CSV file
data.to_csv(f'{name}.csv', index=False)
print(data)
# Round the seconds column to 2 decimal places
data['seconds'] = data['seconds'].round(2)

# Print the data and its statistics
print(data)
print(data.diff())
print(data.diff().describe())
print(1)
