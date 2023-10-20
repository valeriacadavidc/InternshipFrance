import time
import pandas as pd
import numpy as np
import csv
import sched
import cython



def collect_data(devices_dictionary, sample_time, num_samples, path, name):
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses sched
    try:
        # Get the list of devices from the dictionary
        devices_list = list(devices_dictionary.values())
        sample = 0
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in devices_dictionary.keys()]
        # Create a CSV file to save the data
        csv_file_path = f'{path}\\{name}.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header to the CSV file
            csv_writer.writerow(columnas)
            def write_timestamp(sc):
                """This function writes a timestamp and device positions to the CSV file."""
                nonlocal sample
                data_line = [time.perf_counter()] + [device.Position for device in devices_list]
                sample += 1
                csv_writer.writerow(data_line)
                if sample < num_samples:
                    sc.enter(sample_time, 1, write_timestamp, (sc,))
            # Create a scheduler object
            s = sched.scheduler(time.perf_counter, time.sleep)
            # Schedule the function `write_timestamp()` to run immediately and then repeatedly every sample_time seconds
            s.enter(0, 1, write_timestamp, (s,))
            s.run()
        print('valelinda')
        data = pd.read_csv(f'{path}\\{name}.csv')
        data['seconds'] = data['seconds'] - data['seconds'].iloc[0]
        data.to_csv(f'{path}\\{name}_all.csv', index=False)
    except Exception as e:
        # Handle the exception here, you can print an error message or log it
        print(f"An exception occurred: {str(e)}")
        data = pd.read_csv(f'{path}\\{name}.csv')
        data['seconds'] = data['seconds'] - data['seconds'].iloc[0]
        data.to_csv(f'{path}\\{name}_all.csv', index=False)