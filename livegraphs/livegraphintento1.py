# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
import random


import os
import time
import clr

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
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def live_graph(file_path):
    print('live',file_path)
    print('VALELINDA')
    while True:
        try:
            data = pd.read_csv(file_path)
            
            if not data.empty:
                print(data)
                break
        except Exception as e:
            # Handle the exception here, you can print an error message or log it
            print(f"An exception occurred: {str(e)}")
  
            

def collect_data(sample_time, num_samples, path, name):
    
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses sched
    try:
        # Get the list of devices from the dictionary
        devices_list = [1,2,3,4]
        load_cells_list=[1,2]
        sample = 0
        columnas = ['seconds'] + ['real_position_' + str(elemento) +'_mm' for elemento in devices_list]+['load_cell_'+str(loadcell) for loadcell in load_cells_list]
        # Create a CSV file to save the data
        csv_file_path = f'{path}\\{name}.csv'
        print('collect',csv_file_path)
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(columnas) # Write the header to the CSV file
            def write_timestamp(sc):
                """This function writes a timestamp and device positions to the CSV file."""
                #initial_time=time.perf_counter()
                nonlocal sample, initial_time
                # Combine the results into a single array at the same time     
                data_line = [time.perf_counter()-initial_time] + [device*random.randint(1, 25) for device in devices_list]+[loadcell*random.randint(-5, 20) for loadcell in load_cells_list]
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
    
    except Exception as e:
        # Handle the exception here, you can print an error message or log it
        print(f"An exception occurred: {str(e)}")

if __name__ == "__main__":
    sample_time = 0.1  # Adjust as needed
    num_samples = 100  # Adjust as needed
    path = r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\intentolivegraphs"
    name = "intento3"
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        p1=executor.submit(collect_data,sample_time=sample_time, num_samples=num_samples, path=path, name=name)
        p2=executor.submit(live_graph,file_path=f'{path}\\{name}.csv')
        
        # Wait for all of the tasks to complete
        concurrent.futures.wait([p1,p2])
    # Create a thread for collecting data
    

