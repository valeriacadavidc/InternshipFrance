
import os
import time
import clr
import System
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
import matplotlib.pyplot as plt
def collect_data(sample_time, num_samples, data_ready_flag):
    try:
        sample = 0
        columnas = ['seconds'] 
        csv_file_path = 'hola.csv'
        
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(columnas)

            def write_timestamp(sc):
                nonlocal sample, initial_time, t1, t2
                data_line = [time.perf_counter() - initial_time]
                sample += 1
                csv_writer.writerow(data_line)
                if sample < num_samples:
                    t1 = time.perf_counter()
                    sc.enter((data_line[0] + sample_time - time.perf_counter() + initial_time - t2) * 0.9, 1, write_timestamp, (sc,))
                    t2 = time.perf_counter() - t1
                else:
                    # Set the flag to signal that data collection is complete
                    data_ready_flag.set()

            s = sched.scheduler(time.perf_counter, time.sleep)
            initial_time = time.perf_counter()
            t1 = time.perf_counter()
            s.enter(0, 1, write_timestamp, (s,))
            t2 = time.perf_counter() - t1
            s.run()

    except Exception as e:
        print(f"An exception occurred: {str(e)}")

def update_graph(csv_file_path, data_ready_flag):
    try:
        x_vals = []  # To store the x-axis values (seconds)
        y_vals = []  # To store the y-axis values (data)

        plt.ion()  # Turn on interactive mode for matplotlib
        fig, ax = plt.subplots()

        while not data_ready_flag.is_set():
            # Read the CSV file and update the graph with new data
            with open(csv_file_path, mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Skip the header
                for row in csv_reader:
                    seconds, = map(float, row)
                    x_vals.append(seconds)
                    y_vals.append(0)  # You can replace 0 with the actual data value if available

                ax.clear()
                ax.plot(x_vals, y_vals, marker='o', linestyle='-')
                plt.xlabel('Seconds')
                plt.ylabel('Data Value')
                plt.pause(0.1)  # Allow time for the plot to update

        # After data collection is complete, show the final graph
        plt.show()

    except Exception as e:
        print(f"An exception occurred while updating the graph: {str(e)}")

if __name__ == "__main__":
    csv_file_path = 'hola.csv'
    data_ready_flag = threading.Event()

    data_collection_thread = threading.Thread(target=collect_data, args=(0.1, 100, data_ready_flag))  # Adjust sample_time and num_samples accordingly
    graph_update_thread = threading.Thread(target=update_graph, args=(csv_file_path, data_ready_flag))

    data_collection_thread.start()
    graph_update_thread.start()

    data_collection_thread.join()
    graph_update_thread.join()