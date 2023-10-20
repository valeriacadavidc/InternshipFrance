from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("optimized_function.pyx"),
)

#python setup.py build_ext --inplace in the command

# cpdef void collect_data_cython(dict devices_dictionary, double sample_time, int num_samples, str path, str name):
#     # Declare variables and types
#     cdef list devices_list = list(devices_dictionary.values())
#     cdef int sample = 0
#     cdef list columnas = ['seconds'] + ['real_position_' + elemento for elemento in devices_dictionary.keys()]
    
#     # Create a CSV file to save the data
#     cdef str csv_file_path = f'{path}\\{name}.csv'
#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(columnas)
        
#         cdef void write_timestamp(sched.scheduler sc):
#             cdef int sample = 0
#             cdef float data_line_time
#             cdef list data_line_positions
            
#             while sample < num_samples:
#                 data_line_time = time.perf_counter()
#                 data_line_positions = [device.Position for device in devices_list]
#                 csv_writer.writerow([data_line_time] + data_line_positions)
#                 sample += 1
#                 sc.enter(sample_time, 1, write_timestamp, (sc,))
                
#         # Create a scheduler object
#         cdef sched.scheduler s = sched.scheduler(time.perf_counter, time.sleep)
#         # Schedule the function `write_timestamp()` to run immediately and then repeatedly every sample_time seconds
#         s.enter(0, 1, write_timestamp, (s,))
#         s.run()
        
#     # Process the data after the collection
#     data = pd.read_csv(f'{path}\\{name}.csv')
#     data['seconds'] = data['seconds'] - data['seconds'].iloc[0]
#     data.to_csv(f'{path}\\{name}_all.csv', index=False)