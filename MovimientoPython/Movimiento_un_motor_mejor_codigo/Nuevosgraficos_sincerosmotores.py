import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import dask.dataframe as dd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import datetime
import numpy as np
import pingouin as pg
import re
from permetrics.regression import RegressionMetric

import warnings
warnings.filterwarnings("ignore")
directory="C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Taguchi_un_motor_4factores\\9"

# List all files in the specified directory
files_in_directory = os.listdir(directory)

# Filter files to select those ending with 'all.csv' and sort them based on a specific pattern
selected_files = [file for file in files_in_directory if file.endswith('all.csv')]
#selected_files = sorted(selected_files, key=lambda x: int(x.split('_rep_')[1].split('_all.csv')[0])) #carpeta Taguchi_funcion_schedule
selected_files = sorted(selected_files, key=lambda x: int(x.split('_exp_')[1].split('_all.csv')[0])) #crpeta Taguchi_un_motor_4factores

#carpeta Taguchi_funcion_schedule
#From the file name take the settings made for the movement
# velocity = float(re.search(r'v_([\d.]+)', selected_files[0]).group(1)) #velocity
# pf = 25-float(re.search(r'pf_([\d.]+)', selected_files[0]).group(1)) # final position
# frequency = float(re.search(r'freq_(\d+)', selected_files[0]).group(1)) #sample frequency
# sampling_time=1/frequency

#carpeta crpeta Taguchi_un_motor_4factores
velocity = float(re.search(r'vel_([\d.]+)', selected_files[0]).group(1)) #velocity
pf = 25-float(re.search(r'pf_([\d.]+)', selected_files[0]).group(1)) # final position
frequency = float(re.search(r'samplefreq_(\d+)', selected_files[0]).group(1)) #sample frequency
sampling_time=1/frequency
# Initialize lists to store DataFrames and sampling data
dataframes = []
dataframes_vertical = []
sampling = []

# Iterate through the list of selected file names
for filename in selected_files:
    # Build the full file path
    filepath = os.path.join(directory, filename)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)
    
    # Perform column manipulation to save the dataframes to be joined horizontally
    #start_index = filename.find("rep_")#carpeta Taguchi_funcion_schedule
    start_index = filename.find("exp_") #carpeta Taguchi_un_motor_4factores
    end_index = filename.find("_all")
    #result = int(filename[start_index + len("rep_"):end_index])#carpeta Taguchi_funcion_schedule
    result = int(filename[start_index + len("exp_"):end_index])#carpeta Taguchi_un_motor_4factores
    columna = [columna for columna in df.columns if 'real' in columna]
    nueva_col = columna[0] + f'_m_{result}'
    df.rename(columns={columna[0]: nueva_col, 'seconds': f'seconds_m_{result}'}, inplace=True)
    #del df[[columna for columna in df.columns if 'relative' in columna][0]]
    
    # Add the modified DataFrame to the list
    dataframes.append(df)

    # Create a new copy of the DataFrame
    df = pd.read_csv(filepath)
    
    # Add 'cycle' and 'sampling_time' columns
    df['cycle'] = [int(result)] * len(df)
    df['sampling_time'] = df["seconds"].diff()
    #Create the Column sample to calculate later the ICC
    df['Sample'] = range(len(df))
    
    # Append the DataFrame to the vertical data list
    dataframes_vertical.append(df)
    
    
    # Create another copy for the 'sampling' list, to have only the information of the sample time
    df2 = df.copy()
    df2[f'sampling_time_m_{result}'] = df2['sampling_time']
    sampling.append(df2[f'sampling_time_m_{result}'])

# Concatenate DataFrames horizontally, vertically, and for sampling

# Concatenate DataFrames in a horizontal manner (side by side)
concatenated_df = pd.concat(dataframes, axis=1, ignore_index=False)

# Concatenate DataFrames vertically (stacked on top of each other)
concatenated_df_vertical = pd.concat(dataframes_vertical, ignore_index=True)
sampling_horizontal = pd.concat(sampling, axis=1, ignore_index=False)
sampling_horizontal = sampling_horizontal.drop([0])

position_columns = [col for col in concatenated_df.columns if 'real_position' in col]

# Iterate through the rows and find the first index where all positions are equal to pf
matching_index = None
for idx, row in concatenated_df.iterrows():
    if all(row[col] == pf for col in position_columns):
        matching_index = idx
        break  # Break out of the loop when the first match is found

# Slice the DataFrame up to the index where the condition is met
if matching_index is not None:
    concatenated_df = concatenated_df.loc[:matching_index]
    sampling_horizontal=sampling_horizontal.loc[:matching_index-1]
    concatenated_df_vertical=concatenated_df_vertical[concatenated_df_vertical['Sample'] <= matching_index]

# Select columns containing 'real' and 'seconds' in their names
positions = [columna for columna in concatenated_df.columns if 'real' in columna]
times = [columna for columna in concatenated_df.columns if 'seconds' in columna]

# Create DataFrames for position and time data, and their transposed versions
df_positions = concatenated_df[positions].copy()
df_times = concatenated_df[times].copy()
df_positions_T = df_positions.T
df_times_T = df_times.T

# Crear una lista de tiempo que aumente con un paso de 1/100 y redondear a 6 cifras significativas
tiempo_teorico = [round(i/frequency, 6) for i in range(len(df_positions))]

# Crear una lista de posición que comience en 25 y alcance 18.75 con velocidad 1 y redondear a 6 cifras significativas
posicion_teorica = [round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in tiempo_teorico]



# Crear un DataFrame con las listas de tiempo y posición
df_teorico = pd.DataFrame({'Tiempo_teorico': tiempo_teorico, 'Posicion_teorico': posicion_teorica})


# Número de experimentos (columnas en el DataFrame)
num_experimentos = df_positions.shape[1]
# Define a larger font size
font_size = 14

posicion_teorica = np.array(df_teorico['Posicion_teorico'])
tiempo_teorico = np.array(df_teorico['Tiempo_teorico'],)
# Loop through the samples and plot both position and time with different colors
for i in range(num_experimentos):
    tiempo_teorico_graph = [round(i/frequency, 6) for i in range(int(df_times.iloc[-1,i]*frequency))]

    # Crear una lista de posición que comience en 25 y alcance 18.75 con velocidad 1 y redondear a 6 cifras significativas
    posicion_teorica_graph = [round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in tiempo_teorico_graph]

    columns_to_select = df_positions.filter(regex=f'_m_{i+1}$').columns
    non_zero_indices1 = df_positions[columns_to_select[0]] != 0
    plt.figure(figsize=(10, 6))
    plt.plot(df_times.iloc[:,i][non_zero_indices1], df_positions[columns_to_select[0]][non_zero_indices1], label='Real', color='#e85f04')
    plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico', color='#007acc',linestyle='dashed')
    plt.xlabel('Tiempo (s)', fontsize=font_size)  # Increase font size for the x-axis label
    plt.ylabel('Posición (mm)', fontsize=font_size)  # Increase font size for the y-axis label
    plt.title(f'Movimiento real y movimiento teorico obtenido para el experimento numero {i}', fontsize=font_size)  # Increase font size for the title
    plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
    plt.legend(fontsize=font_size)  # Increase font size for the legend

    # Save the plot as an image (you can adjust the extension as needed)
    plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i}_spanish_sinceros.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_times.iloc[:,i][non_zero_indices1], df_positions[columns_to_select[0]][non_zero_indices1], label='Real', color='#e85f04')
    plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Theoretical', color='#007acc',linestyle='dashed')
    plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
    plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
    plt.title(f'Real motion and theoretical motion obtained for experiment number {i}', fontsize=font_size)  # Increase font size for the title
    plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
    plt.legend(fontsize=font_size)  # Increase font size for the legend

    # Save the plot as an image (you can adjust the extension as needed)
    plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i}_english_sinceros.png')
    plt.close()

# Define colors for the lines
colors = ['#007acc', '#e85f04', '#d62728', '#9467bd', '#8c564b',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#bcbd22',
          '#f7b6d2', '#c7c7c7', '#98df8a', '#ff9896', '#ffbb78',
          '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d']
tiempo_teorico_graph = [round(i/frequency, 6) for i in range(int(df_times.iloc[-1,:].max()*frequency))]

# Crear una lista de posición que comience en 25 y alcance 18.75 con velocidad 1 y redondear a 6 cifras significativas
posicion_teorica_graph = [round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in tiempo_teorico_graph]
# Create a larger title, legend, and labels for the final plot
plt.figure(figsize=(10, 6))
for i in range(num_experimentos):
    columns_to_select = df_positions.filter(regex=f'_m_{i+1}$').columns
    non_zero_indices1 = df_positions[columns_to_select[0]] != 0
    plt.plot(df_times.iloc[:, i][non_zero_indices1], df_positions.iloc[:, i][non_zero_indices1], label=f'Experimento {i}', color=colors[i])

plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico',linestyle='dashed' )
plt.xlabel('Tiempo (s)', fontsize=font_size)  # Increase font size for the x-axis label
plt.ylabel('Posición (mm)', fontsize=font_size)  # Increase font size for the y-axis label
plt.title('Movimiento real y movimiento teórico para todos los experimentos', fontsize=font_size)  # Increase font size for the title
plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size*0.8)  # Increase font size for the legend


# Save the plot as an image (you can adjust the extension as needed)
plt.savefig(f'{directory}\\Posicionvstiempo_experimento_all_spanis_sinceros.png', bbox_inches='tight')
plt.close()


# Create a larger title, legend, and labels for the final plot in English
plt.figure(figsize=(10, 6))
for i in range(num_experimentos):
    columns_to_select = df_positions.filter(regex=f'_m_{i+1}$').columns
    non_zero_indices1 = df_positions[columns_to_select[0]] != 0
    plt.plot(df_times.iloc[:, i][non_zero_indices1], df_positions.iloc[:, i][non_zero_indices1], label=f'Experiment {i}', color=colors[i])

plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Theoretical',linestyle='dashed' )
plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
plt.title('Real motion and theoretical motion obtained for all experiments', fontsize=font_size)  # Increase font size for the title
plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size*0.8)  # Increase font size for the legend

plt.savefig(f'{directory}\\Posicionvstiempo_experimento_all_english_sinceros.png', bbox_inches='tight')
plt.close()
print('valelinda')