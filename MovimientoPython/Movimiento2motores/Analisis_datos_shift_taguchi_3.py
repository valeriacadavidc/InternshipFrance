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

# Function to perform Anderson-Darling test
def anderson_darling_test(series):
    result = stats.anderson(series)
    critical_value = result.critical_values[2]  # Use the critical value at significance level 0.05
    if result.statistic < critical_value:
        return "Normal"
    else:
        return "Not normal"
    
def shapiro_p_value(dataframe):
    normality_p_values = {}

    # Loop through each column in the DataFrame
    for column in dataframe.columns:
        _, p_value = stats.shapiro(dataframe[column].dropna())  # Perform the Shapiro-Wilk test
        normality_p_values[column] = p_value  # Store the p-value in a dictionary

    # Create a new DataFrame to store the p-values
    p_value_df = pd.DataFrame.from_dict(normality_p_values, orient='index', columns=['Shapiro-Wilk P-Value'])

    return p_value_df

def shapiro_p_value1(series):
    _, p_value = stats.shapiro(series)
    return p_value
def Agostino_p_value(series):
    _, p_value= stats.normaltest(series)
    return p_value

def text_format(val,value):
    if value==0.2: #Cambie el 0.05 por 0.2 y el lightgreen por lightblue
        color = 'lightblue' if val <0.2 else 'white'
    if value==0.05:
        color = 'lightgreen' if val >0.05 else 'white'
    if value==0.0:
        color = 'lightblue' if np.abs(val)<=0.05 else 'white'
    if value=='anderson':
        color = 'lightblue' if val=='Normal' else 'white'
    return 'background-color: %s' % color


# Set the directory path 
#directory = "C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Prueba_schedule"
#directory=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\\Prueba_schedule_ciclofor_inthecode"
directory="C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Taguchi3_dos_motores_2factores\\3"

# List all files in the specified directory
files_in_directory = os.listdir(directory)

# Filter files to select those ending with 'all.csv' and sort them based on a specific pattern
selected_files = [file for file in files_in_directory if file.endswith('all.csv')]
selected_files = sorted(selected_files, key=lambda x: int(x.split('_exp_')[1].split('_all.csv')[0]))

#From the file name take the settings made for the movement
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
    start_index = filename.find("exp_")
    end_index = filename.find("_all")
    result = int(filename[start_index + len("rep_"):end_index])
    columns_to_rename = [columna for columna in df.columns if 'real' in columna]
    df.rename(columns={'seconds': f'seconds_m_{result}'}, inplace=True)
    for column in columns_to_rename:
        new_column_name = column + f'_m_{result}'
        df.rename(columns={column: new_column_name}, inplace=True)
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

concatenated_df
# Concatenate DataFrames for sampling horizontally (side by side), sample times only
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

# Calculate statistics for 'sampling_time' data for the analysis by row of sampling data
df_sampling_time = concatenated_df_vertical[['sampling_time', 'cycle']]
cycle_groups_sampling = df_sampling_time.groupby('cycle')
descriptions_sampling = cycle_groups_sampling.describe()


cycle_groups = concatenated_df_vertical.groupby('cycle')
descriptions_experiments= cycle_groups.describe()



# Create an Excel writer to save the results
writer1 = pd.ExcelWriter(f'{directory}\\results.xlsx', engine='openpyxl')


# Write concatenated data and descriptions to the Excel file
# concatenated_df_vertical.to_excel(writer1, sheet_name='datos_columnas_vertical')
descriptions_experiments.to_excel(writer1, sheet_name='datos_columnas_vertical', startcol=concatenated_df_vertical.shape[1] + 3) #the description of sampling data

# Calculate statistics for all data in the 'sampling_time' column of concatenated_df_vertical, all the sampling times of the all experiments joined
all_data_sampling_time = concatenated_df_vertical['sampling_time'].copy()
medias = np.mean(all_data_sampling_time, axis=0)
desviaciones = np.std(all_data_sampling_time, axis=0, ddof=1)
n = all_data_sampling_time.count()  # Number of samples
errores_tipicos = desviaciones / np.sqrt(n)
varianzas = np.var(all_data_sampling_time, axis=0, ddof=1)
margen_de_error = 1.96 * errores_tipicos
interval_inf = medias - margen_de_error
interval_sup = medias + margen_de_error
amplitud = interval_sup - interval_inf
amplitud_2 = np.power(amplitud, 2)
muestras = (np.power(1.96, 2) * varianzas) / amplitud_2

# Create a Series to store results
data = {
    'Medias': medias,
    'Desviaciones Estándar': desviaciones,
}


df_resultados_muestras_tiempos = pd.Series(data) #Results just with medias and desviaciones estandar
df_resultados_muestras_tiempos = df_resultados_muestras_tiempos.rename('Sampling time')
mi_dataframe_data_sampling = pd.DataFrame({'data_sampling': all_data_sampling_time})
mi_dataframe_data_sampling=mi_dataframe_data_sampling.describe()
# Write additional statistics to the Excel file
df_resultados_muestras_tiempos.to_excel(writer1, sheet_name='datos_columnas_vertical', startcol=concatenated_df_vertical.shape[1] + 3, startrow=len(descriptions_sampling) + 5)
mi_dataframe_data_sampling.to_excel(writer1, sheet_name='datos_columnas_vertical', startcol=concatenated_df_vertical.shape[1] + 3+3, startrow=len(descriptions_sampling) + 5)
writer1.close()
# Define un tamaño de fuente más grande
font_size = 14

# Configurar el tamaño de fuente global para todos los gráficos
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size*0.8)
plt.rc('ytick', labelsize=font_size*0.8)

# # Create a boxplot of the sample time per experiment
# plt.figure(figsize=(10, 6))
# sns.set_style("white")
# sns.boxplot(x='cycle', y='sampling_time', data=concatenated_df_vertical, color='royalblue')
# plt.ylabel('Sampling time (s)')
# plt.xlabel('Experiment')
# plt.title(f'Sampling time for each experiment\nExpected sampling time: {round(sampling_time,4)}')
# plt.savefig(f'{directory}\\sampletime_english.png')
# plt.close()

# # Create a boxplot of the sample time per experiment spanish version
# plt.figure(figsize=(10, 6))
# sns.set_style("white")
# sns.boxplot(x='cycle', y='sampling_time', data=concatenated_df_vertical, color='royalblue')
# plt.ylabel('Tiempo de muestreo (s)')
# plt.xlabel('Experimento')
# plt.title(f'Tiempo de muestreo para cada experimento\nTiempo de muestreo esperado: {round(sampling_time,4)}')
# plt.savefig(f'{directory}\\sampletime_spanish.png')
# plt.close()

# Create a histogram of the sampling time for all the data, of the 20 experiments
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.histplot(data=concatenated_df_vertical, x='sampling_time', kde=True, palette="mako")  # Adjust the number of bins as needed
plt.title(f'Histogram of sampling time obtained from all experiments\nExpected sampling time: {round(sampling_time,4)}\nMean: {medias:.3f}, SD: {desviaciones:.3f}')
plt.xlabel('Sampling time (s)')
plt.ylabel('Frequency')
plt.savefig(f'{directory}\\histogram_english.png')
plt.close()

# Create a histogram of the sampling time for all the data, of the 20 experiments, spanish version
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.histplot(data=concatenated_df_vertical, x='sampling_time', kde=True, palette="mako")  # Adjust the number of bins as needed
plt.title(f'Histograma del tiempo de muestreo obtenido de todos los experimentos\nTiempo de muestreo esperado: {round(sampling_time,4)}\nPromedio: {medias:.3f}, DE: {desviaciones:.3f}')
plt.xlabel('Tiempo de muestreo (s)')
plt.ylabel('Frecuencia')
plt.savefig(f'{directory}\\histogram_spanish.png')
plt.close()

# Create a FacetGrid for histograms for each experiment
#g = sns.FacetGrid(data=concatenated_df_vertical, col='cycle', col_wrap=5, aspect=1.2)

# Create histograms with KDE for 'sampling_time' in each subplot
# g.map(sns.histplot, 'sampling_time', kde=True, palette="YlGnBu_r")
# plt.suptitle(f'Histogram of Sampling Time Obtained for Each Experiment\nExpected sampling time: {round(sampling_time,4)}', fontsize=font_size*1.5)
# plt.subplots_adjust(top=0.85)
# plt.xlabel('Sampling Time (s)')
# plt.ylabel('Frequency')
# g.set_titles("Experiment {col_name}", fontsize=font_size)
# plt.tight_layout()
# plt.savefig(f'{directory}\\histogram_per_experiment_english.png')
# plt.close()

# # Create a FacetGrid for histograms for each experiment, spanish version
# g = sns.FacetGrid(data=concatenated_df_vertical, col='cycle', col_wrap=5, aspect=1.2)

# # Create histograms with KDE for 'sampling_time' in each subplot
# g.map(sns.histplot, 'sampling_time', kde=True, palette="YlGnBu_r")
# plt.suptitle(f'Histograma del tiempo de muestreo obtenido para cada experimento\nTiempo de muestreo esperado: {round(sampling_time,4)}', fontsize=font_size*1.5)
# plt.subplots_adjust(top=0.85)
# plt.xlabel('Tiempo de muestreo (s)')
# plt.ylabel('Frecuencia')
# g.set_titles("Experimento {col_name}", fontsize=font_size)
# plt.tight_layout()
# plt.savefig(f'{directory}\\histogram_per_experiment_spanish.png')
# plt.close()
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

# Calcular el MAPE
mape_dict_positions_teorica = {}
mape_dict_positions_teorica_ajustada = {}
mape_dict_tiempos = {}
mape_dict_sampling = {}


mape_dict_positions_teorica_multioutput = {}
mape_dict_positions_teorica_ajustada_multioutput = {}
mape_dict_tiempos_multioutput = {}
mape_dict_sampling_multioutput = {}


posicion_teorica = np.array(df_teorico['Posicion_teorico'])
tiempo_teorico = np.array(df_teorico['Tiempo_teorico'],)
# Loop through the samples and plot both position and time with different colors
for i in range(num_experimentos):
    if i>num_experimentos/2-1:
        j=i-5
    else:
        j=i
    posicion_teorica_ajustada = np.array([round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in df_times.iloc[:,j]])
    mape_dict_positions_teorica[f'{df_positions.columns[i]}'] = round(mean_absolute_percentage_error(posicion_teorica, np.array(df_positions.iloc[:, j])) * 100, 5)
    mape_dict_positions_teorica_ajustada[f'{df_positions.columns[i]}'] = round(mean_absolute_percentage_error(posicion_teorica_ajustada, np.array(df_positions.iloc[:, j])) * 100, 5)
    
    #Multioutput
    # Compute values for positions
    mape_dict_positions_teorica_multioutput[f'MAPE_{df_positions.columns[i]}'] = mean_absolute_percentage_error(posicion_teorica.reshape(1, -1), np.array(df_positions.iloc[:, j]).reshape(1, -1), multioutput='raw_values')*100

    # Compute values for adjusted positions
    mape_dict_positions_teorica_ajustada_multioutput[f'MAPE_{df_positions.columns[i]}'] = mean_absolute_percentage_error(posicion_teorica_ajustada.reshape(1, -1), np.array(df_positions.iloc[:, j]).reshape(1, -1), multioutput='raw_values') *100

    if i<num_experimentos/2-1:
        mape_dict_tiempos[f'{df_times.columns[i]}'] = round(mean_absolute_percentage_error(tiempo_teorico, np.array(df_times.iloc[:, i])) * 100, 5)

        mape_dict_sampling[f'{sampling_horizontal.columns[i]}'] = round(mean_absolute_percentage_error(np.array([sampling_time]*len(sampling_horizontal)), np.array(sampling_horizontal.iloc[:, j])) * 100, 5)

        # Compute values for sampling times
        mape_dict_tiempos_multioutput[f'MAPE_{df_times.columns[i]}'] = mean_absolute_percentage_error(tiempo_teorico.reshape(1, -1), np.array(df_times.iloc[:, j]).reshape(1, -1), multioutput='raw_values')*100
    
        # Compute values for sampling
        mape_dict_sampling_multioutput[f'MAPE_{sampling_horizontal.columns[i]}'] = mean_absolute_percentage_error(np.array([sampling_time]*len(sampling_horizontal)).reshape(1, -1), np.array(sampling_horizontal.iloc[:, i]).reshape(1, -1), multioutput='raw_values') *100

        tiempo_teorico_graph = [round(i/frequency, 6) for i in range(int(df_times.iloc[-1,i]*frequency))]

        # Crear una lista de posición que comience en 25 y alcance pf con velocidad yredondear a 6 cifras significativas
        posicion_teorica_graph = [round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in tiempo_teorico_graph]
        columns_to_select = df_positions.filter(regex=f'_m_{i+1}$').columns
        non_zero_indices1 = df_positions[columns_to_select[0]] != 0
        non_zero_indices2 = df_positions[columns_to_select[1]] != 0
        plt.figure(figsize=(10, 6))
        plt.plot(df_times.iloc[:,i][non_zero_indices1], df_positions[columns_to_select[0]][non_zero_indices1], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
        plt.plot(df_times.iloc[:,i][non_zero_indices2], df_positions[columns_to_select[1]][non_zero_indices2], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
        plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico', color='#007acc',linestyle='dashed')
        plt.xlabel('Tiempo (s)', fontsize=font_size)  # Increase font size for the x-axis label
        plt.ylabel('Posición (mm)', fontsize=font_size)  # Increase font size for the y-axis label
        plt.title(f'Movimiento real y movimiento teorico obtenido para el experimento numero {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
        plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
        plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
        plt.legend(fontsize=font_size)  # Increase font size for the legend
        plt.grid(True)
        # Save the plot as an image (you can adjust the extension as needed)
        plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_spanish.png')
        plt.close()
        plt.figure(figsize=(10, 6))
        
        plt.plot(df_times.iloc[:,i][non_zero_indices1], df_positions[columns_to_select[0]][non_zero_indices1], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
        plt.plot(df_times.iloc[:,i][non_zero_indices2], df_positions[columns_to_select[1]][non_zero_indices2], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
        plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico', color='#007acc',linestyle='dashed')
        plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
        plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
        plt.title(f'Real motion and theoretical motion obtained for experiment number {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
        plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
        plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
        plt.legend(fontsize=font_size)  # Increase font size for the legend
        plt.grid(True)

        # Save the plot as an image (you can adjust the extension as needed)
        plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_english.png')
        plt.close()
        # plt.figure(figsize=(10, 6))
        # plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[0]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
        # plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[1]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
        # plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico', color='#007acc',linestyle='dashed')
        # plt.xlabel('Tiempo (s)', fontsize=font_size)  # Increase font size for the x-axis label
        # plt.ylabel('Posición (mm)', fontsize=font_size)  # Increase font size for the y-axis label
        # plt.title(f'Movimiento real y movimiento teorico obtenido para el experimento numero {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
        # plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
        # plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
        # plt.legend(fontsize=font_size)  # Increase font size for the legend
        # plt.grid(True)
        # # Save the plot as an image (you can adjust the extension as needed)
        # plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_spanish.png')
        # plt.close()
        # plt.figure(figsize=(10, 6))
        
        # plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[0]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
        # plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[1]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
        # plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico', color='#007acc',linestyle='dashed')
        # plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
        # plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
        # plt.title(f'Real motion and theoretical motion obtained for experiment number {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
        # plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
        # plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
        # plt.legend(fontsize=font_size)  # Increase font size for the legend
        # plt.grid(True)

        # # Save the plot as an image (you can adjust the extension as needed)
        # plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_english.png')
        # plt.close()


combined_df_positions = pd.DataFrame({
    'MAPE': mape_dict_positions_teorica
})

combined_df_positions_ajustada = pd.DataFrame({
    'MAPE_ajustada': mape_dict_positions_teorica_ajustada
})

combined_df_tiempos = pd.DataFrame({

    'MAPE': mape_dict_tiempos
})

combined_df_sampling = pd.DataFrame({
    'MAPE': mape_dict_sampling
})


mape_df_positions_multioutput = pd.DataFrame(mape_dict_positions_teorica_multioutput)


mape_df_positions_ajustada_multioutput = pd.DataFrame(mape_dict_positions_teorica_ajustada_multioutput)

mape_df_tiempos_multioutput = pd.DataFrame(mape_dict_tiempos_multioutput)

mape_df_sampling_multioutput = pd.DataFrame(mape_dict_sampling_multioutput)

combined_df_positions['SerialNumber'] = list(combined_df_positions.index.str.extract(r'real_position_(\d+)_m_(\d+)')[0])
describe_total = combined_df_positions['MAPE'].describe()
combined_df_positions['SerialNumber'] = combined_df_positions['SerialNumber'].astype(int)
# Describe for groups based on the serial number
describe_grouped = combined_df_positions.groupby('SerialNumber')['MAPE'].describe()


describe_positions = combined_df_positions.describe()
describe_positions_ajustada = combined_df_positions_ajustada.describe()
describe_tiempos = combined_df_tiempos.describe()
describe_sampling = combined_df_sampling.describe()

#PRUEBAS DE NORMALIDAD
data_positions = {
    'Anderson-Darling Test': combined_df_positions.apply(anderson_darling_test),
    'Shapiro-Wilk Test P-Value': combined_df_positions.apply(shapiro_p_value1)
}
df_resultados_muestras_positions = pd.DataFrame(data_positions)

data_positions_ajustada = {
    'Anderson-Darling Test': combined_df_positions_ajustada.apply(anderson_darling_test),
    'Shapiro-Wilk Test P-Value': combined_df_positions_ajustada.apply(shapiro_p_value1)
}
df_resultados_muestras_positions_ajustada = pd.DataFrame(data_positions_ajustada)

data_tiempos = {
    'Anderson-Darling Test': combined_df_tiempos.apply(anderson_darling_test),
    'Shapiro-Wilk Test P-Value': combined_df_tiempos.apply(shapiro_p_value1)
}
df_resultados_muestras_tiempos = pd.DataFrame(data_tiempos)

data_sampling = {
    'Anderson-Darling Test': combined_df_sampling.apply(anderson_darling_test),
    'Shapiro-Wilk Test P-Value': combined_df_sampling.apply(shapiro_p_value1)
}
df_resultados_muestras_sampling = pd.DataFrame(data_sampling)

# Apply styling
table_df_resultados_muestras_positions = df_resultados_muestras_positions.style.map(text_format,value='anderson',subset=['Anderson-Darling Test']).map(text_format,value=0.05,subset=['Shapiro-Wilk Test P-Value'])
table_df_resultados_muestras_positions_ajustada = df_resultados_muestras_positions_ajustada.style.map(text_format,value='anderson',subset=['Anderson-Darling Test']).map(text_format,value=0.05,subset=['Shapiro-Wilk Test P-Value'])
table_df_resultados_muestras_tiempos = df_resultados_muestras_tiempos.style.map(text_format,value='anderson',subset=['Anderson-Darling Test']).map(text_format,value=0.05,subset=['Shapiro-Wilk Test P-Value'])
table_df_resultados_muestras_sampling = df_resultados_muestras_sampling.style.map(text_format,value='anderson',subset=['Anderson-Darling Test']).map(text_format,value=0.05,subset=['Shapiro-Wilk Test P-Value'])


writer1 = pd.ExcelWriter(f'{directory}\\results_positions.xlsx', engine='openpyxl')

#df_positions.to_excel(writer1, sheet_name='Pos_Err_Row')
mape_df_positions_multioutput.to_excel(writer1, sheet_name='Pos_Err_Row',startcol=df_positions.shape[1]+2+mse_df_positions_multioutput.shape[1]+2)
combined_df_positions.to_excel(writer1, sheet_name='Pos_Err_Row',startcol=df_positions.shape[1]+2+mse_df_positions_multioutput.shape[1]+2+mape_df_positions_multioutput.shape[1]+2)
describe_positions.to_excel(writer1, sheet_name='Pos_Err_Row',startrow=combined_df_positions.shape[0] + 2,startcol=df_positions.shape[1]+2+mse_df_positions_multioutput.shape[1]+2+mape_df_positions_multioutput.shape[1]+2)
table_df_resultados_muestras_positions.to_excel(writer1, sheet_name='Pos_Err_Row',startrow=combined_df_positions.shape[0] + 2+describe_positions.shape[0]+2,startcol=df_positions.shape[1]+2+mse_df_positions_multioutput.shape[1]+2+mape_df_positions_multioutput.shape[1]+2)
writer1.close()

writer1 = pd.ExcelWriter(f'{directory}\\results_positions_ajus.xlsx', engine='openpyxl')

#df_positions.to_excel(writer1, sheet_name='Pos_Aj_Err_Row')
mape_df_positions_ajustada_multioutput.to_excel(writer1, sheet_name='Pos_Aj_Err_Row',startcol=df_positions.shape[1]+2+mse_df_positions_ajustada_multioutput.shape[1]+2)
combined_df_positions_ajustada.to_excel(writer1, sheet_name='Pos_Aj_Err_Row',startcol=df_positions.shape[1]+2+mse_df_positions_ajustada_multioutput.shape[1]+2+2+mape_df_positions_ajustada_multioutput.shape[1])
describe_positions_ajustada.to_excel(writer1, sheet_name='Pos_Aj_Err_Row',startcol=df_positions.shape[1]+2+mse_df_positions_ajustada_multioutput.shape[1]+2+2+mape_df_positions_ajustada_multioutput.shape[1], startrow=combined_df_positions_ajustada.shape[0] + 2)
table_df_resultados_muestras_positions_ajustada.to_excel(writer1, sheet_name='Pos_Aj_Err_Row',startcol=df_positions.shape[1]+2+mse_df_positions_ajustada_multioutput.shape[1]+2+2+mape_df_positions_ajustada_multioutput.shape[1], startrow=combined_df_positions_ajustada.shape[0] + 2+describe_positions_ajustada.shape[0]+2)
writer1.close()

writer1 = pd.ExcelWriter(f'{directory}\\results_sampling.xlsx', engine='openpyxl')

#sampling_horizontal.to_excel(writer1, sheet_name='Sampling_Err_Row')
mape_df_sampling_multioutput.to_excel(writer1, sheet_name='Sampling_Err_Row',startcol=sampling_horizontal.shape[1]+2+mse_df_sampling_multioutput.shape[1]+2)
combined_df_sampling.to_excel(writer1, sheet_name='Sampling_Err_Row',startcol=sampling_horizontal.shape[1]+2+mse_df_sampling_multioutput.shape[1]+2+2+mape_df_sampling_multioutput.shape[1])
describe_sampling.to_excel(writer1, sheet_name='Sampling_Err_Row',startcol=sampling_horizontal.shape[1]+2+mse_df_sampling_multioutput.shape[1]+2+2+mape_df_sampling_multioutput.shape[1], startrow=combined_df_sampling.shape[0] + 2)
table_df_resultados_muestras_sampling.to_excel(writer1, sheet_name='Sampling_Err_Row',startcol=sampling_horizontal.shape[1]+2+mse_df_sampling_multioutput.shape[1]+2+2+mape_df_sampling_multioutput.shape[1], startrow=combined_df_sampling.shape[0] + 2+describe_sampling.shape[0]+2)
writer1.close()


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
for i in range(1,num_experimentos+1,1):
    plt.plot(df_times.iloc[:, i], df_positions.iloc[:, i], label=f'Experimento {i}', color=colors[i])

plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico',linestyle='dashed' )
plt.xlabel('Tiempo (s)', fontsize=font_size*1.2)  # Increase font size for the x-axis label
plt.ylabel('Posición (mm)', fontsize=font_size*1.2)  # Increase font size for the y-axis label
plt.title('Movimiento real y movimiento teórico para todos los experimentos', fontsize=font_size*1.5)  # Increase font size for the title
plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size*0.8)  # Increase font size for the legend
plt.grid(True)

# Save the plot as an image (you can adjust the extension as needed)
plt.savefig(f'{directory}\\Posicionvstiempo_experimento_all_spanish.png', bbox_inches='tight')
plt.close()


# Create a larger title, legend, and labels for the final plot in English
plt.figure(figsize=(10, 6))
for i in range(num_experimentos):
    plt.plot(df_times.iloc[:, i], df_positions.iloc[:, i], label=f'Experiment {i}', color=colors[i])

plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Theoretical',linestyle='dashed' )
plt.xlabel('Time (s)', fontsize=font_size*1.2)  # Increase font size for the x-axis label
plt.ylabel('Position (mm)', fontsize=font_size*1.2)  # Increase font size for the y-axis label
plt.title('Real motion and theoretical motion obtained for all experiments', fontsize=font_size*1.5)  # Increase font size for the title
plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size*0.8)  # Increase font size for the legend
plt.grid(True)
plt.savefig(f'{directory}\\Posicionvstiempo_experimento_all_eglish.png', bbox_inches='tight')
plt.close()


print('finalizacion ruta', directory,'hora',datetime.datetime.now().strftime("%H:%M:%S"))