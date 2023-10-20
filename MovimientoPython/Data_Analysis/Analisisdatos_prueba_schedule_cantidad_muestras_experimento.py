import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import dask.dataframe as dd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import numpy as np
import pingouin as pg
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
#    elif value==0.8:
#        if val >=0.7 and val<0.8:
#            color = 'salmon'
#        elif val >=0.8:
#            color = 'lightblue' 
#        else:
#            color='white'

    return 'background-color: %s' % color

# Set the directory path 
#directory = "C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Prueba_schedule"
directory=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\\Prueba_schedule_ciclofor_inthecode"
# List all files in the specified directory
files_in_directory = os.listdir(directory)

# Filter files to select those ending with 'all.csv' and sort them based on a specific pattern
selected_files = [file for file in files_in_directory if file.endswith('all.csv')]
selected_files = sorted(selected_files, key=lambda x: int(x.split('_rep_')[1].split('_all.csv')[0]))

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
    start_index = filename.find("rep_")
    end_index = filename.find("_all")
    result = int(filename[start_index + len("rep_"):end_index])
    columna = [columna for columna in df.columns if 'real' in columna]
    nueva_col = columna[0] + f'_m_{result}'
    df.rename(columns={columna[0]: nueva_col, 'seconds': f'seconds_m_{result}'}, inplace=True)
    del df[[columna for columna in df.columns if 'relative' in columna][0]]
    
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

# Concatenate DataFrames for sampling horizontally (side by side), sample times only
sampling_horizontal = pd.concat(sampling, axis=1, ignore_index=False)

#ICC
columna_posicion = [columna for columna in df.columns if 'real' in columna][0]
icc_positions = pg.intraclass_corr(data=concatenated_df_vertical, targets='Sample', raters='cycle', ratings=columna_posicion ).round(5)

icc_times = pg.intraclass_corr(data=concatenated_df_vertical, targets='Sample', raters='cycle', ratings='seconds').round(5)

# Calculate statistics for 'sampling_time' data for the analysis by row of sampling data
df_sampling_time = concatenated_df_vertical[['sampling_time', 'cycle']]
cycle_groups_sampling = df_sampling_time.groupby('cycle')
descriptions_sampling = cycle_groups_sampling.describe()


cycle_groups = concatenated_df_vertical.groupby('cycle')
descriptions_experiments= cycle_groups.describe()

# Calculate statistics for 'sampling_time' across all cycles
medias = np.mean(sampling_horizontal, axis=0)
desviaciones = np.std(sampling_horizontal, axis=0, ddof=1)
n = sampling_horizontal.shape[0]  # Number of samples
errores_tipicos = desviaciones / np.sqrt(n)
varianzas = np.var(sampling_horizontal, axis=0, ddof=1)
margen_de_error = 1.96 * errores_tipicos
interval_inf = medias - margen_de_error
interval_sup = medias + margen_de_error
amplitud = interval_sup - interval_inf
amplitud_2 = np.power(amplitud, 2)
muestras = (np.power(1.96, 2) * varianzas) / amplitud_2

# Create a DataFrame to store results
data = {
    'Tamaño de Muestra': muestras,
    'Medias': medias,
    'Desviaciones Estándar': desviaciones,
    'Errores Típicos': errores_tipicos,
    'Varianzas': varianzas,
    'Margen de Error': margen_de_error,
    'Intervalo Inferior': interval_inf,
    'Intervalo Superior': interval_sup,
    'Amplitud': amplitud,
    'Amplitud al Cuadrado': amplitud_2,
}
df_resultados_muestras_tiempos = pd.DataFrame(data) 


# Create an Excel writer to save the results
writer1 = pd.ExcelWriter(f'{directory}\\results.xlsx', engine='openpyxl')
icc_positions.to_excel(writer1, sheet_name='datos_icc_posiciones')

icc_times.to_excel(writer1, sheet_name='datos_icc_tiempos')

# Write concatenated data and descriptions to the Excel file
concatenated_df_vertical.to_excel(writer1, sheet_name='datos_columnas_vertical')
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
# Define un tamaño de fuente más grande
font_size = 14

# Configurar el tamaño de fuente global para todos los gráficos
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size*0.8)
plt.rc('ytick', labelsize=font_size*0.8)

# Create a boxplot of the sample time per experiment
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.boxplot(x='cycle', y='sampling_time', data=concatenated_df_vertical, color='royalblue')
plt.ylabel('Sampling time (s)')
plt.xlabel('Experiment')
plt.title('Sampling sime for each experiment')
plt.savefig(f'{directory}\\sampletime_english.png')
plt.close()

# Create a boxplot of the sample time per experiment spanish version
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.boxplot(x='cycle', y='sampling_time', data=concatenated_df_vertical, color='royalblue')
plt.ylabel('Tiempo de muestreo (s)')
plt.xlabel('Experimento')
plt.title('Tiempo de muestreo para cada experimento')
plt.savefig(f'{directory}\\sampletime_spanish.png')
plt.close()

# Create a histogram of the sampling time for all the data, of the 20 experiments
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.histplot(data=concatenated_df_vertical, x='sampling_time', kde=True, palette="mako")  # Adjust the number of bins as needed
plt.title(f'Histogram of sampling time obtained from all experiments\nMean: {medias:.3f}, SD: {desviaciones:.3f}')
plt.xlabel('Sampling time (s)')
plt.ylabel('Frequency')
plt.savefig(f'{directory}\\histogram_english.png')
plt.close()

# Create a histogram of the sampling time for all the data, of the 20 experiments, spanish version
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.histplot(data=concatenated_df_vertical, x='sampling_time', kde=True, palette="mako")  # Adjust the number of bins as needed
plt.title(f'Histograma del tiempo de muestreo obtenido de todos los experimentos\nPromedio: {medias:.3f}, DE: {desviaciones:.3f}')
plt.xlabel('Tiempo de muestreo (s)')
plt.ylabel('Frecuencia')
plt.savefig(f'{directory}\\histogram_spanish.png')
plt.close()

# Create a FacetGrid for histograms for each experiment
g = sns.FacetGrid(data=concatenated_df_vertical, col='cycle', col_wrap=5, aspect=1.2)

# Create histograms with KDE for 'sampling_time' in each subplot
g.map(sns.histplot, 'sampling_time', kde=True, palette="YlGnBu_r")
plt.suptitle('Histogram of Sampling Time Obtained for Each Experiment', fontsize=font_size*1.5)
plt.subplots_adjust(top=0.85)
plt.xlabel('Sampling Time (s)')
plt.ylabel('Frequency')
g.set_titles("Experiment {col_name}", fontsize=font_size)
plt.tight_layout()
plt.savefig(f'{directory}\\histogram_per_experiment_english.png')
plt.close()

# Create a FacetGrid for histograms for each experiment, spanish version
g = sns.FacetGrid(data=concatenated_df_vertical, col='cycle', col_wrap=5, aspect=1.2)

# Create histograms with KDE for 'sampling_time' in each subplot
g.map(sns.histplot, 'sampling_time', kde=True, palette="YlGnBu_r")
plt.suptitle('Histograma del tiempo de muestreo obtenido para cada experimento', fontsize=font_size*1.5)
plt.subplots_adjust(top=0.85)
plt.xlabel('Tiempo de muestreo (s)')
plt.ylabel('Frecuencia')
g.set_titles("Experimento {col_name}", fontsize=font_size)
plt.tight_layout()
plt.savefig(f'{directory}\\histogram_per_experiment_spanish.png')
plt.close()
# Select columns containing 'real' and 'seconds' in their names
positions = [columna for columna in concatenated_df.columns if 'real' in columna]
times = [columna for columna in concatenated_df.columns if 'seconds' in columna]

# Create DataFrames for position and time data, and their transposed versions
df_positions = concatenated_df[positions].copy()
df_times = concatenated_df[times].copy()
df_positions_T = df_positions.T
df_times_T = df_times.T

# Crear una lista de tiempo que aumente con un paso de 1/100 y redondear a 6 cifras significativas
tiempo_teorico = [round(i/100, 6) for i in range(751)]

# Crear una lista de posición que comience en 25 y alcance 18.75 con velocidad 1 y redondear a 6 cifras significativas
posicion_teorica = [round(25 - t, 6) if t <= 6.25 else 18.75 for t in tiempo_teorico]

# Crear un DataFrame con las listas de tiempo y posición
df_teorico = pd.DataFrame({'Tiempo_teorico': tiempo_teorico, 'Posicion_teorico': posicion_teorica})


# Número de experimentos (columnas en el DataFrame)
num_experimentos = df_positions.shape[1]
# Define a larger font size
font_size = 14

# Loop through the 20 samples and plot both position and time with different colors
for i in range(num_experimentos):

    plt.figure(figsize=(10, 6))
    plt.plot(df_teorico['Tiempo_teorico'], df_teorico['Posicion_teorico'], label='Teorico', color='#007acc' )
    plt.plot(df_times.iloc[:,i], df_positions.iloc[:,i], label='Real', color='#e85f04')
    plt.xlabel('Tiempo (s)', fontsize=font_size)  # Increase font size for the x-axis label
    plt.ylabel('Posición (mm)', fontsize=font_size)  # Increase font size for the y-axis label
    plt.title(f'Movimiento real y movimiento teorico obtenido para el experimento numero {i}', fontsize=font_size*1.2)  # Increase font size for the title
    plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
    plt.legend(fontsize=font_size)  # Increase font size for the legend
    plt.grid(True)

    # Save the plot as an image (you can adjust the extension as needed)
    plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i}_spanish.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_teorico['Tiempo_teorico'], df_teorico['Posicion_teorico'], label='Theoretical', color='#007acc')
    plt.plot(df_times.iloc[:,i], df_positions.iloc[:,i], label='Real', color='#e85f04')
    plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
    plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
    plt.title(f'Real motion and theoretical motion obtained for experiment number {i}', fontsize=font_size*1.2)  # Increase font size for the title
    plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
    plt.legend(fontsize=font_size)  # Increase font size for the legend
    plt.grid(True)

    # Save the plot as an image (you can adjust the extension as needed)
    plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i}_english.png')
    plt.close()

# Define colors for the lines
colors = ['#007acc', '#e85f04', '#d62728', '#9467bd', '#8c564b',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#bcbd22',
          '#f7b6d2', '#c7c7c7', '#98df8a', '#ff9896', '#ffbb78',
          '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d']

# Create a larger title, legend, and labels for the final plot
plt.figure(figsize=(10, 6))
for i in range(num_experimentos):
    plt.plot(df_times.iloc[:, i], df_positions.iloc[:, i], label=f'Experimento {i}', color=colors[i])

plt.plot(df_teorico['Tiempo_teorico'], df_teorico['Posicion_teorico'], label='Teorico' )
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

plt.plot(df_teorico['Tiempo_teorico'], df_teorico['Posicion_teorico'], label='Theoretical' )
plt.xlabel('Time (s)', fontsize=font_size*1.2)  # Increase font size for the x-axis label
plt.ylabel('Position (mm)', fontsize=font_size*1.2)  # Increase font size for the y-axis label
plt.title('Real motion and theoretical motion obtained for all experiments', fontsize=font_size*1.5)  # Increase font size for the title
plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size*0.8)  # Increase font size for the legend
plt.grid(True)
plt.savefig(f'{directory}\\Posicionvstiempo_experimento_all_eglish.png', bbox_inches='tight')
plt.close()

# Compute descriptive statistics for time data, for each sample, to see the behaviour of each column (751 samples) each one with 20 samples, of the experiments
tiempos_describe = df_times_T.describe()

# Calculate various statistics for time data, each column (751 samples) of the time data (20 samples) 
medias = np.mean(df_times_T, axis=0)
desviaciones = np.std(df_times_T, axis=0, ddof=1)
n = df_times_T.shape[0]
errores_tipicos = desviaciones / np.sqrt(n)
varianzas = np.var(df_times_T, axis=0, ddof=1)
margen_de_error = 1.96 * errores_tipicos
interval_inf = medias - margen_de_error
interval_sup = medias + margen_de_error
amplitud = interval_sup - interval_inf
amplitud_2 = np.power(amplitud, 2)
muestras = (np.power(1.96, 2) * varianzas) / amplitud_2
normality_test_anderson_times = df_times_T.apply(anderson_darling_test)
normality_test_shapiro_times=df_times_T.apply(shapiro_p_value1)
normality_test_angostino_times=df_times_T.apply(Agostino_p_value)

# Create a DataFrame to store the results for time data
data = {
    'Medias': medias,
    'Desviaciones Estándar': desviaciones,
    'Anderson-Darling Test': normality_test_anderson_times,
    'Shapiro-Wilk Test P-Value': normality_test_shapiro_times,
    'Agostino and Pearson Test':normality_test_angostino_times,
    'Tamaño de Muestra': muestras,
    'Errores Típicos': errores_tipicos,
    'Varianzas': varianzas,
    'Margen de Error': margen_de_error,
    'Intervalo Inferior': interval_inf,
    'Intervalo Superior': interval_sup,
    'Amplitud': amplitud,
    'Amplitud al Cuadrado': amplitud_2,
}
df_resultados_muestras_tiempos = pd.DataFrame(data)

table_df_resultados_muestras_tiempos=df_resultados_muestras_tiempos.style.applymap(text_format,value='anderson',subset=['Anderson-Darling Test']).applymap(text_format,value=0.05,subset=['Shapiro-Wilk Test P-Value']).applymap(text_format,value=0.05,subset=['Agostino and Pearson Test'])




# Write time data and its results to an Excel file
df_times.to_excel(writer1, sheet_name='datos_tiempos')
table_df_resultados_muestras_tiempos.to_excel(writer1, sheet_name='datos_tiempos', startcol=df_times.shape[1] + 2)
tiempos_describe.T.to_excel(writer1, sheet_name='datos_tiempos', startcol=df_times.shape[1] + 2 + df_resultados_muestras_tiempos.shape[1] + 2)

# Calculate various statistics for position data, each column (751 samples) of the time data (20 samples) 
medias = np.mean(df_positions_T, axis=0)
desviaciones = np.std(df_positions_T, axis=0, ddof=1)
n = df_positions_T.shape[0]
errores_tipicos = desviaciones / np.sqrt(n)
varianzas = np.var(df_positions_T, axis=0, ddof=1)
margen_de_error = 1.96 * errores_tipicos
interval_inf = medias - margen_de_error
interval_sup = medias + margen_de_error
amplitud = interval_sup - interval_inf
amplitud_2 = np.power(amplitud, 2)
muestras = (np.power(1.96, 2) * varianzas) / amplitud_2
normality_test_anderson_position = df_positions_T.apply(anderson_darling_test)
normality_test_shapiro_position=df_positions_T.apply(shapiro_p_value1)
normality_test_angostino_position=df_positions_T.apply(Agostino_p_value)

# Create a DataFrame to store the results for time data
data = {
    'Medias': medias,
    'Desviaciones Estándar': desviaciones,
    'Anderson-Darling Test': normality_test_anderson_position,
    'Shapiro-Wilk Test P-Value': normality_test_shapiro_position,
    'Agostino and Pearson Test':normality_test_angostino_position,
    'Tamaño de Muestra': muestras,
    'Errores Típicos': errores_tipicos,
    'Varianzas': varianzas,
    'Margen de Error': margen_de_error,
    'Intervalo Inferior': interval_inf,
    'Intervalo Superior': interval_sup,
    'Amplitud': amplitud,
    'Amplitud al Cuadrado': amplitud_2,
}
df_resultados_muestras_posiciones = pd.DataFrame(data)
table_df_resultados_muestras_posiciones=df_resultados_muestras_posiciones.style.applymap(text_format,value='anderson',subset=['Anderson-Darling Test']).applymap(text_format,value=0.05,subset=['Shapiro-Wilk Test P-Value']).applymap(text_format,value=0.05,subset=['Agostino and Pearson Test'])

# Compute descriptive statistics for time data, for each sample, to see the behaviour of each column (751 samples) each one with 20 samples, of the experiments
posiciones_describe = df_positions_T.describe() 

# Write position data and its results to the same Excel file
df_positions.to_excel(writer1, sheet_name='datos_posiciones')

table_df_resultados_muestras_posiciones.to_excel(writer1, sheet_name='datos_posiciones', startcol=df_positions.shape[1] + 2)
posiciones_describe.T.to_excel(writer1, sheet_name='datos_posiciones', startcol=df_positions.shape[1] + 2 + df_resultados_muestras_posiciones.shape[1] + 2)

# Close the Excel writer
writer1.close()

#Graphs with confidence intervals

# Calculate confidence intervals using bootstrapping
def bootstrap(data, n_bootstrap=1000, alpha=0.05):
    boot_means = []
    boot_medians = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, len(data), replace=True)
        boot_means.append(np.mean(sample))
        boot_medians.append(np.median(sample))

    mean_ci = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    median_ci = np.percentile(boot_medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return mean_ci, median_ci

# Create histograms with confidence intervals for each column
df=df_times.copy()
num_rows_to_select = 10
# Paso específico que deseas utilizar (en este caso, cada quinto valor)
step = len(df) // num_rows_to_select

# Crear una lista de índices de filas que cumplen con el paso
selected_indices = [i for i in range(1, len(df ), step)]

# Seleccionar las filas correspondientes utilizando .iloc
df= df.iloc[selected_indices].copy()
df1=df.T.copy()

df=df_positions.copy()
df= df.iloc[selected_indices].copy()
df2=df.T.copy()

# Copia de los DataFrames
df1_copy = df1.copy()
df2_copy = df2.copy()

# Resetear los índices
df1_copy = df1_copy.reset_index(drop=True)
df2_copy = df2_copy.reset_index(drop=True)

# Ordenar las columnas en orden descendente
# df1_copy = df1_copy[sorted(list(df1_copy.columns), reverse=True)]
# df2_copy = df2_copy[sorted(list(df2_copy.columns), reverse=True)]
print('valelinda')

for i, (col1, col2) in enumerate(zip(df1_copy.columns, df2_copy.columns)):
    data1 = df1_copy[col1]
    data2 = df2_copy[col2]
    bins1=len(data1.unique())
    bins2=len(data2.unique())
  
    # Calcular intervalos de confianza para ambos DataFrames
    mean_ci1, median_ci1 = bootstrap(data1)
    mean_ci2, median_ci2 = bootstrap(data2)

    # Extraer promedio y desviación estándar de los DataFrames respectivos
    mean_time = df_resultados_muestras_tiempos.loc[col1, 'Medias']
    std_dev_time = df_resultados_muestras_tiempos.loc[col1, 'Desviaciones Estándar']
    mean_position = df_resultados_muestras_posiciones.loc[col2, 'Medias']
    std_dev_position = df_resultados_muestras_posiciones.loc[col2, 'Desviaciones Estándar']
    # Crear una figura con subplots para esta iteración
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.set_style("white")
    if bins1 > 3:
        # Gráfica de histograma para data1
        sns.histplot(data=data1, bins=bins1, kde=True, color='steelblue',legend=False,ax=axes[0])
        axes[0].axvline(mean_ci1[0], color='red', linestyle='dashed', linewidth=2, label=f'Mean  ({mean_ci1[0]:.2f}-{mean_ci1[1]:.2f})')
        axes[0].axvline(median_ci1[0], color='green', linestyle='dashed', linewidth=2, label=f'Median  ({median_ci1[0]:.2f}-{median_ci1[1]:.2f})')
        axes[0].set_title(f'Time histogram of sample number {col1}\nMean: {mean_time:.3f}, SD: {std_dev_time:.3f}')
        axes[0].legend(title="95% Confidence Intervals")
        
        # Agregar leyendas
        
    else:
        sns.countplot(data=data1, color='#A2C0D9',legend=False,ax=axes[0])
        axes[0].set_title(f'Time histogram of sample number {col1}\nMean: {mean_time:.3f}, SD: {std_dev_time:.3f}\n95% Confidence Intervals: Median  ({median_ci1[0]:.2f}-{median_ci1[1]:.2f}), Mean: {mean_time:.3f}, SD: {std_dev_time:.3f} ')
        
    if bins2 > 3:
        # Gráfica de histograma para data2
        sns.histplot(data=data2, bins=bins2, kde=True, color='steelblue',legend=False, ax=axes[1])
        axes[1].axvline(mean_ci2[0], color='blue', linestyle='dashed', linewidth=2, label=f'Mean ({mean_ci2[0]:.2f}-{mean_ci2[1]:.2f})')
        axes[1].axvline(median_ci2[0], color='purple', linestyle='dashed', linewidth=2, label=f'Median ({median_ci2[0]:.2f}-{median_ci2[1]:.2f})')
        axes[1].set_title(f'Position histogram of sample number {col2}\nMean: {mean_position:.3f}, SD: {std_dev_position:.3f}')
        axes[1].legend(title="95% Confidence Intervals")
        
    else:
        df2_copy[col2] = df2_copy[col2].astype('category')
        data = {'experiment': list(df2_copy[col2])}
        df_graph = pd.DataFrame(data)
        #color='#9CC1EE'
        sns.countplot(data=df_graph,x='experiment',color='#A2C0D9',legend=False,ax=axes[1])
        axes[1].set_title(f'Position histogram of sample number {col2}\nMean: {mean_position:.3f}, SD: {std_dev_position:.3f}\n95% Confidence Intervals: Mean ({mean_ci2[0]:.2f}-{mean_ci2[1]:.2f}), Median ({median_ci2[0]:.2f}-{median_ci2[1]:.2f})')
  
    axes[0].set_xlabel('Time (s)')
    axes[1].set_xlabel('Position (mm)')
    axes[0].set_ylabel('Frequency')
    axes[1].set_ylabel('Frequency')

    # Ajustar diseño de los subplots
    plt.tight_layout()

    # Definir el nombre de archivo para guardar (inglés)
    output_filename_english = os.path.join(directory, f'histogram_time_position_{i}_english.png')
    # Guardar la figura en inglés
    plt.savefig(output_filename_english)
    plt.close()

    # Crear una figura con subplots para esta iteración
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.set_style("white")
    if bins1 > 3:
        # Gráfica de histograma para data1
        sns.histplot(data=data1, bins=bins1, kde=True, color='steelblue', legend=False, ax=axes[0])
        axes[0].axvline(mean_ci1[0], color='red', linestyle='dashed', linewidth=2, label=f'Promedio  ({mean_ci1[0]:.2f}-{mean_ci1[1]:.2f})')
        axes[0].axvline(median_ci1[0], color='green', linestyle='dashed', linewidth=2, label=f'Mediana  ({median_ci1[0]:.2f}-{median_ci1[1]:.2f})')
        axes[0].set_title(f'Histograma de tiempo del número de muestra  {col1}\nPromedio: {mean_time:.3f}, DE: {std_dev_time:.3f}')
        axes[0].legend(title="Intervalos de confianza del 95%")
            
        
    else:
        sns.countplot(data=data1, color='#A2C0D9',legend=False,ax=axes[0])
        axes[0].set_title(f'Histograma de tiempo del número de muestra  {col1}\nPromedio: {mean_time:.3f}, SD: {std_dev_time:.3f}\nIntervalos de confianza del 95%: Promedio  ({mean_ci1[0]:.2f}-{mean_ci1[1]:.2f}), Mediana  ({median_ci1[0]:.2f}-{median_ci1[1]:.2f})')
    
    if bins2 > 3:
        # Gráfica de histograma para data2
        #, color='steelblue'
        sns.histplot(data=data2, bins=bins2, kde=True, color='steelblue', legend=False,ax=axes[1])
        axes[1].axvline(mean_ci2[0], color='blue', linestyle='dashed', linewidth=2, label=f'Promedio ({mean_ci2[0]:.2f}-{mean_ci2[1]:.2f})')
        axes[1].axvline(median_ci2[0], color='purple', linestyle='dashed', linewidth=2, label=f'Mediana ({median_ci2[0]:.2f}-{median_ci2[1]:.2f})')
        axes[1].set_title(f'Histograma de posición del número de muestra {col2}\nPromedio: {mean_position:.3f}, DE: {std_dev_position:.3f}')
        axes[1].legend(title="Intervalos de confianza del 95%")
        
    else:
        df2_copy[col2] = df2_copy[col2].astype('category')
        data = {'experiment': list(df2_copy[col2])}
        df_graph = pd.DataFrame(data)
        sns.countplot(data=df_graph,x='experiment',color='#A2C0D9',legend=False,ax=axes[1])
        axes[1].set_title(f'Histograma de posición del número de muestra {col2}\nPromedio: {mean_position:.3f}, DE: {std_dev_position:.3f}\nIntervalos de confianza del 95%: Promedio ({mean_ci2[0]:.2f}-{mean_ci2[1]:.2f}), Mediana ({median_ci2[0]:.2f}-{median_ci2[1]:.2f})')

    
    
    axes[0].set_xlabel('Time (s)')
    axes[1].set_xlabel('Position (mm)')
    axes[0].set_ylabel('Frequency')
    axes[1].set_ylabel('Frequency')


    # Ajustar diseño de los subplots
    plt.tight_layout()
    # Definir el nombre de archivo para guardar (español)
    output_filename_spanish = os.path.join(directory, f'histogram_time_position_{i}_spanish.png')
    # Guardar la figura en español
    plt.savefig(output_filename_spanish)
    plt.close()