"""
Se realiza el analisis de los datos que se encuentran en la carpeta de directory

Donde se utilizo el archivo One_linear_stage_intento2, la parte del codigo de repetibilidad por 20 veces moviendo de la posicion real 25 hasta 18.75, 
queriendo un movimiento de 0 a 6.25
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import dask.dataframe as dd
import scipy.stats as stats

#Analisis datos bajo las mismas condicioes (apagando y encendiendo pc)


# Ruta del directorio que deseas explorar
directory ="C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541_iguales_condiciones"
# Obtén una lista de todos los archivos en la ruta especificada
# List files in the directory
files_in_directory = os.listdir(directory)

# Filter files that end with the word "all.csv"
selected_files = [file for file in files_in_directory if file.endswith('all.csv')]
selected_files = sorted(selected_files, key=lambda x: int(x.split('_rep_')[1].split('_all.csv')[0]))

#ANALISIS CONCATENANDO DATOS HORIZONTALMENTE
# List to store DataFrames
dataframes = []

# Iterate through the list of file names
for filename in selected_files:
    # Construir la ruta completa del archivo
    filepath = os.path.join(directory, filename)
    
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(filepath)
    
    # Realizar la manipulación de columnas según sea necesario
    start_index = filename.find("rep_")
    end_index = filename.find("_all")
    result = int(filename[start_index + len("rep_"):end_index])
    columna = [columna for columna in df.columns if 'real' in columna]
    nueva_col = columna[0] + f'_m_{result}'
    df.rename(columns={columna[0]: nueva_col, 'seconds': f'seconds_m_{result}'}, inplace=True)
    del df[[columna for columna in df.columns if 'relative' in columna][0]]
    
    # Añadir el DataFrame modificado a la lista
    dataframes.append(df)

# Concatenate all DataFrames in the list
concatenated_df = pd.concat(dataframes, axis=1,ignore_index=False)
concatenated_df=concatenated_df.iloc[0:int(concatenated_df.describe().T.describe()['count'].loc['min'])]#Dataframe que no tiene datos vacios
num_rows_to_select = 4000

# Paso específico que deseas utilizar (en este caso, cada quinto valor)
step = len(concatenated_df ) // num_rows_to_select

# Crear una lista de índices de filas que cumplen con el paso
selected_indices = [i for i in range(0, len(concatenated_df ), step)]

# Seleccionar las filas correspondientes utilizando .iloc
selected_data = concatenated_df.iloc[selected_indices]
# Seleccionar las filas deseadas del DataFrame original

print('valelinda')

positions= [columna for columna in df.columns if 'real' in columna]
times= [columna for columna in df.columns if 'seconds' in columna]
positions= [columna for columna in concatenated_df.columns if 'real' in columna]
times= [columna for columna in concatenated_df.columns if 'seconds' in columna]
df_positions=selected_data[positions].copy()
df_times=selected_data[times].copy()
df_positions_T=df_positions.T
df_times_T=df_times.T
# Aplicar la función a cada fila del DataFrame de Dask
normality_test_results = pd.DataFrame(columns=df_times_T.columns)

# Perform the Shapiro-Wilk test on each column and store the p-values
for column in df_times_T.columns:
    _, p_value = stats.shapiro(df_times_T[column].dropna())  # Drop NaN values for each column
    normality_test_results.at['Shapiro-Wilk p-value', column] = p_value

# Display the results
print(normality_test_results)
p_values = normality_test_results.T
greater_than_0_05 = (p_values > 0.05).sum()
less_than_0_05 = (p_values < 0.05).sum()
print('TIEMPOS')
print(f'Cantidad de datos distribucion normal: {greater_than_0_05}',f'Cantidad de datos distribucion no normal: {less_than_0_05}')
print(df_times_T.describe())

normality_test_results = pd.DataFrame(columns=df_times_T.columns)

# Perform the Shapiro-Wilk test on each column and store the p-values
for column in df_times_T.columns:
    _, p_value = stats.shapiro(df_positions_T[column].dropna())  # Drop NaN values for each column
    normality_test_results.at['Shapiro-Wilk p-value', column] = p_value

# Display the results
print(normality_test_results)
p_values = normality_test_results.T
greater_than_0_05 = (p_values > 0.05).sum()
less_than_0_05 = (p_values < 0.05).sum()
print('POSICION')
print(f'Cantidad de datos distribucion normal: {greater_than_0_05}',f'Cantidad de datos distribucion no normal: {less_than_0_05}')
print(df_positions_T.describe())
print('valelinda')

#ANALISIS CONCATENANDO DATOS VERTICALMENTE
# List to store DataFrames
dataframes = []

# Iterate through the list of file names
for filename in selected_files :
    # Construct the full file path
    filepath = os.path.join(directory, filename)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)
    df['sampling_time']=df["seconds"].diff()
    start_index = filename.find("rep_")
    end_index = filename.find("_all")# Find the position of "_all" in the string
    # Extract the substring between "rep_" and "_all"
    result = int(filename[start_index + len("rep_"):end_index])
    df['cycle']=[result]*len(df)

    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames in the list
concatenated_df = pd.concat(dataframes, ignore_index=True)
del concatenated_df["relative_position_27259541"]
concatenated_df = concatenated_df.sort_values(by='cycle')
#shapiro_wilk_statistic, shapiro_wilk_pvalue = stats.shapiro(concatenated_df)
cycle_groups = concatenated_df.groupby('cycle')
descriptions = cycle_groups.describe()
descriptions2 = cycle_groups.agg(['mean', 'std'])

descriptions3=descriptions.describe() #describe promedio de promedios de descriptions
descriptions4=descriptions2.describe() #describe promedio de descriptions 2

writer = pd.ExcelWriter(f'{directory}\\resultados_estadisticos.xlsx', engine='openpyxl')


descriptions.to_excel(writer, sheet_name='datos_columnas')
descriptions3.to_excel(writer, sheet_name='datos_columnas',startrow=(len(descriptions)+3))

descriptions2.to_excel(writer, sheet_name='datos_columnas',startcol=(len(descriptions.columns)+2))
descriptions4.to_excel(writer, sheet_name='datos_columnas',startrow=(len(descriptions)+3),startcol=(len(descriptions.columns)+2))
# Guardar el archivo Excel
writer.close()

# plt.figure(figsize=(8, 6))
# concatenated_df.boxplot(column='real_position_27259541', by='cycle')
# plt.xlabel('Cycle')
# plt.ylabel('Real Position')
# plt.title('Box Plot of Real Position by Cycle')
# plt.show()
print('valelinda')



# print('valelindapreciosadivina')
#Easy code
# sns.set(style="whitegrid")
# sns.relplot(x='x', y='y', hue='hue_column', col='col_column', kind='line', data=df)

# # Mostrar el gráfico
# plt.show()
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
sns.lineplot(x='seconds', y='real_position_27259541', hue='cycle',  data=concatenated_df)
plt.savefig('line_plot.png')
plt.show()


print('valelindapreciosadivina')
#Analisis datos ciclo for
# Ruta del directorio que deseas explorar
directory ="C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Prueba_1motor_20veces_0-6.25mmo25-18.75mm_1mms_motor_27259541"
# Obtén una lista de todos los archivos en la ruta especificada
# List files in the directory
files_in_directory = os.listdir(directory)

# Filter files that end with the word "all.csv"
selected_files = [file for file in files_in_directory if file.endswith('all.csv')]

# List to store DataFrames
dataframes = []

# Iterate through the list of file names
for filename in selected_files :
    # Construct the full file path
    filepath = os.path.join(directory, filename)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)
    start_index = filename.find("rep_")

# Find the position of "_all" in the string
    end_index = filename.find("_all")

    # Extract the substring between "rep_" and "_all"
    result = int(filename[start_index + len("rep_"):end_index])
    df['cycle']=[int(result)]*len(df)

    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames in the list
concatenated_df = pd.concat(dataframes, ignore_index=True)
concatenated_df = concatenated_df.sort_values(by='cycle')
# print('valelindapreciosadivina')
#Easy code
# sns.set(style="whitegrid")
# sns.relplot(x='x', y='y', hue='hue_column', col='col_column', kind='line', data=df)

# # Mostrar el gráfico
# plt.show()
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
sns.lineplot(x='seconds', y='real_position_27259541', hue='cycle',  data=concatenated_df)
plt.savefig('line_plot.png')
plt.show()
