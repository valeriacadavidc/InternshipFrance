
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import dask.dataframe as dd
import scipy.stats as stats


# Ruta del directorio que deseas explorar
directory ="C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Prueba_schedule"
# Obtén una lista de todos los archivos en la ruta especificada
# List files in the directory
files_in_directory = os.listdir(directory)

# Filter files that end with the word "all.csv"
selected_files = [file for file in files_in_directory if file.endswith('all.csv')]
selected_files = sorted(selected_files, key=lambda x: int(x.split('_rep_')[1].split('_all.csv')[0]))

#ANALISIS CONCATENANDO DATOS HORIZONTALMENTE
# List to store DataFrames
dataframes = []
dataframes_vertical=[]
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

    df = pd.read_csv(filepath)
    df['cycle']=[int(result)]*len(df)
    df['sampling_time']=df["seconds"].diff()
    dataframes_vertical.append(df)


# Concatenate all DataFrames in the list
concatenated_df = pd.concat(dataframes, axis=1,ignore_index=False)
concatenated_df_vertical=pd.concat(dataframes_vertical, ignore_index=True)


cycle_groups = concatenated_df_vertical.groupby('cycle')
descriptions = cycle_groups.describe()
descriptions2 = cycle_groups.agg(['mean', 'std'])

writer = pd.ExcelWriter(f'{directory}\\resultados_estadisticos.xlsx', engine='openpyxl')


descriptions.to_excel(writer, sheet_name='datos_columnas_vertical')
descriptions2.to_excel(writer, sheet_name='datos_columnas_vertical',startcol=(len(descriptions.columns)+2))




print('valelinda')

positions= [columna for columna in concatenated_df.columns if 'real' in columna]
times= [columna for columna in concatenated_df.columns if 'seconds' in columna]
df_positions=concatenated_df[positions].copy()
df_times=concatenated_df[times].copy()
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
tiempos=df_times_T.describe()



tiempos.to_excel(writer, sheet_name='tiempos_horizontal')
normality_test_results.to_excel(writer, sheet_name='tiempos_horizontal',startrow=len(tiempos)+3)

normality_test_results = pd.DataFrame(columns=df_positions_T.columns)
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
posiciones=df_positions_T.describe()
posiciones.to_excel(writer, sheet_name='posiciones_horizontal')
normality_test_results.to_excel(writer, sheet_name='posiciones_horizontal',startrow=len(posiciones)+3)




writer.close()

print('valelinda')