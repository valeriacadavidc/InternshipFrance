"""
Se realiza el analisis de los datos que se encuentran en la carpeta de directory

Donde se utilizo el archivo One_linear_stage_intento2, la parte del codigo de repetibilidad por 20 veces moviendo de la posicion real 25 hasta 18.75, 
queriendo un movimiento de 0 a 6.25
"""

import os
import pandas as pd
#import matplotlib.pyplot as plt

print('valelindapreciosadivina')
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
    df['cycle']=[result]*len(df)

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

