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
import re

directory=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\Intentos_precarga"
filename="histeresis_v_0.25_pi_0_pf_2.5_freq_20_ciclos_10_intento_1_all.csv"
filepath = os.path.join(directory, filename)
df = pd.read_csv(filepath)
columnas = [columna for columna in df.columns if 'real' in columna]
# Define colors for the lines
colors = ['#007acc', '#e85f04', '#d62728', '#9467bd', '#8c564b',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#bcbd22',
          '#f7b6d2', '#c7c7c7', '#98df8a', '#ff9896', '#ffbb78',
          '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d']
font_size = 14
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10))]

# Create a larger title, legend, and labels for the final plot
plt.figure(figsize=(10, 6))
for i in range(len(columnas)):
    plt.plot(df['seconds'], df[columnas[i]], label=f'Device {i}', color=colors[i],linestyle=linestyles[i])

plt.xlabel('Tiempo (s)', fontsize=font_size*1.2)  # Increase font size for the x-axis label
plt.ylabel('Posición (mm)', fontsize=font_size*1.2)  # Increase font size for the y-axis label
plt.title('Movimiento de las etapas lineales motorizadas con la funcion de histeresis', fontsize=font_size*1.5)  # Increase font size for the title
plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size*0.8)  # Increase font size for the legend
plt.tight_layout()
plt.savefig(f'{directory}\\{filename}.png')
plt.close()

fig, axs = plt.subplots(len(columnas), 1, figsize=(10, 10), sharex=True)

for i in range(len(columnas)):
    axs[i].plot(df['seconds'], df[columnas[i]], label=f'Device {i}', color=colors[i])
    axs[i].set_ylabel('Posición (mm)', fontsize=font_size )  # Ajustar el tamaño de la fuente para la etiqueta del eje Y
    axs[i].set_title(f'Device {i}', fontsize=font_size )
    axs[i].tick_params(axis='x', labelsize=font_size * 0.8)  # Ajustar el tamaño de la fuente para los ticks del eje X
    axs[i].tick_params(axis='y', labelsize=font_size * 0.8)  # Ajustar el tamaño de la fuente para los ticks del eje Y

axs[-1].set_xlabel('Tiempo (s)', fontsize=font_size )  # Ajustar el tamaño de la fuente para la etiqueta del eje X
fig.suptitle('Movimiento de las etapas lineales motorizadas con la función de histéresis', fontsize=font_size * 1.5)  # Añadir un título general
plt.subplots_adjust(hspace=0.4)  # Ajustar el espacio entre subplots

# Guardar el gráfico
plt.savefig(f'{directory}\\{filename}_subplots.png')
plt.close()