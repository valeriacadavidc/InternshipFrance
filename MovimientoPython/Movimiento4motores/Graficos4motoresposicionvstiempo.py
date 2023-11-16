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

def plot_graph(df, columns, colors, directory, filename, language='spanish', exclude_zero=False):
    font_size = 14
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10))]

    # GRAFICO 1
    plt.figure(figsize=(10, 6))
    if exclude_zero:
        for i in range(len(columns)):
            non_zero_indices1 = df[columns[i]] != 0
            plt.plot(df['seconds'][non_zero_indices1], df[columns[i]][non_zero_indices1], label='Device '+re.sub(f'real_position_', '', columns[i]), color=colors[i], linestyle=linestyles[i])
    else:
        for i in range(len(columns)):
            plt.plot(df['seconds'], df[columns[i]], label='Device '+re.sub(f'real_position_', '', columns[i]), color=colors[i], linestyle=linestyles[i])

    if language == 'spanish':
        plt.xlabel('Tiempo (s)', fontsize=font_size * 1.2)
        plt.ylabel('Posición (mm)', fontsize=font_size * 1.2)
        plt.title('Movimiento de las etapas lineales motorizadas', fontsize=font_size)
    elif language == 'english':
        plt.xlabel('Time (s)', fontsize=font_size * 1.2)
        plt.ylabel('Position (mm)', fontsize=font_size * 1.2)
        plt.title('Motion of motorized linear stages', fontsize=font_size)

    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size * 0.8)
    plt.tight_layout()

    language_suffix = 'spanish' if language == 'spanish' else 'english'
    exclude_zero_suffix = 'sinceros' if exclude_zero else 'conceros'
    plt.savefig(f'{directory}\\{filename}_{language_suffix}_{exclude_zero_suffix}.png')
    plt.close()
    # GRAFICO 2
    fig, axs = plt.subplots(len(columns), 1, figsize=(10, 10), sharex=True)
    if exclude_zero:
        for i in range(len(columns)):
            non_zero_indices1 = df[columns[i]] != 0
            axs[i].plot(df['seconds'][non_zero_indices1], df[columns[i]][non_zero_indices1],  color=colors[i])
            axs[i].tick_params(axis='x', labelsize=font_size * 0.8)  # Ajustar el tamaño de la fuente para los ticks del eje X
            axs[i].tick_params(axis='y', labelsize=font_size * 0.8)
    else:
        for i in range(len(columns)):
            axs[i].plot(df['seconds'], df[columns[i]],  color=colors[i])
            axs[i].tick_params(axis='x', labelsize=font_size * 0.8)  # Ajustar el tamaño de la fuente para los ticks del eje X
            axs[i].tick_params(axis='y', labelsize=font_size * 0.8)
    if language == 'spanish':
        for i in range(len(columns)):
            axs[i].set_ylabel('Posición (mm)', fontsize=font_size )  # Ajustar el tamaño de la fuente para la etiqueta del eje Y
            axs[i].set_title('Device '+re.sub(f'real_position_', '', columns[i]), fontsize=font_size )
        axs[-1].set_xlabel('Tiempo (s)', fontsize=font_size )  # Ajustar el tamaño de la fuente para la etiqueta del eje X
        fig.suptitle('Movimiento de las etapas lineales motorizadas', fontsize=font_size ) 
       
    elif language == 'english':
        for i in range(len(columns)):
            axs[i].set_ylabel('Position(mm)', fontsize=font_size )  # Ajustar el tamaño de la fuente para la etiqueta del eje Y
            axs[i].set_title('Device '+re.sub(f'real_position_', '', columns[i]), fontsize=font_size )
        axs[-1].set_xlabel('Time (s)', fontsize=font_size )  # Ajustar el tamaño de la fuente para la etiqueta del eje X
        fig.suptitle('Motion of motorized linear stages', fontsize=font_size)
    plt.subplots_adjust(hspace=0.4) 
  
    language_suffix = 'spanish' if language == 'spanish' else '_english'
    exclude_zero_suffix = 'sinceros' if exclude_zero else 'conceros'

    plt.savefig(f'{directory}\\{filename}_subplots_{language_suffix}_{exclude_zero_suffix}.png')
    plt.close()


directory=r"C:\Users\valeria.cadavid\Documents\RepositorioCodigos\Resultados\Movimiento\intentos_stress_relaxation"
#directory=r"C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Intentos_precarga"
filename="stress_relaxation_vel_0.5_pi_0_pollrate_1_samplefreq_100_cycles_10_fp_1_waittime_10_exp_1_all.csv"
filepath = os.path.join(directory, filename)
df = pd.read_csv(filepath)
columns = [columna for columna in df.columns if 'real' in columna]
# Define colors for the lines
colors = ['#007acc', '#e85f04', '#d62728', '#9467bd', '#8c564b',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#bcbd22',
          '#f7b6d2', '#c7c7c7', '#98df8a', '#ff9896', '#ffbb78',
          '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d']
# Example usage
# Supongamos que df, columns, colors, directory, y filename están definidos

# Generar gráficos en español sin excluir ceros
plot_graph(df, columns, colors, directory, filename, language='spanish', exclude_zero=False)
plot_graph(df, columns, colors, directory, filename, language='english', exclude_zero=False)
plot_graph(df, columns, colors, directory, filename, language='spanish', exclude_zero=True)
plot_graph(df, columns, colors, directory, filename, language='english', exclude_zero=True)
print('valelinda')

