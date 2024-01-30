import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import dask.dataframe as dd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

import datetime
import numpy as np
import pingouin as pg
import re
from scipy.stats import zscore
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

def anderson_darling_test2(dataframe):
    normality_p_values = {}

    # Loop through each column in the DataFrame
    for column in dataframe.columns:
        result = stats.anderson(dataframe[column].dropna())
        critical_value = result.critical_values[2]  # Use the critical value at significance level 0.05
        if result.statistic < critical_value:
            normality_p_values[column] = "Normal"
        else:
            normality_p_values[column] = "Not normal"
    # Create a new DataFrame to store the p-values
    p_value_df = pd.DataFrame.from_dict(normality_p_values, orient='index', columns=['Shapiro-Wilk P-Value'])

    return p_value_df

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
directory_base="C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\Resultados\\Movimiento\\Taguchi3_dos_motores_2factores"

for i in range(1,10,1):
    print('INICIO ',i)

    directory=f"{directory_base}\\{i}"
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

    data_dict = {}
    slopes_df = pd.DataFrame()
    finalpositions_df= pd.DataFrame()

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

        position_columns = [col for col in df.columns if 'real_position' in col]
        slopes=[]
        position_coherent=[]
        for columna in position_columns:
        
            non_zero_indices1 = df[columna] != 0
            x= df['seconds'][non_zero_indices1].values.reshape(-1, 1)
            y = df[columna ][non_zero_indices1].values

            first_occurrence_index = np.where(y == pf)[0][0]
            model = LinearRegression()
            model.fit(x[:first_occurrence_index+1], y[:first_occurrence_index+1])
            pos=y[first_occurrence_index+1]
            # Obtener la pendiente (velocidad)
            slope = model.coef_[0]
            nuevoindex=first_occurrence_index
            while True:
                #print(np.abs((np.abs(slope)-velocity)/velocity)*100)
                if np.abs((np.abs(slope)-velocity)/velocity)*100>9:
                    #nuevo_valor=y[first_occurrence_index-1]
                    nuevoindex=nuevoindex-1
                    #first_occurrence_index = np.where(y == nuevo_valor)[0][0]
                    model = LinearRegression()
                    model.fit(x[:nuevoindex+1], y[:nuevoindex+1])
                    #print(y[nuevoindex+1])
                    pos=y[nuevoindex+1]
                    slope = model.coef_[0]
                else:
                    break
            position_coherent.append(pos)
            slopes.append(slope)
        slopes_dict = dict(zip(position_columns, slopes))
            # Crear el DataFrame
        df_slope = pd.DataFrame([slopes_dict])
        df_slope['cycle'] =[int(result)]
        df_slope['mean_slope']=np.mean(slopes)
        slopes_df=pd.concat([slopes_df, df_slope], ignore_index=True)

        finalpositions_dict = dict(zip(position_columns, position_coherent))
        df_finalpositions = pd.DataFrame([finalpositions_dict])
        df_finalpositions['cycle'] =[int(result)]
        finalpositions_df=pd.concat([finalpositions_df, df_finalpositions], ignore_index=True)

        
        # Create another copy for the 'sampling' list, to have only the information of the sample time
        df2 = df.copy()
        df2[f'sampling_time_m_{result}'] = df2['sampling_time']
        sampling.append(df2[f'sampling_time_m_{result}'])


    print('VALELINDA\n posiciones finales que no danan la pendiente')
    print(finalpositions_df)
    # Concatenate DataFrames horizontally, vertically, and for sampling

    # Concatenate DataFrames in a horizontal manner (side by side)
    concatenated_df = pd.concat(dataframes, axis=1, ignore_index=False)

    # Concatenate DataFrames vertically (stacked on top of each other)
    concatenated_df_vertical = pd.concat(dataframes_vertical, ignore_index=True)

    position_columns = [col for col in concatenated_df_vertical.columns if 'real_position' in col]

    concatenated_df_vertical['Diferencias posicion']=np.abs(concatenated_df_vertical[position_columns[0]]-concatenated_df_vertical[position_columns[1]])

    # Calculate Z-scores for the column 'Diferencias posicion'
    z_scores = zscore(concatenated_df_vertical['Diferencias posicion'])

    # Define a threshold for Z-scores to identify outliers (e.g., 3 standard deviations)
    threshold = 3

    # Create a new column without outliers, replacing them with NaN
    concatenated_df_vertical['Diferencias_posicion_sin_outliers'] = \
        concatenated_df_vertical['Diferencias posicion'].where(abs(z_scores) <= threshold, other=None)

    merged_df = pd.merge(concatenated_df_vertical, slopes_df[['cycle', 'mean_slope']], on='cycle', how='left')
    concatenated_df_vertical=merged_df.copy()
    # Calculate the temporal_delays by dividing Diferencias_sin_outliers by mean_slope
    concatenated_df_vertical['temporal_delays'] = np.abs(concatenated_df_vertical['Diferencias_posicion_sin_outliers'] / concatenated_df_vertical['mean_slope'])

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

    concatenated_df_vertical_normalidad = {
        'Anderson-Darling Test': concatenated_df_vertical.groupby(['cycle'])[['Diferencias posicion','Diferencias_posicion_sin_outliers','temporal_delays']].apply(anderson_darling_test2),
        'Shapiro-Wilk Test P-Value': concatenated_df_vertical.groupby(['cycle'])[['Diferencias posicion','Diferencias_posicion_sin_outliers','temporal_delays']].apply(shapiro_p_value)
    }
    df_concatenated_df_vertical_normalidad=pd.concat(list(concatenated_df_vertical_normalidad.values()), axis=1)





    # Create an Excel writer to save the results
    writer1 = pd.ExcelWriter(f'{directory}\\results.xlsx', engine='openpyxl')


    # Write concatenated data and descriptions to the Excel file
    # concatenated_df_vertical.to_excel(writer1, sheet_name='datos_columnas_vertical')
    #descriptions_experiments.to_excel(writer1, sheet_name='datos_columnas_vertical', startcol=concatenated_df_vertical.shape[1] + 3) #the description of sampling data
    descriptions_experiments.to_excel(writer1, sheet_name='datos_columnas_vertical') #the description of sampling data

    # Calculate statistics for all data in the 'sampling_time' column of concatenated_df_vertical, all the sampling times of the all experiments joined
    all_data_sampling_time = concatenated_df_vertical['sampling_time'].copy()
    medias = np.mean(all_data_sampling_time, axis=0)
    desviaciones = np.std(all_data_sampling_time, axis=0, ddof=1)

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
    df_resultados_muestras_tiempos.to_excel(writer1, sheet_name='datos_columnas_vertical', startrow=len(descriptions_sampling) + 5)
    mi_dataframe_data_sampling.to_excel(writer1, sheet_name='datos_columnas_vertical',startcol=3, startrow=len(descriptions_sampling) + 5)
    slopes_df.to_excel(writer1, sheet_name='datos_columnas_vertical', startcol=3+2+mi_dataframe_data_sampling.shape[1], startrow=len(descriptions_sampling) + 5)
    df_concatenated_df_vertical_normalidad.to_excel(writer1, sheet_name='datos_columnas_vertical', startcol=3+2+mi_dataframe_data_sampling.shape[1]+slopes_df.shape[1]+3, startrow=len(descriptions_sampling) + 5)
    
    writer1.close()
    # Define un tamaño de fuente más grande
    font_size = 14

    # Configurar el tamaño de fuente global para todos los gráficos
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size*0.8)
    plt.rc('ytick', labelsize=font_size*0.8)

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


    #Encontrar los tiempos en los que se alcanza la posicion deseada

    position_columns = [col for col in concatenated_df_vertical.columns if col.startswith("real_position_")]

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=["cycle"])

    # Loop through each position column
    for position_column in position_columns:
        # Create a boolean mask for rows where the column has the value pf
        mask = (concatenated_df_vertical[position_column] == np.max(finalpositions_df[position_column])) | (concatenated_df_vertical[position_column] == np.min(finalpositions_df[position_column]))
        
        #este era originalmente, pero ajuste a la posicion a la que llega segun la pendiente coherente, para ser coherentes con los graficos
        #ya que por 0.0002 mm aproximadamente generaba desfase mayor, que no era coherente
        #mask = (concatenated_df_vertical[position_column] == pf)) 

        # Use the mask to filter rows and then group by 'cycle' to get the first occurrence in each cycle
        result = concatenated_df_vertical[mask].groupby("cycle").head(1)

        # Extract 'cycle', 'seconds', and the value from the position column for the result
        result_data = result[["cycle", "seconds"]]
        result_data.columns = ["cycle", "seconds_" + position_column]

        # Merge the result_data with the result_df on 'cycle' and 'seconds'
        result_df = pd.merge(result_df, result_data, on=["cycle"], how="outer")

    # Sort the result_df by 'cycle' and 'seconds'
    result_df = result_df.sort_values(by=["cycle"]).reset_index(drop=True)



    # Crear una lista de tiempo que aumente con un paso de 1/100 y redondear a 6 cifras significativas

    first_occurrence = df_teorico[df_teorico['Posicion_teorico'] == pf].head(1)['Tiempo_teorico'].values[0]
    result_df['tiempo_teorico'] =[first_occurrence]*len(result_df)
    for position_column in position_columns:
        result_df['diff_' + position_column] = result_df['seconds_' + position_column] - result_df['tiempo_teorico']

    for i in range(len(position_columns)):
        if i==1:
            break
        result_df['diff_absoluta_between_' + position_columns[i+1]+'_and_'+position_columns[i]] =np.abs(result_df['diff_' + position_columns[i+1]] - result_df['diff_' + position_columns[i]])

    # Display the result_df
    print(result_df)

    # Número de experimentos (columnas en el DataFrame)
    num_experimentos = df_positions.shape[1]
    # Define a larger font size
    font_size = 14

    # Calcular el MAPE
    mape_dict_positions_teorica = {}
    mape_dict_positions_teorica_ajustada = {}
    mape_dict_positions_teorica_ajustada_sin_outliers={}
    mape_dict_tiempos = {}
    mape_dict_sampling = {}


    mape_dict_positions_teorica_multioutput = {}
    mape_dict_positions_teorica_ajustada_multioutput = {}
    mape_dict_tiempos_multioutput = {}
    mape_dict_sampling_multioutput = {}


    posicion_teorica = np.array(df_teorico['Posicion_teorico'])
    tiempo_teorico = np.array(df_teorico['Tiempo_teorico'],)
    j=0
    k=0
    # Loop through the samples and plot both position and time with different colors
    for i in range(num_experimentos):
        if i>0 :
            if i% 2 != 0:
                k+=1
            j=i-k
        
        posicion_teorica_ajustada = np.array([round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in df_times.iloc[:,j]])

        non_zero_indices = df_positions.iloc[:, i] != 0

        posicion_teorica_ajustada_sin_outliers = np.array([round(25 - t * velocity, 6) if t * velocity <= (25 - pf) else pf for t in df_times.iloc[:,j][non_zero_indices]])



        print(i,df_times.iloc[:,j].name)
        print(i,df_positions.iloc[:, i].name)
        mape_dict_positions_teorica[f'{df_positions.columns[i]}'] = round(mean_absolute_percentage_error(posicion_teorica, np.array(df_positions.iloc[:, i])) * 100, 5)
        mape_dict_positions_teorica_ajustada[f'{df_positions.columns[i]}'] = round(mean_absolute_percentage_error(posicion_teorica_ajustada, np.array(df_positions.iloc[:, i])) * 100, 5)
        mape_dict_positions_teorica_ajustada_sin_outliers[f'{df_positions.columns[i]}'] = round(mean_absolute_percentage_error(posicion_teorica_ajustada_sin_outliers, np.array(df_positions.iloc[:, i][non_zero_indices])) * 100, 5)
        
        #Multioutput
        # Compute values for positions
        mape_dict_positions_teorica_multioutput[f'MAPE_{df_positions.columns[i]}'] = mean_absolute_percentage_error(posicion_teorica.reshape(1, -1), np.array(df_positions.iloc[:, i]).reshape(1, -1), multioutput='raw_values')*100

        # Compute values for adjusted positions
        mape_dict_positions_teorica_ajustada_multioutput[f'MAPE_{df_positions.columns[i]}'] = mean_absolute_percentage_error(posicion_teorica_ajustada.reshape(1, -1), np.array(df_positions.iloc[:, i]).reshape(1, -1), multioutput='raw_values') *100

        if i<num_experimentos/2-1:#se divide por dos por que son 2 motores
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
            # Save the plot as an image (you can adjust the extension as needed)
            plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_spanish_sin ceros.png')
            plt.close()
            plt.figure(figsize=(10, 6))
            
            plt.plot(df_times.iloc[:,i][non_zero_indices1], df_positions[columns_to_select[0]][non_zero_indices1], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
            plt.plot(df_times.iloc[:,i][non_zero_indices2], df_positions[columns_to_select[1]][non_zero_indices2], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
            plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Theoretical', color='#007acc',linestyle='dashed')
            plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
            plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
            plt.title(f'Real motion and theoretical motion obtained for experiment number {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
            plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
            plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
            plt.legend(fontsize=font_size)  # Increase font size for the legend


            # Save the plot as an image (you can adjust the extension as needed)
            plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_english_sinceros.png')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[0]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
            plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[1]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
            plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Teorico', color='#007acc',linestyle='dashed')
            plt.xlabel('Tiempo (s)', fontsize=font_size)  # Increase font size for the x-axis label
            plt.ylabel('Posición (mm)', fontsize=font_size)  # Increase font size for the y-axis label
            plt.title(f'Movimiento real y movimiento teorico obtenido para el experimento numero {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
            plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
            plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
            plt.legend(fontsize=font_size)  # Increase font size for the legend
        
            # Save the plot as an image (you can adjust the extension as needed)
            plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_spanish.png')
            plt.close()
            plt.figure(figsize=(10, 6))
            
            plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[0]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[0]), color='#e85f04')
            plt.plot(df_times.iloc[:,i], df_positions[columns_to_select[1]], label='Device '+re.sub(f'real_position_|_m_{i+1}', '', columns_to_select[1]), color='lightgreen',linestyle='dashed')
            plt.plot(tiempo_teorico_graph, posicion_teorica_graph, label='Theoretical', color='#007acc',linestyle='dashed')
            plt.xlabel('Time (s)', fontsize=font_size)  # Increase font size for the x-axis label
            plt.ylabel('Position (mm)', fontsize=font_size)  # Increase font size for the y-axis label
            plt.title(f'Real motion and theoretical motion obtained for experiment number {i+1}', fontsize=font_size*1.2)  # Increase font size for the title
            plt.xticks(fontsize=font_size*0.8)  # Increase font size for x-axis ticks
            plt.yticks(fontsize=font_size*0.8)  # Increase font size for y-axis ticks
            plt.legend(fontsize=font_size)  # Increase font size for the legend
            

            # Save the plot as an image (you can adjust the extension as needed)
            plt.savefig(f'{directory}\\Posicionvstiempo_experimento_{i+1}_english.png')
            plt.close()


    combined_df_positions = pd.DataFrame({
        'MAPE': mape_dict_positions_teorica
    })

    combined_df_positions_ajustada = pd.DataFrame({
        'MAPE': mape_dict_positions_teorica_ajustada
    })

    combined_df_positions_ajustada_sin_outliers = pd.DataFrame({
        'MAPE': mape_dict_positions_teorica_ajustada_sin_outliers
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

    combined_df_positions_ajustada['SerialNumber'] = list(combined_df_positions_ajustada.index.str.extract(r'real_position_(\d+)_m_(\d+)')[0])
    combined_df_positions_ajustada['Exp'] = list(combined_df_positions_ajustada.index.str.extract(r'_m_(\d+)')[0])
    combined_df_positions_ajustada['SerialNumber'] = combined_df_positions_ajustada['SerialNumber'].astype(int)
    combined_df_positions_ajustada['Exp'] = combined_df_positions_ajustada['Exp'].astype(int)
    grouped_df_positions_ajustada = combined_df_positions_ajustada.groupby(['Exp', 'SerialNumber']).mean().T
    grouped_df_positions_ajustada_describe_by_motor=combined_df_positions_ajustada.groupby(['SerialNumber'])['MAPE'].describe()[['mean','std']]
    grouped_df_describe_perexperiment_positions_ajust = combined_df_positions_ajustada.groupby('Exp').describe()['MAPE'][['mean','std']]
    describe_means_per_experiment=grouped_df_describe_perexperiment_positions_ajust.describe().loc[['mean', 'std']]

    combined_df_positions_ajustada_sin_outliers['SerialNumber'] = list(combined_df_positions_ajustada_sin_outliers.index.str.extract(r'real_position_(\d+)_m_(\d+)')[0])
    combined_df_positions_ajustada_sin_outliers['Exp'] = list(combined_df_positions_ajustada_sin_outliers.index.str.extract(r'_m_(\d+)')[0])
    combined_df_positions_ajustada_sin_outliers['SerialNumber'] = combined_df_positions_ajustada_sin_outliers['SerialNumber'].astype(int)
    combined_df_positions_ajustada_sin_outliers['Exp'] = combined_df_positions_ajustada_sin_outliers['Exp'].astype(int)
    grouped_df_positions_ajustada_sin_outliers = combined_df_positions_ajustada_sin_outliers.groupby(['Exp', 'SerialNumber']).mean().T
    grouped_df_positions_ajustada_describe_by_motor_sin_outliers=combined_df_positions_ajustada_sin_outliers.groupby(['SerialNumber'])['MAPE'].describe()[['mean','std']]
    grouped_df_describe_perexperiment_positions_ajust_sin_outliers = combined_df_positions_ajustada_sin_outliers.groupby('Exp').describe()['MAPE'][['mean','std']]
    describe_means_per_experiment_sin_outliers=grouped_df_describe_perexperiment_positions_ajust_sin_outliers.describe().loc[['mean', 'std']]



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

    normality_data_positions_ajustada_per_device = {
        'Anderson-Darling Test': combined_df_positions_ajustada.groupby(['SerialNumber'])['MAPE'].apply(anderson_darling_test),
        'Shapiro-Wilk Test P-Value': combined_df_positions_ajustada.groupby(['SerialNumber'])['MAPE'].apply(shapiro_p_value1)
    }
    df_normality_resultados_muestras_positions_per_device = pd.DataFrame(normality_data_positions_ajustada_per_device)

    normality_describe_perexperiment_positions_ajust = {
        'Anderson-Darling Test': pd.DataFrame(grouped_df_describe_perexperiment_positions_ajust['mean']).apply(anderson_darling_test),
        'Shapiro-Wilk Test P-Value': pd.DataFrame(grouped_df_describe_perexperiment_positions_ajust['mean']).apply(shapiro_p_value1)
    }
    df_normality_describe_perexperiment_positions_ajust =pd.DataFrame(normality_describe_perexperiment_positions_ajust)

    data_positions_ajustada_sin_outliers = {
        'Anderson-Darling Test': combined_df_positions_ajustada_sin_outliers.apply(anderson_darling_test),
        'Shapiro-Wilk Test P-Value': combined_df_positions_ajustada_sin_outliers.apply(shapiro_p_value1)
    }
    df_resultados_muestras_positions_ajustada_sin_outliers = pd.DataFrame(data_positions_ajustada_sin_outliers)

    normality_data_positions_ajustada_per_device_sin_outliers = {
        'Anderson-Darling Test': combined_df_positions_ajustada_sin_outliers.groupby(['SerialNumber'])['MAPE'].apply(anderson_darling_test),
        'Shapiro-Wilk Test P-Value': combined_df_positions_ajustada_sin_outliers.groupby(['SerialNumber'])['MAPE'].apply(shapiro_p_value1)
    }
    df_normality_resultados_muestras_positions_per_device_sin_outliers = pd.DataFrame(normality_data_positions_ajustada_per_device_sin_outliers)

    normality_describe_perexperiment_positions_ajust_sin_outliers = {
        'Anderson-Darling Test': pd.DataFrame(grouped_df_describe_perexperiment_positions_ajust_sin_outliers['mean']).apply(anderson_darling_test),
        'Shapiro-Wilk Test P-Value': pd.DataFrame(grouped_df_describe_perexperiment_positions_ajust_sin_outliers['mean']).apply(shapiro_p_value1)
    }
    df_normality_describe_perexperiment_positions_ajust_sin_outliers =pd.DataFrame(normality_describe_perexperiment_positions_ajust_sin_outliers)

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

    #Guardaren excel
    #MAPE por experimento y motor (serial number) [todos los resultados]
    grouped_df_positions_ajustada
    #MAPE promedio por los resultados de cada motor, es decir, si son 3 motores, 3 promedio
    grouped_df_positions_ajustada_describe_by_motor 
    #Normalidad MAPE por motor de los experimentos
    df_normality_resultados_muestras_positions_per_device
    #MAPE promedio por experimento, es decir promedio entre el MAPE obtenido de los motores que se estan moviendo en un experimento
    grouped_df_describe_perexperiment_positions_ajust 
    print(grouped_df_describe_perexperiment_positions_ajust )
    grouped_df_describe_perexperiment_positions_ajust.columns = pd.MultiIndex.from_tuples([
        ('Average MAPE of the motors per experiment','Mean'),
        ('Average MAPE of the motors per experiment','Std')
    ])
    print(grouped_df_describe_perexperiment_positions_ajust )
    #MAPE promedio de los promedios por experimento
    describe_means_per_experiment
    print(describe_means_per_experiment)
    describe_means_per_experiment.columns = pd.MultiIndex.from_tuples([
        ('Average MAPE averages of the motors per experiment','Mean' ),
        ('Average MAPE averages of the motors per experiment','Std' )
    ])
    print(describe_means_per_experiment)
    #Normalidad MAPE por el promedio del MAPE de los motores por experimento 
    df_normality_describe_perexperiment_positions_ajust
    
    #SIN OUTLIERS
    #MAPE por experimento y motor (serial number) [todos los resultados]
    grouped_df_positions_ajustada_sin_outliers
    #MAPE promedio por los resultados de cada motor, es decir, si son 3 motores, 3 promedio
    grouped_df_positions_ajustada_describe_by_motor_sin_outliers
    #Normalidad MAPE por motor de los experimentos
    df_normality_resultados_muestras_positions_per_device_sin_outliers
    #MAPE promedio por experimento, es decir promedio entre el MAPE obtenido de los motores que se estan moviendo en un experimento
    grouped_df_describe_perexperiment_positions_ajust_sin_outliers 
    print(grouped_df_describe_perexperiment_positions_ajust_sin_outliers )
    grouped_df_describe_perexperiment_positions_ajust_sin_outliers.columns = pd.MultiIndex.from_tuples([
        ('Average MAPE of the motors per experiment','Mean'),
        ('Average MAPE of the motors per experiment','Std')
    ])
    print(grouped_df_describe_perexperiment_positions_ajust_sin_outliers )
    #MAPE promedio de los promedios por experimento
    describe_means_per_experiment_sin_outliers
    print(describe_means_per_experiment_sin_outliers)
    describe_means_per_experiment_sin_outliers.columns = pd.MultiIndex.from_tuples([
        ('Average MAPE averages of the motors per experiment','Mean' ),
        ('Average MAPE averages of the motors per experiment','Std' )
    ])
    print(describe_means_per_experiment_sin_outliers)
    #Normalidad MAPE por el promedio del MAPE de los motores por experimento 
    df_normality_describe_perexperiment_positions_ajust_sin_outliers


    writer1 = pd.ExcelWriter(f'{directory}\\results_summary.xlsx', engine='openpyxl')

    grouped_df_positions_ajustada.to_excel(writer1, sheet_name='MAPE_POS')
    grouped_df_positions_ajustada_describe_by_motor.to_excel(writer1, sheet_name='MAPE_POS',startrow=grouped_df_positions_ajustada.shape[0]+3+1)
    df_normality_resultados_muestras_positions_per_device.to_excel(writer1, sheet_name='MAPE_POS',startrow=grouped_df_positions_ajustada.shape[0]+3+1+grouped_df_positions_ajustada_describe_by_motor.shape[0]+1+1)
    grouped_df_describe_perexperiment_positions_ajust.to_excel(writer1, sheet_name='MAPE_POS',startcol=grouped_df_positions_ajustada.shape[1]+2)
    describe_means_per_experiment.to_excel(writer1, sheet_name='MAPE_POS',startcol=grouped_df_positions_ajustada.shape[1]+2,startrow=grouped_df_describe_perexperiment_positions_ajust.shape[0]+4)
    df_normality_describe_perexperiment_positions_ajust.to_excel(writer1, sheet_name='MAPE_POS',startcol=grouped_df_positions_ajustada.shape[1]+2,startrow=grouped_df_describe_perexperiment_positions_ajust.shape[0]+4+describe_means_per_experiment.shape[0]+4)

    #sin outliers
    grouped_df_positions_ajustada_sin_outliers.to_excel(writer1, sheet_name='MAPE_POS_sin_outliers')
    grouped_df_positions_ajustada_describe_by_motor_sin_outliers.to_excel(writer1, sheet_name='MAPE_POS_sin_outliers',startrow=grouped_df_positions_ajustada_sin_outliers.shape[0]+3+1)
    df_normality_resultados_muestras_positions_per_device_sin_outliers.to_excel(writer1, sheet_name='MAPE_POS_sin_outliers',startrow=grouped_df_positions_ajustada_sin_outliers.shape[0]+3+1+grouped_df_positions_ajustada_describe_by_motor_sin_outliers.shape[0]+1+1)
    grouped_df_describe_perexperiment_positions_ajust_sin_outliers.to_excel(writer1, sheet_name='MAPE_POS_sin_outliers',startcol=grouped_df_positions_ajustada_sin_outliers.shape[1]+2)
    describe_means_per_experiment_sin_outliers.to_excel(writer1, sheet_name='MAPE_POS_sin_outliers',startcol=grouped_df_positions_ajustada_sin_outliers.shape[1]+2,startrow=grouped_df_describe_perexperiment_positions_ajust_sin_outliers.shape[0]+4)
    df_normality_describe_perexperiment_positions_ajust_sin_outliers.to_excel(writer1, sheet_name='MAPE_POS_sin_outliers',startcol=grouped_df_positions_ajustada_sin_outliers.shape[1]+2,startrow=grouped_df_describe_perexperiment_positions_ajust_sin_outliers.shape[0]+4+describe_means_per_experiment_sin_outliers.shape[0]+4)


    combined_df_sampling.to_excel(writer1, sheet_name='MAPE_Sampling')
    result_df.to_excel(writer1, sheet_name='Diferencias de tiempo')
    writer1.close()



    print('finalizacion ruta', directory,'hora',datetime.datetime.now().strftime("%H:%M:%S"))