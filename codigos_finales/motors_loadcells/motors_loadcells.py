"""
    The following code is employed for the simultaneous and synchronized control of motorized linear stages and load cells. 
    It facilitates various movements and enables the acquisition of force information from the load cells and position data from the motors.

    Author: Valeria Cadavid Castro
    Last Modified: January 18, 2024
"""
# Importing necessary libraries and modules

import time
import System
import pandas as pd
import datetime
import concurrent.futures
import numpy as np
import csv
import sched
import re
from datetime import datetime
import queue
import matplotlib.pyplot as plt


# Importing shared variables from the 'motors_loadcells' module
import process.motors_loadcells.shared_variables as shared_variables


def force_units_and_conversion_factor(from_unit='lb', to_unit='N'):
    """
    Retrieve a list of force units and the conversion factor from specified units.

    Args:
        from_unit (str): The unit to convert from (default: 'lb').
        to_unit (str): The unit to convert to (default: 'N').

    Returns:
        tuple: A tuple containing a list of force units and the conversion factor.

    This function takes the 'from_unit' and 'to_unit' arguments, which represent force units,
    and returns a tuple containing a list of available force units and the conversion factor
    between the specified units.

    """
    # List used to create a dataframe with all the units and conversion factors to convert from one unit of force to another.
    data_units_force = [
        ('Force',90,'dyne','dyn','dyne','dyn',1,0),
('Force',91,'dyne','dyn','kilodyne','kdyn',0.001,0),
('Force',92,'dyne','dyn','megadyne','Mdyn',0.000001,0),
('Force',93,'dyne','dyn','newton','N',0.00001,0),
('Force',94,'dyne','dyn','kilonewton','kN',0.00000001,0),
('Force',95,'dyne','dyn','microgram','µg',1019.716,0),
('Force',96,'dyne','dyn','milligram','mg',1.019716,0),
('Force',97,'dyne','dyn','gram','g',0.001019716,0),
('Force',98,'dyne','dyn','kilogram','kg',0.000001019716,0),
('Force',99,'dyne','dyn','metric ton','MT',0.000000001019716,0),
('Force',100,'dyne','dyn','ounce','oz',0.00003596942,0),
('Force',101,'dyne','dyn','pound','lb',0.000002248089,0),
('Force',102,'dyne','dyn','kilopound','klb',0.000000002248089,0),
('Force',103,'dyne','dyn','short ton (US)','ST',0.000000001124044,0),
('Force',104,'dyne','dyn','long ton (UK)','LT',0.000000001003611,0),
('Force',105,'kilodyne','kdyn','dyne','dyn',1000,0),
('Force',106,'kilodyne','kdyn','kilodyne','kdyn',1,0),
('Force',107,'kilodyne','kdyn','megadyne','Mdyn',0.001,0),
('Force',108,'kilodyne','kdyn','newton','N',0.01,0),
('Force',109,'kilodyne','kdyn','kilonewton','kN',0.00001,0),
('Force',110,'kilodyne','kdyn','microgram','µg',1019716,0),
('Force',111,'kilodyne','kdyn','milligram','mg',1019.716,0),
('Force',112,'kilodyne','kdyn','gram','g',1.019716,0),
('Force',113,'kilodyne','kdyn','kilogram','kg',0.001019716,0),
('Force',114,'kilodyne','kdyn','metric ton','MT',0.000001019716,0),
('Force',115,'kilodyne','kdyn','ounce','oz',0.03596942,0),
('Force',116,'kilodyne','kdyn','pound','lb',0.002248089,0),
('Force',117,'kilodyne','kdyn','kilopound','klb',0.000002248089,0),
('Force',118,'kilodyne','kdyn','short ton (US)','ST',0.000001124044,0),
('Force',119,'kilodyne','kdyn','long ton (UK)','LT',0.000001003611,0),
('Force',120,'megadyne','Mdyn','dyne','dyn',1000000,0),
('Force',121,'megadyne','Mdyn','kilodyne','kdyn',1000,0),
('Force',122,'megadyne','Mdyn','megadyne','Mdyn',1,0),
('Force',123,'megadyne','Mdyn','newton','N',10,0),
('Force',124,'megadyne','Mdyn','kilonewton','kN',0.01,0),
('Force',125,'megadyne','Mdyn','microgram','µg',1019716000,0),
('Force',126,'megadyne','Mdyn','milligram','mg',1019716,0),
('Force',127,'megadyne','Mdyn','gram','g',1019.716,0),
('Force',128,'megadyne','Mdyn','kilogram','kg',1.019716,0),
('Force',129,'megadyne','Mdyn','metric ton','MT',0.001019716,0),
('Force',130,'megadyne','Mdyn','ounce','oz',35.96942,0),
('Force',131,'megadyne','Mdyn','pound','lb',2.248089,0),
('Force',132,'megadyne','Mdyn','kilopound','klb',0.002248089,0),
('Force',133,'megadyne','Mdyn','short ton (US)','ST',0.001124044,0),
('Force',134,'megadyne','Mdyn','long ton (UK)','LT',0.001003611,0),
('Force',135,'newton','N','dyne','dyn',100000,0),
('Force',136,'newton','N','kilodyne','kdyn',100,0),
('Force',137,'newton','N','megadyne','Mdyn',0.1,0),
('Force',138,'newton','N','newton','N',1,0),
('Force',139,'newton','N','kilonewton','kN',0.001,0),
('Force',140,'newton','N','microgram','µg',101971600,0),
('Force',141,'newton','N','milligram','mg',101971.6,0),
('Force',142,'newton','N','gram','g',101.9716,0),
('Force',143,'newton','N','kilogram','kg',0.1019716,0),
('Force',144,'newton','N','metric ton','MT',0.0001019716,0),
('Force',145,'newton','N','ounce','oz',3.596942,0),
('Force',146,'newton','N','pound','lb',0.2248089,0),
('Force',147,'newton','N','kilopound','klb',0.0002248089,0),
('Force',148,'newton','N','short ton (US)','ST',0.0001124044,0),
('Force',149,'newton','N','long ton (UK)','LT',0.0001003611,0),
('Force',150,'kilonewton','kN','dyne','dyn',100000000,0),
('Force',151,'kilonewton','kN','kilodyne','kdyn',100000,0),
('Force',152,'kilonewton','kN','megadyne','Mdyn',100,0),
('Force',153,'kilonewton','kN','newton','N',1000,0),
('Force',154,'kilonewton','kN','kilonewton','kN',1,0),
('Force',155,'kilonewton','kN','microgram','µg',101971600000,0),
('Force',156,'kilonewton','kN','milligram','mg',101971600,0),
('Force',157,'kilonewton','kN','gram','g',101971.6,0),
('Force',158,'kilonewton','kN','kilogram','kg',101.9716,0),
('Force',159,'kilonewton','kN','metric ton','MT',0.1019716,0),
('Force',160,'kilonewton','kN','ounce','oz',3596.942,0),
('Force',161,'kilonewton','kN','pound','lb',224.8089,0),
('Force',162,'kilonewton','kN','kilopound','klb',0.2248089,0),
('Force',163,'kilonewton','kN','short ton (US)','ST',0.1124044,0),
('Force',164,'kilonewton','kN','long ton (UK)','LT',0.1003611,0),
('Force',165,'microgram','µg','dyne','dyn',0.0009806652,0),
('Force',166,'microgram','µg','kilodyne','kdyn',0.0000009806652,0),
('Force',167,'microgram','µg','megadyne','Mdyn',0.0000000009806652,0),
('Force',168,'microgram','µg','newton','N',0.000000009806652,0),
('Force',169,'microgram','µg','kilonewton','kN',0.000000000009806652,0),
('Force',170,'microgram','µg','microgram','µg',1,0),
('Force',171,'microgram','µg','milligram','mg',0.001,0),
('Force',172,'microgram','µg','gram','g',0.000001,0),
('Force',173,'microgram','µg','kilogram','kg',0.000000001,0),
('Force',174,'microgram','µg','metric ton','MT',0.000000000001,0),
('Force',175,'microgram','µg','ounce','oz',0.00000003527396,0),
('Force',176,'microgram','µg','pound','lb',0.000000002204623,0),
('Force',177,'microgram','µg','kilopound','klb',0.000000000002204623,0),
('Force',178,'microgram','µg','short ton (US)','ST',0.000000000001102311,0),
('Force',179,'microgram','µg','long ton (UK)','LT',9.842065E-13,0),
('Force',180,'milligram','mg','dyne','dyn',0.9806652,0),
('Force',181,'milligram','mg','kilodyne','kdyn',0.0009806652,0),
('Force',182,'milligram','mg','megadyne','Mdyn',0.0000009806652,0),
('Force',183,'milligram','mg','newton','N',0.000009806652,0),
('Force',184,'milligram','mg','kilonewton','kN',0.000000009806652,0),
('Force',185,'milligram','mg','microgram','µg',1000,0),
('Force',186,'milligram','mg','milligram','mg',1,0),
('Force',187,'milligram','mg','gram','g',0.001,0),
('Force',188,'milligram','mg','kilogram','kg',0.000001,0),
('Force',189,'milligram','mg','metric ton','MT',0.000000001,0),
('Force',190,'milligram','mg','ounce','oz',0.00003527396,0),
('Force',191,'milligram','mg','pound','lb',0.000002204623,0),
('Force',192,'milligram','mg','kilopound','klb',0.000000002204623,0),
('Force',193,'milligram','mg','short ton (US)','ST',0.000000001102311,0),
('Force',194,'milligram','mg','long ton (UK)','LT',0.0000000009842065,0),
('Force',195,'gram','g','dyne','dyn',980.6652,0),
('Force',196,'gram','g','kilodyne','kdyn',0.9806652,0),
('Force',197,'gram','g','megadyne','Mdyn',0.0009806652,0),
('Force',198,'gram','g','newton','N',0.009806652,0),
('Force',199,'gram','g','kilonewton','kN',0.000009806652,0),
('Force',200,'gram','g','microgram','µg',1000000,0),
('Force',201,'gram','g','milligram','mg',1000,0),
('Force',202,'gram','g','gram','g',1,0),
('Force',203,'gram','g','kilogram','kg',0.001,0),
('Force',204,'gram','g','metric ton','MT',0.000001,0),
('Force',205,'gram','g','ounce','oz',0.03527396,0),
('Force',206,'gram','g','pound','lb',0.002204623,0),
('Force',207,'gram','g','kilopound','klb',0.000002204623,0),
('Force',208,'gram','g','short ton (US)','ST',0.000001102311,0),
('Force',209,'gram','g','long ton (UK)','LT',0.0000009842065,0),
('Force',210,'kilogram','kg','dyne','dyn',980665.2,0),
('Force',211,'kilogram','kg','kilodyne','kdyn',980.6652,0),
('Force',212,'kilogram','kg','megadyne','Mdyn',0.9806652,0),
('Force',213,'kilogram','kg','newton','N',9.806652,0),
('Force',214,'kilogram','kg','kilonewton','kN',0.009806652,0),
('Force',215,'kilogram','kg','microgram','µg',1000000000,0),
('Force',216,'kilogram','kg','milligram','mg',1000000,0),
('Force',217,'kilogram','kg','gram','g',1000,0),
('Force',218,'kilogram','kg','kilogram','kg',1,0),
('Force',219,'kilogram','kg','metric ton','MT',0.001,0),
('Force',220,'kilogram','kg','ounce','oz',35.27396,0),
('Force',221,'kilogram','kg','pound','lb',2.204623,0),
('Force',222,'kilogram','kg','kilopound','klb',0.002204623,0),
('Force',223,'kilogram','kg','short ton (US)','ST',0.001102311,0),
('Force',224,'kilogram','kg','long ton (UK)','LT',0.0009842065,0),
('Force',225,'metric ton','MT','dyne','dyn',980665200,0),
('Force',226,'metric ton','MT','kilodyne','kdyn',980665.2,0),
('Force',227,'metric ton','MT','megadyne','Mdyn',980.6652,0),
('Force',228,'metric ton','MT','newton','N',9806.652,0),
('Force',229,'metric ton','MT','kilonewton','kN',9.806652,0),
('Force',230,'metric ton','MT','microgram','µg',1000000000000,0),
('Force',231,'metric ton','MT','milligram','mg',1000000000,0),
('Force',232,'metric ton','MT','gram','g',1000000,0),
('Force',233,'metric ton','MT','kilogram','kg',1000,0),
('Force',234,'metric ton','MT','metric ton','MT',1,0),
('Force',235,'metric ton','MT','ounce','oz',35273.96,0),
('Force',236,'metric ton','MT','pound','lb',2204.623,0),
('Force',237,'metric ton','MT','kilopound','klb',2.204623,0),
('Force',238,'metric ton','MT','short ton (US)','ST',1.102311,0),
('Force',239,'metric ton','MT','long ton (UK)','LT',0.9842065,0),
('Force',240,'ounce','oz','dyne','dyn',27801.39,0),
('Force',241,'ounce','oz','kilodyne','kdyn',27.80139,0),
('Force',242,'ounce','oz','megadyne','Mdyn',0.02780139,0),
('Force',243,'ounce','oz','newton','N',0.2780139,0),
('Force',244,'ounce','oz','kilonewton','kN',0.0002780139,0),
('Force',245,'ounce','oz','microgram','µg',28349520,0),
('Force',246,'ounce','oz','milligram','mg',28349.52,0),
('Force',247,'ounce','oz','gram','g',28.34952,0),
('Force',248,'ounce','oz','kilogram','kg',0.02834952,0),
('Force',249,'ounce','oz','metric ton','MT',0.00002834952,0),
('Force',250,'ounce','oz','ounce','oz',1,0),
('Force',251,'ounce','oz','pound','lb',0.0625,0),
('Force',252,'ounce','oz','kilopound','klb',0.0000625,0),
('Force',253,'ounce','oz','short ton (US)','ST',0.00003125,0),
('Force',254,'ounce','oz','long ton (UK)','LT',2.79017857142857E-05,0),
('Force',255,'pound','lb','dyne','dyn',444822.2,0),
('Force',256,'pound','lb','kilodyne','kdyn',444.8222,0),
('Force',257,'pound','lb','megadyne','Mdyn',0.4448222,0),
('Force',258,'pound','lb','newton','N',4.448222,0),
('Force',259,'pound','lb','kilonewton','kN',0.004448222,0),
('Force',260,'pound','lb','microgram','µg',453592400,0),
('Force',261,'pound','lb','milligram','mg',453592.4,0),
('Force',262,'pound','lb','gram','g',453.5924,0),
('Force',263,'pound','lb','kilogram','kg',0.4535924,0),
('Force',264,'pound','lb','metric ton','MT',0.0004535924,0),
('Force',265,'pound','lb','ounce','oz',16,0),
('Force',266,'pound','lb','pound','lb',1,0),
('Force',267,'pound','lb','kilopound','klb',0.001,0),
('Force',268,'pound','lb','short ton (US)','ST',0.0005,0),
('Force',269,'pound','lb','long ton (UK)','LT',0.000446428571428571,0),
('Force',270,'kilopound','klb','dyne','dyn',444822200,0),
('Force',271,'kilopound','klb','kilodyne','kdyn',444822.2,0),
('Force',272,'kilopound','klb','megadyne','Mdyn',444.8222,0),
('Force',273,'kilopound','klb','newton','N',4448.222,0),
('Force',274,'kilopound','klb','kilonewton','kN',4.448222,0),
('Force',275,'kilopound','klb','microgram','µg',453592400000,0),
('Force',276,'kilopound','klb','milligram','mg',453592400,0),
('Force',277,'kilopound','klb','gram','g',453592.4,0),
('Force',278,'kilopound','klb','kilogram','kg',453.5924,0),
('Force',279,'kilopound','klb','metric ton','MT',0.4535924,0),
('Force',280,'kilopound','klb','ounce','oz',16000,0),
('Force',281,'kilopound','klb','pound','lb',1000,0),
('Force',282,'kilopound','klb','kilopound','klb',1,0),
('Force',283,'kilopound','klb','short ton (US)','ST',0.5,0),
('Force',284,'kilopound','klb','long ton (UK)','LT',0.446428571428571,0),
('Force',285,'short ton (US)','ST','dyne','dyn',889644500,0),
('Force',286,'short ton (US)','ST','kilodyne','kdyn',889644.5,0),
('Force',287,'short ton (US)','ST','megadyne','Mdyn',889.6445,0),
('Force',288,'short ton (US)','ST','newton','N',8896.445,0),
('Force',289,'short ton (US)','ST','kilonewton','kN',8.896445,0),
('Force',290,'short ton (US)','ST','microgram','µg',907184700000,0),
('Force',291,'short ton (US)','ST','milligram','mg',907184700,0),
('Force',292,'short ton (US)','ST','gram','g',907184.7,0),
('Force',293,'short ton (US)','ST','kilogram','kg',907.1847,0),
('Force',294,'short ton (US)','ST','metric ton','MT',0.9071847,0),
('Force',295,'short ton (US)','ST','ounce','oz',32000,0),
('Force',296,'short ton (US)','ST','pound','lb',2000,0),
('Force',297,'short ton (US)','ST','kilopound','klb',2,0),
('Force',298,'short ton (US)','ST','short ton (US)','ST',1,0),
('Force',299,'short ton (US)','ST','long ton (UK)','LT',0.892857142857142,0),
('Force',300,'long ton (UK)','LT','dyne','dyn',996401900,0),
('Force',301,'long ton (UK)','LT','kilodyne','kdyn',996401.9,0),
('Force',302,'long ton (UK)','LT','megadyne','Mdyn',996.4019,0),
('Force',303,'long ton (UK)','LT','newton','N',9964.019,0),
('Force',304,'long ton (UK)','LT','kilonewton','kN',9.964019,0),
('Force',305,'long ton (UK)','LT','microgram','µg',1016047000000,0),
('Force',306,'long ton (UK)','LT','milligram','mg',1016047000,0),
('Force',307,'long ton (UK)','LT','gram','g',1016047,0),
('Force',308,'long ton (UK)','LT','kilogram','kg',1016.047,0),
('Force',309,'long ton (UK)','LT','metric ton','MT',1.016047,0),
('Force',310,'long ton (UK)','LT','ounce','oz',35840,0),
('Force',311,'long ton (UK)','LT','pound','lb',2240,0),
('Force',312,'long ton (UK)','LT','kilopound','klb',2.24,0),
('Force',313,'long ton (UK)','LT','short ton (US)','ST',1.12,0),
('Force',314,'long ton (UK)','LT','long ton (UK)','LT',1,0),
    ]
    # Define the column names for the force unit conversion data
    columns = ["Unit Type", "Conversion", "Convert From", "From Unit", "Convert To", "To Unit", "Scale Factor", "Offset"]
    # Create a DataFrame using the provided force unit conversion data and column names
    df_units_force = pd.DataFrame(data_units_force,columns=columns)
    # Retrieve the conversion factor for the specified 'from_unit' to 'to_unit'
    conversion_factor = df_units_force[((df_units_force['From Unit'] == from_unit) | (df_units_force["Convert From"] == from_unit)) & (df_units_force['To Unit'] == to_unit)]['Scale Factor'].values[0]
    # Obtain a list of unique force units available in the DataFrame
    force_units=list(df_units_force["From Unit"].unique())
    return force_units,conversion_factor

def plot_graph_positions(df, path, language='spanish'):
    """
    Plot the graph of motorized linear stages positions over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data for plotting.
    - path (str): Filepath to save the generated plot.
    - language (str, optional): Language for plot labels ('spanish' or 'english'). Default is 'spanish'.

    Returns:
    - tuple: Filepath of the saved plot and the Figure instance.
    """

    colors = ["#20B2AA","#FF6347",'#007acc', '#e85f04',"#ff7f0e","#ff7f0e",'#d62728',"#4B0082",  '#9467bd', '#8c564b',
              '#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', '#bcbd22',
              '#f7b6d2', '#c7c7c7', '#98df8a', '#ff9896', '#ffbb78',
              '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d']
    font_size = 14
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10))]
    # Extract columns related to linear stage positions
    columns=[col for col in df.columns if 'Linear Stage Position' in col]
    # Create a Figure instance
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot each linear stage position over time
    for i in range(len(columns)):
        ax.plot(df['seconds'], df[columns[i]], label=re.findall(r"\d+", columns[i])[0], color=colors[i], linestyle=linestyles[i])

    # Set plot labels and title based on language
    if language == 'spanish':
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size * 0.8)
        legend.set_title('Etapa lineal', prop={'size': font_size * 0.8})
        ax.set_xlabel('Tiempo transcurrido (s)', fontsize=font_size * 1.2)
        ax.set_ylabel('Posición (mm)', fontsize=font_size * 1.2)
        ax.set_title('Movimiento de las etapas lineales motorizadas', fontsize=font_size)
    elif language == 'english':
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size * 0.8)
        legend.set_title('Linear stage', prop={'size': font_size * 0.8})
        ax.set_xlabel('Time Elapsed (s)', fontsize=font_size * 1.2)
        ax.set_ylabel('Position (mm)', fontsize=font_size * 1.2)
        ax.set_title('Motion of motorized linear stages', fontsize=font_size)
    # Set tick parameters
    ax.tick_params(axis='x', labelsize=font_size * 0.8)
    ax.tick_params(axis='y', labelsize=font_size * 0.8)
    # Adjust layout for better appearance
    plt.tight_layout()
     # Save the plot with language-specific suffix
    language_suffix = 's' if language == 'spanish' else 'e'

    plt.savefig(f'{path}_{language_suffix}_positions.png')
    # Return the Figure instance
    return f'{path}_{language_suffix}_positions.png',fig

def plot_graph_forces(df,path,language='spanish'):
    """
    Plot the graph of forces measured by load cells over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data for plotting.
    - path (str): Filepath to save the generated plot.
    - language (str, optional): Language for plot labels ('spanish' or 'english'). Default is 'spanish'.

    Returns:
    - tuple: Filepath of the saved plot and the Figure instance.

    """


    colors = ["#1f77b4", "#d62728", "#2ca02c", '#ADD8E6',"#ff7f0e","#9ACD32", "#000080","#FFA07A","#90EE90","#B0C4DE","#FF4500","#2E8B57", "#008080", "#ff4500",
                  "#3cb371", "#ff6347", "#32cd32", "#ff0000", "#4682b4", "#dc143c",
                  "#20b2aa", "#8b4513","#DC143C", "#00fa9a", "#ff8c00", "#4169e1", "#ff1493",
                  "#00ced1", "#800000", "#556b2f", "#008000", "#b22222"]

    font_size = 14
    linestyles = ['-', '--', '-.','-','--','-.', '-', '--', '-.','-','--','-.',':', (0, (3, 1, 1, 1, 1, 1)),'-' ,(0, (5, 10)), (0, (3, 10, 1, 10)),'-',
               '--', '-.','-', ':', (0, (3, 1, 1, 1, 1, 1)), '-',(0, (5, 10)), (0, (3, 10, 1, 10)),'-',
             '--', '-.', ':']
    # Extract columns related to tracking, peak, and valley forces
    tracking_columns = [col for col in df.columns if 'Tracking' in col]
    peak_columns = [col for col in df.columns if 'Peak' in col]
    valley_columns = [col for col in df.columns if 'Valley' in col]

    # Organize columns based on the numbers extracted
    columns = sorted(tracking_columns + peak_columns + valley_columns, key=lambda col: int(col.split('-')[-1]))

    # Create a Figure instance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each force measurement over time
    for i in range(len(columns)):
        match = re.findall(r"\w+", columns[i])
        first_word, serial_number = match[0],match[-1]
    
        ax.plot(df['seconds'], df[columns[i]], label=f"{serial_number} ({first_word})", color=colors[i], linestyle=linestyles[i])
    
    # Extract the unit from the first column, the unit of force
    unit=re.search(r'\(([^)]+)\)', columns[0]).group(1)
    
    # Set plot labels and title based on language
    if language == 'spanish':
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size * 0.8)
        legend.set_title('Celda de carga', prop={'size': font_size * 0.8})
        ax.set_xlabel('Tiempo transcurrido (s)', fontsize=font_size * 1.2)
        ax.set_ylabel(f'Fuerza ({unit})', fontsize=font_size * 1.2)
        ax.set_title('Fuerza vs Tiempo transcurrido', fontsize=font_size)
    elif language == 'english':
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size * 0.8)
        legend.set_title('Load cell', prop={'size': font_size * 0.8})
        ax.set_xlabel('Time elapsed (s)', fontsize=font_size * 1.2)
        ax.set_ylabel(f'Force ({unit})', fontsize=font_size * 1.2)
        ax.set_title('Force vs Time elapsed', fontsize=font_size)
    # Set tick parameters
    ax.tick_params(axis='x', labelsize=font_size * 0.8)
    ax.tick_params(axis='y', labelsize=font_size * 0.8)
    # Adjust layout for better appearance
    plt.tight_layout()
    language_suffix = 's' if language == 'spanish' else 'e'
     # Save the plot with language-specific suffix
    plt.savefig(f'{path}_{language_suffix}_forces.png')
    # Return the Figure instance
    return f'{path}_{language_suffix}_forces.png',fig
  
def collect_data(queue_obj,motor_devices_dictionary,load_cells_list, sample_time, num_samples,execution_time,path):
    """
    This function collects data from motorized devices and load cells, obteining data of position of the load cells and the forces,
    saving it to a CSV file.

    Parameters:
    - queue_obj: A queue object for interprocess communication, used to pass the initial date and time to be used in the function collected_data_modification.
    - motor_devices_dictionary: A dictionary containing motorized devices with keys as identifiers and values as device objects.
    - load_cells_list: A list of load cell objects.
    - sample_time: The time interval between consecutive data samples.
    - num_samples: The maximum number of samples to collect.
    - execution_time: The total time duration for data collection.
    - path: The path where the CSV file will be saved.

    Returns:
    - None

    Note: This function uses the sched module for scheduling data collection at regular intervals.

    """
        
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses sched
    try:
        # Get the list of devices from the dictionary
        devices_list = list(motor_devices_dictionary.values())
        sample = 0

        # Define column names for the CSV file
        columnas = ['Time Elapsed'] +['Tracking Value (' +str(loadcell.GetChannelXUnitOfMeasure(0))+')-'+loadcell.GetInstrumentSerialNumber() for loadcell in load_cells_list]+ ['Linear Stage Position (mm)-' + elemento for elemento in motor_devices_dictionary.keys()]

        # Get the current date and time
        initial_date_hour = datetime.now()
        
        # Create a CSV file to save the data
        csv_file_path = f'{path}.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header to the CSV file
            csv_writer.writerow(columnas) 
       
            def write_timestamp(sc):
                """
                This function writes a time elapsed in seconds, linear stages positions and load cells forces to the CSV file.

                Parameters:
                - sc: A scheduler object.

                Returns:
                - None
                """

                nonlocal sample, initial_time,t1,t2

                # Combine the results into a single array at the same time     
                data_line = [time.perf_counter()-initial_time]+[loadcell.GetChannelXReading(0) for loadcell in load_cells_list] + [device.Position for device in devices_list]              
                # Writes a single row of data (data_line) to the CSV file
                csv_writer.writerow(data_line)

                # Schedule the next data collection if the conditions are met
                save_data=shared_variables.get_movements_motors_completed()
                # Time, force, and position data are stored at specified time intervals. This function will only stop collecting data when the desired
                # movements for the motors are completed, or when the Stop button on the interface is pressed.
                if not save_data:
                    t1=time.perf_counter()
                    # Schedule the next data collection based on the sample_time
                    # This code does not allow data to be stored at a higher frequency than the time it takes to read the forces and positions.
                    sc.enter((data_line[0]+sample_time-time.perf_counter()+initial_time-t2) * 0.9 if sample_time != 0 else 0, 1, write_timestamp, (sc,))
                    # Saves the time it takes to execute the previous line of code
                    t2=time.perf_counter()-t1
                   
            # Create a scheduler object
            s = sched.scheduler(time.perf_counter, time.sleep)
            # Schedule the function `write_timestamp()` to run immediately and then repeatedly every sample_time seconds
            initial_time = time.perf_counter()
            t1=time.perf_counter()
            s.enter(0, 1, write_timestamp, (s,))
            t2=time.perf_counter()-t1
            s.run()

        # Put the initial date and time into the queue for other processes to access
        queue_obj.put(initial_date_hour)
        print("Fin collected data")

    except Exception as e:
        # Handle the exception 
        print(f"An exception occurred: {str(e)}")

def collected_data_modification(path,initial_date_hour,desired_uni_force,serial_numbers_initial_positions, initial_positions,serial_numbers_initial_forces,initial_forces):
    """
    Modify the collected raw data from a CSV file, adjusting positions and forces to the desired values,
    and save the modified data to an Excel file.

    Parameters:
    - path (str): Filepath to the input CSV file.
    - initial_date_hour (str): Initial date and hour in the format 'YYYY-MM-DD HH:mm:ss'.
    - desired_unit_force (str): Desired unit for force values.
    - serial_numbers_initial_positions (list): List of serial numbers corresponding to initial positions.
    - initial_positions (list): List of initial positions corresponding to serial numbers.
    - serial_numbers_initial_forces (list): List of serial numbers corresponding to initial forces.
    - initial_forces (list): List of initial forces corresponding to serial numbers.

    Returns:
    - pd.DataFrame: Modified DataFrame containing adjusted positions and forces.


    This function was created with the purpose of changing the recorded values of positions of the motorized linear stages 
    by the relative values, i.e. the values of the movement performed in the experiment, where depending on the position 
    where the linear stages started the movement, this is the initial position, i.e. the values obtained should be changed 
    depending on this value. In other words, its purpose is to correct the absolute values of the recorded positions, based on 
    the relative values of the movement performed in a specific experiment.

    Likewise, the force values recorded are always in the unit in which the load cells were calibrated, however, 
    in this post-processing it is possible to change the units to a selected unit of force, as well as to obtain the maximum 
    and minimum values recorded by the load cells.

    """

    try:
        print("start collected data modification")
        # Read the CSV data into a DataFrame.
        data = pd.read_csv(f'{path}.csv')
        # Create a new column 'Date' by adding the initial date and time to the time elapsed values
        data['Date']=initial_date_hour +pd.to_timedelta(data['Time Elapsed'], unit='s')
        # Create a new column 'seconds' and copy the values from 'Time Elapsed'
        data['seconds']=data['Time Elapsed'].copy()
        # Convert 'Time Elapsed' column to datetime format, extract time component, and update the column
        data['Time Elapsed']=pd.to_datetime(data['Time Elapsed'], unit='s').dt.time

        # View the columns that contain 0 and delete the corresponding rows
        # This is due to the fact that there is a data sending error for a short period of time,
        # where the linear stage controller sends zeros instead of the actual position.

        # Initialize an empty list to store indices that need to be deleted
        index_to_delete=[]

        # Iterate through serial numbers associated with initial positions
        for serial_number in serial_numbers_initial_positions:

            # Iterate through columns in the DataFrame
            for column in data.columns:

                # Check if the serial number is present in the column name
                if serial_number in column:

                    try:
                        # Identify indices where there is a significant change (difference > 1 or < -1) in column values
                        positive_index=[elemento -1 for elemento in list(data[data[column].diff()>1].index)]
                        negative_index= list(data[data[column].diff()<-1].index)
                        index_only_column=positive_index+negative_index

                        # Filter indices where the column value is 0
                        index_only_column_veryfied=data[column][index_only_column].index[data[column][index_only_column] == 0].tolist()

                        # Add the identified indices to the list of indices to delete
                        index_to_delete=index_to_delete+index_only_column_veryfied
                        # Subtract the initial position from the column values
                        data[column]=float(str(initial_positions[serial_numbers_initial_positions.index(serial_number)]))-data[column]
             
                    except Exception as e:
                        print(f"An exception processing positions: {str(e)}")

        # Remove rows with identified indices to delete and reset the DataFrame index
        data = data.drop(list(set(index_to_delete)), axis=0)
        data = data.reset_index(drop=True)

        # Initialize an empty list to store column names related to load cells
        load_cells_columns=[]

        # Iterate through each column in the DataFrame
        for column in data.columns:
            # Check if the column name contains 'Tracking Value'
            if 'Tracking Value' in column:
                
                # Use regular expressions to extract unit and serial number from the column name
                match = re.search(r'\(([^)]+)\)-(\d+)', column)
                if match:
                    # Extract unit and serial number
                    unit= match.group(1)
                    serial_number  = match.group(2)
                    # Adjust column values by subtracting the corresponding initial force
                    data[column]=data[column]-initial_forces[serial_numbers_initial_forces.index(serial_number)]
                    # Retrieve conversion factor for unit desired conversion
                    _,conversion_factor=force_units_and_conversion_factor(from_unit=unit, to_unit=desired_uni_force)
                    # Create a new column with adjusted values in the desired force unit
                    new_column_name = f'Tracking Value ({desired_uni_force})-{serial_number}'
                    data[new_column_name]=data[column]*conversion_factor
                    # Calculate Peak Values
                    current_peak_index = 0
                    current_peak_value = data.loc[current_peak_index, new_column_name]

                    # Iterate over the DataFrame
                    for index, value in enumerate(data[new_column_name]):
                        # Check if the current value is greater than the current peak value
                        if value > current_peak_value:
                            current_peak_index = index
                            current_peak_value = value

                        # Update the DataFrame with the current peak value
                        data.loc[index, f'Peak Value ({desired_uni_force})-{serial_number}'] = current_peak_value

                    # Calculate Valley Value
                    current_valley_index = 0
                    current_valley_value = data.loc[current_valley_index, new_column_name]

                    # Iterate over the DataFrame
                    for index, value in enumerate(data[new_column_name]):
                        # Check if the current value is smaller than the current valley value
                        if value < current_valley_value:
                            current_valley_index = index
                            current_valley_value = value

                        # Update the DataFrame with the current valley value
                        data.loc[index, f'Valley Value ({desired_uni_force})-{serial_number}'] = current_valley_value

                    # Add the column names to the list load_cells_columns, to have an order of the data forces in the Excel
                    load_cells_columns.append(new_column_name)
                    load_cells_columns.append(f'Peak Value ({desired_uni_force})-{serial_number}')
                    load_cells_columns.append(f'Valley Value ({desired_uni_force})-{serial_number}')
                    
                    # Drop the original column from the DataFrame, it is say, the force data in de original unit it is not taken into account
                    # Just the coolumns of force with the desired unit are taken into account
                    data=data.drop(column,axis=1)

        # Add a new column 'Sample Number' containing an array of integers from 0 to the length of the DataFrame
        data['Sample Number']=np.arange(0,len(data))

        # Define the desired order of columns in the final DataFrame
        desired_order=['Sample Number']+ load_cells_columns +[col for col in data.columns if 'Linear Stage Position (mm)' in col]+['Date','Time Elapsed']
        
        # Create a new DataFrame 'data_save' containing columns in the desired order
        data_save=data[desired_order]

        # Create a writer object for Excel using the 'openpyxl' engine
        writer = pd.ExcelWriter(f'{path}.xlsx', engine='openpyxl')

        # Write the 'data_save' DataFrame to an Excel sheet named 'Data'
        data_save.to_excel(writer, sheet_name='Data')

        
        # Close the Excel writer
        writer.close()

        # Return the modified DataFrame 'data', this data is used later in the code to make the plots of the registered positions and forces.
        return data
       
    except Exception as e:
        print(f"An exception occurred in the function collected_data_modification: {str(e)}")
    
def synchronized_displacement(path,thorlabs_devices,devices_FUTEK, velocity,final_position,frequency,force_unit):
    """
    Perform a synchronized displacement experiment with Thorlabs devices and FUTEK load cells and allows to save the data 
    of positions and forces

    Parameters:
    - path: The path to save the collected data.
    - thorlabs_devices: An object containing Thorlabs devices.
    - devices_FUTEK: A list of FUTEK load cell devices.
    - velocity: The displacement velocity desired to move the devices in mm/s.
    - final_position: The target final position for the displacement in mm.
    - frequency: The sampling frequency for data collection.
    - force_unit: The desired unit for force values.

    Returns:
    - data: Modified and processed data after the experiment.

    Note: This function orchestrates the entire experiment, including setting velocities, positions,
    collecting data, and post-processing the results.
    """
    # Frequency and Sample Time Calculation
    if frequency>=60:
            frequency=60
            sample_time = 0
    else:
        sample_time = 1 / frequency

    # Obtaining the initial position
    initial_position=np.mean([float(str(device.Position)) for device in list(thorlabs_devices.devices.values())])  

    # Final Position Adjustment
    if final_position>=initial_position:
        final_position=0
    else:
        final_position=initial_position-final_position
    # Execution Time, Wait Timeout, Number of Samples Calculation
    execution_time=(initial_position-final_position)/velocity
    waitTimeout=execution_time*1000+2000
    waitTimeout+=waitTimeout*0.8
    waitTimeout=System.Convert.ToUInt64(waitTimeout)
    execution_time=execution_time*1.085
    num_samples = int((execution_time) * frequency)

    # Thorlabs Devices Velocity Setting
    # The desired velocity for the movement of the devices is set
    # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:   
        # Initialize an empty list to store the future objects for each device velocity setting task         
        futures_velocity = []
         # Iterate over Thorlabs devices
        for device in thorlabs_devices.devices.values():
            # Submit a velocity setting task to the executor and append the returned Future object to the list
            futures_velocity.append(executor.submit(thorlabs_devices.set_velocity,device, velocity))
        # Wait for all submitted tasks (setting device velocities) to complete
        concurrent.futures.wait(futures_velocity)

    # Queue Initialization
    queue_data_collected = queue.Queue()

    # Motor, Load Cell Serials, and Initial Positions/Forces Extraction
    motor_serials=list(thorlabs_devices.devices.keys())
    motor_positions=[device.Position for device in list(thorlabs_devices.devices.values())]              
    loadcells_serials=[loadcell.GetInstrumentSerialNumber() for loadcell in devices_FUTEK] 
    loadcells_forces= [loadcell.GetChannelXReading(0) for loadcell in devices_FUTEK]
    
    # Data Collection and Displacement Tasks
    # Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a data collection task to the executor and obtain a Future object (p1)
        p1=executor.submit(collect_data,queue_data_collected,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,execution_time=execution_time, path=path)
        # Start the tasks in futures
        futures = []
        # Start displacement tasks for each Thorlabs device and obtain Future objects for each task
        for device in thorlabs_devices.devices.values():
            futures.append(executor.submit(thorlabs_devices.shif_device, device, final_position,waitTimeout))
        # Wait for all of the tasks to complete
        concurrent.futures.wait(futures)
        # Once the movement of the linear stages are finished (shif_device finished), the variable movements_motors_completed change to True, 
        # to indicate that the movement is completed
        shared_variables.set_movements_motors_completed(True)
    
    # Retrieve the initial date and hour from the queue
    initial_date_hour = queue_data_collected.get()

    # Data Post-Processing
    data=collected_data_modification(path,initial_date_hour=initial_date_hour,desired_uni_force=force_unit,serial_numbers_initial_positions=motor_serials,initial_positions=motor_positions,
                                        serial_numbers_initial_forces=loadcells_serials,initial_forces=loadcells_forces)

    return data

def synchronized_hysteresis(path,thorlabs_devices,devices_FUTEK,velocity,final_position,cycles,frequency,force_unit):
    """
    Perform a synchronized hysteresis experiment with Thorlabs devices and FUTEK load cells.

    Parameters:
    - path: The path to save the collected data.
    - thorlabs_devices: An object containing Thorlabs devices.
    - devices_FUTEK: A list of FUTEK load cell devices.
    - velocity: The displacement velocity in mm/s.
    - final_position: The target final position for the hysteresis in mm.
    - cycles: The number of hysteresis cycles.
    - frequency: The sampling frequency for data collection.
    - force_unit: The desired unit for force values.

    Returns:
    - data: Modified and processed data after the experiment.

    Note: This function orchestrates the entire hysteresis experiment, including setting velocities,final positions,
    collecting data, and post-processing the results.
    """

    # Frequency and Sample Time Calculation
    if frequency>=60:
            frequency=60
            sample_time = 0
    else:
        sample_time = 1 / frequency

    # Obtaining the initial position
    initial_position=np.mean([float(str(device.Position)) for device in list(thorlabs_devices.devices.values())])  

    # Final Position Adjustment
    if final_position>=initial_position:
        final_position=0
    else:
        final_position=initial_position-final_position

    # Execution Time, Wait Timeout, Number of Samples Calculation
    execution_time= (initial_position-final_position)*2*cycles/velocity
    waitTimeout=(initial_position-final_position)*1000/velocity+2000
    waitTimeout+=waitTimeout*0.8
    waitTimeout=System.Convert.ToUInt64(waitTimeout)#Duration to wait for command to execute
    execution_time=execution_time*1.08
    num_samples = int((execution_time +2) * frequency)
    
    # Thorlabs Devices Velocity Setting
    # The desired velocity for the movement of the devices is set
    # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:   
        # Initialize an empty list to store the future objects for each device velocity setting task         
        futures_velocity = []
         # Iterate over Thorlabs devices
        for device in thorlabs_devices.devices.values():
            # Submit a velocity setting task to the executor and append the returned Future object to the list
            futures_velocity.append(executor.submit(thorlabs_devices.set_velocity,device, velocity))
        # Wait for all submitted tasks (setting device velocities) to complete
        concurrent.futures.wait(futures_velocity)

    # Queue Initialization
    queue_data_collected = queue.Queue()

    # Motor, Load Cell Serials, and Initial Positions/Forces Extraction
    motor_serials=list(thorlabs_devices.devices.keys())
    motor_positions=[device.Position for device in list(thorlabs_devices.devices.values())]              
    loadcells_serials=[loadcell.GetInstrumentSerialNumber() for loadcell in devices_FUTEK] 
    loadcells_forces= [loadcell.GetChannelXReading(0) for loadcell in devices_FUTEK]

    # Data Collection and Displacement Tasks (hysteresis)
    # Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a data collection task to the executor and obtain a Future object (p1)
        p1=executor.submit(collect_data,queue_data_collected,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,execution_time=execution_time, path=path)
        # Start the tasks in futures
        futures = []
        # Start displacement tasks for each Thorlabs device and obtain Future objects for each task
        for device in thorlabs_devices.devices.values():
            futures.append(executor.submit(thorlabs_devices.hysteresis,device,initial_position, final_position, cycles,waitTimeout))
        # Wait for all of the tasks to complete
        concurrent.futures.wait(futures)
        # Once the movement of the linear stages are finished (hysteresis finished), the variable movements_motors_completed change to True, 
        # to indicate that the movement is completed
        shared_variables.set_movements_motors_completed(True)

    # Retrieve the initial date and hour from the queue
    initial_date_hour = queue_data_collected.get()

    # Data Post-Processing
    data=collected_data_modification(path,initial_date_hour=initial_date_hour,desired_uni_force=force_unit,serial_numbers_initial_positions=motor_serials,initial_positions=motor_positions,
                                        serial_numbers_initial_forces=loadcells_serials,initial_forces=loadcells_forces)

    return data

def synchronized_hysteresis_fp(path,thorlabs_devices,devices_FUTEK,velocity,final_position,cycles,frequency,force_unit):
    """
    Perform a synchronized hysteresis experiment with Thorlabs devices and FUTEK load cells.
    The key distinction from the function 'synchronized_hysteresis' is that this function
    does not return the linear stage to the initial position in the last cycle; it concludes
    at the final position.

    Parameters:
    - path: The path to save the collected data.
    - thorlabs_devices: An object containing Thorlabs devices.
    - devices_FUTEK: A list of FUTEK load cell devices.
    - velocity: The displacement velocity in mm/s.
    - final_position: The target final position for the hysteresis in mm.
    - cycles: The number of hysteresis cycles.
    - frequency: The sampling frequency for data collection.
    - force_unit: The desired unit for force values.

    Returns:
    - data: Modified and processed data after the experiment.

    Note: This function orchestrates the entire hysteresis experiment, including setting velocities,final positions,
    collecting data, and post-processing the results.
    """

    # Frequency and Sample Time Calculation
    if frequency>=60:
            frequency=60
            sample_time = 0
    else:
        sample_time = 1 / frequency

    # Obtaining the initial position
    initial_position=np.mean([float(str(device.Position)) for device in list(thorlabs_devices.devices.values())])  

    # Final Position Adjustment
    if final_position>=initial_position:
        final_position=0
    else:
        final_position=initial_position-final_position

    # Execution Time, Wait Timeout, Number of Samples Calculation
    execution_time= (initial_position-final_position)*2*cycles/velocity
    waitTimeout=(initial_position-final_position)*1000/velocity+2000
    waitTimeout+=waitTimeout*0.8
    waitTimeout=System.Convert.ToUInt64(waitTimeout)#Duration to wait for command to execute
    execution_time=execution_time*1.08
    num_samples = int((execution_time +2) * frequency)
    
    # Thorlabs Devices Velocity Setting
    # The desired velocity for the movement of the devices is set
    # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:   
        # Initialize an empty list to store the future objects for each device velocity setting task         
        futures_velocity = []
         # Iterate over Thorlabs devices
        for device in thorlabs_devices.devices.values():
            # Submit a velocity setting task to the executor and append the returned Future object to the list
            futures_velocity.append(executor.submit(thorlabs_devices.set_velocity,device, velocity))
        # Wait for all submitted tasks (setting device velocities) to complete
        concurrent.futures.wait(futures_velocity)

    # Queue Initialization
    queue_data_collected = queue.Queue()

    # Motor, Load Cell Serials, and Initial Positions/Forces Extraction
    motor_serials=list(thorlabs_devices.devices.keys())
    motor_positions=[device.Position for device in list(thorlabs_devices.devices.values())]              
    loadcells_serials=[loadcell.GetInstrumentSerialNumber() for loadcell in devices_FUTEK] 
    loadcells_forces= [loadcell.GetChannelXReading(0) for loadcell in devices_FUTEK]

    # Data Collection and Displacement Tasks (hysteresis)
    # Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a data collection task to the executor and obtain a Future object (p1)
        p1=executor.submit(collect_data,queue_data_collected,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,execution_time=execution_time, path=path)
        # Start the tasks in futures
        futures = []
        # Start displacement tasks for each Thorlabs device and obtain Future objects for each task
        for device in thorlabs_devices.devices.values():
            futures.append(executor.submit(thorlabs_devices.hysteresis_fp,device,initial_position, final_position, cycles,waitTimeout))
        # Wait for all of the tasks to complete
        concurrent.futures.wait(futures)
        # Once the movement of the linear stages are finished (hysteresis finished), the variable movements_motors_completed change to True, 
        # to indicate that the movement is completed
        shared_variables.set_movements_motors_completed(True)

    # Retrieve the initial date and hour from the queue
    initial_date_hour = queue_data_collected.get()

    # Data Post-Processing
    data=collected_data_modification(path,initial_date_hour=initial_date_hour,desired_uni_force=force_unit,serial_numbers_initial_positions=motor_serials,initial_positions=motor_positions,
                                        serial_numbers_initial_forces=loadcells_serials,initial_forces=loadcells_forces)

    return data

def synchronized_stress_relaxation(path,thorlabs_devices,devices_FUTEK,velocity,forward_position,waiting_time,cycles,frequency,force_unit):
    """
    Perform a synchronized stress relaxation experiment with Thorlabs devices and FUTEK load cells.

    Parameters:
    - path: The path to save the collected data.
    - thorlabs_devices: An object containing Thorlabs devices.
    - devices_FUTEK: A list of FUTEK load cell devices.
    - velocity: The displacement velocity in mm/s.
    - forward_position: The target forward position or step size for stress relaxation in mm, displacemento to move in each cycle from the current position.
    - waiting_time: The time to wait at the final position during each cycle, this time is in seconds.
    - cycles: The number of stress relaxation cycles.
    - frequency: The sampling frequency for data collection in Hz.
    - force_unit: The desired unit for force values.

    Returns:
    - data: Modified and processed data after the experiment.

    Note: This function orchestrates the entire stress relaxation experiment,
    including setting velocities, collecting data, and post-processing the results.
    """

   # Initial Position Adjustment based on Cycles and Forward Position
    initial_position=np.mean([float(str(device.Position)) for device in list(thorlabs_devices.devices.values())] )  
    while (True):
        # If the total required displacement exceeds the maximum allowable displacement of the device, 
        # the number of cycles is decreased until the number of cycles does not exceed the allowable displacement.
        if forward_position*cycles>initial_position:
            cycles-=1
        if forward_position*cycles<=initial_position:
            break

    # Frequency and Sample Time Calculation
    if frequency>=60:
        frequency=60
        sample_time = 0
    else:
        sample_time = 1 / frequency
    
    # Execution Time, Wait Timeout, Number of Samples Calculation
    execution_time_waiting=(waiting_time)*cycles
    execution_time_movement=(forward_position/velocity)*cycles
    waitTimeout=forward_position/velocity*1000
    waitTimeout+=waitTimeout*0.8
    waitTimeout=System.Convert.ToUInt64(waitTimeout)
    execution_time_movement=execution_time_movement*1.08
    num_samples = num_samples = int((execution_time_movement+execution_time_waiting+2) * frequency)
    execution_time=execution_time_waiting+execution_time_movement


    # Thorlabs Devices Velocity Setting
    # The desired velocity for the movement of the devices is set
    # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:   
        # Initialize an empty list to store the future objects for each device velocity setting task         
        futures_velocity = []
         # Iterate over Thorlabs devices
        for device in thorlabs_devices.devices.values():
            # Submit a velocity setting task to the executor and append the returned Future object to the list
            futures_velocity.append(executor.submit(thorlabs_devices.set_velocity,device, velocity))
        # Wait for all submitted tasks (setting device velocities) to complete
        concurrent.futures.wait(futures_velocity)

    # Queue Initialization
    queue_data_collected = queue.Queue()

    # Motor, Load Cell Serials, and Initial Positions/Forces Extraction
    motor_serials=list(thorlabs_devices.devices.keys())
    motor_positions=[device.Position for device in list(thorlabs_devices.devices.values())]              
    loadcells_serials=[loadcell.GetInstrumentSerialNumber() for loadcell in devices_FUTEK] 
    loadcells_forces= [loadcell.GetChannelXReading(0) for loadcell in devices_FUTEK]

    # Data Collection and Displacement Tasks (hysteresis)
    # Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a data collection task to the executor and obtain a Future object (p1)
        p1=executor.submit(collect_data,queue_data_collected,thorlabs_devices.devices,devices_FUTEK,sample_time=sample_time, num_samples=num_samples,execution_time=execution_time, path=path)
        # Start the tasks in futures
        futures = []
        # Start displacement tasks for each Thorlabs device and obtain Future objects for each task
        for device in thorlabs_devices.devices.values():
            futures.append(executor.submit(thorlabs_devices.stress_relaxation,device,initial_position,forward_position, waiting_time, cycles,waitTimeout))
        # Wait for all of the tasks to complete
        concurrent.futures.wait(futures)

        # Once the movement of the linear stages are finished (stress_relaxation finished), the variable movements_motors_completed change to True, 
        # to indicate that the movement is completed
        shared_variables.set_movements_motors_completed(True)

    # Retrieve the initial date and hour from the queue
    initial_date_hour = queue_data_collected.get()

    # Data Post-Processing
    data=collected_data_modification(path,initial_date_hour=initial_date_hour,desired_uni_force=force_unit,serial_numbers_initial_positions=motor_serials,initial_positions=motor_positions,
                                        serial_numbers_initial_forces=loadcells_serials,initial_forces=loadcells_forces)

    return data