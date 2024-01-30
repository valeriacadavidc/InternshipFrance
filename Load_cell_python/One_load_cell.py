import clr
import time
import numpy as np
import matplotlib.pyplot as plt
import System
clr.AddReference("C:\\Users\\valec\\Documents\\Practicas_Academicas_Francia\\Codigo\\InternshipFrance\\Load_cell_python\\FUTEK_USB_DLL.dll")
clr.AddReference(r'C:\Users\valec\Documents\Practicas_Academicas_Francia\Codigo\InternshipFrance\Load_cell_python\FUTEK.Devices.dll')
clr.AddReference(r'C:\Users\valec\Documents\Practicas_Academicas_Francia\Codigo\InternshipFrance\Load_cell_python\FTD2XX_NET.dll')
import FUTEK_USB_DLL
import FUTEK.Devices
#from FUTEK_USB_DLL import USB_DLL

repo = FUTEK.Devices.DeviceRepository()
devices = list(repo.DetectDevices())
#devices[0].SetChannelXUnitOfMeasure(0,FUTEK.Devices.UnitsOfMeasure(5))
serialNumber = devices[0].GetInstrumentSerialNumber()
modelNumber = devices[0].GetModelNumber()
unitOfMeasure = devices[0].GetChannelXUnitOfMeasure(0)
reading = devices[0].GetChannelXReading(0)
print('Model Number')
print(modelNumber)
print('Instrument Serial Number')
print(serialNumber)
print('Reading')
print(reading)
print('Unit Of Measure')
print(unitOfMeasure)
# Crear un vector para almacenar los tiempos
tiempos = []
readings=[]
for _ in range(10000):
    #devices[0].SetChannelXUnitOfMeasure(0,FUTEK.Devices.UnitsOfMeasure(i))
    time1 = time.perf_counter()
    reading = devices[0].GetChannelXReading(0)
    time2 = time.perf_counter() - time1
    tiempos.append(time2)
    readings.append(reading)
    # print(reading)
    # print("tiempo", time2)

# Calcular el promedio y la desviación estándar
promedio = np.mean(tiempos)
desviacion_estandar = np.std(tiempos)
# Imprimir el promedio y la desviación estándar
print('GetChannelXReading(0)')
print(f"Promedio: {promedio} segundos")
print(f"Desviación estándar: {desviacion_estandar} segundos")
print(f"Promedio reading: {np.mean(readings)} segundos")
#Intento streaming
t=[]
readings2=[]
devices[0].PreStreamingOperations()
for _ in range(1000):
    t1=time.perf_counter()
    #devices[0].PreStreamingOperations()
    points = devices[0].GetStreamingDataConverted()
    if points:
        a=min([p.ConvertedValue for p in points])
    else:
        a=None
    #devices[0].PostStreamingOperations()
    t2 = time.perf_counter() - t1
    t.append(t2)
    if a is not None:
        readings2.append(float(a))
    else:
        readings2.append(None)
devices[0].PostStreamingOperations()
promedio1 = np.mean(t)
desviacion_estandar1 = np.std(t)
print('GetStreamingDataConverted()')
print(f"Promedio: {promedio1} segundos")
print(f"Desviación estándar: {desviacion_estandar1} segundos")
print(readings2)
# Crear un histograma
plt.hist(tiempos, bins=5, color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Histograma de tiempos de lectura con reading')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.show()

# Crear un histograma
plt.hist(t, bins=5, color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Histograma de tiempos de lectura con streaming')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.show()

repo.DisconnectDevice( modelNumber,serialNumber)
print('valelinda')
 

#print('ADC',devices[0].GetChannelXNormalDataRequest(0))
#devices[0].SetChannelXUnitOfMeasure(0,FUTEK.Devices.UnitsOfMeasure(18))
# sampling_rate_initial=devices[0].GetChannelXSamplingRate(0)
# list_sampling_rates=list(devices[0].GetChannelXSamplingRatePossibleValues(0))
# #devices[0].SetChannelXSamplingRate(0,list_sampling_rates[11]) #Change the sample rate
# sampling_rate_new=devices[0].GetChannelXSamplingRate(0)
# #print('ADC',devices[0].GetChannelXNormalDataRequest(0))
# print('Initial sampling rate',sampling_rate_initial)
# print('Sampling rates available',list_sampling_rates)
# print('New sampling rate',sampling_rate_new)

# repo.DisconnectDevice( modelNumber,serialNumber)

def collect_data(devices_dictionary, sample_time, num_samples, path, name):
    #Save the data directly in a csv later ir read the csv and save it in a beter way to understand the results, and uses sched
    try:
        # Get the list of devices from the dictionary
        devices_list = list(devices_dictionary.values())
        sample = 0
        columnas = ['seconds'] + ['real_position_' + elemento for elemento in devices_dictionary.keys()]
        # Create a CSV file to save the data
        csv_file_path = f'{path}\\{name}.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(columnas) # Write the header to the CSV file
            def write_timestamp(sc):
                """This function writes a timestamp and device positions to the CSV file."""
                #initial_time=time.perf_counter()
                nonlocal sample, initial_time,t1,t2
                data_line = [time.perf_counter()-initial_time] + [device.Position for device in devices_list]
                sample += 1
                csv_writer.writerow(data_line)
                if sample < num_samples:
                    t1=time.perf_counter()
                    sc.enter((data_line[0]+sample_time-time.perf_counter()+initial_time-t2)*0.9, 1, write_timestamp, (sc,))
                    t2=time.perf_counter()-t1
            # Create a scheduler object
            s = sched.scheduler(time.perf_counter, time.sleep)
            # Schedule the function `write_timestamp()` to run immediately and then repeatedly every sample_time seconds
            initial_time = time.perf_counter()
            t1=time.perf_counter()
            s.enter(0, 1, write_timestamp, (s,))
            t2=time.perf_counter()-t1
            s.run()
    
    except Exception as e:
        # Handle the exception here, you can print an error message or log it
        print(f"An exception occurred: {str(e)}")