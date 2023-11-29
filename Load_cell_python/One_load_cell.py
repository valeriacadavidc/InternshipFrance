import clr
import time
import numpy as np
import matplotlib.pyplot as plt
import System
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK_USB_DLL.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK.Devices.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FTD2XX_NET.dll")
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
for _ in range(1000):
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

