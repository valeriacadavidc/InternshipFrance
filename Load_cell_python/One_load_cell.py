import clr
import System
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK_USB_DLL.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK.Devices.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FTD2XX_NET.dll")
import FUTEK_USB_DLL
import FUTEK.Devices
#from FUTEK_USB_DLL import USB_DLL
from enum import Enum
repo = FUTEK.Devices.DeviceRepository()
devices = list(repo.DetectDevices())

devices[0].SetChannelXUnitOfMeasure(0,FUTEK.Devices.UnitsOfMeasure(32))
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
print('ADC',devices[0].GetChannelXNormalDataRequest(0))
#devices[0].SetChannelXUnitOfMeasure(0,FUTEK.Devices.UnitsOfMeasure(18))
sampling_rate_initial=devices[0].GetChannelXSamplingRate(0)
list_sampling_rates=list(devices[0].GetChannelXSamplingRatePossibleValues(0))
devices[0].SetChannelXSamplingRate(0,list_sampling_rates[11]) #Change the sample rate
sampling_rate_new=devices[0].GetChannelXSamplingRate(0)
unitOfMeasure = devices[0].GetChannelXUnitOfMeasure(0)
reading = devices[0].GetChannelXReading(0)
print('Reading')
print(reading)
print('Unit Of Measure')
print(unitOfMeasure)
print('ADC',devices[0].GetChannelXNormalDataRequest(0))
print('Initial sampling rate',sampling_rate_initial)
print('Sampling rates available',list_sampling_rates)
print('New sampling rate',sampling_rate_new)
repo.DisconnectDevice( modelNumber,serialNumber)

