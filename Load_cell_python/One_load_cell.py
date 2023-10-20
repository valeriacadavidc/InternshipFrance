import clr
import System
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK_USB_DLL.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FUTEK.Devices.dll")
clr.AddReference("C:\\Users\\valeria.cadavid\\Documents\\RepositorioCodigos\\InternshipFrance\\Load_cell_python\\FTD2XX_NET.dll")
import FUTEK_USB_DLL
import FUTEK.Devices
from FUTEK_USB_DLL import USB_DLL

repo = FUTEK.Devices.DeviceRepository()
devices = list(repo.DetectDevices())

#Para ver el numero serial de los dispositivos USB225 print(devices[1].GetInstrumentSerialNumber())
print('valelinda')