using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input; // Agrega este using para acceder a la clase Keyboard


using System.Threading; //enables use of Thread.Sleep() “wait” method
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.GenericMotorCLI.Settings; //this will specifically target only the commands contained within the .Settings sub-class library in *.GenericMotorCLI.dll.
using Thorlabs.MotionControl.KCube.DCServoCLI;
using System.Security.Cryptography;
using Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor;
using Thorlabs.MotionControl.GenericMotorCLI;
using System.Collections;
using System.Runtime;
using static System.Net.Mime.MediaTypeNames;
using Thorlabs.MotionControl.GenericMotorCLI.ControlParameters;

namespace ejemplo1tutorialmanual
{
    internal class Program
    {
        static List<KCubeDCServo> dispositivosCreados = new List<KCubeDCServo>();
        static Barrier barrier = new Barrier(2); // 2 tareas



        static void Main(string[] args)
        {
            Task tarea1 = Task.Run(() => EjecutarFuncionParaTarea("27259483"));
            Task tarea2 = Task.Run(() => EjecutarFuncionParaTarea("27260278"));

            Task.WaitAll(tarea1, tarea2);

            Console.WriteLine("Ambas tareas 1 han terminado.");

            Task tarea3 = Task.Run(() => recorrido("27259483"));
            Task tarea4 = Task.Run(() => recorrido("27260278"));
            Task.WaitAll(tarea3, tarea4);
            Console.ReadLine();
        }

        static void recorrido(string serialNo)

        {
            barrier.SignalAndWait(); // Esperar a que ambas tareas se inicien
            KCubeDCServo device = Crear_Obtener_Dispositivo(serialNo);
            //Sets the velocity parameters in Real World Units. virtual void SetVelocityParams  ( Decimal  maxVelocity,   Decimal acceleration) 
            // maxVelocity The maximum velocity in Real World Units. 
            //acceleration The acceleration in Real World Units.
            //Gets the velocity parameters in Real World Units.
            VelocityParameters velocidadParameters = device.GetVelocityParams();
            device.SetVelocityParams((decimal)1.5, (decimal)4.5);

            //Gets the velocity parameters in Real World Units.
            velocidadParameters = device.GetVelocityParams();
            decimal maxvel = velocidadParameters.MaxVelocity;
            decimal ace = velocidadParameters.Acceleration;
            decimal minVelocity = velocidadParameters.MinVelocity;
            Console.WriteLine($"MaxVelocity: {maxvel}");
            Console.WriteLine($"Aceleration:  {ace}");
            Console.WriteLine($"MinVelocity: {minVelocity}");
            // Obtiene la hora exacta
            string currentTime = DateTime.Now.ToString("HH:mm:ss.fff");
            // Imprime la hora exacta
            Console.WriteLine("Hora exacta inicio: " + currentTime);
            Thread.Sleep(5000);
            device.MoveTo(18, 60000);
            Thread.Sleep(2000);
            device.MoveTo(15, 60000);

            currentTime = DateTime.Now.ToString("HH:mm:ss.fff");
            // Imprime la hora exacta
            Console.WriteLine("Hora exacta 15mm: " + currentTime);

            decimal currentPosition = device.Position;
            Console.WriteLine($"Posicion del dispositivo con serial {serialNo}: {currentPosition}");
            // Stop polling the device.
            device.StopPolling();
            // This shuts down the controller. This will use the Disconnect() function to  close communications &will then close the used library.
            device.ShutDown();
            // Desconectar dispositivo
            device.Disconnect(true);
            Console.WriteLine("Finalizado,", serialNo);
        }
        static void EjecutarFuncionParaTarea(string serialNo)
        {
            

            // Detect available devices
            DeviceManagerCLI.BuildDeviceList();    
            //USBDeviceManager.Instance().
            List<string> availableDevices = DeviceManagerCLI.GetDeviceList(); // lista de dispositivos disponibles
            Console.WriteLine("Elementos en la lista: " + string.Join(", ", availableDevices));

            // We create the serial number string of your connected controller. This will
            // be used as an argument for LoadMotorConfiguration(). You can replace this
            // serial number with the number printed on your device.
            //string serialNo = availableDevices[0];
            //string serialNo = "27259483"; // serial number del controlador actual



            //device.Disconnect(true);
            //Console.WriteLine(DeviceManagerCLI.IsDeviceConnected(serialNo));
            //if (!DeviceManagerCLI.IsDeviceConnected(serialNo)) //si el dispositivo no esta conectado
           
            // This creates an instance of KCubeDCServo class, passing in the Serial Number parameter. 
             //KCubeDCServo.CreateKCubeDCServo(serialNo);
            KCubeDCServo device = Crear_Obtener_Dispositivo(serialNo);
            // We tell the user that we are opening connection to the device.
            //Console.WriteLine("Opening device {0}", serialNo);
            //Console.WriteLine(device.IsConnected);
            //// This connects to the device.
            //device.Connect(serialNo);
            //Console.WriteLine(device.IsConnected);
            Console.WriteLine("Identificando dispositivo");

            //device.IdentifyDevice();//para hacer que titilee el equipo o controlador en cuestion
            
            Console.WriteLine("Valelinda");
            // Get device name
            string deviceName = device.DeviceName;
            Console.WriteLine($"Device name: {deviceName}");
            // Get device info for the device
            DeviceInfo deviceInfo = device.GetDeviceInfo();
            Console.WriteLine($"Device info: {deviceInfo}");

            // Wait for the device settings to initialize. We ask the device to
            // throw an exception if this takes more than 5000ms (5s) to complete.
            device.WaitForSettingsInitialized(5000);
            // This calls LoadMotorConfiguration on the device to initialize the DeviceUnitConverter object required for real world unit parameters.
            // initializes the current motor configuration.
            //This will load the settings appropriate for the motor / stage combination as defined in the DeviceConfiguration settings.
            //This should only be called once.Subsequently, the MotorConfiguration can be obtained using MotorConfiguration.
            MotorConfiguration motorSettings = device.LoadMotorConfiguration(serialNo, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings); //original
                                                                                                                                                            //MotorConfiguration motorSettings = device.LoadMotorConfiguration(serialNo); tambien funciona

            if (motorSettings != null)
            {
                Console.WriteLine("valelinda");
                string actuator = motorSettings.DeviceSettingsName;
                Console.WriteLine($"Device connected to the driver {actuator}");
                // Get Settings for device, observando NO SON RELEVANTES PARA MI POR AHORA
                DCMotorSettings settings = DCMotorSettings.GetSettings(motorSettings);
                Console.WriteLine(settings);

            }


            //Gets the velocity parameters in Real World Units.
            VelocityParameters velocidadParameters = device.GetVelocityParams();
            decimal maxvel = velocidadParameters.MaxVelocity;
            decimal ace = velocidadParameters.Acceleration;
            decimal minVelocity = velocidadParameters.MinVelocity;
            Console.WriteLine($"MaxVelocity: {maxvel}");
            Console.WriteLine($"Aceleration:  {ace}");
            Console.WriteLine($"MinVelocity: {minVelocity}");

            //Checks that the current motor settings are valid.true if the current mnotor settings are valid.
            //IsMotorSettingsValid; 
            bool isMotorSettingsValid = device.IsMotorSettingsValid;
            Console.WriteLine($"is valid motor sttings: {isMotorSettingsValid}");

            // Get the DeviceSettings configured for the motor
            MotorDeviceSettings motorDeviceSettings = device.MotorDeviceSettings;


            // This starts polling the device at intervals of 250ms (0.25s).
            device.StartPolling(250);
            // We are now able to Enable the device otherwise any move is ignored. You should see a physical response from your controller.
            device.EnableDevice();
            Console.WriteLine("Device Enabled");
            // Needs a delay to give time for the device to be enabled.
            Thread.Sleep(500);
            // Home the stage/actuator. 
            Console.WriteLine("Actuator is Homing");
            device.Home(60000);
            // Move the stage/actuator to 5mm (or degrees depending on the device connected).
            Console.WriteLine("Actuator is Moving");
            device.MoveTo(25, 60000);
            // Obtiene la hora exacta
            string currentTime = DateTime.Now.ToString("HH:mm:ss.fff");
            // Imprime la hora exacta
            Console.WriteLine("Hora exacta 25 mm: " + currentTime);


            //virtual void Stop(int waitTimeout)
            //Stops the current motor move.
            //Parameters
            //waitTimeout The wait timeout
            //If this is value is 0 then the function will return immediately.
            //If this value is non zero, then the function will wait until the move completes or the timeout elapses, whichever comes first.

            //device.Stop(0);

            //device.Home(60000) or device.Move(position, 60000). These functions will execute the command and will wait for completion.
            //    //device.Home() or device.Move(position). These functions will execute the command but will not wait for completion.
            //}




        }
        static KCubeDCServo Crear_Obtener_Dispositivo(string serialNo)
        {
            DeviceManagerCLI.BuildDeviceList();
            string serialNumber = DeviceManagerCLI.GetDeviceList().FirstOrDefault(s => s == serialNo);

            if (serialNumber != null)
            {
                KCubeDCServo deviceConectado = dispositivosCreados.FirstOrDefault(d => d.DeviceID == serialNumber);

                if (deviceConectado != null)
                {
                    Console.WriteLine($"Dispositivo con SerialNo {serialNo} ya está conectado.");
                    return deviceConectado;
                }
                else
                {
                    KCubeDCServo nuevoDevice = KCubeDCServo.CreateKCubeDCServo(serialNumber);

                    if (!nuevoDevice.IsConnected)
                    {
                        Console.WriteLine($"Creando y conectando dispositivo con SerialNo {serialNo}");
                        nuevoDevice.Connect(serialNo);
                        dispositivosCreados.Add(nuevoDevice);
                        Console.WriteLine($"Dispositivo con SerialNo {serialNo} conectado.");
                    }

                    return nuevoDevice;
                }
            }
            else
            {
                Console.WriteLine($"No se encontró un dispositivo con SerialNo {serialNo}.");
                return null;
            }
        }
        
        
        
    }
}
