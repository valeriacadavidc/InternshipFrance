using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


using System.Threading; //enables use of Thread.Sleep() “wait” method
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.GenericMotorCLI.Settings; //this will specifically target only the commands contained within the .Settings sub-class library in *.GenericMotorCLI.dll.
using Thorlabs.MotionControl.KCube.DCServoCLI;
using System.Security.Cryptography;
using Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor;
using Thorlabs.MotionControl.GenericMotorCLI;

namespace ejemplo1tutorialmanual
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // We create the serial number string of your connected controller. This will
            // be used as an argument for LoadMotorConfiguration(). You can replace this
            // serial number with the number printed on your device.
            string serialNo = "27259483";
            // This instructs the DeviceManager to build and maintain the list of
            // devices connected.
            DeviceManagerCLI.BuildDeviceList();
            // This creates an instance of KCubeDCServo class, passing in the Serial Number parameter. 
            KCubeDCServo device = KCubeDCServo.CreateKCubeDCServo(serialNo);
            // We tell the user that we are opening connection to the device.
            Console.WriteLine("Opening device {0}", serialNo);
            // This connects to the device.
            device.Connect(serialNo);
            // Wait for the device settings to initialize. We ask the device to
            // throw an exception if this takes more than 5000ms (5s) to complete.
            device.WaitForSettingsInitialized(5000);
            // This calls LoadMotorConfiguration on the device to initialize the DeviceUnitConverter object required for real world unit parameters.
            MotorConfiguration motorSettings = device.LoadMotorConfiguration(serialNo,    DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings);
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

            device.SetVelocityParams((decimal)0.01, 1);


            device.MoveTo(14, 60000);
            // Stop polling the device.
            device.StopPolling();
            // This shuts down the controller. This will use the Disconnect() function to  close communications &will then close the used library.
            device.ShutDown();
            // Click any key at the end of the program to exit.
            Console.WriteLine("Process complete. Press any key to exit");
            Console.ReadKey();

        }
    }
}
