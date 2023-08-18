using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.KCube.DCServoCLI;
using Thorlabs.MotionControl.KCube.DCServoUI;
namespace kinesisinterface
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }
        //The field _kCubeDCServo is an instance field of KCubeDCServo. This builds an empty reference _kCubeDCServo.
        //This reference will later be assigned to create an instance of the   KCubeDCServo class object variable which we will interact
        //within MainWindow_OnLoaded() & MainWindow_OnClosed()
         KCubeDCServo _kCubeDCServo = null;

        private void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            // This instructs the DeviceManager to build and maintain the list of
            // devices connected. We then print a list of device name strings called 
            // “devices” which contain the prefix “27”
            DeviceManagerCLI.BuildDeviceList();
            List<string> devices = DeviceManagerCLI.GetDeviceList(27);
            // IF statement – if the number of devices connected is zero, the Window
            // will display “No Devices”.
            if (devices.Count == 0)
            {
                MessageBox.Show("No Devices");
                return;
            }
            // Selects the first device serial number from “devices” list. 
            string serialNo = devices[0];
            // Creates the device. We assign an instance of the device to _kCubeDCServo 
            KCubeDCServo _kCubeDCServo = KCubeDCServo.CreateKCubeDCServo(serialNo);
            // Connect to the device & wait for initialisation. This is contained in a 
            // Try/Catch Error Handling Statement.
            try
            {
                _kCubeDCServo.Connect(serialNo);
                // wait for settings to be initialized
                _kCubeDCServo.WaitForSettingsInitialized(5000);
            }
            catch (DeviceException ex)
            {
                MessageBox.Show(ex.Message);
                return;
            }
            // Create the Kinesis Panel View for KDC101
            _contentControl.Content = KCubeDCServoUI.CreateLargeView(_kCubeDCServo);


        }

        private void MainWindow_OnClosed(object sender, EventArgs e)
        {
            // Disconnect device after closing the Window.
            if ((_kCubeDCServo != null) && _kCubeDCServo.IsConnected)
            {
                _kCubeDCServo.Disconnect(true);
            }
        }
    }
}
