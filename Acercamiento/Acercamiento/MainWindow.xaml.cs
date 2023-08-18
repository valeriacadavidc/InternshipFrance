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
using System.Reflection;
using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.DeviceManagerUI;


namespace Acercamiento
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private GenericDeviceHolder.GenericDevice _genericDevice;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            // Register device DLLs so they can ba accessed by the DeviceManager
            // These devices are self referencing and do not need to be referenced by the project
            // This info could be supplied from a config file.
            // Registration (with the device manager) enables a UI Factory to create the UI freom a generic device
            string path = System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            DeviceManager.RegisterLibrary(null, System.IO.Path.Combine(path, "Thorlabs.MotionControl.KCube.DCServoCLI"), " Thorlabs.MotionControl.KCube.DCServoUI");

            // Get list of devices
            DeviceManagerCLI.BuildDeviceList();
            // Tell the device manager the device types we are interested in 
            // (37=filter flipper, 67=TCube Brushless motor, 73=Benchtop Brushless motor, 83=TCube DC Servo)
            List<string> devices = DeviceManagerCLI.GetDeviceList(new List<int> { 37, 67, 73, 83, 27});
            if (devices.Count == 0)
            {
                MessageBox.Show("No Devices");
                return;
            }

            // Populate the ComboBox - with the list of devices
            _devices.ItemsSource = devices;
            _devices.SelectedIndex = 0;

            // Get first serial number in list
            string serialNo = devices[0];
            // and create that device
            ConnectDevice(serialNo);
        }

        private void MainWindow_OnClosed(object sender, EventArgs e)
        {
            // Disconnect any connected device
            DisconnectDevice();
            // Unregister devices before exit
            string path = System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            DeviceManager.UnregisterLibrary(null, System.IO.Path.Combine(path, "Thorlabs.MotionControl.KCube.DCServoCLI"), " Thorlabs.MotionControl.KCube.DCServoUI");
        }

        private void _devices_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Get the serial number from the combobox
            string serialNo = _devices.SelectedItem as string;
            if (_genericDevice != null && _genericDevice.CoreDevice.DeviceID == serialNo)
            {
                // The device is already connected so leave it alone
                return;
            }

            // Connect the device
            ConnectDevice(serialNo);
        }

        private void ConnectDevice(string serialNo)
        {
            // Unload any currently connected device if not of the desired type
            if (_genericDevice != null)
            {
                if (_genericDevice.CoreDevice.DeviceID == serialNo)
                {
                    return;
                }
                DisconnectDevice();
            }

            // Create the new device anonymously - i.e. create device as generic interface from factory using serial number
            IGenericCoreDeviceCLI device = DeviceFactory.CreateDevice(serialNo);
            // Create a generic device holder to hold the device
            GenericDeviceHolder devices = new GenericDeviceHolder(device);
            // NOTE channel 1 is always available as TCubes are treated as single channel devices
            // For Benchtops, check that the channel exists before accessing it;
            _genericDevice = devices[1];
            if (_genericDevice == null)
            {
                MessageBox.Show("Unknown Device Type");
                return;
            }

            // Connect the device via its serial number, by accessing the core device functions
            _genericDevice.CoreDevice.Connect(serialNo);
            bool connected = _genericDevice.CoreDevice.IsConnected;
            if (!connected)
            {
                MessageBox.Show("Failed to connect");
                return;
            }

            // Wait for settings to be initialized (on the channel)
            _genericDevice.Device.WaitForSettingsInitialized(5000);

            // Create user interface (WPF view) via the DeviceManager
            // Get the User Interface Factory for the device
            IUIFactory factory = DeviceManager.GetUIFactory(_genericDevice.CoreDevice.DeviceID);

            // Create and initialize the view model with configuration settings, to connect the View to the Model, the view model can be 'Full' or 'preview' (small) option
            //IDeviceViewModel viewModel = factory.CreateViewModel(DisplayTypeEnum.Full, _genericDevice);      
            //viewModel.Initialize();

            // Determine the appropriate startup settings mode based on your application logic
            DeviceConfiguration.DeviceSettingsUseOptionType startupSettingsMode = DeviceConfiguration.DeviceSettingsUseOptionType.UseDeviceSettings;

            // Create and initialize the view model with configuration settings
            IDeviceViewModel viewModel = factory.CreateViewModel(DisplayTypeEnum.Full, _genericDevice);
            viewModel.Initialize(startupSettingsMode);

            // Create the view using the UI factory and attach it to our display
            _contentControl.Content = factory.CreateLargeView(viewModel);
        }

        private void DisconnectDevice()
        {
            if ((_genericDevice != null) && _genericDevice.CoreDevice.IsConnected)
            {
                _genericDevice.CoreDevice.Disconnect(true);
                _genericDevice = null;
            }
        }
    }
}

