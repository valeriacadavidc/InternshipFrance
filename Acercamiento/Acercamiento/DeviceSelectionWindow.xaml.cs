using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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


namespace Acercamiento
{
    public partial class DeviceSelectionWindow : Window
    {
        public ObservableCollection<DeviceListItem> AvailableDevices { get; private set; } = new ObservableCollection<DeviceListItem>();

        public List<string> SelectedDevices { get; private set; } = new List<string>();

        public DeviceSelectionWindow()
        {
            InitializeComponent();
            PopulateDeviceList();// Llenar la lista de dispositivos disponibles
        }

        private void PopulateDeviceList()
        {
            // Detect available devices
            DeviceManagerCLI.BuildDeviceList();
            List<string> availableDevices = DeviceManagerCLI.GetDeviceList();

            foreach (string device in availableDevices)
            {
                AvailableDevices.Add(new DeviceListItem
                {
                    Serial = device,
                    Description = "Device Description Placeholder"
                });
            }
            DataGrid.ItemsSource = AvailableDevices;
            Console.WriteLine(AvailableDevices);
        }

        private void ConfirmConnectButton_Click(object sender, RoutedEventArgs e)
        {
            SelectedDevices = AvailableDevices.Where(device => device.IsSelected).Select(device => device.Serial).ToList();
            DialogResult = true;
            Close();
        }

        public class DeviceListItem
        {
            public bool IsSelected { get; set; }
            public string Serial { get; set; }
            public string Description { get; set; }
        }

        
    }
}
