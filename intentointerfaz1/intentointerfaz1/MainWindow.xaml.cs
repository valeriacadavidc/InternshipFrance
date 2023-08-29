using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading; //enables use of Thread.Sleep() “wait” method

using Thorlabs.MotionControl.DeviceManagerCLI;
using Thorlabs.MotionControl.GenericMotorCLI.Settings; //this will specifically target only the commands contained within the .Settings sub-class library in *.GenericMotorCLI.dll.
using Thorlabs.MotionControl.KCube.DCServoCLI;
using System.Security.Cryptography;
using Thorlabs.MotionControl.GenericMotorCLI.AdvancedMotor;
using Thorlabs.MotionControl.GenericMotorCLI;


namespace intentointerfaz1
{
    public partial class MainWindow : Window
    {
        private KCubeDCServo _device;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            // serial number with the number printed on your device.
            string serialNo = "27259483";
            // Initialize and connect to the device
            DeviceManagerCLI.BuildDeviceList();
            _device = KCubeDCServo.CreateKCubeDCServo(serialNo);
            _device.Connect(serialNo);
            // This calls LoadMotorConfiguration on the device to initialize the DeviceUnitConverter object required for real world unit parameters.
            MotorConfiguration motorSettings = _device.LoadMotorConfiguration(serialNo, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings);
            _device.WaitForSettingsInitialized(5000);
            _device.StartPolling(250);
            _device.EnableDevice();
            _device.Home(60000);
        }

        private async void MoveButton_Click(object sender, RoutedEventArgs e)
        {
            decimal targetPosition = 6; // Set your target position here

            // Ejecutar el movimiento y la actualización de la interfaz simultáneamente
            await MoveAndShowPosition(targetPosition);
        }
        private async Task WaitForDeviceMovementAsync()
        {
            // Esperar hasta que el movimiento haya comenzado (el estado IsMoving cambia a verdadero).
            while (!_device.Status.IsMoving)
            {
                await Task.Delay(100); // Pequeña pausa para no saturar el CPU.
            }
            // Mostrar la posición en tiempo real mientras el dispositivo está en movimiento.
            while (_device.Status.IsMoving)
            {
                decimal currentPosition = _device.Position;
                UpdatePositionUI(currentPosition); // Actualizar la interfaz con la posición actual.
                await Task.Delay(50); // Pequeña pausa para no saturar el CPU.
            }
        }
        /// <summary>
        /// Mueve el dispositivo a la posición objetivo y muestra la posición en tiempo real durante el movimiento.
        /// </summary>
        /// <param name="targetPosition">Posición a la que se debe mover el dispositivo.</param>
        private async Task MoveAndShowPosition(decimal targetPosition)
        {
            // Iniciar una tarea en segundo plano para mover el dispositivo a la posición objetivo.
            Task moveTask = Task.Run(() => _device.MoveTo(targetPosition, 60000));

            // Esperar hasta que el movimiento haya comenzado (el estado IsMoving cambia a verdadero).
            while (!_device.Status.IsMoving)
            {
                await Task.Delay(100); // Pequeña pausa para no saturar el CPU.
            }

            // Esperar hasta que el movimiento haya comenzado (el estado IsMoving cambia a verdadero).
            await WaitForDeviceMovementAsync();

            // Mostrar la posición final una vez que el movimiento se haya completado.
            decimal finalPosition = _device.Position;
            UpdatePositionUI(finalPosition);

            // Esperar a que termine la tarea de movimiento en segundo plano.
            await moveTask;
        }

        /// <summary>
        /// Actualiza la interfaz de usuario con la posición proporcionada.
        /// </summary>
        /// <param name="position">Posición a mostrar en la interfaz.</param>
        private void UpdatePositionUI(decimal position)
        {
            Dispatcher.Invoke(() =>
            {
                PositionTextBlock.Text = $"Position: {position}";
            });
        }

    }
}