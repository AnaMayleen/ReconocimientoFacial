import serial
import time

# Cambia COM8 si tu ESP32 usa otro puerto
esp32 = serial.Serial("COM8", 115200)

# Espera a que el ESP32 termine de reiniciar
time.sleep(2)

print("Conectado al ESP32")

while True:
    comando = input("Escribe ABRIR para activar el rele: ")

    if comando.upper() == "ABRIR":
        esp32.write(b"ABRIR\n")
        print("Señal enviada al ESP32")