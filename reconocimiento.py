import serial
import time

esp32 = serial.Serial("COM11", 115200)

time.sleep(2)

print("Conectado al ESP32")

while True:

    comando = input("Escribe ROSTRO para simular reconocimiento: ")

    if comando.upper() == "ROSTRO":

        esp32.write(b"ROSTRO_OK\n")

        print("Rostro autorizado enviado al ESP32")