import cv2
import os
import sys
import threading
import queue
import time
import numpy as np
import serial
import logging


# Optimización de OpenCV

cv2.setUseOptimized(True)


# Rutas

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

dataPath = os.path.join(
    BASE_DIR,
    'Data'
)

REPO_SRC = os.path.join(
    BASE_DIR,
    'a9-v720',
    'src'
)

sys.path.insert(0, REPO_SRC)


# Ocultar mensajes de depuración de la cámara A9

from log import log

# Oculta los mensajes internos de depuración de la cámara.
log.set_log_lvl(logging.WARNING)


# Librerías de la cámara

from netcl_tcp import netcl_tcp
from v720_ap import v720_ap
import cmd_udp


# Configuración de la cámara

CAMERA_IP = '192.168.169.1'
CAMERA_PORT = 6123

VIDEO_TIMEOUT = 10
RECONNECT_DELAY = 2


# Configuración de video

ANCHO_PROCESAMIENTO = 480

# Umbral LBPH: menor valor = reconocimiento más estricto.
UMBRAL_RECONOCIMIENTO = 70


# Configuración del ESP32 y relé

ESP32_PORT = "COM8"
ESP32_BAUD = 115200

# Evita activar el relé repetidamente con el mismo rostro.
TIEMPO_ENTRE_ACCESOS = 5


# Conexión con el ESP32

try:

    esp32 = serial.Serial(
        ESP32_PORT,
        ESP32_BAUD,
        timeout=1
    )

    # Espera el reinicio del ESP32 al abrir el puerto.
    time.sleep(2)

    print(
        f"ESP32 conectado correctamente en {ESP32_PORT}."
    )

except serial.SerialException as error:

    print(
        f"ERROR: No se pudo abrir {ESP32_PORT}: {error}"
    )

    print(
        "Cierra el Monitor Serial de Arduino IDE y vuelve a ejecutar."
    )

    raise


ultimo_acceso = 0


# Personas registradas

imagePaths = sorted([
    nombre
    for nombre in os.listdir(dataPath)
    if os.path.isdir(
        os.path.join(dataPath, nombre)
    )
])

print(
    "Personas registradas:",
    imagePaths
)


# Cola de frames

frame_queue = queue.Queue(
    maxsize=1
)

stop_event = threading.Event()


# Guardar solo el frame más reciente

def guardar_frame(frame):

    if frame is None:
        return

    # Descarta el frame anterior.
    try:
        frame_queue.get_nowait()

    except queue.Empty:
        pass

    # Guarda el frame más reciente.
    try:
        frame_queue.put_nowait(
            frame
        )

    except queue.Full:
        pass


# Hilo de la cámara

def camera_worker():

    conexion = 0

    while not stop_event.is_set():

        conexion += 1

        print()
        print(
            "=============================================="
        )

        print(
            f"Conectando cámara A9 - intento {conexion}"
        )

        print(
            "=============================================="
        )

        connection_stop = threading.Event()

        try:

            # Conectar socket

            with netcl_tcp(
                CAMERA_IP,
                CAMERA_PORT
            ) as sock:

                print(
                    "Socket conectado."
                )


                # Inicializar cámara

                cam = v720_ap(
                    sock
                )

                cam.init_live_motion()

                print(
                    "Cámara A9 detectada."
                )


                # Ajustes de la cámara

                try:

                    cam.ir_led(False)

                except Exception:
                    pass


                try:

                    cam.flip(False)

                except Exception:
                    pass


                # Buffer JPEG

                jpeg_buffer = bytearray()


                estado = {
                    'ultimo_frame': time.monotonic()
                }


                # Control de pérdida de video

                def watchdog():

                    while not connection_stop.wait(1):

                        tiempo_sin_video = (
                            time.monotonic()
                            - estado['ultimo_frame']
                        )

                        if (
                            tiempo_sin_video
                            >= VIDEO_TIMEOUT
                        ):

                            print()

                            print(
                                f"⚠ Cámara sin imágenes durante "
                                f"{tiempo_sin_video:.1f} segundos."
                            )

                            print(
                                "Forzando reconexión..."
                            )

                            try:

                                sock.close()

                            except Exception:
                                pass

                            return


                watchdog_thread = threading.Thread(
                    target=watchdog,
                    daemon=True
                )

                watchdog_thread.start()


                # Recibir datos de la cámara

                def on_rcv(cmd, data):

                    # Procesa solo paquetes JPEG.
                    if (
                        cmd
                        != cmd_udp.P2P_UDP_CMD_JPEG
                    ):
                        return

                    if not data:
                        return


                    # Añade datos al buffer.
                    jpeg_buffer.extend(
                        data
                    )


                    # Buscar imágenes JPEG completas

                    while True:

                        inicio = jpeg_buffer.find(
                            b'\xff\xd8'
                        )

                        # Si no hay inicio JPEG, espera más datos.
                        if inicio == -1:

                            if (
                                len(jpeg_buffer)
                                > 2_000_000
                            ):

                                jpeg_buffer.clear()

                            return


                        fin = jpeg_buffer.find(
                            b'\xff\xd9',
                            inicio + 2
                        )


                        # Espera hasta recibir el JPEG completo.
                        if fin == -1:

                            if inicio > 0:

                                del jpeg_buffer[
                                    :inicio
                                ]


                            if (
                                len(jpeg_buffer)
                                > 2_000_000
                            ):

                                jpeg_buffer.clear()

                            return


                        # Extraer JPEG

                        jpeg_bytes = bytes(
                            jpeg_buffer[
                                inicio:fin + 2
                            ]
                        )


                        del jpeg_buffer[
                            :fin + 2
                        ]


                        # Decodificar JPEG con OpenCV

                        jpeg_array = np.frombuffer(
                            jpeg_bytes,
                            dtype=np.uint8
                        )


                        frame = cv2.imdecode(
                            jpeg_array,
                            cv2.IMREAD_COLOR
                        )


                        if frame is None:
                            continue


                        estado['ultimo_frame'] = (
                            time.monotonic()
                        )


                        guardar_frame(
                            frame
                        )


                # Iniciar transmisión de video

                print(
                    "Iniciando transmisión de vídeo..."
                )

                estado['ultimo_frame'] = (
                    time.monotonic()
                )


                cam.cap_live(
                    on_rcv
                )


                print()
                print(
                    "La transmisión terminó."
                )


        except Exception as error:

            print()

            print(
                "⚠ Conexión con cámara interrumpida:"
            )

            print(
                error
            )


        finally:

            connection_stop.set()


        # Reconectar cámara

        if not stop_event.is_set():

            print(
                f"Reconectando en "
                f"{RECONNECT_DELAY} segundos..."
            )

            time.sleep(
                RECONNECT_DELAY
            )


# Iniciar hilo de la cámara

cam_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

cam_thread.start()


# Modelo LBPH

face_recognizer = (
    cv2.face.LBPHFaceRecognizer_create()
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    'modeloLBPHFace.xml'
)


face_recognizer.read(
    MODEL_PATH
)


# Clasificador de rostros

faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + 'haarcascade_frontalface_default.xml'
)


# Ventana de video

cv2.namedWindow(
    'Reconocimiento Facial',
    cv2.WINDOW_NORMAL
)


# Variables de control

ultimo_aviso = 0

fps = 0

contador_fps = 0

tiempo_fps = (
    time.monotonic()
)


# Bucle principal

try:

    while True:

        # Obtener frame

        try:

            frame = frame_queue.get(
                timeout=0.05
            )

        except queue.Empty:

            tecla = (
                cv2.waitKey(1)
                & 0xFF
            )


            # Salir con ESC.
            if tecla == 27:
                break


            ahora = (
                time.monotonic()
            )


            # Muestra el aviso cada 5 segundos.
            if (
                ahora
                - ultimo_aviso
                >= 5
            ):

                print(
                    "Esperando imágenes de la cámara..."
                )

                ultimo_aviso = ahora

            continue


        # Redimensionar frame

        alto_original, ancho_original = (
            frame.shape[:2]
        )


        nuevo_ancho = (
            ANCHO_PROCESAMIENTO
        )


        nuevo_alto = int(
            alto_original
            * nuevo_ancho
            / ancho_original
        )


        frame = cv2.resize(
            frame,
            (
                nuevo_ancho,
                nuevo_alto
            ),
            interpolation=cv2.INTER_AREA
        )


        # Convertir a escala de grises

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # Detectar rostros

        faces = (
            faceClassif.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60)
            )
        )


        # Reconocer rostros detectados

        for (
            x,
            y,
            w,
            h
        ) in faces:


            rostro = gray[
                y:y + h,
                x:x + w
            ]


            if rostro.size == 0:
                continue


            rostro = cv2.resize(
                rostro,
                (150, 150),
                interpolation=cv2.INTER_CUBIC
            )


            # Reconocimiento LBPH

            result = (
                face_recognizer.predict(
                    rostro
                )
            )


            id_persona = (
                result[0]
            )


            confianza = (
                result[1]
            )


            # Persona reconocida

            if (
                confianza
                < UMBRAL_RECONOCIMIENTO
            ):

                if (
                    0
                    <= id_persona
                    < len(imagePaths)
                ):

                    nombre = (
                        imagePaths[
                            id_persona
                        ]
                    )


                    # Activar relé mediante el ESP32

                    ahora_acceso = (
                        time.monotonic()
                    )


                    if (
                        ahora_acceso
                        - ultimo_acceso
                        >= TIEMPO_ENTRE_ACCESOS
                    ):

                        print(
                            f"Acceso autorizado: {nombre}"
                        )


                        esp32.write(
                            b"ABRIR\n"
                        )


                        ultimo_acceso = (
                            ahora_acceso
                        )


                else:

                    nombre = (
                        "ID desconocido"
                    )


                # Verde: persona reconocida.
                color = (
                    0,
                    255,
                    0
                )


            # Persona desconocida

            else:

                nombre = (
                    "Desconocido"
                )

                # Rojo: persona desconocida.
                color = (
                    0,
                    0,
                    255
                )


            # Dibujar recuadro del rostro

            cv2.rectangle(
                frame,
                (
                    x,
                    y
                ),
                (
                    x + w,
                    y + h
                ),
                color,
                2
            )


            # Mostrar nombre

            cv2.putText(
                frame,
                nombre,
                (
                    x,
                    max(
                        y - 25,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )


            # Mostrar nivel de confianza

            cv2.putText(
                frame,
                f'Conf: {confianza:.1f}',
                (
                    x,
                    max(
                        y - 5,
                        40
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    255,
                    255,
                    0
                ),
                1,
                cv2.LINE_AA
            )


        # Calcular FPS

        contador_fps += 1

        ahora = (
            time.monotonic()
        )


        diferencia = (
            ahora
            - tiempo_fps
        )


        if diferencia >= 1:

            fps = (
                contador_fps
                / diferencia
            )


            contador_fps = 0

            tiempo_fps = ahora




        # Mostrar video

        cv2.imshow(
            'Reconocimiento Facial',
            frame
        )


        # Salir con ESC

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        if tecla == 27:
            break


# Detener con Ctrl + C

except KeyboardInterrupt:

    print(
        "Programa detenido."
    )


# Cerrar recursos

finally:

    stop_event.set()

    cv2.destroyAllWindows()


    if (
        'esp32' in globals()
        and esp32.is_open
    ):

        esp32.close()


    print(
        "Reconocimiento facial finalizado."
    )