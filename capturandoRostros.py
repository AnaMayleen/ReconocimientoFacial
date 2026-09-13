import cv2
import os
import sys
import threading
import queue
import time
import numpy as np


# Optimización de OpenCV

cv2.setUseOptimized(True)


# Persona y carpeta de captura
# Nombre de la persona a registrar
personName = input("Ingrese el nombre de la persona: ").strip()

dataPath = 'Data'

personPath = os.path.join(
    dataPath,
    personName
)

if not os.path.exists(personPath):
    os.makedirs(personPath)
    print("Carpeta creada:", personPath)
else:
    print("La persona ya tiene una carpeta:", personPath)

if not os.path.exists(personPath):

    print(
        'Carpeta creada:',
        personPath
    )

    os.makedirs(
        personPath
    )


# Rutas

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REPO_SRC = os.path.join(
    BASE_DIR,
    'a9-v720',
    'src'
)

sys.path.insert(
    0,
    REPO_SRC
)


# Librerías de la cámara

from netcl_tcp import netcl_tcp
from v720_ap import v720_ap
import cmd_udp


# Configuración de la cámara

CAMERA_IP = '192.168.169.1'

CAMERA_PORT = 6123

# Reconecta si no llegan imágenes durante 10 segundos.
VIDEO_TIMEOUT = 10

# Espera antes de reconectar.
RECONNECT_DELAY = 2


# Configuración de captura

ANCHO_PROCESAMIENTO = 640

MAX_ROSTROS = 300


# Cola de frames

# Evita retraso conservando solo el frame más reciente.

frame_queue = queue.Queue(
    maxsize=1
)

stop_event = threading.Event()


# Guardar el frame más reciente

def guardar_frame(frame):

    if frame is None:
        return

    try:

        frame_queue.get_nowait()

    except queue.Empty:

        pass

    try:

        frame_queue.put_nowait(
            frame
        )

    except queue.Full:

        pass


# Conexión con la cámara

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

                # Ajustes de cámara

                try:

                    cam.ir_led(
                        False
                    )

                except Exception:

                    pass

                try:

                    cam.flip(
                        False
                    )

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


                # Recibir datos

                def on_rcv(cmd, data):

                    # Procesar solo paquetes JPEG.

                    if (
                        cmd
                        != cmd_udp.P2P_UDP_CMD_JPEG
                    ):

                        return

                    if not data:

                        return


                    # Añadir datos al buffer.

                    jpeg_buffer.extend(
                        data
                    )


                    # Buscar JPEG completos

                    while True:

                        inicio = jpeg_buffer.find(
                            b'\xff\xd8'
                        )

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


                        # Decodificar JPEG

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


                        # Actualizar control de video.

                        estado['ultimo_frame'] = (
                            time.monotonic()
                        )


                        # Guardar el frame más reciente.

                        guardar_frame(
                            frame
                        )


                # Iniciar transmisión

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


# Iniciar cámara

cam_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

cam_thread.start()


# Detector de rostros

faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + 'haarcascade_frontalface_default.xml'
)


# Ventana

cv2.namedWindow(
    'Captura de Rostros',
    cv2.WINDOW_NORMAL
)


# Contador

count = 0

ultimo_aviso = 0


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

            if tecla == 27:

                break


            ahora = time.monotonic()

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


        # Redimensionar

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


        # Copia para guardar rostros.

        auxFrame = frame.copy()


        # Escala de grises

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # Detectar rostros

        faces = faceClassif.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # Guardar rostros

        for (
            x,
            y,
            w,
            h
        ) in faces:

            # Marcar rostro.

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
                (
                    0,
                    255,
                    0
                ),
                2
            )


            # Extraer rostro.

            rostro = auxFrame[
                y:y + h,
                x:x + w
            ]


            if rostro.size == 0:

                continue


            rostro = cv2.resize(
                rostro,
                (
                    150,
                    150
                ),
                interpolation=cv2.INTER_CUBIC
            )


            # Guardar imagen

            ruta_rostro = os.path.join(
                personPath,
                f'rostro_{count}.jpg'
            )

            cv2.imwrite(
                ruta_rostro,
                rostro
            )


            count += 1


            print(
                f"Rostro capturado: "
                f"{count}/{MAX_ROSTROS}"
            )


            # Mostrar contador.

            cv2.putText(
                frame,
                f'Capturas: {count}/{MAX_ROSTROS}',
                (
                    10,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    0
                ),
                2,
                cv2.LINE_AA
            )


            if count >= MAX_ROSTROS:

                break


        # Mostrar contador

        cv2.putText(
            frame,
            f'Capturas: {count}/{MAX_ROSTROS}',
            (
                10,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA
        )


        # Mostrar video

        cv2.imshow(
            'Captura de Rostros',
            frame
        )


        # Salir

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ESC para salir.
        if tecla == 27:

            break


        # Finalizar al completar las capturas.

        if count >= MAX_ROSTROS:

            print()

            print(
                f"Captura completada: "
                f"{count} rostros guardados."
            )

            break


except KeyboardInterrupt:

    print(
        "Programa detenido manualmente."
    )


finally:

    stop_event.set()

    cv2.destroyAllWindows()

    print(
        "Captura de rostros finalizada."
    )