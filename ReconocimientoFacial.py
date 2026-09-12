import cv2
import os
import sys
import threading
import queue
import time
import numpy as np
import serial


# ==========================================================
# OPTIMIZACIÓN DE OPENCV
# ==========================================================

cv2.setUseOptimized(True)


# ==========================================================
# RUTAS
# ==========================================================

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


# ==========================================================
# LIBRERÍAS DE LA CÁMARA
# ==========================================================

from netcl_tcp import netcl_tcp
from v720_ap import v720_ap
import cmd_udp


# ==========================================================
# CONFIGURACIÓN DE LA CÁMARA
# ==========================================================

CAMERA_IP = '192.168.169.1'
CAMERA_PORT = 6123

VIDEO_TIMEOUT = 10

RECONNECT_DELAY = 2


# ==========================================================
# CONFIGURACIÓN DE VIDEO
# ==========================================================

ANCHO_PROCESAMIENTO = 480

# Umbral LBPH:
# menor = más estricto
# mayor = más permisivo
UMBRAL_RECONOCIMIENTO = 70

# ==========================================================
# CONFIGURACIÓN DEL ESP32 / RELÉ
# ==========================================================

ESP32_PORT = "COM8"
ESP32_BAUD = 115200

# Evita activar el relé muchas veces seguidas mientras
# el mismo rostro permanece frente a la cámara.
TIEMPO_ENTRE_ACCESOS = 5

try:
    esp32 = serial.Serial(
        ESP32_PORT,
        ESP32_BAUD,
        timeout=1
    )

    # Al abrir el puerto serial, el ESP32 puede reiniciarse.
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



# ==========================================================
# PERSONAS REGISTRADAS
# ==========================================================

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


# ==========================================================
# COLA DE FRAMES
# ==========================================================

frame_queue = queue.Queue(
    maxsize=1
)

stop_event = threading.Event()


# ==========================================================
# GUARDAR SOLO EL FRAME MÁS NUEVO
# ==========================================================

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


# ==========================================================
# HILO DE LA CÁMARA
# ==========================================================

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

            # ==================================================
            # CONECTAR SOCKET
            # ==================================================

            with netcl_tcp(
                CAMERA_IP,
                CAMERA_PORT
            ) as sock:

                print(
                    "Socket conectado."
                )


                # ==================================================
                # INICIALIZAR CÁMARA
                # ==================================================

                cam = v720_ap(
                    sock
                )

                cam.init_live_motion()

                print(
                    "Cámara A9 detectada."
                )


                # ==================================================
                # CONFIGURACIÓN OPCIONAL
                # ==================================================

                try:
                    cam.ir_led(False)
                except Exception:
                    pass

                try:
                    cam.flip(False)
                except Exception:
                    pass


                # ==================================================
                # BUFFER JPEG
                # ==================================================

                jpeg_buffer = bytearray()


                estado = {
                    'ultimo_frame': time.monotonic()
                }


                # ==================================================
                # WATCHDOG
                # ==================================================

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


                # ==================================================
                # RECIBIR DATOS
                # ==================================================

                def on_rcv(cmd, data):

                    # Solo JPEG
                    if (
                        cmd
                        != cmd_udp.P2P_UDP_CMD_JPEG
                    ):
                        return

                    if not data:
                        return


                    # Añadir datos al buffer
                    jpeg_buffer.extend(
                        data
                    )


                    # ==================================================
                    # BUSCAR IMÁGENES JPEG COMPLETAS
                    # ==================================================

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


                        # ==================================================
                        # EXTRAER JPEG
                        # ==================================================

                        jpeg_bytes = bytes(
                            jpeg_buffer[
                                inicio:fin + 2
                            ]
                        )


                        del jpeg_buffer[
                            :fin + 2
                        ]


                        # ==================================================
                        # DECODIFICAR CON OPENCV
                        # ==================================================

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


                # ==================================================
                # INICIAR VIDEO
                # ==================================================

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


        # ==================================================
        # RECONECTAR
        # ==================================================

        if not stop_event.is_set():

            print(
                f"Reconectando en "
                f"{RECONNECT_DELAY} segundos..."
            )

            time.sleep(
                RECONNECT_DELAY
            )


# ==========================================================
# INICIAR HILO DE CÁMARA
# ==========================================================

cam_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

cam_thread.start()


# ==========================================================
# MODELO LBPH
# ==========================================================

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


# ==========================================================
# CLASIFICADOR DE ROSTROS
# ==========================================================

faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + 'haarcascade_frontalface_default.xml'
)


# ==========================================================
# VENTANA
# ==========================================================

cv2.namedWindow(
    'Reconocimiento Facial',
    cv2.WINDOW_NORMAL
)


# ==========================================================
# VARIABLES
# ==========================================================

ultimo_aviso = 0

fps = 0

contador_fps = 0

tiempo_fps = (
    time.monotonic()
)


# ==========================================================
# BUCLE PRINCIPAL
# ==========================================================

try:

    while True:

        # ==================================================
        # OBTENER FRAME
        # ==================================================

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


            ahora = (
                time.monotonic()
            )


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


        # ==================================================
        # REDIMENSIONAR
        # ==================================================

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


        # ==================================================
        # ESCALA DE GRISES
        # ==========================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ==========================================================
        # DETECTAR ROSTROS EN CADA FRAME
        # ==========================================================

        faces = (
            faceClassif.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60)
            )
        )


        # ==========================================================
        # RECONOCER CADA ROSTRO EN ESTE MISMO FRAME
        # ==========================================================

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


            # ==================================================
            # RECONOCIMIENTO LBPH
            # ==================================================

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


            # ==================================================
            # CONOCIDO
            # ==================================================

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

                    # ==============================================
                    # ACTIVAR RELÉ MEDIANTE EL ESP32
                    # ==============================================

                    ahora_acceso = time.monotonic()

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

                        ultimo_acceso = ahora_acceso

                else:

                    nombre = (
                        "ID desconocido"
                    )


                color = (
                    0,
                    255,
                    0
                )


            # ==================================================
            # DESCONOCIDO
            # ==================================================

            else:

                nombre = (
                    "Desconocido"
                )

                color = (
                    0,
                    0,
                    255
                )


            # ==================================================
            # RECTÁNGULO
            # ==================================================

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


            # ==================================================
            # NOMBRE
            # ==================================================

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


            # ==================================================
            # CONFIANZA
            # ==================================================

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


        # ==================================================
        # FPS
        # ==================================================

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


        # ==================================================
        # MOSTRAR FPS
        # ==================================================

        cv2.putText(
            frame,
            f'FPS: {fps:.1f}',
            (
                10,
                25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA
        )


        # ==================================================
        # MOSTRAR VIDEO
        # ==================================================

        cv2.imshow(
            'Reconocimiento Facial',
            frame
        )


        # ==================================================
        # TECLA ESC
        # ==================================================

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        if tecla == 27:
            break


except KeyboardInterrupt:

    print(
        "Programa detenido."
    )


finally:

    stop_event.set()

    cv2.destroyAllWindows()

    if 'esp32' in globals() and esp32.is_open:
        esp32.close()

    print(
        "Reconocimiento facial finalizado."
    )