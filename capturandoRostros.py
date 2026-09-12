import cv2
import os
import sys
import threading
import queue
import time
import numpy as np


# ==========================================================
# OPTIMIZACIÓN DE OPENCV
# ==========================================================

cv2.setUseOptimized(True)


# ==========================================================
# CONFIGURACIÓN DE PERSONA
# ==========================================================

personName = 'Ashley'

dataPath = 'Data'

personPath = os.path.join(
    dataPath,
    personName
)

if not os.path.exists(personPath):

    print(
        'Carpeta creada:',
        personPath
    )

    os.makedirs(
        personPath
    )


# ==========================================================
# RUTAS
# ==========================================================

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

# Si pasan 10 segundos sin recibir imágenes,
# se fuerza una reconexión.
VIDEO_TIMEOUT = 10

# Tiempo antes de intentar conectarse nuevamente.
RECONNECT_DELAY = 2


# ==========================================================
# CONFIGURACIÓN DE CAPTURA
# ==========================================================

ANCHO_PROCESAMIENTO = 640

MAX_ROSTROS = 300


# ==========================================================
# COLA DE FRAMES
# ==========================================================

# Solo conservamos el frame más nuevo.
# Esto evita acumulación y retraso en el vídeo.

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
# HILO DE CONEXIÓN CON LA CÁMARA
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
                # RECIBIR DATOS DE LA CÁMARA
                # ==================================================

                def on_rcv(cmd, data):

                    # Solo nos interesan paquetes JPEG.

                    if (
                        cmd
                        != cmd_udp.P2P_UDP_CMD_JPEG
                    ):

                        return

                    if not data:

                        return


                    # Añadir datos recibidos al buffer.

                    jpeg_buffer.extend(
                        data
                    )


                    # ==================================================
                    # BUSCAR JPEG COMPLETOS
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


                        # Actualizar watchdog.

                        estado['ultimo_frame'] = (
                            time.monotonic()
                        )


                        # Guardar solamente el frame más nuevo.

                        guardar_frame(
                            frame
                        )


                # ==================================================
                # INICIAR TRANSMISIÓN
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
    'Captura de Rostros',
    cv2.WINDOW_NORMAL
)


# ==========================================================
# CONTADOR
# ==========================================================

count = 0

ultimo_aviso = 0


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


        # ==================================================
        # REDIMENSIONAR
        # ==========================================================

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


        # Copia limpia para guardar los rostros.

        auxFrame = frame.copy()


        # ==================================================
        # ESCALA DE GRISES
        # ==========================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ==================================================
        # DETECTAR ROSTROS
        # ==========================================================

        faces = faceClassif.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # ==================================================
        # GUARDAR ROSTROS
        # ==========================================================

        for (
            x,
            y,
            w,
            h
        ) in faces:

            # Dibujar rectángulo.

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


            # ==================================================
            # GUARDAR IMAGEN
            # ==================================================

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


        # ==================================================
        # MOSTRAR CONTADOR AUNQUE NO HAYA ROSTRO
        # ==========================================================

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


        # ==================================================
        # MOSTRAR VÍDEO
        # ==========================================================

        cv2.imshow(
            'Captura de Rostros',
            frame
        )


        # ==================================================
        # SALIR
        # ==========================================================

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        # ESC
        if tecla == 27:

            break


        # Terminamos automáticamente al llegar a 300.

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