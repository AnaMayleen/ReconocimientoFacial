"""
Interfaz web para el sistema de reconocimiento facial (LBPH + cámara A9 + ESP32).

Ejecutar:   python app_web.py
Abrir:      http://127.0.0.1:5000

Desde la página puedes: ver el video en vivo, registrar personas,
entrenar el modelo y activar el reconocimiento (con envío opcional al ESP32).

Fuente de video: cámara A9 o webcam de la PC (se cambia desde la página).
Opciones de arranque:
    python app_web.py --camara pc            # empezar con la webcam de la PC
    python app_web.py --camara pc --indice 1 # usar la segunda webcam
"""

import argparse
import json
import logging
import os
import queue
import re
import sys
import threading
import time

import cv2
import numpy as np
import serial
from flask import Flask, Response, jsonify, request


# ======================================================
# CONFIGURACIÓN
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenCV en Windows no puede leer/escribir archivos si la ruta absoluta contiene
# caracteres no ASCII (por ejemplo "Año 6" o tildes). Para evitarlo, el programa
# trabaja desde la carpeta del proyecto y usa rutas RELATIVAS en todo lo que
# toca OpenCV (imágenes y modelo).
os.chdir(BASE_DIR)

DATA_PATH = 'Data'
MODEL_PATH = 'modeloLBPHFace.xml'
NAMES_PATH = 'modeloLBPHFace.names.json'

CAMERA_IP = '192.168.169.1'
CAMERA_PORT = 6123
VIDEO_TIMEOUT = 10
RECONNECT_DELAY = 2

ESP32_PORT = os.environ.get('ESP32_PORT', 'COM11')
ESP32_BAUD = 115200

UMBRAL_RECONOCIMIENTO = 70      # menor = más estricto
ANCHO_CAPTURA = 640
ANCHO_RECONOCIMIENTO = 480
INTERVALO_CAPTURA = 0.2         # segundos entre fotos guardadas
IMG_EXT = ('.jpg', '.jpeg', '.png')

# Solo accesible desde esta PC. NO cambiar a 0.0.0.0 sin añadir autenticación:
# esta página puede activar el ESP32 (relé / acceso).
HOST = '127.0.0.1'
PORT = 5000

sys.path.insert(0, os.path.join(BASE_DIR, 'a9-v720', 'src'))

# La librería de la cámara A9 es opcional: sin ella solo se puede usar la webcam.
try:
    from log import log                  # noqa: E402
    log.set_log_lvl(logging.WARNING)

    from netcl_tcp import netcl_tcp      # noqa: E402
    from v720_ap import v720_ap          # noqa: E402
    import cmd_udp                       # noqa: E402

    A9_AVAILABLE = True
except ImportError as error:
    A9_AVAILABLE = False
    print(f'⚠ Librería de la cámara A9 no disponible ({error}).')
    print('  Solo estará disponible la webcam de la PC.')

cv2.setUseOptimized(True)

if not hasattr(cv2, 'face'):
    sys.exit(
        '\nERROR: tu OpenCV no incluye el módulo "face" (reconocedor LBPH).\n'
        'Tienes instalado "opencv-python" en lugar de "opencv-contrib-python".\n'
        'Solución (ejecuta estos comandos y vuelve a intentar):\n\n'
        '    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless\n'
        '    pip install "opencv-contrib-python<5"\n'
    )
logging.getLogger('werkzeug').setLevel(logging.WARNING)


# ======================================================
# ESTADO COMPARTIDO
# ======================================================

lock = threading.RLock()
stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=1)

state = {
    'mode': 'idle',            # idle | capturing | recognizing
    'message': 'Listo.',
    'camera': 'Conectando...',
    'source': 'a9' if A9_AVAILABLE else 'pc',   # a9 | pc
    'pc_index': 0,             # índice de la webcam de la PC
    'esp32': 'Desconectado',
    'training': False,
    'send_esp32': True,
    'authorized': False,       # True tras enviar ROSTRO_OK, hasta recibir LISTO
    'captured': 0,
    'target': 0,
    'person': '',
}

capture = {'dir': '', 'index': 0, 'last': 0.0}

recognizer = None
names = []
esp32 = None

latest_jpeg = None
jpeg_version = 0
jpeg_cond = threading.Condition()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    sys.exit(
        f'\nERROR: no se pudo cargar el detector de rostros (OpenCV {cv2.__version__}).\n'
        'Las versiones 5.x de OpenCV no incluyen los archivos haarcascade.\n'
        'Solución: instala OpenCV 4.x:\n\n'
        '    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless\n'
        '    pip install "opencv-contrib-python<5"\n'
    )


def set_state(**kwargs):
    with lock:
        state.update(kwargs)


# ======================================================
# CÁMARAS (A9 y webcam de la PC)
# ======================================================

def get_source():
    return state['source']


def push_frame(frame):
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass


def a9_camera_worker():
    """Se conecta a la A9 solo mientras la fuente seleccionada sea 'a9'."""
    attempt = 0

    while not stop_event.is_set():
        if get_source() != 'a9':
            attempt = 0
            time.sleep(0.3)
            continue

        attempt += 1
        set_state(camera=f'Conectando A9 (intento {attempt})...')
        conn_stop = threading.Event()

        try:
            with netcl_tcp(CAMERA_IP, CAMERA_PORT) as sock:
                cam = v720_ap(sock)
                cam.init_live_motion()

                for ajuste in (lambda: cam.ir_led(False),
                               lambda: cam.flip(False)):
                    try:
                        ajuste()
                    except Exception:
                        pass

                buf = bytearray()
                est = {'last': time.monotonic()}

                def watchdog():
                    while not conn_stop.wait(1):
                        cambio = get_source() != 'a9'
                        sin_video = (time.monotonic() - est['last']
                                     >= VIDEO_TIMEOUT)

                        if cambio or sin_video:
                            if sin_video and not cambio:
                                print('Cámara sin imágenes; reconectando...')
                            try:
                                sock.close()
                            except Exception:
                                pass
                            return

                threading.Thread(target=watchdog, daemon=True).start()

                def on_rcv(cmd, data):
                    if get_source() != 'a9':
                        return
                    if cmd != cmd_udp.P2P_UDP_CMD_JPEG or not data:
                        return

                    buf.extend(data)

                    while True:
                        start = buf.find(b'\xff\xd8')
                        if start == -1:
                            if len(buf) > 2_000_000:
                                buf.clear()
                            return

                        end = buf.find(b'\xff\xd9', start + 2)
                        if end == -1:
                            if start > 0:
                                del buf[:start]
                            if len(buf) > 2_000_000:
                                buf.clear()
                            return

                        jpg = bytes(buf[start:end + 2])
                        del buf[:end + 2]

                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            continue

                        est['last'] = time.monotonic()
                        push_frame(frame)
                        if state['camera'] != 'Conectada':
                            set_state(camera='Conectada')

                est['last'] = time.monotonic()
                cam.cap_live(on_rcv)

        except Exception as error:
            if get_source() == 'a9':
                print('⚠ Conexión con cámara interrumpida:', error)

        finally:
            conn_stop.set()

        if get_source() == 'a9':
            set_state(camera='Desconectada')
            if not stop_event.is_set():
                time.sleep(RECONNECT_DELAY)


def pc_camera_worker():
    """Lee la webcam de la PC solo mientras la fuente seleccionada sea 'pc'."""
    cap = None
    opened_index = None

    while not stop_event.is_set():
        if get_source() != 'pc':
            if cap is not None:
                cap.release()
                cap, opened_index = None, None
            time.sleep(0.2)
            continue

        index = state['pc_index']

        # (Re)abrir si aún no está abierta o cambió el índice
        if cap is None or opened_index != index:
            if cap is not None:
                cap.release()

            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            opened_index = index

            if not cap.isOpened():
                cap.release()
                cap, opened_index = None, None
                set_state(camera=f'Webcam {index} no disponible')
                time.sleep(RECONNECT_DELAY)
                continue

            set_state(camera=f'Abriendo webcam {index}...')

        ok, frame = cap.read()

        if not ok or frame is None:
            cap.release()
            cap, opened_index = None, None
            set_state(camera=f'Webcam {index} sin imagen')
            time.sleep(RECONNECT_DELAY)
            continue

        if get_source() == 'pc' and state['pc_index'] == index:
            if state['camera'] != 'Conectada':
                set_state(camera='Conectada')
            push_frame(frame)

    if cap is not None:
        cap.release()


# ======================================================
# ESP32
# ======================================================

def connect_esp32():
    global esp32
    try:
        esp32 = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        time.sleep(2)  # el ESP32 se reinicia al abrir el puerto
        set_state(esp32=f'Conectado ({ESP32_PORT})')
        print(f'ESP32 conectado en {ESP32_PORT}.')
    except serial.SerialException as error:
        esp32 = None
        set_state(esp32=f'No disponible ({ESP32_PORT})')
        print(f'⚠ No se pudo abrir {ESP32_PORT}: {error}')

        texto = str(error).lower()
        if 'filenotfound' in texto or 'could not open port' in texto and 'permission' not in texto:
            print('  El puerto no existe: conecta el ESP32 por USB o indica el puerto correcto')
            print('  (Administrador de dispositivos > Puertos COM), por ejemplo:')
            print('      set ESP32_PORT=COM5      (Windows, cmd)')
            print('      $env:ESP32_PORT="COM5"   (Windows, PowerShell)')
        else:
            print('  El puerto está ocupado: cierra el Monitor Serial de Arduino IDE.')
        print('  Continuando sin ESP32 (el reconocimiento funciona, pero no se envía ROSTRO_OK).')


def read_esp32():
    global esp32
    if esp32 is None:
        return
    try:
        while esp32.in_waiting > 0:
            msg = esp32.readline().decode('utf-8', errors='ignore').strip()
            if msg == 'LISTO':
                set_state(authorized=False)
                print('ESP32: listo para otro rostro.')
    except (serial.SerialException, OSError):
        esp32 = None
        set_state(esp32='Error de comunicación')


def send_authorization():
    """Envía ROSTRO_OK una sola vez, hasta que el ESP32 responda LISTO."""
    with lock:
        if state['authorized']:
            return
        state['authorized'] = True
        enviar = state['send_esp32']

    if enviar and esp32 is not None:
        try:
            esp32.write(b'ROSTRO_OK\n')
            print('ROSTRO_OK enviado al ESP32.')
        except (serial.SerialException, OSError):
            set_state(esp32='Error de comunicación')


# ======================================================
# MODELO
# ======================================================

def list_people():
    result = []
    if os.path.isdir(DATA_PATH):
        for name in sorted(os.listdir(DATA_PATH)):
            path = os.path.join(DATA_PATH, name)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path)
                             if f.lower().endswith(IMG_EXT)])
                result.append({'name': name, 'count': count})
    return result


def load_model():
    global recognizer, names

    if not os.path.exists(MODEL_PATH):
        return

    rec = cv2.face.LBPHFaceRecognizer_create()
    try:
        rec.read(MODEL_PATH)
    except cv2.error as error:
        print(f'⚠ No se pudo cargar {MODEL_PATH}: {error}')
        print('  Entrena el modelo de nuevo desde la página.')
        return

    # Si existe el archivo de nombres (generado al entrenar desde la web)
    # se usa; si no, se usa el orden alfabético de Data/ como en el script original.
    if os.path.exists(NAMES_PATH):
        with open(NAMES_PATH, encoding='utf-8') as f:
            lista = json.load(f)
    else:
        lista = [p['name'] for p in list_people()]

    with lock:
        recognizer = rec
        names = lista


def train_model():
    global recognizer, names

    try:
        people = [p['name'] for p in list_people()]
        faces, labels = [], []

        for label, name in enumerate(people):
            folder = os.path.join(DATA_PATH, name)
            for fname in os.listdir(folder):
                if not fname.lower().endswith(IMG_EXT):
                    continue
                img = cv2.imread(os.path.join(folder, fname), 0)
                if img is None:
                    continue
                faces.append(img)
                labels.append(label)

        if not faces:
            set_state(message='No hay imágenes para entrenar.')
            return

        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(faces, np.array(labels))
        rec.write(MODEL_PATH)

        with open(NAMES_PATH, 'w', encoding='utf-8') as f:
            json.dump(people, f, ensure_ascii=False)

        with lock:
            recognizer = rec
            names = people

        set_state(message=f'Modelo entrenado: {len(people)} persona(s), '
                          f'{len(faces)} imágenes.')

    except Exception as error:
        set_state(message=f'Error al entrenar: {error}')

    finally:
        set_state(training=False)


# ======================================================
# PROCESAMIENTO DE VIDEO
# ======================================================

def publish(frame):
    global latest_jpeg, jpeg_version
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    with jpeg_cond:
        latest_jpeg = buf.tobytes()
        jpeg_version += 1
        jpeg_cond.notify_all()


def placeholder(text):
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 185), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def detect_faces(gray):
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )


def process_capture(frame):
    aux = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    with lock:
        done, target = state['captured'], state['target']

    if len(faces) > 0:
        # Solo el rostro más grande (evita guardar a otras personas al fondo)
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.monotonic()
        if now - capture['last'] >= INTERVALO_CAPTURA and done < target:
            rostro = cv2.resize(aux[y:y + h, x:x + w], (150, 150),
                                interpolation=cv2.INTER_CUBIC)
            path = os.path.join(capture['dir'],
                                f"rostro_{capture['index']}.jpg")
            if cv2.imwrite(path, rostro):
                capture['index'] += 1
                capture['last'] = now
                done += 1
                set_state(captured=done)

    cv2.putText(frame, f'Capturadas: {done}/{target}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if done >= target:
        set_state(mode='idle',
                  message=f'Captura completada: {done} imágenes. '
                          f'Ahora entrena el modelo.')
    return frame


def process_recognition(frame):
    with lock:
        rec, lista = recognizer, list(names)

    if rec is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in detect_faces(gray):
        rostro = cv2.resize(gray[y:y + h, x:x + w], (150, 150),
                            interpolation=cv2.INTER_CUBIC)
        id_persona, confianza = rec.predict(rostro)

        if confianza < UMBRAL_RECONOCIMIENTO and 0 <= id_persona < len(lista):
            nombre = lista[id_persona]
            color = (0, 255, 0)
            send_authorization()
        else:
            nombre = 'Desconocido'
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, nombre, (x, max(y - 25, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f'Conf: {confianza:.1f}', (x, max(y - 5, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                    cv2.LINE_AA)
    return frame


def vision_worker():
    last_frame = 0.0
    placeholder_shown = False
    publish(placeholder('Esperando camara...'))

    while not stop_event.is_set():
        read_esp32()

        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            if time.monotonic() - last_frame > 3 and not placeholder_shown:
                publish(placeholder('Sin senal de la camara'))
                placeholder_shown = True
            continue

        last_frame = time.monotonic()
        placeholder_shown = False

        with lock:
            mode = state['mode']

        ancho = ANCHO_CAPTURA if mode == 'capturing' else ANCHO_RECONOCIMIENTO
        alto = int(frame.shape[0] * ancho / frame.shape[1])
        frame = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)

        if mode == 'capturing':
            frame = process_capture(frame)
        elif mode == 'recognizing':
            frame = process_recognition(frame)

        publish(frame)


# ======================================================
# SERVIDOR WEB
# ======================================================

app = Flask(__name__)

NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9 _-]{0,39}')


@app.route('/')
def index():
    return Response(INDEX_HTML, mimetype='text/html')


@app.route('/video_feed')
def video_feed():
    def stream():
        last = -1
        while not stop_event.is_set():
            with jpeg_cond:
                jpeg_cond.wait_for(lambda: jpeg_version != last, timeout=2)
                if jpeg_version == last:
                    continue
                last, jpg = jpeg_version, latest_jpeg
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + jpg + b'\r\n')

    return Response(stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with lock:
        data = dict(state)
        data['model_ready'] = recognizer is not None
        data['a9_available'] = A9_AVAILABLE
    data['people'] = list_people()
    return jsonify(data)


def reply(ok, message, code=200):
    set_state(message=message)
    return jsonify({'ok': ok, 'message': message}), (code if ok else 400)


@app.route('/api/source', methods=['POST'])
def api_source():
    body = request.get_json(silent=True) or {}
    source = body.get('source')

    try:
        index = int(body.get('index', 0))
    except (TypeError, ValueError):
        return reply(False, 'El índice de la webcam debe ser un número.')

    if source not in ('a9', 'pc'):
        return reply(False, 'Fuente de video inválida.')
    if source == 'a9' and not A9_AVAILABLE:
        return reply(False, 'La librería de la cámara A9 no está disponible.')
    if not 0 <= index <= 9:
        return reply(False, 'El índice de la webcam debe estar entre 0 y 9.')

    with lock:
        if state['mode'] != 'idle' or state['training']:
            return reply(False, 'Detén la operación actual antes de '
                                'cambiar de cámara.')
        nombre = 'Cámara A9' if source == 'a9' else f'Webcam de la PC ({index})'
        state.update(source=source, pc_index=index,
                     camera='Cambiando de cámara...',
                     message=f'Fuente: {nombre}. Usa la misma cámara para '
                             f'registrar y reconocer.')

    # Descartar frames de la fuente anterior
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass

    return jsonify({'ok': True})


@app.route('/api/capture', methods=['POST'])
def api_capture():
    body = request.get_json(silent=True) or {}
    name = str(body.get('name', '')).strip()

    try:
        count = int(body.get('count', 100))
    except (TypeError, ValueError):
        return reply(False, 'La cantidad debe ser un número.')

    if not NAME_RE.fullmatch(name):
        return reply(False, 'Nombre inválido: usa solo letras (sin tildes), '
                            'números, espacios, "_" o "-" (máx. 40).')
    if not 1 <= count <= 1000:
        return reply(False, 'La cantidad debe estar entre 1 y 1000.')

    with lock:
        if state['mode'] != 'idle' or state['training']:
            return reply(False, 'Hay otra operación en curso.')

        folder = os.path.join(DATA_PATH, name)
        os.makedirs(folder, exist_ok=True)

        indices = []
        for f in os.listdir(folder):
            base = os.path.splitext(f)[0]
            if base.startswith('rostro_') and base.split('_')[-1].isdigit():
                indices.append(int(base.split('_')[-1]))

        capture.update(dir=folder,
                       index=max(indices) + 1 if indices else 0,
                       last=0.0)
        state.update(mode='capturing', captured=0, target=count,
                     person=name,
                     message=f'Capturando rostros de {name}...')

    return jsonify({'ok': True})


@app.route('/api/train', methods=['POST'])
def api_train():
    with lock:
        if state['mode'] != 'idle' or state['training']:
            return reply(False, 'Hay otra operación en curso.')
        state.update(training=True, message='Entrenando modelo...')

    threading.Thread(target=train_model, daemon=True).start()
    return jsonify({'ok': True})


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    body = request.get_json(silent=True) or {}

    with lock:
        if state['mode'] != 'idle' or state['training']:
            return reply(False, 'Hay otra operación en curso.')
        if recognizer is None:
            return reply(False, 'Primero entrena el modelo.')
        state.update(mode='recognizing',
                     send_esp32=bool(body.get('send_esp32', True)),
                     authorized=False,
                     message='Reconocimiento activo.')

    return jsonify({'ok': True})


@app.route('/api/send_esp32', methods=['POST'])
def api_send_esp32():
    body = request.get_json(silent=True) or {}
    set_state(send_esp32=bool(body.get('enabled', True)))
    return jsonify({'ok': True})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    set_state(mode='idle', message='Detenido.')
    return jsonify({'ok': True})


# ======================================================
# PÁGINA
# ======================================================

INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reconocimiento Facial</title>
<style>
  :root { --bg:#f5f6f8; --card:#fff; --text:#1c1e21; --muted:#667085;
          --line:#e1e4e8; --accent:#2563eb; --ok:#16a34a; --bad:#dc2626; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --card:#1e2127; --text:#e8eaed; --muted:#9aa3af;
            --line:#2f343c; --accent:#60a5fa; --ok:#4ade80; --bad:#f87171; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,sans-serif; background:var(--bg);
         color:var(--text); }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           background:var(--card); display:flex; flex-wrap:wrap; gap:8px 20px;
           align-items:center; }
  header h1 { font-size:18px; margin:0 auto 0 0; }
  .pill { font-size:13px; color:var(--muted); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         background:var(--bad); margin-right:5px; }
  .dot.on { background:var(--ok); }
  main { display:grid; grid-template-columns:minmax(0,2fr) minmax(280px,1fr);
         gap:16px; padding:16px; max-width:1200px; margin:auto; }
  @media (max-width:800px) { main { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:14px; }
  .card h2 { font-size:15px; margin:0 0 10px; }
  .video { width:100%; border-radius:8px; background:#000; display:block;
           aspect-ratio:4/3; object-fit:contain; }
  #msg { margin-top:10px; font-size:14px; color:var(--muted); min-height:20px; }
  .side { display:flex; flex-direction:column; gap:16px; }
  label { display:block; font-size:13px; color:var(--muted); margin:8px 0 3px; }
  input[type=text], input[type=number] { width:100%; padding:8px;
          border:1px solid var(--line); border-radius:6px; font-size:14px;
          background:var(--bg); color:var(--text); }
  button { padding:9px 14px; border:0; border-radius:6px; font-size:14px;
           cursor:pointer; background:var(--accent); color:#fff;
           margin-top:10px; }
  button.secondary { background:transparent; color:var(--text);
           border:1px solid var(--line); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
  .check { display:flex; align-items:center; gap:8px; font-size:14px;
           margin-top:10px; color:var(--text); }
  ul { list-style:none; margin:0; padding:0; font-size:14px; }
  li { display:flex; justify-content:space-between; padding:5px 0;
       border-bottom:1px solid var(--line); }
  li:last-child { border:0; }
  progress { width:100%; margin-top:10px; }
  select, #pcIndex { padding:4px 6px; border:1px solid var(--line);
         border-radius:6px; background:var(--bg); color:var(--text);
         font-size:13px; width:auto; }
  #pcIndex { width:54px; }
</style>
</head>
<body>
<header>
  <h1>Reconocimiento Facial</h1>
  <span class="pill"><span class="dot" id="dotCam"></span>Cámara: <span id="cam">-</span></span>
  <span class="pill">Fuente:
    <select id="source">
      <option value="a9">Cámara A9</option>
      <option value="pc">Webcam de la PC</option>
    </select>
    <input type="number" id="pcIndex" value="0" min="0" max="9"
           title="Índice de la webcam (0 = la primera)" hidden>
  </span>
  <span class="pill"><span class="dot" id="dotEsp"></span>ESP32: <span id="esp">-</span></span>
</header>

<main>
  <section class="card">
    <img class="video" src="/video_feed" alt="Video en vivo">
    <div id="msg"></div>
  </section>

  <div class="side">
    <section class="card">
      <h2>1. Registrar persona</h2>
      <label for="name">Nombre (sin tildes)</label>
      <input type="text" id="name" maxlength="40" placeholder="Ej: Gaby">
      <label for="count">Cantidad de fotos</label>
      <input type="number" id="count" value="100" min="1" max="1000">
      <progress id="prog" value="0" max="1" hidden></progress>
      <button id="btnCapture">Iniciar captura</button>
    </section>

    <section class="card">
      <h2>2. Entrenar modelo</h2>
      <div class="pill">Repite este paso cada vez que agregues personas o fotos.</div>
      <button id="btnTrain">Entrenar</button>
    </section>

    <section class="card">
      <h2>3. Reconocimiento</h2>
      <label class="check"><input type="checkbox" id="chkEsp" checked>
        Enviar ROSTRO_OK al ESP32</label>
      <div class="row">
        <button id="btnRec">Activar</button>
        <button id="btnStop" class="secondary">Detener</button>
      </div>
    </section>

    <section class="card">
      <h2>Personas registradas</h2>
      <ul id="people"><li>Sin personas</li></ul>
    </section>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);

async function post(url, body) {
  const r = await fetch(url, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body || {})});
  const d = await r.json();
  if (!d.ok) $('msg').textContent = d.message;
  refresh();
}

$('btnCapture').onclick = () =>
  post('/api/capture', {name: $('name').value, count: $('count').value});
$('btnTrain').onclick = () => post('/api/train');
$('btnRec').onclick = () => post('/api/recognize', {send_esp32: $('chkEsp').checked});
$('btnStop').onclick = () => post('/api/stop');
const changeSource = () =>
  post('/api/source', {source: $('source').value, index: $('pcIndex').value});
$('source').onchange = changeSource;
$('pcIndex').onchange = changeSource;
$('chkEsp').onchange = () => post('/api/send_esp32', {enabled: $('chkEsp').checked});

function setDot(id, on) { $(id).classList.toggle('on', on); }

async function refresh() {
  try {
    const s = await (await fetch('/api/status')).json();
    const busy = s.mode !== 'idle' || s.training;

    $('cam').textContent = s.camera;
    $('esp').textContent = s.esp32;
    setDot('dotCam', s.camera === 'Conectada');
    setDot('dotEsp', s.esp32.startsWith('Conectado'));
    $('msg').textContent = s.message;

    const optA9 = $('source').querySelector('option[value=a9]');
    optA9.disabled = !s.a9_available;
    $('source').disabled = busy;
    $('pcIndex').disabled = busy;
    $('pcIndex').hidden = s.source !== 'pc';
    if (document.activeElement !== $('source')) $('source').value = s.source;
    if (document.activeElement !== $('pcIndex')) $('pcIndex').value = s.pc_index;

    $('btnCapture').disabled = busy;
    $('btnTrain').disabled = busy;
    $('btnRec').disabled = busy || !s.model_ready;
    $('btnStop').disabled = s.mode === 'idle';

    const prog = $('prog');
    prog.hidden = s.mode !== 'capturing';
    prog.max = s.target || 1;
    prog.value = s.captured;

    if (document.activeElement !== $('chkEsp')) $('chkEsp').checked = s.send_esp32;

    $('people').innerHTML = s.people.length
      ? s.people.map(p => `<li><span>${p.name}</span><span>${p.count} fotos</span></li>`).join('')
      : '<li>Sin personas</li>';
  } catch (e) {
    $('msg').textContent = 'Sin conexión con el servidor.';
  }
}

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


# ======================================================
# INICIO
# ======================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interfaz web de reconocimiento facial')
    parser.add_argument('--camara', choices=['a9', 'pc'],
                        help='fuente de video inicial (por defecto: a9)')
    parser.add_argument('--indice', type=int, default=None,
                        help='índice de la webcam de la PC (por defecto: 0)')
    args = parser.parse_args()

    if args.camara:
        set_state(source=args.camara)
    if args.indice is not None:
        set_state(pc_index=args.indice)
    if state['source'] == 'a9' and not A9_AVAILABLE:
        set_state(source='pc')

    os.makedirs(DATA_PATH, exist_ok=True)

    connect_esp32()
    load_model()

    if A9_AVAILABLE:
        threading.Thread(target=a9_camera_worker, daemon=True).start()
    threading.Thread(target=pc_camera_worker, daemon=True).start()
    threading.Thread(target=vision_worker, daemon=True).start()

    print(f'\nAbre http://{HOST}:{PORT} en tu navegador.\n')

    try:
        app.run(host=HOST, port=PORT, threaded=True)
    finally:
        stop_event.set()
        if esp32 is not None and esp32.is_open:
            esp32.close()