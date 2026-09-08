import cv2
import os
import sys
import threading
import queue
import runpy
import imutils

# --- Configuración ---
personName = 'Raul'
dataPath = 'Data'  # Cambia a tu ruta
personPath = dataPath + '/' + personName
REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a9-v720', 'src')

if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)
    
sys.path.insert(0, REPO_SRC)

# --- Puente entre el script de la cámara y nuestro pipeline ---
frame_queue = queue.Queue(maxsize=1)
_original_imshow = cv2.imshow  # guardamos la función real para usarla nosotros

def _capture_imshow(winname, frame):
    # Se ejecuta cada vez que a9_naxclow.py muestra un frame en vivo
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # prueba esto
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put(frame.copy())

cv2.imshow = _capture_imshow  # interceptamos, no dejamos que abra su propia ventana

def run_camera_script():
    sys.argv = ['a9_naxclow.py', '-l']
    runpy.run_path(os.path.join(REPO_SRC, 'a9_naxclow.py'), run_name='__main__')

cam_thread = threading.Thread(target=run_camera_script, daemon=True)
cam_thread.start()

# --- Tu lógica original, intacta ---
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    try:
        frame = frame_queue.get(timeout=5)
    except queue.Empty:
        print('No llegaron frames en 5s. Revisa que estés conectado a la red Nax_ de la cámara.')
        continue

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1

    _original_imshow('frame', frame)  # esta sí abre la ventana de verdad

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cv2.destroyAllWindows()