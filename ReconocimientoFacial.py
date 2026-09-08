import cv2
import os
import sys
import threading
import queue
import runpy
import imutils

# --- Configuración (misma ruta que en captura) ---
dataPath = 'C:\\Users\\ashle\\OneDrive\\Documentos\\EscaneoFacial'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a9-v720', 'src')
sys.path.insert(0, REPO_SRC)

# --- Puente entre el script de la cámara y nuestro pipeline (igual que en captura) ---
frame_queue = queue.Queue(maxsize=1)
_original_imshow = cv2.imshow

def _capture_imshow(winname, frame):
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    frame_queue.put(frame_bgr.copy())

cv2.imshow = _capture_imshow

def run_camera_script():
    sys.argv = ['a9_naxclow.py', '-l']
    runpy.run_path(os.path.join(REPO_SRC, 'a9_naxclow.py'), run_name='__main__')

cam_thread = threading.Thread(target=run_camera_script, daemon=True)
cam_thread.start()

# --- Reconocedor ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    try:
        frame = frame_queue.get(timeout=5)
    except queue.Empty:
        print('No llegaron frames en 5s. Revisa que estés conectado a la red Nax_ de la cámara.')
        continue

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    _original_imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()