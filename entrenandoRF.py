import cv2
import os
import numpy as np

dataPath = 'Data'

# Personas registradas
peopleList = sorted([
    nombre
    for nombre in os.listdir(dataPath)
    if os.path.isdir(os.path.join(dataPath, nombre))
])

print('Lista de personas:', peopleList)

labels = []
facesData = []
label = 0

# Leer imágenes de cada persona
for nameDir in peopleList:

    personPath = os.path.join(
        dataPath,
        nameDir
    )

    print('Leyendo imágenes de:', nameDir)

    for fileName in os.listdir(personPath):

        print('Rostro:', nameDir + '/' + fileName)

        labels.append(label)

        facesData.append(
            cv2.imread(
                os.path.join(personPath, fileName),
                0
            )
        )

    label += 1

# Crear reconocedor LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenar
print("Entrenando...")

face_recognizer.train(
    facesData,
    np.array(labels)
)

# Guardar modelo
face_recognizer.write(
    'modeloLBPHFace.xml'
)

print("Modelo almacenado.")