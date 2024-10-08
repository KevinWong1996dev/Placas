import cv2
import numpy as np
import pytesseract
from PIL import Image

#----Realizamos videocaptura-----#
cap = cv2.VideoCapture()
cap.open('rtsp://admin:leteragosl2024@192.168.1.134:554/Streaming/channels/2/')

Ctexto = ''

#----Creamos el While True----#
while True:
    #----Lectura de la videocaptura----#
    ret, frame = cap.read()

    if ret == False:
        break

    #----Dibujamos un rectangulo----#
    cv2.rectangle(frame,(870,750),(1070,850),(0,0,0),cv2.FILLED)
    cv2.putText(frame, Ctexto[0:7], (900,810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #----Extraemos el ancho y el alto de los fotogramas----#
    al, an, c = frame.shape

    #----Tomar el centro de la imagen----#
    #----En x:----#
    x1 = int(an / 4) # Tomamos el 1/3 de la imagen
    x2 = int(x1 * 2) # Hasta el inicio del 3/3 de la imagen

    #En y:
    y1 = int(al / 8 ) #Tomamos el 1/3 de la imagen
    y2 = int(y1 * 7) #Hasta el inicio del 3/3 de la imagen

    # Texto
    cv2.rectangle(frame, (x1 + 160, y1 + 500), (1120, 940), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, 'Procesando Placa', (x1 + 180, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Ubicamos el rectangulo en las zonas extraidas
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)

    # Realizamos un recorte a nuestra zona de interés
    recorte = frame[y1:y2, x1:x2]

    #Preprocesamiento de la zona de interes
    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    #Color
    Color = cv2.absdiff(mG, mB)

    #Binarizamos la imagen
    _, umbral = cv2.threshold(Color, 40, 255, cv2.THRESH_BINARY)

    #Extraemos los contornos de la zona seleccionada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Primero los ordenamos del mas grande al mas pequeño
    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    #Dibujamos los contornos extraidos
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500 and area < 5000:
            #Detectamos la placa
            x, y, ancho, alto = cv2.boundingRect(contorno)

            #Extraemos las cordenadas
            xpi = x + x1  #Coordenada de la placa en X inicial
            ypi = y + y1  #Coordenada de la placa en Y inicial

            xpf = x + ancho + x1  #Coordenada de la placa en X final
            ypf = y + ancho + y1  #Coordenada de la placa en Y final

            #Dibujamos el rectangulo
            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            #Extraemos los pixeles
            placa = frame[ypi:ypf, xpi:xpf]

            #Extraemos el ancho y el alto de los fotogramas
            alp, anp, cp = placa.shape

            #Procesamos los pixeles para extraer los valores de las placas
            Mva = np.zeros((alp, anp))

            #Normalizamos las matrices
            mBp = np.matrix(placa[:, :,0])
            mGp = np.matrix(placa[:, :,1])
            mRp = np.matrix(placa[:, :,2])

            #Creamos una mascara
            for col in range(0,alp):
                for fil in range(0,anp):
                    Max = max(mRp[col,fil] ,mGp[col,fil], mBp[col,fil])
                    Mva[col,fil] = 255 - Max

            #Binarizamos la imagen
            _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

            #Convertimos la matriz en imagen
            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")

            #Nos aseguramos de tener un buen tamaño de placa
            if alp >= 36 and anp >=82:

                #Declaramos la direccion pytesseract
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

                #Extraemos el texto
                config = "--psm 1"
                texto = pytesseract.pytesseract.image_to_string(bin, config=config)

                #if para no mostrar basura
                if len(texto) >= 7:
                    #Print(texto[0:7])

                    Ctexto = texto

                    #Mostramos los valores que nos interesan
                    #cv2.putText(frame, Ctexto[0:7], (910, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 255, 0),2)

            break

    #Mostramos el recorte
    cv2.imshow("Vehiculos", frame)

    #Leemos una tecla
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()