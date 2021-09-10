import numpy as np
import cv2
import imutils
import easyocr
import sys
import pytesseract
import pandas as pd
import time


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Lê e redimensiona a imagem para o temanho desejado
image = cv2.imread('imagem2.jpg')
image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)

# Converte para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversion", gray)

# blur to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", gray)

# Aplica um filtro Gaussiano para reduzir reuído
gray = cv2.GaussianBlur(gray, (3 , 3), 0)
cv2.imshow("Gaussian Filter", gray)

# Realiza a detecção de bordas
v = np.median(gray)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(gray, lower, upper)
cv2.imshow("Canny Edges", edged)

cv2.waitKey(0)

# Encontra os contornos na imagem das bordas
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

NumberPlateCnt = None 
count = 0
# loop over contours
for c in cnts:
	# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the approximated contour has four points, then assume that screen is found
    if cv2.isContourConvex(approx):
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

# mask the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final Image",cv2.WINDOW_NORMAL)
cv2.imshow("Final Image",new_image)
cv2.waitKey(0)

# configuration for tesseract
config = ('-l por+deu --oem 1 --psm 10')
text = pytesseract.image_to_string(new_image, config=config)
texto = "".join(caractere for caractere in text if caractere.isalnum())


# print recognized text
print(texto)

