import mtcnn
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import numpy as np 
import cv2
import time

detector = MTCNN()
def extract_face(img, required_size=(160, 160)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = asarray(img)
    results = detector.detect_faces(pixels)
    people = len(results)
    if results != []:
        list_face = []
        X1, X2, Y1, Y2 = [], [], [], []
        for result in results:
            box = result['box']
            x1 = box[0]
            y1 = box[1]
            width = box[2]
            height = box[3]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            list_face.append(face_array)
    return list_face, X1, X2, Y1, Y2
img = cv2.imread('1.jpg')
list_face, X1, X2, Y1, Y2 = extract_face(img)
for i in range(len(list_face)):
    cv2.rectangle(img,(X1[i],Y1[i]),(X2[i],Y2[i]),(0,155,255),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,str(i),(int(X1[i]-10),Y1[i]), font, 1,(0,255,0),2,cv2.LINE_AA)

cv2.imshow('img',img)
cv2.waitKey(0)
