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
###--------------------------------------------------------------------------####
# load the facenet model
model_facenet = load_model('model/facenet_keras.h5')
print('Loaded Model')
#Load data
dataset = load('dataset.npz')
testX_faces = dataset['arr_2']
embeddings = load('embeddings.npz')
trainX, trainy, testX, testy = embeddings['arr_0'], embeddings['arr_1'], embeddings['arr_2'], embeddings['arr_3']
#Model
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
###--------------------------------------------------------------------------####

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

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

def euclidean(a,b):
	return np.sqrt(np.sum((a-b)**2))

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    list_face, X1, X2, Y1, Y2 = extract_face(frame)
    for i in range(len(list_face)):
        cv2.rectangle(frame,(X1[i],Y1[i]),(X2[i],Y2[i]),(0,155,255),2)
    for i in range(len(list_face)):
        embedding = get_embedding(model_facenet, list_face[i])
        ###------------------------------------------------------------------------------------------------####
        samples = expand_dims(embedding, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_put = int(X1[i]-10)
        y_put = int(Y1[i])
        #cv2.putText(frame,"unknown",(int(X1[i]-10),Y1[i]), font, 0.5,(0,255,0),1,cv2.LINE_AA)
        ###------------------------------------------------------------------------------------------------####
        distance = []
        for i in range(testX.shape[0]):
            d = euclidean(embedding,testX[i])
            distance.append(d)

        min_d = np.min(distance)
        
        if min_d < 5:
            cv2.putText(frame,str(predict_names[0]),(x_put,y_put), font, 0.5,(0,255,0),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,"unknown",(x_put,y_put), font, 0.5,(0,255,0),1,cv2.LINE_AA)
        

        ###------------------------------------------------------------------------------------------------####


        # # test model on a random example from the test dataset
        # selection = np.argmin(distance)

        # random_face_pixels = testX_faces[selection]
        # random_face_emb = testX[selection]
        # random_face_class = testy[selection]
        # random_face_name = out_encoder.inverse_transform([random_face_class])
        # # prediction for the face
        # samples = expand_dims(random_face_emb, axis=0)
        # print(samples.shape)
        # yhat_class = model.predict(samples)
        # yhat_prob = model.predict_proba(samples)
        # # get name
        # class_index = yhat_class[0]
        # class_probability = yhat_prob[0,class_index] * 100
        # predict_names = out_encoder.inverse_transform(yhat_class)
        # print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        # print('Expected: %s' % random_face_name[0])

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()