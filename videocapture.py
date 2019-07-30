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
import imutils
from keras.models import model_from_json
from keras.preprocessing import image
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
## Model Age
faceProto = "Age_model/opencv_face_detector.pbtxt"
faceModel = "Age_model/opencv_face_detector_uint8.pb"

ageProto = "Age_model/age_deploy.prototxt"
ageModel = "Age_model/age_net.caffemodel"

genderProto = "Age_model/gender_deploy.prototxt"
genderModel = "Age_model/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

#face expression recognizer initialization
model_emotion = model_from_json(open("emotion_model/facial_expression_model_structure.json", "r").read())
model_emotion.load_weights('emotion_model/facial_expression_model_weights.h5') #load weights
###--------------------------------------------------------------------------####

detector = MTCNN()
def extract_face(img, required_size=(160, 160)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = asarray(img)
    results = detector.detect_faces(pixels)
    people = len(results)
    list_face = []
    X1, X2, Y1, Y2 = [], [], [], []
    if people == 0:
        return None,None,None,None,None
    else:
        if results != []:
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
    if list_face == None:
        cv2.putText(frame,"No people in the frame",(10,30), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    else:
        for i in range(len(list_face)):
            cv2.rectangle(frame,(X1[i],Y1[i]),(X2[i],Y2[i]),(0,155,255),2)
            face = frame[Y1[i]:Y2[i],X1[i]:X2[i]]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{}".format(age)
            x_put_age = int(X1[i])
            y_put_age = int(Y1[i]-20)
            cv2.putText(frame, label, (x_put_age, y_put_age), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            ###------------------------------------------------------------------------------------------------####
            detected_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
            
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            
            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
            
            predictions = model_emotion.predict(img_pixels) #store probabilities of 7 expressions
            
            #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])
            
            emotion = emotions[max_index]
            x_put_emotion = int(X1[i])
            y_put_emotion = int(Y1[i]+10)
            #write emotion text above rectangle
            cv2.putText(frame, emotion, (x_put_emotion,y_put_emotion), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
		

        for i in range(len(list_face)):
            embedding = get_embedding(model_facenet, list_face[i])
            ###------------------------------------------------------------------------------------------------####
            samples = expand_dims(embedding, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = out_encoder.inverse_transform(yhat_class)
            font = cv2.FONT_HERSHEY_SIMPLEX
            x_put = int(X1[i])
            y_put = int(Y1[i]-5)
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
                
    frame = imutils.resize(frame,width=1024)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()