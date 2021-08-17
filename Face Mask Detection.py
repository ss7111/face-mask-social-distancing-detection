'''#importing requierd modules'''
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

'''#Serach for pre-trained module (haarcascaed_frontalface_akt2.xml) used for face detection
#cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml'''
cascPath = "D:\\Project\\Enviroments\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
'''#importing pre_trained mask detection module'''
model = load_model("mask_recog1.h5")
print("module loaded")
'''#start capturing video'''
video_capture = cv2.VideoCapture('mask_test2.mp4')
print("video stream started")
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #convert color image to gray image
    cv2.imshow('1 i/p image', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #DETECT FACES (get face coordinates)
    cv2.imshow('2 gray scale', gray)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    #cv2.imshow('3 faces in image', faces)
    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
        #
        face_frame = frame[y:y + h, x:x + w]
        #convert captures frame BGR to RGB
        cv2.imshow('3 face_frame', face_frame)
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('4 RGB face frame', face_frame)
        #resize frame
        face_frame = cv2.resize(face_frame, (224, 224))
        cv2.imshow('5 resized frame to 224 244', face_frame)
        #convert image into array
        face_frame = img_to_array(face_frame)


        #expand array and insert new axis
        face_frame = np.expand_dims(face_frame, axis=0)

        #PREPROCESS IMAGE AS PER REQUIRED FOR MODULE
        face_frame = preprocess_input(face_frame)

        cv2.imshow('Video', frame)
        #APPEND IN FACE_LIST
        faces_list.append(face_frame)
        #check
        if len(faces_list) > 0:
            #prediction for mask
            # model.predict return list of vaules
            preds = model.predict(faces_list)
        for pred in preds:
            #store value in mask and withoutMsk variables
            (mask, withoutMask) = pred
        #select lable according to prediction
        label = "Mask" if mask > withoutMask else "No Mask"
        #select color RED if mask not present or else selsct green
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        #lable to frame
        per=max(mask,withoutMask)*100
        label = "{}: {:.2f}%".format(label, per)
        #label2 = "full face cover" if mask > withoutMask else "Full face not cover"
        label2 = "full face cover" if (color==(0,255,0) and per > 90) else "Full face not cover"
        #formating for rectangle and lable
        cv2.putText(frame, label2, (x, y - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()