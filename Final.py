# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from social import social_distancing_config as config
from social.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

#Serach for pre-trained module (haarcascaed_frontalface_akt2.xml) used for face detection

cascPath = "C:\\Users\\rudralachake\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml"

#cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#import model for mask detection
model = load_model("mask_recog_ver2.0.h5")

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture('pedestrians.MP4')
writer = None
#capture video for mask detection
video_capture = cv2.VideoCapture(1)

# loop over the frames from the video stream
while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()
	# convert color image to gray image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# DETECT FACES (get face coordinates)
	faces = faceCascade.detectMultiScale(gray,
										 scaleFactor=1.1,
										 minNeighbors=5,
										 minSize=(60, 60),
										 flags=cv2.CASCADE_SCALE_IMAGE)
	faces_list = []
	preds = []
	for (x, y, w, h) in faces:
		#
		face_frame = frame[y:y + h, x:x + w]
		# convert captures frame BGR to RGB
		face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
		# resize frame
		face_frame = cv2.resize(face_frame, (224, 224))
		# convert image into array
		face_frame = img_to_array(face_frame)
		# expand array and insert new axis
		face_frame = np.expand_dims(face_frame, axis=0)
		# PREPROCESS IMAGE AS PER REQUIRED FOR MODULE
		face_frame = preprocess_input(face_frame)
		# APPEND IN FACE_LIST
		faces_list.append(face_frame)
		# check if
		if len(faces_list) > 0:
			# prediction for mask
			# model.predict return list of vaules
			preds = model.predict(faces_list)
		for pred in preds:
			# store value in mask and withoutMsk variables
			(mask, withoutMask) = pred
		# select lable according to prediction
		label = "Mask" if mask > withoutMask else "No Mask"
		# select color RED if mask not present or else selsct green
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# lable to frame
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# formating for rectangle and lable
		cv2.putText(frame, label, (x, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	# Display the resulting frame
	cv2.imshow('Video', frame)

	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
	#print(len(results))
	text1 = "Number of People: {}".format(len(results))
	cv2.putText(frame, text1, (10, frame.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (105, 45, 255), 3)
	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			video_capture.release()
			cv2.destroyAllWindows()
			break
	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output1"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)
video_capture.release()
vs.release()
cv2.destroyAllWindows()