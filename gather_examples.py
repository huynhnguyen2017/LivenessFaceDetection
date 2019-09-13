import numpy as np
import cv2
import os
import time 

# load our serialized face detector from disk
print("[info] loading face detector")
protoPath = "face_detector/deploy.prototxt"
modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

vs = cv2.VideoCapture(0)
read = 0
saved = 0
# loop over frames from the video file stream
while True:
  time.sleep(1)
  # grap frame form the file
  (grabbed, frame) = vs.read()
  #if the frame is not grabbed, we can have reached the end of the stream
  if not grabbed:
    break
  read += 1
  
  #check to see if we should process this frame
  #if read % 1 != 0:
  #  continue  
  #grab the frame dimensions and construct a blob from the frame
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300,300), (104.0, 177.0, 123.0))

  #pass the blob through the network and obtain the detections and prediction
  net.setInput(blob)
  detections = net.forward()

  #ensure at least one face was found
  if len(detections) > 0:
    # we're making the assumption that each image has only ONE face 
    # so find the bounding box with the largest probability
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    # ensure that the detection with the largest probability also 
    # means our minimum probability test (thus helping filter out 
    # weak detections)
    if confidence > 0.5:
      # compute the (x, y)-coordinates of the bounding box for 
      # the face and extract the face ROI
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      face = frame[startY:endY, startX:endX]

      # write the frame to disk 
      p = "dataset/fake/img_{}.png".format(saved)
      cv2.imwrite(p, face)
      saved += 1
      print("[INFO] {}".format(p))
vs.release()
cv2.destroyAllWindow()
