import numpy as np
import time
import cv2
import os
from gtts import gTTS
from playsound import playsound


labelsPath = "coco.names"
weightsPath ="yolov3.weights"
configPath = "yolov3.cfg"
file_path = "object_detection.mp3"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")




print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image = cv2.imread("images/17119.png")
(H, W) = image.shape[:2]


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()


print("[INFO] YOLO took {:.6f} seconds".format(end - start))


boxes = []
confidences = []
classIDs = []
ID = 0


for output in layerOutputs:

    for detection in output:

        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]


        if confidence > 0.5:

            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")




            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))


            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)


idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
    0.3)


if len(idxs) > 0:
    list1 = []
    for i in idxs.flatten():

        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        centerx = round((2*x + w)/2)
        centery = round((2*y + h)/2)
        if centerX <= W/3:
            W_pos = "left "
        elif centerX <= (W/3 * 2):
            W_pos = "center "
        else:
            W_pos = "right "

        if centerY <= H/3:
            H_pos = "top "
        elif centerY <= (H/3 * 2):
            H_pos = "mid "
        else:
            H_pos = "bottom "
        list1.append("There is a " +LABELS[classIDs[i]]+" at "+H_pos + W_pos)

    description = ', '.join(list1)

    myobj = gTTS(text=description, lang="en")

    myobj.save(file_path)
    
    playsound(file_path)
    
    os.remove(file_path) 
