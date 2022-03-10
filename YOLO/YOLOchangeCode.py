import cv2
import numpy as np
import evaltool

thres = 0.50 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
classNames= []

PREFIX = "camRelated\\YOLO\\"
VERSION = "-320"

WEIGHTS = PREFIX + "yolov3" + VERSION + ".weights"
CFG = PREFIX + "yolov3" + VERSION + ".cfg"
with open(PREFIX+"coco.names", "r") as f:
    classNames = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classNames), 3))
 
net = cv2.dnn_DetectionModel(WEIGHTS,CFG)
net.setInputSize(416,416)
net.setInputScale(0.00392)
net.setInputMean((0, 0, 0))
net.setInputSwapRB(True)
# is equal to blob = cv2.dnn.blobFromImage(img, 1.0/127.5, (320,320), (127.5, 127.5, 127.5), True, crop=False)

f = evaltool.FileStat("yolov3" + VERSION + ".txt")
f.fileClose()
f.fstatWriteSecArgs(3)
while True:
    success,img = cap.read()

    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold = nms_threshold)
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

    for i in indices:
        if classNames[classIds[i]].upper() == "PERSON":
            x,y,w,h = bbox[i]
            color = colors[classIds[i]]
            cv2.rectangle(img, (x,y),(x+w,h+y), color, thickness=2)
            cv2.putText(img,classNames[classIds[i]].upper() + ":" + str(confs[i] * 100)[:4],(x+5,y+20),
            cv2.FONT_ITALIC,0.67,color,2)
            f.setKwargs(score=str(confs[i] * 100)[0:4])
    f.calTime()
    # cv2.putText(img,str(fps.FPS),(15,30), cv2.FONT_ITALIC,1,(255,127,127),4)

    
    cv2.imshow("Output",img)
    cv2.waitKey(1)