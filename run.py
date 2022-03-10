import cv2
import numpy as np
import evaltool
import os, sys

PREFIX = os.path.dirname(os.path.realpath(__file__))
SUBDIR = ""
VERSION = {
    "yolov3-320": 
    {
        "weights": "yolov3-320.weights",
        "cfg": "yolov3-320.cfg"
    },
    "yolov3-tiny": 
    {
        "weights": "yolov3-tiny.weights",
        "cfg": "yolov3-tiny.cfg"
    },
    "mobilenet": 
    {
        "weights": "frozen_inference_graph.pb",
        "cfg": "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    }
}

if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()

if sys.argv[1] == "h" or sys.argv[1] == "help":
    print("    List of Models     -> yolov3-320     -> yolov3-tiny     -> mobilenet     ")
    sys.exit()



if sys.argv[1] == "yolov3-320":
    CFG = VERSION["yolov3-320"]["cfg"]
    WEIGHTS = VERSION["yolov3-320"]["weights"]
    SUBDIR = "YOLO"
elif sys.argv[1] == "yolov3-tiny":
    CFG = VERSION["yolov3-tiny"]["cfg"]
    WEIGHTS = VERSION["yolov3-tiny"]["weights"]
elif sys.argv[1] == "mobilenet":
    CFG = VERSION["mobilenet"]["cfg"]
    WEIGHTS = VERSION["mobilenet"]["weights"]
    SUBDIR = "MobileNet"
else:
    sys.exit()

MODELNAME = sys.argv[1]
MODELPATH = os.path.join(PREFIX, SUBDIR)

classNames= []
classFile = os.path.join(MODELPATH, 'coco.names')
with open(classFile,'rt') as f:
    classNames = [line.strip() for line in f.readlines()]
configPath = os.path.join(MODELPATH, CFG)
weightsPath = os.path.join(MODELPATH, WEIGHTS)

# print(configPath, weightsPath)
# sys.exit()

thres = 0.50 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)

colors = np.random.uniform(0, 255, size=(len(classNames), 3))
 
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# is equal to blob = cv2.dnn.blobFromImage(img, 1.0/127.5, (320,320), (127.5, 127.5, 127.5), True, crop=False)

f = evaltool.FileStat(MODELNAME + "Eval.txt")
f.fileClose()
f.fstatWriteSecArgs(3)

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
delay = round(1000/fps)
w, h = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output-' + MODELNAME + '.avi', fourcc, fps, (w, h))


while True:
    success,img = cap.read()
    # print(success, img)

    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold = nms_threshold)
    if isinstance(img, np.ndarray):
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        for i in indices:
            if classNames[classIds[i]].upper() == "PERSON":
                f.setKwargs(score=str(confs[i] * 100)[0:4])
            x,y,w,h = bbox[i]
            color = colors[classIds[i]]
            cv2.rectangle(img, (x,y),(x+w,h+y), color, thickness=2)
            cv2.putText(img,classNames[classIds[i]].upper() + ":" + str(confs[i] * 100)[:4],(x+5,y+20),
            cv2.FONT_ITALIC,0.67,color,2)
                
        f.calTime()
        cv2.putText(img,str(f.FPS),(15,30), cv2.FONT_ITALIC,1,(255,127,127),4)

        out.write(img)
        cv2.imshow("Output",img)
        if cv2.waitKey(1) == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
f.closeAll()
sys.exit(0)