import cv2
import numpy as np
import evaltool

PREFIX = "camRelated\\YOLO\\"
VERSION = "-tiny"

WEIGHTS = PREFIX + "yolov3" + VERSION + ".weights"
CFG = PREFIX + "yolov3" + VERSION + ".cfg"

# Load Yolo
net = cv2.dnn.readNet(WEIGHTS, CFG)
classes = []
with open(PREFIX+"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,360)
# cap.set(10,150)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# i - 1 parts was i[0] - 1 in original code
colors = np.random.uniform(0, 255, size=(len(classes), 3))

f = evaltool.filestat("yolov3" + VERSION + ".txt")
f.fileClose()
f.fstatWriteSec(3)

# Loading image
# img = cv2.imread("room_ser.jpg")
while True:
    success,img = cap.read()
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    font = cv2.FONT_HERSHEY_PLAIN

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    f.fps.calTime()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()