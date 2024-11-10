import cv2
import urllib.request
import numpy as np
import time
#preferred version == 3.6.3
# this is working on 3.6.3  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] with CV 2
# this "SHOULD" work in python ==3.8 and higher [layer_name for layer_name in net.getUnconnectedOutLayersNames()]

url='http://192.168.1.23/jpg'    # base IP http://192.168.0.39/cam-hi.jpg

# Load the YOLO model
net = cv2.dnn.readNet("./weights/yolov3.weights", "./configuration/yolov3.cfg")
classes = []
with open("./configuration/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load webcam
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
labelv = 0

while True:
    # Read webcam
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Visualising data
    class_ids = []
    confidences = []
    boxes = []
    w = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    if w < 1:
        F = (w*42)/8
        print(F)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            frame = cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 3)
            if label != labelv:
                print(label)
            labelv = label

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    frame = cv2.putText(frame, "FPS: " + str(round(fps, 2)), (40, 670), font, .7, (0, 255, 255), 1)
    frame = cv2.putText(frame, "press [esc] to exit", (40, 690), font, .45, (0, 255, 255), 1)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        print("[button pressed] ///// [esc].")
        print("[feedback] ///// Videocapturing succesfully stopped")
        break

cv2.destroyAllWindows()