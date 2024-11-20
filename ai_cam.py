import cv2
import urllib.request
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time
#preferred version == 3.6.3
# this is working on 3.6.3  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] with CV 2
# this "SHOULD" work in python ==3.8 and higher [layer_name for layer_name in net.getUnconnectedOutLayersNames()]

#from patrickelectric video_udp.py repo

class Video():
    """BlueRov video capture class constructor
    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)soft
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary
        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600) folloing the guidlines of patrickelectric (might not be the best for the ST card
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array
        Args:
            sample (TYPE): Description
        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame
        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available
        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK


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

video = Video()

while True:
    # Read webcam
    if not video.frame_available():
        continue
    imgNp = video.frame()
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