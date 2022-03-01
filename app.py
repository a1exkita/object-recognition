from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(1)  # use 0 for web camera
camera.set(3, 1280)
camera.set(4, 720)
camera.set(10, 70)

classNames = []
classFile = 'coco.names'

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, img = camera.read()  # read the camera frame

        classIds, confidences, bbox = net.detect(img, confThreshold=0.5)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds, confidences, bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img,
                            classNames[classId-1].upper(),
                            (box[0]+10, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img,
                            str(round(confidence, 4)),
                            (box[0]+10, box[1]+60),
                            cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0), 2)
        img = cv2.flip(img, 1)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
