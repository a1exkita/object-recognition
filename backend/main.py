import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
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
    cv2.imshow("Output", img)
    cv2.waitKey(1)


# For image
# img = cv2.imread('us.jpeg')
# cv2.imshow("Output", img)
# cv2.waitKey(0)
# for classId, confidence, box in zip(classIds, confidences, bbox):
#     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
#     cv2.putText(img,
#                 classNames[classId-1].upper(),
#                 (box[0]+10, box[1]+30),
#                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(img,
#                 str(round(confidence, 4)),
#                 (box[0]+10, box[1]+60),
#                 cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0), 2)
# cv2.imshow("Output", img)
# cv2.waitKey(0)
