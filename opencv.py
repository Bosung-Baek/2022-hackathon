import cv2
import numpy as np
from tts import say
import time

# 웹캠 신호 받기
VideoSignal = cv2.VideoCapture(0)
# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

# YOLO NETWORK 재구성
classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

direction = ''
latest_time = time.time()
while True:
    # 웹캠 프레임
    ret, frame = VideoSignal.read()
    h, w, c = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
    True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # objection position : left
                if center_x < w/3:
                    if center_y > 2*h/3:
                        # print("left down")
                        direction = 'left down'
                    elif center_y < h/3:
                        # print("left up")
                        direction = 'left up'
                    else:
                        # print("left")
                        direction = 'left'

                # objection position : center
                elif center_x>w/3 and center_x<2*w/3:
                    if center_y > 2*h/3:
                        # print("center down")
                        direction = 'center down'
                    elif center_y < h/3:
                        # print("center up")
                        direction = 'center up'
                    else:
                        # print("center")
                        direction = 'center'

                # objection position : right
                else:
                    if center_y > 2*h/3:
                        # print("right down")
                        direction = 'right down'
                    elif center_y < h/3:
                        # print("right up")
                        direction = 'right up'
                    else:
                        # print("right")
                        direction = 'right'

                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            #라벨 음성출력
            print(label)
            elapsed_time = time.time() - latest_time
            if elapsed_time >= 3:    # 안내 나온지 3초 이상 지났을 때 다시 출력
                say(label, direction)
                latest_time = time.time()



    # 영상 edge 구분
    def grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    def gaussian_blur(frame, kernel_size):
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    def canny(frame, low_threshold, high_threshold):
        return cv2.Canny(frame, low_threshold, high_threshold)
    kernel_size = 5
    blur_gray = gaussian_blur(grayscale(frame), kernel_size)

    low_threshold = 50
    high_threshold = 200
    edges = canny(blur_gray, low_threshold, high_threshold)
    #영상출력
    cv2.imshow("YOLOv3", edges)

    if cv2.waitKey(100) > 0:
        break