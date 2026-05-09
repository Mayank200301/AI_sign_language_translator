import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

# Load labels dynamically
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Real-time smoothing settings
last_prediction = []
smoothed_label = "None"

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']

            x_min = max(0, min(x1, x2) - offset)
            y_min = max(0, min(y1, y2) - offset)
            x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + offset)
            y_max = min(img.shape[0], max(y1 + h1, y2 + h2) + offset)

            imgCrop = img[y_min:y_max, x_min:x_max]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            h, w = imgCrop.shape[:2]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index] if prediction[index] > 0.8 else "Uncertain"

            last_prediction.append(label)
            if len(last_prediction) > 5:
                last_prediction.pop(0)
            smoothed_label = max(set(last_prediction), key=last_prediction.count)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        else:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index] if prediction[index] > 0.8 else "Uncertain"

            last_prediction.append(label)
            if len(last_prediction) > 5:
                last_prediction.pop(0)
            smoothed_label = max(set(last_prediction), key=last_prediction.count)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # === Centered Smoothed Label Box ===
    box_w, box_h = 250, 100
    x_box, y_box = int((imgOutput.shape[1] - box_w) / 2), 30

    # Shadow
    cv2.rectangle(imgOutput, (x_box + 4, y_box + 4), (x_box + box_w + 4, y_box + box_h + 4), (50, 50, 50), cv2.FILLED)
    # Box
    cv2.rectangle(imgOutput, (x_box, y_box), (x_box + box_w, y_box + box_h), (0, 0, 0), cv2.FILLED)

    # Text
    font_scale = 2.5 if len(smoothed_label) == 1 else 1.5
    text_size = cv2.getTextSize(smoothed_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
    text_x = x_box + (box_w - text_size[0]) // 2
    text_y = y_box + (box_h + text_size[1]) // 2 - 5
    cv2.putText(imgOutput, smoothed_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 5, cv2.LINE_AA)

    # Show final output
    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
