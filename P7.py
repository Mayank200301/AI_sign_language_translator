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

# FPS setup
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 2:
            # === New logic: combine both hands ===
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

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if prediction[index] > 0.8:  # 80% confidence
                label = labels[index]
            else:
                label = "Uncertain"

            # Smoothing the output by averaging predictions
            last_prediction.append(label)
            if len(last_prediction) > 5:
                last_prediction.pop(0)  # Keep the last 5 predictions

            # Calculate the most frequent prediction from the last 5 predictions
            smoothed_label = max(set(last_prediction), key=last_prediction.count)

            # Display the smoothed label only
            cv2.putText(imgOutput, smoothed_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the crop and white image (if you want, for debugging purposes)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        else:
            # === Single hand logic remains unchanged ===
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

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if prediction[index] > 0.8:  # 80% confidence
                label = labels[index]
            else:
                label = "Uncertain"

            # Smoothing the output by averaging predictions
            last_prediction.append(label)
            if len(last_prediction) > 5:
                last_prediction.pop(0)  # Keep the last 5 predictions

            # Calculate the most frequent prediction from the last 5 predictions
            smoothed_label = max(set(last_prediction), key=last_prediction.count)

            # Display the smoothed label only
            cv2.putText(imgOutput, smoothed_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the crop and white image (if you want, for debugging purposes)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the final output with just the smoothed label
    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
