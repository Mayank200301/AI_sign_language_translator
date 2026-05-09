import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Make sure we allow detecting two hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

# Load labels dynamically
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# FPS setup
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    predictions = []

    if hands:
        # Process each detected hand
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Safe cropping with boundaries
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropShape = imgCrop.shape
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
            label = labels[index]
            predictions.append(label)  # Store the prediction of the current hand

            # Drawing
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Show crop images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # If two hands are detected, combine the predictions
    if len(predictions) == 2:
        combined_label = ''.join(sorted(predictions))  # Sort to maintain order (e.g., "AW" instead of "WA")
        cv2.putText(imgOutput, combined_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    elif len(predictions) == 1:
        # Only one hand detected, display that prediction
        cv2.putText(imgOutput, predictions[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show final output
    cv2.imshow("Image", imgOutput)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
