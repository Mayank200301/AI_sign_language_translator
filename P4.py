import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import deque, Counter, defaultdict

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
confidence_threshold = 0.75

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Smoothing buffers
smooth_label_buffer = deque(maxlen=5)
confidence_buffer = defaultdict(lambda: deque(maxlen=5))

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 2:
            # Combine bounding boxes of both hands
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            x_min = max(0, min(x1, x2) - offset)
            y_min = max(0, min(y1, y2) - offset)
            x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + offset)
            y_max = min(img.shape[0], max(y1 + h1, y2 + h2) + offset)
            imgCrop = img[y_min:y_max, x_min:x_max]
        else:
            # Single hand
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x_min = max(0, x - offset)
            y_min = max(0, y - offset)
            x_max = min(img.shape[1], x + w + offset)
            y_max = min(img.shape[0], y + h + offset)
            imgCrop = img[y_min:y_max, x_min:x_max]

        # Preprocess for prediction
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        h, w = imgCrop.shape[:2]
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = prediction[index]
        raw_label = labels[index]

        # Update buffers
        smooth_label_buffer.append(raw_label)
        confidence_buffer[raw_label].append(confidence)

        # Apply smoothing
        if len(smooth_label_buffer) == smooth_label_buffer.maxlen:
            most_common_label, _ = Counter(smooth_label_buffer).most_common(1)[0]
            avg_conf = sum(confidence_buffer[most_common_label]) / len(confidence_buffer[most_common_label])
            if avg_conf > confidence_threshold:
                label = most_common_label
            else:
                label = "Uncertain"
        else:
            label = "Uncertain"

        # Visualization feedback
        if label == "Uncertain":
            box_color = (0, 255, 255)  # Yellow
            text_color = (0, 0, 0)
        else:
            box_color = (255, 0, 255)  # Purple
            text_color = (255, 255, 255)

        # Draw label box
        cv2.rectangle(imgOutput, (x_min, y_min - 50), (x_min + 180, y_min), box_color, cv2.FILLED)
        cv2.putText(imgOutput, label, (x_min + 10, y_min - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)

        # Draw bounding box
        cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), box_color, 4)

        # Show processed inputs
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
