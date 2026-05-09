import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import deque
from collections import Counter

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

# Load labels dynamically
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Smooth parameters
smooth_label_buffer = deque(maxlen=10)  # Store last 10 predictions
confidence_buffer = {label: [] for label in labels}  # Store confidence scores for each label
confidence_threshold = 0.8  # 80% confidence threshold for smoothing

# FPS setup
pTime = 0
show_smoothed = True  # Toggle for showing smoothed predictions

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
            confidence = prediction[index]
            raw_label = labels[index]

            # Update buffers
            smooth_label_buffer.append(raw_label)
            confidence_buffer[raw_label].append(confidence)

            # Apply smoothing
            if len(smooth_label_buffer) == smooth_label_buffer.maxlen:
                most_common_label, _ = Counter(smooth_label_buffer).most_common(1)[0]
                avg_conf = sum(confidence_buffer[most_common_label]) / len(confidence_buffer[most_common_label])
                smoothed_label = most_common_label if avg_conf > confidence_threshold else "Uncertain"
            else:
                smoothed_label = "Uncertain"

            # Visualization feedback
            if smoothed_label == "Uncertain":
                box_color = (0, 255, 255)  # Yellow
                text_color = (0, 0, 0)
            else:
                box_color = (255, 0, 255)  # Purple
                text_color = (255, 255, 255)

            # Raw vs Smoothed (side-by-side)
            raw_img = img.copy()
            smoothed_img = img.copy()

            # Display Raw Prediction (top-left)
            cv2.putText(raw_img, f'Raw: {raw_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Display Smoothed Prediction (top-left)
            cv2.putText(smoothed_img, f'Smoothed: {smoothed_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            # Draw raw label (top-left)
            cv2.putText(raw_img, f'Raw: {raw_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Draw smoothed label
            cv2.putText(smoothed_img, f'Smoothed: {smoothed_label}', (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            # Main label above hand
            cv2.rectangle(raw_img, (x_min, y_min - 50), (x_min + 200, y_min), box_color, cv2.FILLED)
            cv2.putText(raw_img, smoothed_label, (x_min + 10, y_min - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)

            # Combine Raw & Smoothed
            combined = np.hstack((raw_img, smoothed_img))
            cv2.imshow("Raw vs Smoothed Prediction", combined)

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
            confidence = prediction[index]
            raw_label = labels[index]

            # Update buffers
            smooth_label_buffer.append(raw_label)
            confidence_buffer[raw_label].append(confidence)

            # Apply smoothing
            if len(smooth_label_buffer) == smooth_label_buffer.maxlen:
                most_common_label, _ = Counter(smooth_label_buffer).most_common(1)[0]
                avg_conf = sum(confidence_buffer[most_common_label]) / len(confidence_buffer[most_common_label])
                smoothed_label = most_common_label if avg_conf > confidence_threshold else "Uncertain"
            else:
                smoothed_label = "Uncertain"

            # Visualization feedback
            if smoothed_label == "Uncertain":
                box_color = (0, 255, 255)  # Yellow
                text_color = (0, 0, 0)
            else:
                box_color = (255, 0, 255)  # Purple
                text_color = (255, 255, 255)

            # Raw vs Smoothed (side-by-side)
            raw_img = img.copy()
            smoothed_img = img.copy()

            # Display Raw Prediction (top-left)
            cv2.putText(raw_img, f'Raw: {raw_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Display Smoothed Prediction (top-left)
            cv2.putText(smoothed_img, f'Smoothed: {smoothed_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            # Combine Raw & Smoothed
            combined = np.hstack((raw_img, smoothed_img))
            cv2.imshow("Raw vs Smoothed Prediction", combined)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", imgOutput)

    # Toggle 'r' for raw/smoothed toggle
    if cv2.waitKey(1) & 0xFF == ord('r'):
        show_smoothed = not show_smoothed

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
