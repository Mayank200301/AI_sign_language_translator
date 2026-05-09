import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Webcam setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300

label = "Y"  # Change this per gesture
folder = f"Data/{label}"
counter = 0
save_interval = 1  # Save every 10 frames
frame_count = 0

# Ensure save folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        x_list = []
        y_list = []

        for hand in hands:
            x, y, w, h = hand['bbox']
            x_list.extend([x, x + w])
            y_list.extend([y, y + h])
        
        x_min = max(min(x_list) - offset, 0)
        x_max = min(max(x_list) + offset, img.shape[1])
        y_min = max(min(y_list) - offset, 0)
        y_max = min(max(y_list) + offset, img.shape[0])

        imgCrop = img[y_min:y_max, x_min:x_max]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        crop_h, crop_w = imgCrop.shape[:2]
        aspectRatio = crop_h / crop_w

        if aspectRatio > 1:
            k = imgSize / crop_h
            wCal = math.ceil(k * crop_w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / crop_w
            hCal = math.ceil(k * crop_h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Show processed output
        cv2.imshow("CombinedCrop", imgCrop)
        cv2.imshow("CombinedWhite", imgWhite)

        # Auto-save logic
        frame_count += 1
        if frame_count % save_interval == 0:
            timestamp = time.time()
            filename = f'{folder}/Image_{timestamp}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"[Saved] {filename}")
            counter += 1

        # Display counter on screen
        cv2.putText(img, f"Saved: {counter}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Webcam", img)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
time.sleep(0.01)
cv2.destroyAllWindows()
