import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# This file is an example of how to easily use the the module "HandTrackingModule".

prevTime = 0
currentTime = 0
cap = cv2.VideoCapture(1)
# Only change is to use the imported module here we imported as 'htm'
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        print(lmList[4])

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)