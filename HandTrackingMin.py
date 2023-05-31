import cv2
import mediapipe as mp
import time

# This script is the bare bones minimum code in order for hand tracking to work.

# Read in video feed
cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    # Convert from standard BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        # iterate through each hand detected
        for handLmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLmarks.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # id is based on a preassigned point on the hand
                if id==0:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    # Print the fps to the screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)