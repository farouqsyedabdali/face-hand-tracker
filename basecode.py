import cv2
import mediapipe as mp
import time

vidcap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = vidcap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

