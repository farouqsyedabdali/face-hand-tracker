import cv2
import mediapipe as mp
import time

vidcap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
curTime = 0


while True:
    success, img = vidcap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                #print(id, lm)

                hei, wid, ch = img.shape
                cx, cy = int(lm.x*wid), int(lm.y*hei)
                print(id, cx, cy)

                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 
                3, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

