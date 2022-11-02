import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
                mode=False,
                max_hands=2,
                complexity=1,
                detecConfidence=0.5,
                trackConfidence=0.5):

        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detecConfidence = detecConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.complexity, 
                                        self.detecConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, 
                                                self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNum=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]


            for id, lm in enumerate(myHand.landmark):

                hei, wid, ch = img.shape
                cx, cy = int(lm.x*wid), int(lm.y*hei)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        return lmList

    


def main(): # Main program

    # FPS calculating variables
    prevTime = 0
    curTime = 0

    vidcap = cv2.VideoCapture(0) # Video Capture

    detector = handDetector() # Creates an object of the class above

    while True:
        success, img = vidcap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        # FPS in the corner of the screen
        curTime = time.time()
        fps = 1/(curTime-prevTime)
        prevTime = curTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()