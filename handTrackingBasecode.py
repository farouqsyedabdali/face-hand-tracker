import cv2
import mediapipe as mp
import time


class HandDetector:
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

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(handLMS.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

        return img

    def find_position(self, img, handNum=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNum]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
            except IndexError:
                pass

        return lmList
    

def main(): # Main program

    # FPS calculating variables
    prev_time = 0
    cur_time = 0

    vidcap = cv2.VideoCapture(0) # Video Capture

    detector = HandDetector() # Creates an object of the class above

    while True:
        success, img = vidcap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        if len(lm_list) != 0:
            print(lm_list[4])

        # FPS in the corner of the screen
        cur_time = time.time()
        fps = 1/(cur_time-prev_time)
        prev_time = cur_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()