# import necessary functions and classes from hand-tracking and face-tracking files
from handTrackingBasecode import HandDetector
from faceTrackingBasecode import FaceDetector
import cv2
import time

# create instances of hand and face detectors
hand_detector = HandDetector()
face_detector = FaceDetector()

# main function to run the hand and face detectors
def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        # detect hands and faces in the frame
        frame = hand_detector.find_hands(frame)
        frame = face_detector.find_faces(frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
