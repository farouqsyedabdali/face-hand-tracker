import cv2
import time
import numpy as np
import faceTrackingBasecode as ftb
import handTrackingBasecode as htb

# Face detection variables
face_mesh = ftb.mp.solutions.face_mesh.FaceMesh()
prev_face_time = 0

# Hand detection variables
detector = htb.HandDetector()
prev_hand_time = 0

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    
    # Face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    curr_time = time.time()
    face_fps = 1 / (curr_time - prev_face_time)
    prev_face_time = curr_time
    face_fps_text = "Face FPS: {:.2f}".format(face_fps)
    cv2.putText(img, face_fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hand detection
    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        for lm in lm_list:
            x, y = lm[1], lm[2]
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    curr_time = time.time()
    hand_fps = 1 / (curr_time - prev_hand_time)
    prev_hand_time = curr_time
    hand_fps_text = "Hand FPS: {:.2f}".format(hand_fps)
    cv2.putText(img, hand_fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("Image", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
