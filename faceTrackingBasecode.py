import cv2
import mediapipe as mp
import time

# Define a class to handle face detection
class FaceDetector:
    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Set parameters for the face detector
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Load the face mesh detection module from Mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_image_mode,
                                                    max_num_faces=self.max_num_faces,
                                                    min_detection_confidence=self.min_detection_confidence,
                                                    min_tracking_confidence=self.min_tracking_confidence)
        # Load the drawing utilities module from Mediapipe
        self.mp_draw = mp.solutions.drawing_utils

    # Define a method to detect faces in an input image
    def find_faces(self, image):
        # Convert the image from BGR (OpenCV's default color format) to RGB (which is required by Mediapipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image using the face mesh detector
        self.results = self.face_mesh.process(image_rgb)

        # If one or more faces are detected in the image, draw the landmarks on the faces
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1))

        # Return the image with landmarks drawn on any detected faces
        return image



# Define the main function to run the face detection application
def main():
    # Define variables for calculating FPS (frames per second)
    prev_time = 0
    curr_time = 0

    # Open the default video capture device (usually the computer's built-in camera)
    cap = cv2.VideoCapture(0)

    # Create a FaceDetector object to perform face detection
    face_detector = FaceDetector()

    # Continuously capture frames from the video capture device and process them for face detection
    while True:
        # Read a frame from the video capture device
        ret, frame = cap.read()

        # If there was an error reading the frame, break out of the loop
        if not ret:
            break

        # Process the current frame using the FaceDetector object to detect and draw landmarks on any faces
        frame = face_detector.find_faces(frame)

        # Calculate the current FPS and display it on the frame
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the current frame in a window named "Face Detection"
        cv2.imshow('Face Detection', frame)

        # If the user presses the 'q' key, break out of the loop and exit the application
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
