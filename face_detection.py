import cv2


def detect():
    # Load the Haar cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Draw rectangles around the detected faces and eyes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Face and Eye Detection', frame)
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

# Call the detect function to start the detection process
detect()
