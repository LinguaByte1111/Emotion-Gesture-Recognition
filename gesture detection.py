# Import required libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Mediapipe Hands model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load pre-trained gesture recognition model
model = load_model('mp_hand_gesture')

# Load gesture class names from file
f = open('gesture.names', 'r')
classNames = f.read().split('\n') # Read each gesture name
f.close()
print(classNames) # Print gesture names for verification

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, frame = cap.read()
    
    # Get frame dimensions
    x, y, c = frame.shape

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to RGB format for Mediapipe processing
    result = hands.process(framergb)    
    className = '' # Initialize gesture name

    # Process the frame to detect hands
    if result.multi_hand_landmarks:
        landmarks = [] # List to hold hand landmark coordinates
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:

                # Convert normalized coordinates to pixel values
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy]) # Append each landmark

            # Draw hand landmarks on the frame
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

    # Display the gesture name on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output frame
    cv2.imshow("Output", frame) 
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()  # Fixed typo here from 'destroyAllWindow'
