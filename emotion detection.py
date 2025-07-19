# Import necessary libraries
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Load Haar Cascade face detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
classifier =load_model('./Emotion_Detection.h5')

# Define the labels corresponding to the output classes of the model
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)


# Start the main loop to process video frames continuously
while True:
    # Capture each frame from the video stream
    ret, frame = cap.read()
    labels = []
    # Convert the captured frame to grayscale as the classifier expects gray images
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame using the Haar Cascade model
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    # Loop through each detected face
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Extract the region of interest (ROI) which is the face area
        roi_gray = gray[y:y+h,x:x+w]

        # Resize ROI to 48x48 which is the input size expected by the model
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        # Ensure the ROI is not empty
        if np.sum([roi_gray])!=0:
            # Normalize the ROI pixels and convert to array
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)  # Add batch dimension

            # Predict the emotion from the ROI using the loaded model
            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)

            # Get the label with the highest probability
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)

            # Display the predicted label on the original frame
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            # If no face is found, display a message
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

