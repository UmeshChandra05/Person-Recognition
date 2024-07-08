import cv2
import os
import numpy as np
from datetime import datetime, timedelta

# Function to get greeting based on current time
def get_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good Morning"
    elif 12 <= current_hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"

# Creating the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Loading the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize lists to hold the faces and labels and dictionary to map label names to numeric fields
known_faces = []
known_labels = []
label_map = {}

# Loading known faces and assigning labels (names)
label = 0
for person_folder in os.listdir('Household_Members'):
    person_path = os.path.join('Household_Members', person_folder)
    
    if os.path.isdir(person_path):
        # Assigning label for each person folder
        label_map[person_folder] = label
        label += 1
        
        # Loading images from each person's folder
        for filename in os.listdir(person_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Reading the image
                img_path = os.path.join(person_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Detect faces in the image
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                
                for (x, y, w, h) in faces:
                    # Extract the face region of interest (ROI)
                    face = image[y:y+h, x:x+w]
                    known_faces.append(face)
                    known_labels.append(label_map[person_folder])

# Training the recognizer with the known faces and labels
recognizer.train(known_faces, np.array(known_labels))

# Initializing the camera
cap = cv2.VideoCapture(0)

start_time = datetime.now()
person_detected = False

while True:
    # Capturing a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Converting frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecting faces in grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        person_detected = True

    if len(faces) == 0:
        cv2.putText(frame, "No Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        # Extracting the face ROI from the grayscale frame
        face = gray_frame[y:y+h, x:x+w]
        # Predicting the label and confidence for the face
        label, confidence = recognizer.predict(face)
        name = "Unknown"

        if confidence < 50:  # Confidence threshold
            name = list(label_map.keys())[list(label_map.values()).index(label)]
            print(f"Person recognized as {name}. Opening the door.")
            cv2.putText(frame, f"recognized {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            main_image_path = None
            for ext in ['jpg', 'jpeg', 'png']:
                main_image_path = os.path.join('Household_Members', name, f"{name}_Main.{ext}")
                if os.path.exists(main_image_path):
                    break
            
            if main_image_path is not None and os.path.exists(main_image_path):
                main_image = cv2.imread(main_image_path)
                if main_image is not None:
                    # Resize the main image to fit in the left side of the screen
                    main_image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    main_image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
                    main_image = cv2.resize(main_image, (int(main_image_width), int(main_image_height)))

                    # Create the welcome message image with SmokyWhite background
                    welcome_image = np.full((int(main_image_height), int(main_image_width), 3), (245, 245, 245), dtype=np.uint8)  # SmokyWhite background
                    greeting = get_greeting()
                    lines = [greeting, name, "Welcome Home!"]
                    line_height = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] + 10

                    for i, line in enumerate(lines):
                        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_x = (welcome_image.shape[1] - text_size[0]) // 2
                        text_y = (welcome_image.shape[0] - len(lines) * line_height) // 2 + (i * line_height)
                        cv2.putText(welcome_image, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text color

                    # Combine the main image and the welcome message
                    combined_image = np.hstack((main_image, welcome_image))
                    
                    # Show the combined image in full screen
                    cv2.namedWindow('Recognized Person', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('Recognized Person', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('Recognized Person', combined_image)
                    cv2.waitKey(5000)  # Display for 5 seconds
                else:
                    print(f"Main image for {name} could not be read.")
            else:
                print(f"Main image for {name} not found.")
            
            # Release the camera and close windows
            cap.release()
            cv2.destroyAllWindows()
            exit()
        else:
            print("Person not recognized. Ringing the doorbell.")
            cv2.putText(frame, "Person not recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Drawing a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Drawing a filled rectangle below the face
        cv2.rectangle(frame, (x, y+h+10), (x+w, y+h+35), (255, 0, 0), cv2.FILLED)
        # Putting the name of person on the rectangle
        cv2.putText(frame, name, (x + 6, y + h + 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame in full screen
    cv2.namedWindow('Camera Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Camera Feed', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if 15 seconds have passed
    if datetime.now() - start_time > timedelta(seconds = 15):
        
        # Create the message image with SmokyWhite background
        message_image = np.full((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), (245, 245, 245), dtype=np.uint8)  # SmokyWhite background
        
        if person_detected:
            lines = ["Person Not Recognised", "Ringing the Door Bell"]
        else:
            lines = ["No Person Detected", "Turning off Camera"]
        
        line_height = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] + 10

        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (message_image.shape[1] - text_size[0]) // 2
            text_y = (message_image.shape[0] - len(lines) * line_height) // 2 + (i * line_height)
            cv2.putText(message_image, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text color
            
        # Show the message image in full screen
        cv2.namedWindow('UnRecognised Person/No Person Detected', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('UnRecognised Person/No Person Detected', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('UnRecognised Person/No Person Detected', message_image)
        cv2.waitKey(5000)
        cap.release()
        cv2.destroyAllWindows()
        exit()