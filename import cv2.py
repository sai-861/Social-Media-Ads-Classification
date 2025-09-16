import cv2
import mediapipe as mp
import numpy as np
import winsound
import time
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define EAR calculation function
def calculate_ear(landmarks, eye_points):
    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (A + B) / (2.0 * C)

# Define eye landmark indexes for EAR calculation
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

# Initialize GUI window
root = Tk()
root.title("Driver Drowsiness Detection")
root.geometry("1000x700")
root.configure(bg="lightblue")

# Add a label for the video feed
video_label = Label(root)
video_label.pack(padx=20, pady=20)

# Add a status label
status_label = Label(root, text="Status: WAITING FOR START", font=("Helvetica", 16), bg="lightblue")
status_label.pack(pady=10)

# Define start and stop variables
cap = None
running = False

# Define the start detection function
def start_detection():
    global cap, running, COUNTER
    running = True
    cap = cv2.VideoCapture(0)
    COUNTER = 0
    status_label.config(text="Status: Detecting...", fg="orange")
    detect()

# Define the stop detection function
def stop_detection():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="Status: Stopped", fg="red")

# Define the main detection loop
def detect():
    global COUNTER, running
    if not running:
        return
    
    ret, frame = cap.read()
    if not ret:
        stop_detection()
        return

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # Draw eyes on the frame
            for point in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[point], 5, (0, 255, 0), -1)

            # Check if EAR is below the threshold
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    status_label.config(text="Status: DROWSINESS ALERT!", fg="red")
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    winsound.Beep(2500, 1000)  # Beep once for the alert
            else:
                COUNTER = 0
                status_label.config(text="Status: Detecting...", fg="orange")

    # Convert frame to a format compatible with Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Continue updating the frame
    video_label.after(10, detect)

# Add start and stop buttons
button_frame = Frame(root, bg="lightblue")
button_frame.pack(pady=20)

start_button = Button(button_frame, text="Start Detection", font=("open sans", 14), command=start_detection, bg="green", fg="white")
start_button.pack(side=LEFT, padx=10)

stop_button = Button(button_frame, text="Stop Detection", font=("open sans", 14), command=stop_detection, bg="red", fg="white")
stop_button.pack(side=LEFT, padx=10)

# Run the GUI event loop
root.mainloop()