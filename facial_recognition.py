import face_recognition
import cv2
import numpy as np
import time
import pickle

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)  # 0 for default webcam, change if using an external one

if not video_capture.isOpened():
    print("[ERROR] Could not access the webcam.")
    exit()

# Initialize variables
cv_scaler = 3  # Scale factor to reduce processing load
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame(frame):
    """Processes the frame: detects faces and recognizes them."""
    global face_locations, face_encodings, face_names

    # Resize the frame to improve processing speed
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))

    # Convert the frame from BGR to RGB (required by face_recognition)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute face encodings
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    for face_encoding in face_encodings:
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_names.append(name)

    return frame

def draw_results(frame):
    """Draws bounding boxes and labels around detected faces."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale face locations back up
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)

        # Draw label
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return frame

def calculate_fps():
    """Calculates the FPS for performance tracking."""
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Main loop
while True:
    ret, frame = video_capture.read()  # Read frame from webcam

    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    # Process frame
    processed_frame = process_frame(frame)

    # Draw results
    display_frame = draw_results(processed_frame)

    # Calculate and display FPS
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow('Face Recognition', display_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
