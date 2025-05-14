import cv2
import os
from datetime import datetime
from deepface import DeepFace

# Change this to the name of the person being photographed
PERSON_NAME = "aryan"

# Update dataset location for laptop
DATASET_PATH = r"C:\Users\aryan\OneDrive\Desktop\Face Recognition\dataset"

def create_folder(name):
    person_folder = os.path.join(DATASET_PATH, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def analyze_face(image_path):
    try:
        image = cv2.imread(image_path)
        result = DeepFace.analyze(image, actions=['age', 'gender', 'emotion', 'race'])
        print(f"Analysis result for {image_path}:\n", result)
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")

def capture_photos(name):
    folder = create_folder(name)
    
    # Initialize the webcam (0 is the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    photo_count = 0
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space key to capture image
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")
            analyze_face(filepath)  # Perform DeepFace analysis

        elif key == ord('q'):  # 'q' key to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)
