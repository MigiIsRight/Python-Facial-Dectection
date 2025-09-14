import cv2
import os
from datetime import datetime

# Configuration
folder_path = "c:/Users/peyto/Python-Facial-Detection/database"
save_every_n_frames = 30  # Save every 30 frames (about every 1 second at 30fps)

# Create directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Quick Face Capture Test")
print("This will save an image every ~1 second when faces are detected")
print("Press 'q' to quit, 's' to save immediately")

counter = 0
saved_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display info on frame
        cv2.putText(frame, f'Faces: {len(faces)} | Saved: {saved_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Detection Test', frame)

        # Save logic - save when faces detected and enough frames passed
        if len(faces) > 0 and counter % save_every_n_frames == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = os.path.join(folder_path, f"test_capture_{saved_count:03d}_{timestamp}.jpg")
            
            if cv2.imwrite(filename, frame):
                print(f"Saved: {filename}")
                saved_count += 1
            
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(folder_path, f"manual_{saved_count:03d}_{timestamp}.jpg")
            if cv2.imwrite(filename, frame):
                print(f"Manual save: {filename}")
                saved_count += 1

        counter += 1
        
        # Stop after saving 20 images for testing
        if saved_count >= 20:
            print("Reached 20 saved images. Stopping for test.")
            break

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTest completed: {saved_count} images saved in {folder_path}")
    
    # List files in directory
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Total image files in database folder: {len(files)}")
        if files:
            print("Files:")
            for f in sorted(files):
                print(f"  - {f}")