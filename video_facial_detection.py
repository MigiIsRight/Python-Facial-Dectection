import cv2 as cv
import os
from datetime import datetime


class DetectionConditions:
    def __init__(self):
        self.face_detected = False
        self.eyes_detected = False   
        self.mouth_detected = False
        self.left_eye_detected = False
        self.right_eye_detected = False
        self.profile_face_detected = False
    
    def all_features_detected(self):
        """Return True if all facial features are detected"""
        return (self.face_detected and self.eyes_detected and self.mouth_detected 
                and self.left_eye_detected and self.right_eye_detected
                and self.profile_face_detected)
    
    def face_and_any_feature(self):
        """Return True if face and at least one other feature is detected"""
        return self.face_detected and (self.eyes_detected or self.mouth_detected or 
                                     self.left_eye_detected or self.right_eye_detected or 
                                     self.profile_face_detected)
    def facefound(self):
        return self.face_detected
# Configuration
folder_path = "ssdatabase"
save_every_n_frames = 30  # Save every 30 frames (about every 1 second at 30fps)

# Create directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)
print(f"✓ Database folder ready: {os.path.abspath(folder_path)}")

# Load cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')
left_eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_righteye_2splits.xml')
profile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_profileface.xml')

# Check if cascades loaded properly
cascades_loaded = all([
    not face_cascade.empty(),
    not eye_cascade.empty(),
    not mouth_cascade.empty(),
    not left_eye_cascade.empty(),
    not right_eye_cascade.empty(),
    not profile_cascade.empty()
])

if not cascades_loaded:
    print("Warning: Some cascades failed to load. Using face detection only.")

# Initialize camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Multi-Feature Face Capture")
print("Will save when face + other features detected")
print("Press 'q' to quit, 's' to save immediately")

counter = 0
saved_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Create detection condition object
        conditions = DetectionConditions()
        
        # Detect all features
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        left_eyes = left_eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        right_eyes = right_eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Update conditions
        conditions.face_detected = len(faces) > 0
        conditions.eyes_detected = len(eyes) > 0   
        conditions.mouth_detected = len(mouths) > 0
        conditions.left_eye_detected = len(left_eyes) > 0
        conditions.right_eye_detected = len(right_eyes) > 0
        conditions.profile_face_detected = len(profiles) > 0
        
        # Draw rectangles around detected features
        # Faces - Green
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, 'Face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Eyes - Blue
        for (x, y, w, h) in eyes:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv.putText(frame, 'Eye', (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Left Eye - Cyan
        """for (x, y, w, h) in left_eyes:
            cv.circle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
            cv.putText(frame, 'Left Eye', (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Right Eye - Magenta
        for (x, y, w, h) in right_eyes:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)
            cv.putText(frame, 'Right Eye', (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1"""
        
        # Profile Face - Orange 
        for (x, y, w, h) in profiles:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 1)
            cv.putText(frame, 'Profile', (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
        
        # Mouth - Yellow
        """for (x, y, w, h) in mouths:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
            cv.putText(frame, 'Mouth', (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)"""

        # Display detection status
        status_text = f"Face: {'Yes' if conditions.face_detected else 'No'} | Eyes: {'Yes' if conditions.eyes_detected else 'No'} | Mouth: {'Yes' if conditions.mouth_detected else 'No'}"
        cv.putText(frame, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Additional status line for left/right eye and profile
        status_text2 = f"L.Eye: {'Yes' if conditions.left_eye_detected else 'No'} | R.Eye: {'Yes' if conditions.right_eye_detected else 'No'} | Profile: {'Yes' if conditions.profile_face_detected else 'No'}"
        cv.putText(frame, status_text2, (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show save condition status
        save_ready = conditions.facefound()  # Change this to conditions.all_features_detected() if you want all features
        save_status = "READY TO SAVE" if save_ready else "WAITING..."
        color = (0, 255, 0) if save_ready else (0, 0, 255)
        cv.putText(frame, save_status, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv.imshow('Multi-Feature Detection', frame)

        # Save logic - save when conditions met and enough frames passed
        if save_ready and counter % save_every_n_frames == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(folder_path, f"capture_{saved_count:03d}_{timestamp}.jpg")
            
            if cv.imwrite(filename, frame):
                print(f"✓ Saved: {os.path.basename(filename)}")
                print(f"  - Features: Face({conditions.face_detected}), Eyes({conditions.eyes_detected}), Mouth({conditions.mouth_detected}), L.Eye({conditions.left_eye_detected}), R.Eye({conditions.right_eye_detected}), Profile({conditions.profile_face_detected})")
                saved_count += 1
            else:
                print("✗ Failed to save image")
            
        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(folder_path, f"manual_{saved_count:03d}_{timestamp}.jpg")
            if cv.imwrite(filename, frame):
                print(f"✓ Manual save: {os.path.basename(filename)}")
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
    cv.destroyAllWindows()
    print(f"\nTest completed: {saved_count} images saved")
    print(f"Folder: {os.path.abspath(folder_path)}")
    
    # List files in directory
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Total image files in database folder: {len(files)}")
        if files:
            print("Recent files:")
            for f in sorted(files)[-10:]:  # Show last 10 files
                file_path = os.path.join(folder_path, f)
                size = os.path.getsize(file_path)
                print(f"  - {f} ({size:,} bytes)")
    
 
    print("✓ Program ended successfully")