import cv2 as cv
import numpy as np



cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
 

     # When everything done, release the capture
cap.release()
















output_folder="facial_photos"

counter=1
# Create directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)
image_filename = f'facial_photo{counter}.jpg'
image_path = os.path.join(folder_path, image_filename)