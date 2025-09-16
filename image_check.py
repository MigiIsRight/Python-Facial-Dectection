import os
import cv2 as cv
from datetime import datetime
from PIL import Image
from deepface import DeepFace as df


def brighten(image, bfactor=1.2):
    pass








def analyze_image(image_path):

    """Analyze the image for brightness and blurriness."""
    image = cv.imread(image_path)
    if image is None:
        return False, False  # Unable to read image

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Brightness check
    brightness = cv.mean(gray)[0]
    is_bright = brightness > 100  # Threshold for brightness

    # Blurriness check using variance of Laplacian
    laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
    is_blurry = laplacian_var < 100  # Threshold for blurriness

    return is_bright, not is_blurry  # Return True if not blurry


def zoom_on_face(image, face_loc, zoom_factor=1.5):
    
    """Zoom into the face region."""
    (x, y, w, h) = face_loc
    center_x, center_y = x + w // 2, y + h // 2
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)

    start_x = max(center_x - new_w // 2, 0)
    start_y = max(center_y - new_h // 2, 0)
    end_x = min(center_x + new_w // 2, image.shape[1])
    end_y = min(center_y + new_h // 2, image.shape[0])

    return image[start_y:end_y, start_x:end_x]


def remove_background(image_path, output_path):
    """Remove background using PIL's alpha channel."""
    image = Image.open(image_path).convert("RGBA")
    datas = image.getdata()

    newData = []
    for item in datas:
        # Change all white (also shades of whites)
        # to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    image.putdata(newData)
    image.save(output_path, "PNG")

def IsNewFace(image_path, known_faces_dir='known_faces'):
    """Check if the face in the image is new using DeepFace."""
    try:
        df.find(img_path=image_path, db_path=known_faces_dir)
        return False  # Face found in known faces
    except:
        return True  # Face not found, hence new face
    

def newFace_newFolder(base_dir='known_faces'):
    """Create a new folder for a new face."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_path = os.path.join(base_dir, f"face_{timestamp}")
    os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path

def multiple_faces(image_path):
    """Check if multiple faces are present in the image."""
    try:
        analysis = df.analyze(img_path=image_path, actions=['age'], enforce_detection=False)
        if isinstance(analysis, list):
            return len(analysis) > 1
        return False
    except:
        return False
    
