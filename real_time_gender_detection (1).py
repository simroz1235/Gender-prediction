import cv2
import numpy as np

# Model and weight file paths
FACE_PROTO = "deploy.prototxt.txt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_MODEL = 'deploy_gender.prototxt'
GENDER_PROTO = 'gender_net.caffemodel'

# Mean values for image preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

# Initialize frame size
frame_width = 1280
frame_height = 720

# Load face and gender detection models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                             frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()

def display_img(title, img, width=None, height=None):
    if width is None and height is None:
        cv2.imshow(title, img)
    else:
        h, w = img.shape[:2]
        if width is None:
            width = int(h * (height / float(w)))
        elif height is None:
            height = int(w * (width / float(h)))
        
        resized_img = cv2.resize(img, (width, height))
        cv2.imshow(title, resized_img)
    
    key = cv2.waitKey(1)  # Keep window open and refresh rate
    if key == ord('q') or cv2.getWindowProperty('Gender Detection', cv2.WND_PROP_VISIBLE) < 1:  # Exit on 'q' key press
        cv2.destroyAllWindows()
        return False  # Signal to exit the main loop
    return True


def predict_gender():
    cap = cv2.VideoCapture(0)  # 0 for default camera, change to 1, 2, etc. for other cameras

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame.shape[1] > frame_width:
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        faces = get_faces(frame)
        for (start_x, start_y, end_x, end_y) in faces:
            face_img = frame[start_y:end_y, start_x:end_x]
            gender_preds = get_gender_predictions(face_img)
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence_score = gender_preds[0][i]
            label = f"{gender}-{gender_confidence_score*100:.1f}%"
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            
            font_scale = 1.0 if gender == "Male" else 1.0
            box_color = (0, 255, 255) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 7)
            cv2.putText(frame, label, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
        
        should_continue = display_img("Gender Detection", frame, width=800, height=600)
        if not should_continue:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_gender()

