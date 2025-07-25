import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
from keras.models import load_model


# -------- to prevent .h5 incompatibility error (from stackoverflow) ------------- #

import h5py

f = h5py.File("keras_model.h5", mode="r+")
model_config_string = f.attrs.get("model_config")

if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
f.attrs.modify('model_config', model_config_string)
f.flush()

model_config_string = f.attrs.get("model_config")

assert model_config_string.find('"groups": 1,') == -1

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# ['0 ka kyee\n', '1 kha khway\n', ... ]
class_names = open("labels.txt", "r").readlines()

# ---------------------------------------------------- #


def restart_canvas(chn3=True):
    h, w = int(safe_h), int(safe_w)
    if chn3:
        return np.zeros((h, w, 3), dtype=np.uint8)
        # return np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        return np.zeros((h, w), dtype=np.uint8)
        # return np.zeros((480, 640, 3), dtype=np.uint8)


def preprocess_alphabet(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # model requires RGB
    resized = cv2.resize(image, (224, 224))  # Resize to 224x224
    # cv2.imshow('224x224', resized)
    resized = resized.astype('float32') / 255.0  # Normalize to [0, 1]
    resized = np.expand_dims(resized, axis=0)  # Shape: (1, 224, 224, 3) - Add batch dimension
    return resized


def save_userinput(class_name, image):
    space_index = class_name.find(' ')
    folder_name = class_name[space_index+1:-1].strip()
    random_filename = str(uuid.uuid4())

    if folder_name not in os.listdir('./userinput'):
        os.makedirs(f'./userinput/{folder_name}', exist_ok=True)

    return cv2.imwrite(f'./userinput/{folder_name}/{random_filename}.jpg', image)


def predict_alphabet(image):
    prediction = model.predict(image)  # class probabilites [x, x, ...] 
    index      = np.argmax(prediction)  # argmax flattens and get index
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    cv2.putText(canvas, f'{class_name}-{confidence_score:.2f}', (x_min-10, y_min-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return class_name


desired_w = 1331  # desired frame size
desired_h = 222
safe_w = 640
safe_h = 480

vc = cv2.VideoCapture(index=0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)
success, _ = vc.read()

if success:
    safe_w = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    safe_h = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

# print(safe_w, safe_h)
# print(vc.get(cv2.CAP_PROP_FPS))

canvas, contour_canvas, clean_canvas = restart_canvas(chn3=True), restart_canvas(chn3=False), restart_canvas(chn3=True)

save_mode  = True
brush_mode = True

delay_frames = 0  # delay counter before drawing contours

threshold  = 50  # between middle and index fingers tips
brush_size = 5

cropped = None
prev_x, prev_y = None, None  # previous finger position

# Solutions API

# Initialize MediaPipe Hands
mp_hands   = mp.solutions.hands

# Instantiate the Hands class
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video streams (better performance)
    max_num_hands=1,               # Max number of hands to detect
    min_detection_confidence=0.5,  # Confidence threshold to start tracking
    min_tracking_confidence=0.5    # Confidence threshold to continue tracking
)

# For drawing landmarks
mp_drawing = mp.solutions.drawing_utils  

while vc.isOpened():
    _, frame = vc.read()

    frame = cv2.flip(src=frame, flipCode=1)

    # media pipe requires RGB
    rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

    detect = hands.process(rgb_frame)

    # frame height, width and number of channels (3)
    h, w, ch = frame.shape

    # if hand is detected
    if detect.multi_hand_landmarks:

        for hand_landmarks in detect.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Landmark color
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)   # Connection color
            )

            # landmark node (x, y) is standarized to be [0, 1], 
            # so to get its position on frame, multiply it with frame values (finger.x * frame.x)
            finger_tip = hand_landmarks.landmark[8] 
            thumb_tip  = hand_landmarks.landmark[12]

            (finger_x, finger_y) = (int(finger_tip.x * w), int(finger_tip.y * h))
            (thumb_x, thumb_y)   = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            # Compute Euclidean distance between index and thumb tips
            distance = np.linalg.norm(np.array([finger_x, finger_y]) - np.array([thumb_x, thumb_y]))

            if distance > threshold:

                delay_frames = 0

                cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 255), -1) # yellow preview circle
                
                # draw on canvas if finger is moving
                if prev_x and prev_y:
                    cv2.line(canvas, (prev_x, prev_y), (finger_x, finger_y), (255, 255, 255), brush_size)
                    cv2.line(clean_canvas, (prev_x, prev_y), (finger_x, finger_y), (255, 255, 255), brush_size)
                
                prev_x, prev_y = finger_x, finger_y

            else:
                # red circle to show that currently not drawing
                cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)

                if brush_mode:
                    prev_x, prev_y = None, None

                delay_frames += 1 if brush_mode else 0.2

    # if hand is not detected
    else:
        delay_frames += 1 if brush_mode else 0.2

    if delay_frames > 20:
        # find contours from clean_canvas
        contours, _ = cv2.findContours(cv2.cvtColor(clean_canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_canvas, contours, -1, 255) # -1 means draw all contours

        if contours:
            # smallest x-axis and y-axis value out of all contours drawn on contour_canvas currently
            # boundingRect(c) -> top left x,y values and width and height of c
            x_min = min(cv2.boundingRect(c)[0] for c in contours)
            y_min = min(cv2.boundingRect(c)[1] for c in contours)
            x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
            y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

            cropped = clean_canvas.copy()[y_min: y_max, x_min: x_max]
            cropped = cv2.resize(cropped, (83, 84))  

            # cv2.imshow('83x84', cropped)

            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), 255, 3)
            clean_canvas, contour_canvas = restart_canvas(chn3=True), restart_canvas(chn3=False)

            preprocessed_img = preprocess_alphabet(image=cropped)
            predicted_class = predict_alphabet(image=preprocessed_img)

            if save_mode: 
                save_userinput(class_name=predicted_class, image=cropped)

    # overlay two images
    frame = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
    
    if brush_mode:
        mode = 'brush'
    else:
        mode = 'line'
    cv2.putText(frame, f'Mode:{mode}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'threshold:{threshold}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # cv2.imshow('canvas', canvas)
    # cv2.imshow('clean canvas', clean_canvas)
    # cv2.imshow('contour canvas', contour_canvas)

    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    elif key == ord('m'): # m - 109
        threshold += 1
    elif key == ord('n'): # n - 110
        threshold -= 1
    elif key == ord('d'):
        canvas = restart_canvas(chn3=True)
    elif key == ord('s'):
        save_mode = not save_mode
    elif key == ord('p'):
        brush_mode = not brush_mode

vc.release()
cv2.destroyAllWindows()