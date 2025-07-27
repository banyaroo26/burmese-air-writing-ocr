import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
from keras.models import load_model
import joblib

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
keras_model  = load_model('./models/teachable_machine/keras_model.h5', compile=True)
mnv2_model   = load_model('./models/mobile_net_v2/mobilenetv2_burmese.h5', compile=True)
logreg_model = joblib.load('./models/logistic_regression/sk_model_logreg.joblib')

# label encoder for sklearn
label_encoder = joblib.load('./models/logistic_regression/sk_label_encoder.joblib')

# print(model.summary())

# ['0 ka kyee\n', '1 kha khway\n', ... ]
keras_class_names = open('./models/teachable_machine/keras_labels.txt', 'r').readlines()
mnv2_class_names = open('./models/mobile_net_v2/mobilenetv2_labels.txt', 'r').readlines()

save_mode, brush_mode = False, True

threshold = 50 # between middle and index fingers tips

brush_size = 5

safe_w, safe_h = 640, 480


def restart_canvas(chn3=True):
    h, w = int(safe_h), int(safe_w)
    if chn3:
        return np.zeros((h, w, 3), dtype=np.uint8)
        # return np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        return np.zeros((h, w), dtype=np.uint8)
        # return np.zeros((480, 640, 3), dtype=np.uint8)


# contour_canvas = restart_canvas(chn3=False)
canvas, clean_canvas = restart_canvas(chn3=True), restart_canvas(chn3=True)


def strip_class_name(class_name):
    space_index = class_name.find(' ')
    stripped_name = class_name[space_index+1:-1].strip()

    return stripped_name


def preprocess_alphabet(image):
    # imitates teachable machines preprocessing steps
    #image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    #image = (image / 127.5) - 1  # [-1, 1]
    #image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    image = cv2.resize(image, (96, 96))
    image = image / 255.0  # [0, 1]
    image = np.asarray(image, dtype=np.float32).reshape(1, 96, 96, 3)

    return image


def save_userinput(class_name, image):
    folder_name = strip_class_name(class_name=class_name)
    random_filename = str(uuid.uuid4())

    if folder_name not in os.listdir('./userinput'):
        os.makedirs(f'./userinput/{folder_name}', exist_ok=True)

    return cv2.imwrite(f'./userinput/{folder_name}/{random_filename}.jpg', image)


def predict_alphabet(image, text_loc):
    prediction = mnv2_model.predict(image)  # class probabilites [x, x, ...]
    index      = np.argmax(prediction)  # argmax flattens the array and return index
    class_name = mnv2_class_names[index]
    confidence_score = prediction[0][index]

    text = f'{strip_class_name(class_name)}-{str(np.round(confidence_score * 100))[:-2]}%'
    cv2.putText(canvas, text, (text_loc[0]-10, text_loc[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return class_name


def run():

    # global contour_canvas
    global canvas, clean_canvas
    global save_mode, brush_mode, threshold, brush_size

    delay_frames = 0  # delay counter before drawing contours

    prev_x, prev_y = None, None  # previous finger position

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

    vc = cv2.VideoCapture(index=0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, safe_w)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, safe_h)

    while vc.isOpened():
        _, frame = vc.read()

        frame = cv2.flip(src=frame, flipCode=1)

        # media pipe requires RGB
        rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        detect = hands.process(rgb_frame)

        # frame height, width and number of channels (3)
        h, w, _ = frame.shape

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
                thumb_tip  = hand_landmarks.landmark[12] # middle finger, not thumb

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

        # print(delay_frames)

        if delay_frames > 20:
            # find contours from clean_canvas
            contours, _ = cv2.findContours(cv2.cvtColor(clean_canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(contour_canvas, contours, -1, 255) # -1 means draw all contours

            if contours:
                # smallest x-axis and y-axis value out of all contours drawn currently
                # boundingRect(c) -> top left x,y values and width and height of c
                x_min = min(cv2.boundingRect(c)[0] for c in contours)
                y_min = min(cv2.boundingRect(c)[1] for c in contours)
                x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
                y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

                cropped = clean_canvas.copy()[y_min: y_max, x_min: x_max]
                cropped = cv2.resize(cropped, (83, 84), interpolation=cv2.INTER_LANCZOS4)  

                # print(cropped.shape)

                # test with logistic regression before 
                log_reg_prediction = logreg_model.predict([np.array(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)/255.0).flatten()])
                print('Logistic Regression ' + str(label_encoder.classes_[log_reg_prediction][0]))

                # cv2.imshow('83x84', cropped)

                cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), 255, 3)
                # clean_canvas, contour_canvas = restart_canvas(chn3=True), restart_canvas(chn3=False)
                clean_canvas = restart_canvas(chn3=True)

                preprocessed_img = preprocess_alphabet(image=cropped)
                predicted_class = predict_alphabet(image=preprocessed_img, text_loc=(x_min, y_min))

                if save_mode: 
                    save_userinput(class_name=predicted_class, image=cropped)

        # overlay two images
        frame = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
        
        if brush_mode:
            draw_mode = 'brush'
        else:
            draw_mode = 'line'

        cv2.putText(frame, f'Mode:{draw_mode}, press `p` to switch mode', (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'threshold:{threshold}, press `m` to increase and `n` to decrease', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'You are using a model based on MobileNetV2 model', (5,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        if save_mode: 
            cv2.putText(frame, 'User inputs are being saved', (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

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

if __name__ == '__main__':
    run()