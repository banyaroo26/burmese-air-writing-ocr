import cv2
import mediapipe as mp
import numpy as np
from model import predict_alphabet, preprocess_alphabet, save_userinput

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

save_mode, brush_mode = False, True

threshold = 50 # between middle and index fingers tips

brush_size = 5

safe_w, safe_h = 640, 480

to_predict = []

def clear_canvas(chn3=True):
    h, w = int(safe_h), int(safe_w)
    if chn3:
        return np.zeros((h, w, 3), dtype=np.uint8)
        # return np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        return np.zeros((h, w), dtype=np.uint8)
        # return np.zeros((480, 640, 3), dtype=np.uint8)

# contour_canvas = clear_canvas(chn3=False)
canvas, clean_canvas = clear_canvas(chn3=True), clear_canvas(chn3=True)

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
vc.set(cv2.CAP_PROP_FPS, 15)
vc.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def run():

    global canvas, clean_canvas, delay_frames
    global save_mode, brush_mode, threshold, brush_size

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
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),  # Landmark color
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)   # Connection color
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

                    cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), -1) # yellow preview circle
                    
                    # draw on canvas if finger is moving
                    if prev_x and prev_y:
                        cv2.line(canvas, (prev_x, prev_y), (finger_x, finger_y), (255, 255, 255), brush_size)
                        cv2.line(clean_canvas, (prev_x, prev_y), (finger_x, finger_y), (255, 255, 255), brush_size)
                    
                    prev_x, prev_y = finger_x, finger_y

                else:
                    # red circle to show that currently not drawing
                    cv2.circle(frame, (finger_x, finger_y), 5, (255, 0, 0), -1)

                    if brush_mode:
                        prev_x, prev_y = None, None

                    delay_frames += 1 if brush_mode else 0.2

        # if hand is not detected
        else:
            delay_frames += 1 if brush_mode else 0.2

        # print(delay_frames)

        if delay_frames > 50:
            # find contours from clean_canvas
            contours, _ = cv2.findContours(cv2.cvtColor(clean_canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(contour_canvas, contours, -1, 255) # -1 means draw all contours

            if contours:
                # smallest x-axis and y-axis value out of all contours drawn currently
                # boundingRect(c) -> top left x,y values and width and height of c
                '''
                x_min = min(cv2.boundingRect(c)[0] for c in contours)
                y_min = min(cv2.boundingRect(c)[1] for c in contours)
                x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
                y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
                '''

                for c in contours:

                    print('do')

                    x_min = cv2.boundingRect(c)[0]
                    y_min = cv2.boundingRect(c)[1]
                    w     = cv2.boundingRect(c)[2]
                    h     = cv2.boundingRect(c)[3]
                    x_max = x_min + w
                    y_max = y_min + h

                    cropped = clean_canvas.copy()[y_min: y_max, x_min: x_max]
                    cropped = cv2.resize(cropped, (83, 84), interpolation=cv2.INTER_LANCZOS4) 

                    # Remove contour area by filling it with black (canvas color)
                    cv2.rectangle(clean_canvas, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=cv2.FILLED)

                    cv2.imshow('', cropped)

                    cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), 255, 3)
                    # clean_canvas, contour_canvas = clear_canvas(chn3=True), clear_canvas(chn3=False)
                    # clean_canvas = clear_canvas(chn3=True)

                    preprocessed_img = preprocess_alphabet(image=cropped)
                    predicted_class = predict_alphabet(canvas=canvas, image=preprocessed_img, text_loc=(x_min, y_min))

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
        cv2.imshow('canvas', canvas)
        cv2.imshow('clean canvas', clean_canvas)
        # cv2.imshow('contour canvas', contour_canvas)

        # yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        key = cv2.waitKey(1)
        if key == 27: # esc
            break
        elif key == ord('m'): # m - 109
            threshold += 1
        elif key == ord('n'): # n - 110
            threshold -= 1
        elif key == ord('d'):
            canvas = clear_canvas(chn3=True)
        elif key == ord('s'):
            save_mode = not save_mode
        elif key == ord('p'):
            brush_mode = not brush_mode

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()