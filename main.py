import cv2
import mediapipe as mp
import numpy as np
from model import predict_alphabet, preprocess_alphabet, save_userinput

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

save_mode = False
brush_mode = True
move_mode = False

threshold = 50 # between middle and index fingers tips

brush_size = 5

safe_w, safe_h = 640, 480

to_predict = []

# Dictionary to store drawn objects (both contours and their labels)
drawn_objects = {}
current_object_id = 0
selected_object_id = None
drag_offset_x, drag_offset_y = 0, 0


# when a shape is classified, save it
def save_drawn_object(contour, label, label_pos):
    global current_object_id
    x, y, w, h = cv2.boundingRect(contour)
    drawn_objects[current_object_id] = {
        'contour': contour,
        'label': label,
        'label_pos': label_pos,
        'position': (x, y, w, h),
        'image': canvas[y: y+h, x: x+w].copy() # snip the image
    }
    current_object_id += 1
    # cv2.imshow('', drawn_objects[current_object_id-1]['image'])
    return current_object_id - 1


# when index and middle fingers are on one of the object's area
def select_object_at_position(x, y):
    global selected_object_id, drag_offset_x, drag_offset_y
    for object_id, object in drawn_objects.items():
        ox, oy, ow, oh = object['position']
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            selected_object_id = object_id  # mark it as selected object
            drag_offset_x = x - ox
            drag_offset_y = y - oy
            return True
    return False


# move the object selected by function above
# move to index finger position
def move_selected_object(dest_x, dest_y):
    global selected_object_id
    if selected_object_id is not None and selected_object_id in drawn_objects:
        obj = drawn_objects[selected_object_id]
        # old_x = obj['position'][0]
        # old_y = obj['position'][1]
        old_w = obj['position'][2]   # do noting to w and h
        old_h = obj['position'][3]
        new_x = dest_x - drag_offset_x
        new_y = dest_y - drag_offset_y
        drawn_objects[selected_object_id]['position'] = (new_x, new_y, old_w, old_h)
        # Redraw everything (simpler but less efficient approach)
        redraw_all_objects()
    print(f'moved to {new_x} and {new_y}')


def redraw_all_objects():
    global canvas, clean_canvas
    # Clear both canvases
    canvas = clear_canvas(chn3=True)
    clean_canvas = clear_canvas(chn3=True)
    
    # Redraw all objects
    for obj_id, obj in drawn_objects.items():
        x, y, w, h = obj['position']
        
        # Draw the image on clean_canvas (for prediction)
        if obj['image'] is not None:
            canvas[y: y+h, x: x+w] = obj['image']  # just paste the snipped image 
        
        # Draw the label
        if obj['label']:
            label_x = x + obj['label_pos'][0]
            label_y = y + obj['label_pos'][1]
            color = (0, 0, 255) if obj_id == selected_object_id else (255, 255, 255)
            cv2.putText(canvas, obj['label'], (label_x, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    
# reset the canvas 
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

    global canvas, clean_canvas
    global selected_object_id, drawn_objects
    global save_mode, brush_mode, move_mode
    global threshold, brush_size, delay_frames

    while vc.isOpened():
        ok, frame = vc.read()

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
                middle_tip = hand_landmarks.landmark[12] # middle finger

                (finger_x, finger_y) = (int(finger_tip.x * w), int(finger_tip.y * h))
                (middle_x, middle_y) = (int(middle_tip.x * w), int(middle_tip.y * h))

                # Compute Euclidean distance between index and thumb tips
                distance = np.linalg.norm(np.array([finger_x, finger_y]) - np.array([middle_x, middle_y]))

                if not move_mode:

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

                else:

                    if distance > threshold:

                        if selected_object_id is not None:
                            move_selected_object(finger_x, finger_y)

                    else:

                        if selected_object_id is None:
                            if select_object_at_position(finger_x, finger_y):
                                print('selected')

                        else:
                            if not select_object_at_position(finger_x, finger_y):
                                selected_object_id = None
                                print('de-selected')

        # if hand is not detected
        else:
            delay_frames += 1 if brush_mode else 0.2

        # print(delay_frames)

        if delay_frames > 50 and clean_canvas.any():
            # find contours from clean_canvas
            contours, _ = cv2.findContours(cv2.cvtColor(clean_canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(contour_canvas, contours, -1, 255) # -1 means draw all contours

            if contours:
                # smallest x-axis and y-axis value out of all contours drawn currently
                # boundingRect(c) -> top left x,y values and width and height of c

                for c in contours:

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

                    # cv2.imshow('', cropped)

                    cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), 255, 3)
                    # clean_canvas, contour_canvas = clear_canvas(chn3=True), clear_canvas(chn3=False)
                    # clean_canvas = clear_canvas(chn3=True)

                    preprocessed_img = preprocess_alphabet(image=cropped)
                    predicted_class = predict_alphabet(canvas=canvas, image=preprocessed_img, text_loc=(x_min, y_min))

                    # Add to drawn objects (both contour and label)
                    label_pos = (-10, -10)  # Relative position for label
                    save_drawn_object(contour=c, label=predicted_class, label_pos=label_pos)

                    if save_mode: 
                        save_userinput(class_name=predicted_class, image=cropped)

        # overlay two images
        frame = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
        
        if brush_mode:
            draw_mode = 'brush'
        else:
            draw_mode = 'line'

        cv2.putText(frame, f'Mode: {draw_mode}, press `p` to switch mode', (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'threshold: {threshold}, press `m` to increase and `n` to decrease', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'You are using a model based on MobileNetV2 model', (5,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        if move_mode:
            cv2.putText(frame, 'Move mode enabled', (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        if save_mode: 
            cv2.putText(frame, 'User inputs are being saved', (5,75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        # cv2.imshow('canvas', canvas)
        # cv2.imshow('clean canvas', clean_canvas)
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
            drawn_objects = {}
        elif key == ord('s'):
            save_mode = not save_mode
        elif key == ord('b'):
            brush_mode = not brush_mode
        elif key == ord('t'):
            move_mode = not move_mode

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()