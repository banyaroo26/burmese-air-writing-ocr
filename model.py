from keras.models import load_model
import joblib
import cv2
import numpy as np
import os
import uuid

# Load the model
# keras_model  = load_model('./models/teachable_machine/keras_model.h5', compile=True)
# logreg_model = joblib.load('./models/logistic_regression/sk_model_logreg.joblib')

# label encoder for sklearn
# label_encoder = joblib.load('./models/logistic_regression/sk_label_encoder.joblib')

# print(model.summary())

# ['0 ka kyee\n', '1 kha khway\n', ... ]
# keras_class_names = open('./models/teachable_machine/keras_labels.txt', 'r').readlines()


mnv2_model = load_model('./models/augmented/augmented_burmese.h5', compile=True)
mnv2_class_names = open('./models/augmented/labels.txt', 'r').readlines()


def strip_class_name(class_name):
    space_index = class_name.find(' ')
    stripped_name = class_name[space_index+1:-1].strip()

    return stripped_name


def predict_alphabet(canvas, image, text_loc):
    prediction = mnv2_model.predict(image)  # class probabilites [x, x, ...]
    index      = np.argmax(prediction)  # argmax flattens the array and return index
    class_name = mnv2_class_names[index]
    confidence_score = prediction[0][index]

    text = f'{strip_class_name(class_name)}-{str(np.round(confidence_score * 100))[:-2]}%'
    cv2.putText(canvas, text, (text_loc[0]-10, text_loc[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return class_name


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


# test with logistic regression before 
# log_reg_prediction = logreg_model.predict([np.array(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)/255.0).flatten()])
# print('Logistic Regression ' + str(label_encoder.classes_[log_reg_prediction][0]))