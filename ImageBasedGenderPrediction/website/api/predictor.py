
from keras.models import load_model
import numpy as np
import cv2
from matplotlib import pyplot as plt
import base64
# %matplotlib inline
cascade = cv2.CascadeClassifier('detector.xml')
model = load_model('model.h5')
model._make_predict_function()


def outputImg(img):
    try:
        retval, buffered = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffered)
        return jpg_as_text
    except:
        return "error"


def processImage(img):
    font = cv2.FONT_HERSHEY_COMPLEX
    crop_margin = 0.2

    height, width, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    person_num = 0
    for (x, y, w, h) in faces:
        person_num += 1

        x1 = int(x - crop_margin * w)
        y1 = int(y - crop_margin * h)
        x2 = int(x + (1 + crop_margin) * w)
        y2 = int(y + (1 + crop_margin) * h)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height

        crop_img = gray_img[y1:y2, x1:x2]

        try:
            #         print(f'Processing face #{person_num}: ({x1},{y1}),({x2},{y2})')
            resized_normalized = cv2.resize(crop_img, (100, 100))/255
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            for_predict = resized_normalized.reshape(-1, 100, 100, 1)
            prediction = model.predict(for_predict)[0][0]

            if (prediction >= 0.5):
                text = f'M({prediction:.2f})'
                cv2.putText(img, text, (x, y), font, w/120, (255, 200, 0), 2, cv2.LINE_AA)
            else:
                text = f'F({prediction:.2f})'
                cv2.putText(img, text, (x, y), font, w/120, (150, 150, 255), 2, cv2.LINE_AA)

        except:
            print("failed to predict")
            #         print(f'Image({width},{height}) failed to crop face #{person_num}: ({x1},{y1}),({x2},{y2})')
            pass


def makePrediction(img_data):
    img_decoded = base64.b64decode(img_data)
    npimg = np.fromstring(img_decoded, dtype=np.uint8)
    img = cv2.imdecode(npimg, 1)
    processImage(img)
    return outputImg(img)
