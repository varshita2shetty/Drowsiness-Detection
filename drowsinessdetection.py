
from flask import Flask, render_template, Response
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

app = Flask(__name__)

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']
model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

no_face_detected_start_time = None
face_turn_start_time = None
face_turned = False

def detect_drowsiness(frame):
    global count, score, rpred, lpred, thicc, no_face_detected_start_time, face_turn_start_time, face_turned

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    if len(faces) == 0:
        if no_face_detected_start_time is None:
            no_face_detected_start_time = time.time()
        elif time.time() - no_face_detected_start_time > 15:
            try:
                sound.play()
            except:
                pass
    else:
        no_face_detected_start_time = None

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        left_eye = leye.detectMultiScale(roi_gray)
        right_eye = reye.detectMultiScale(roi_gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        face_center_x = x + w // 2
        frame_center_x = width // 2
        if abs(face_center_x - frame_center_x) > frame_center_x // 4:
            if face_turn_start_time is None:
                face_turn_start_time = time.time()
            elif time.time() - face_turn_start_time > 15:
                face_turned = True
                cv2.putText(frame, "Face Turned", (10, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                try:
                    sound.play()
                except:
                    pass
        else:
            face_turn_start_time = None
            face_turned = False

        for (ex, ey, ew, eh) in right_eye:
            r_eye = roi_color[ey:ey + eh, ex:ex + ew]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)
            if rpred[0] == 1:
                lbl = 'Open'
            if rpred[0] == 0:
                lbl = 'Closed'
            break

        for (ex, ey, ew, eh) in left_eye:
            l_eye = roi_color[ey:ey + eh, ex:ex + ew]
            count += 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=-1)
            if lpred[0] == 1:
                lbl = 'Open'
            if lpred[0] == 0:
                lbl = 'Closed'
            break

        if rpred[0] == 0 and lpred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score > 15 and not face_turned:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass
            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_drowsiness(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
