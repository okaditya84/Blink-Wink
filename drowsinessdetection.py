
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
eyes = 2
rpred = [99]
lpred = [99]

# Lists to store eye status and fps over time
time_values = []
status_values = []
fps_values = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
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

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
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
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 15:
        # Person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if eyes < 16:
            eyes = eyes + 2
        else:
            eyes = eyes - 2
            if eyes < 2:
                eyes = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), eyes)

    # Calculate and display FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(frame, 'FPS: ' + str(fps), (10, 40), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    fps_values.append(fps)

    # Show "Press 'q' to exit" text on the camera feed
    cv2.putText(frame, "Press 'q' to exit", (width - 180, height - 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Keep track of time and eye status
    time_values.append(time.time())
    status_values.append(score)

    # Press 'q' to exit the loop and stop capturing frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# Plot the eye status and FPS graph
fig, ax1 = plt.subplots()

ax1.plot(time_values, status_values, label='Eye Status', color='blue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Eye Status (Score)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a twin Axes sharing the x-axis
ax2 = ax1.twinx()
ax2.plot(time_values, fps_values, label='FPS', color='red')
ax2.set_ylabel('Frames per Second (FPS)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Adding legends and title to the graph
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Eye Status and FPS Over Time')
plt.tight_layout()

# Save the graph as a JPG image
plt.savefig('eye_status_and_fps_graph.jpg')

# Show the graph
plt.show()
