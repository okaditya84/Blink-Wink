# import cv2
# import os
# import matplotlib.pyplot as plt
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# import time

# # Initialize the sound mixer
# mixer.init()
# sound = mixer.Sound('alarm.wav')

# # Load Haar cascade classifiers for face and eyes
# # Haar cascade classifiers are pre-trained models used for face and eye detection.
# face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
# left_eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
# right_eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# # Labels for eye status
# eye_status_labels = ['Close', 'Open']

# # Load the CNN model
# # The CNN (Convolutional Neural Network) model is a deep learning model used for image classification tasks.
# model = load_model('models/cnncat2.h5')
# path = os.getcwd()

# # Open the camera
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 60)

# # Initialize variables
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# eyes = 2
# rpred = [99]
# lpred = [99]

# # Lists to store eye status and fps over time
# time_values = []
# status_values = []
# fps_values = []

# while True:
#     # Capture frame from the camera
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally
#     # Flipping the frame horizontally to get a mirror image.
#     frame = cv2.flip(frame, 1)
#     height, width = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces and eyes
#     # Using Haar cascade classifiers to detect faces and eyes in the frame.
#     faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
#     left_eyes = left_eye_cascade.detectMultiScale(gray)
#     right_eyes = right_eye_cascade.detectMultiScale(gray)

#     # Draw a rectangle at the bottom of the frame
#     # Drawing a black rectangle at the bottom of the frame to display text on it.
#     cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

#     # Detect and label faces
#     # Drawing rectangles around detected faces and labeling them as "Face".
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

#     # Detect and label right eye
#     # Detecting the right eye in the frame and labeling it as "Open" or "Closed".
#     for (x, y, w, h) in right_eyes:
#         right_eye = frame[y:y + h, x:x + w]
#         count = count + 1
#         right_eye_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
#         right_eye_resized = cv2.resize(right_eye_gray, (24, 24))
#         right_eye_normalized = right_eye_resized / 255
#         right_eye_input = right_eye_normalized.reshape(24, 24, -1)
#         right_eye_input = np.expand_dims(right_eye_input, axis=0)
#         rpred = np.argmax(model.predict(right_eye_input), axis=-1)
#         if rpred[0] == 1:
#             eye_status = 'Open'
#         if rpred[0] == 0:
#             eye_status = 'Closed'
#         break

#     # Detect and label left eye
#     # Detecting the left eye in the frame and labeling it as "Open" or "Closed".
#     for (x, y, w, h) in left_eyes:
#         left_eye = frame[y:y + h, x:x + w]
#         count = count + 1
#         left_eye_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
#         left_eye_resized = cv2.resize(left_eye_gray, (24, 24))
#         left_eye_normalized = left_eye_resized / 255
#         left_eye_input = left_eye_normalized.reshape(24, 24, -1)
#         left_eye_input = np.expand_dims(left_eye_input, axis=0)
#         lpred = np.argmax(model.predict(left_eye_input), axis=-1)
#         if lpred[0] == 1:
#             eye_status = 'Open'
#         if lpred[0] == 0:
#             eye_status = 'Closed'
#         break

#     # Update eye status score
#     # Keeping track of the eye status score based on the eye predictions.
#     if rpred[0] == 0 and lpred[0] == 0:
#         score = score + 1
#         cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#     else:
#         score = score - 1
#         cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score < 0:
#         score = 0
#     cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#     if score > 15:
#         # Person is feeling sleepy, so sound the alarm
#         # If the eye status score indicates drowsiness for a certain duration, sound an alarm.
#         cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
#         try:
#             sound.play()
#         except:
#             pass
#         if eyes < 16:
#             eyes = eyes + 2
#         else:
#             eyes = eyes - 2
#             if eyes < 2:
#                 eyes = 2
#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), eyes)

#     # Calculate and display FPS
#     # Calculating the frames per second and displaying it on the frame.
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     cv2.putText(frame, 'FPS: ' + str(fps), (10, 40), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
#     fps_values.append(fps)

#     # Display the frame
#     # Showing the frame on the screen.
#     cv2.imshow('frame', frame)

#     # Keep track of time and eye status
#     # Recording the time and eye status score for plotting graphs later.
#     time_values.append(time.time())
#     status_values.append(score)

#     # Press 'q' to exit the loop and stop capturing frames
#     # Pressing 'q' will close the camera window and stop the program.
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#         break

# # Plot the eye status and FPS graph
# # Plotting the eye status and frames per second (FPS) over time using Matplotlib.
# fig, ax1 = plt.subplots()

# ax1.plot(time_values, status_values, label='Eye Status', color='blue')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Eye Status (Score)', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Create a twin Axes sharing the x-axis
# # Adding another Y-axis for plotting FPS.
# ax2 = ax1.twinx()
# ax2.plot(time_values, fps_values, label='FPS', color='red')
# ax2.set_ylabel('Frames per Second (FPS)', color='red')
# ax2.tick_params(axis='y', labelcolor='red')

# # Adding legends and title to the graph
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# plt.title('Eye Status and FPS Over Time')
# plt.tight_layout()

# # Save the graph as a JPG image
# plt.savefig('eye_status_and_fps_graph.jpg')

# # Show the graph
# plt.show()

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
