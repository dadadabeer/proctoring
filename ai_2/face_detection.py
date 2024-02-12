import cv2
import time
import random
import os

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

# Create a directory for snapshots if it doesn't exist
snapshots_dir = "snapshots"
if not os.path.exists(snapshots_dir):
    os.makedirs(snapshots_dir)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# Initialize the time tracker and set a random interval
start_time = time.time()
interval = random.randint(0, 20)

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    faces = detect_bounding_box(video_frame)
    face_count = len(faces)  # Define face_count here to ensure it's always defined

    # Check if the random interval has passed
    if (time.time() - start_time) > interval:
        print(f"Interval reached: {interval} seconds. Faces detected: {face_count}")

        # Take a snapshot if there are 0 or more than 2 faces detected at this interval
        if face_count == 0 or face_count > 2:
            snapshot_filename = os.path.join(snapshots_dir, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snapshot_filename, video_frame)
            print(f"Snapshot taken: {snapshot_filename}")

        # Reset the timer and set a new random interval
        start_time = time.time()
        interval = random.randint(0, 20)

    cv2.putText(video_frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("My Face Detection Project", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
