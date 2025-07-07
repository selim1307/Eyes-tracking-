import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import csv, os
import joblib
from datetime import datetime

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Landmarks
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Gaze smoothing
gaze_history = deque(maxlen=7)
gaze_counter = {"Left": 0, "Right": 0, "Center": 0}
YELLOW_THRESHOLD = 60   # ~2 seconds at 30 FPS
RED_THRESHOLD = 150     # ~5 seconds at 30 FPS

# Iris history
iris_history_left = deque(maxlen=60)
iris_history_right = deque(maxlen=60)

# Load model if exists
model = None
if os.path.exists("eye_anomaly_model.pkl"):
    model = joblib.load("eye_anomaly_model.pkl")

# Create CSV if not exists
if not os.path.exists("features_log.csv"):
    with open("features_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gaze",
            "mean_x_l", "mean_y_l", "std_x_l", "std_y_l", "maxdev_x_l", "maxdev_y_l",
            "mean_x_r", "mean_y_r", "std_x_r", "std_y_r", "maxdev_x_r", "maxdev_y_r",
            "anomaly"
        ])

def get_normalized_horizontal_ratio(eye_corner1, eye_corner2, iris_center):
    x1, x2 = eye_corner1[0], eye_corner2[0]
    x_iris = iris_center[0]
    return (x_iris - x1) / (x2 - x1)

def get_eye_points(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        mesh = result.multi_face_landmarks[0].landmark

        left_eye = get_eye_points(mesh, LEFT_EYE, w, h)
        right_eye = get_eye_points(mesh, RIGHT_EYE, w, h)
        left_iris = (int(mesh[LEFT_IRIS_CENTER].x * w), int(mesh[LEFT_IRIS_CENTER].y * h))
        right_iris = (int(mesh[RIGHT_IRIS_CENTER].x * w), int(mesh[RIGHT_IRIS_CENTER].y * h))

        # Gaze estimation
        left_ratio = get_normalized_horizontal_ratio(left_eye[0], left_eye[1], left_iris)
        right_ratio = get_normalized_horizontal_ratio(right_eye[0], right_eye[1], right_iris)
        avg_ratio = (left_ratio + right_ratio) / 2

        # Classify gaze direction
        if avg_ratio < 0.35:
            gaze = "Right"
        elif avg_ratio > 0.65:
            gaze = "Left"
        else:
            gaze = "Center"

        gaze_history.append(gaze)
        smoothed_gaze = Counter(gaze_history).most_common(1)[0][0]

        # Update counters
        for direction in gaze_counter:
            if direction == smoothed_gaze:
                gaze_counter[direction] += 1
            else:
                gaze_counter[direction] = 0

        # Draw eye landmarks
        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (255, 0, 0), -1)
        cv2.circle(frame, left_iris, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_iris, 3, (0, 255, 0), -1)

        # Show gaze text
        cv2.putText(frame, f"Gaze: {smoothed_gaze}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.putText(frame, f"Iris Ratio: {avg_ratio:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 100), 2)

        # Show alerts
        if gaze_counter["Left"] >= RED_THRESHOLD or gaze_counter["Right"] >= RED_THRESHOLD:
            cv2.putText(frame, "ALERT: Looked away too long!", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        elif gaze_counter["Left"] >= YELLOW_THRESHOLD or gaze_counter["Right"] >= YELLOW_THRESHOLD:
            cv2.putText(frame, "WARNING: Eyes off screen", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Feature extraction every 60 frames
        iris_history_left.append(left_iris)
        iris_history_right.append(right_iris)

        if len(iris_history_left) == 60:
            l = np.array(iris_history_left)
            r = np.array(iris_history_right)

            features = np.concatenate([
                np.mean(l, axis=0),
                np.std(l, axis=0),
                np.max(np.abs(l - np.mean(l, axis=0)), axis=0),
                np.mean(r, axis=0),
                np.std(r, axis=0),
                np.max(np.abs(r - np.mean(r, axis=0)), axis=0),
            ])

            anomaly = "Unknown"
            if model is not None:
                prediction = model.predict([features])
                if prediction[0] == -1:
                    anomaly = "Yes"
                    cv2.putText(frame, "ANOMALY DETECTED!", (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    anomaly = "No"

            with open("features_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    smoothed_gaze,
                    *features,
                    anomaly
                ])

    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
