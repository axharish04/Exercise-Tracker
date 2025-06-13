import os
import cv2
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

base_dir = "data/5exercise"        # Path to the correct exercise folders
landmarks_dir = "data/landmarks"  # Output folder for landmarks
os.makedirs(landmarks_dir, exist_ok=True)

def extract_landmarks_to_csv(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    data = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            row = [frame_idx]
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            data.append(row)
        frame_idx += 1
    cap.release()

    # Save landmarks to CSV
    columns = ["frame"] + [f"{axis}_{i}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

# Process all videos in each exercise folder
for exercise in os.listdir(base_dir):
    exercise_path = os.path.join(base_dir, exercise)
    if os.path.isdir(exercise_path):
        for video_file in os.listdir(exercise_path):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(exercise_path, video_file)
                output_csv = os.path.join(landmarks_dir, f"{exercise}_{video_file.split('.')[0]}.csv")
                print(f"Processing {video_file} for {exercise}...")
                extract_landmarks_to_csv(video_path, output_csv)
print("Landmark extraction completed!")
