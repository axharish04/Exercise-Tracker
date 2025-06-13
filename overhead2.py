import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time


class ShoulderPressFormCorrection:
    def __init__(self, reference_folder, output_csv="shoulder_press_landmarks.csv", threshold=0.3):
        # MediaPipe initialization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # State thresholds (defined FIRST)
        self.state_change_threshold = 160  # Angle threshold for full extension
        self.down_state_threshold = 70  # Angle threshold for down position

        # Reference data loading (uses thresholds above)
        self.reference_landmarks_up, self.reference_landmarks_down = self.load_all_reference_landmarks(reference_folder)
        self.avg_up_landmarks = self.compute_average_landmarks(self.reference_landmarks_up)
        self.avg_down_landmarks = self.compute_average_landmarks(self.reference_landmarks_down)

        # Configuration
        self.threshold = threshold
        self.body_parts = {
            11: "left shoulder", 12: "right shoulder",
            13: "left elbow", 14: "right elbow",
            15: "left wrist", 16: "right wrist",
            23: "left hip", 24: "right hip"
        }

        # Data collection
        self.output_csv = output_csv
        self.landmark_data = []

        # Exercise state tracking
        self.rep_count = 0
        self.press_state = "down"
        self.motion_complete = False

        # Rep accuracy tracking
        self.rep_scores = []
        self.current_rep_errors = []
        self.tracking_errors = False

        # Feedback system
        self.feedback = ""
        self.feedback_display_time = 0
        self.feedback_duration = 1  # Seconds between feedback updates

        # Person tracking
        self.main_person_id = None
        self.confidence_threshold = 0.5
        self.person_lost_frames = 0
        self.max_lost_frames = 30

    def load_all_reference_landmarks(self, folder_path):
        reference_landmarks_up = []
        reference_landmarks_down = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Reference folder not found: {folder_path}")

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                path = os.path.join(folder_path, file)
                try:
                    frames = self.load_reference_landmarks(path)
                    for frame in frames:
                        # Use class-defined thresholds for classification
                        left_angle = self.calculate_angle(frame[13], frame[11], frame[23])
                        right_angle = self.calculate_angle(frame[14], frame[12], frame[24])
                        avg_angle = (left_angle + right_angle) / 2

                        if avg_angle > self.state_change_threshold:
                            reference_landmarks_up.append(frame)
                        elif avg_angle < self.down_state_threshold:
                            reference_landmarks_down.append(frame)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")

        return reference_landmarks_up, reference_landmarks_down

    def load_reference_landmarks(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            frames = []
            for frame_idx in range(len(df)):
                frame_data = df.iloc[frame_idx]
                landmarks = []
                for i in range(33):
                    landmarks.append({
                        'x': frame_data.get(f'x_{i}', 0.0),
                        'y': frame_data.get(f'y_{i}', 0.0)
                    })
                frames.append(landmarks)
            return frames
        except Exception as e:
            print(f"Error parsing {csv_path}: {str(e)}")
            return []

    def compute_average_landmarks(self, frames):
        if not frames:
            return None

        avg_landmarks = [{'x': 0.0, 'y': 0.0} for _ in range(33)]
        num_frames = len(frames)

        for frame in frames:
            for idx, lm in enumerate(frame):
                avg_landmarks[idx]['x'] += lm['x']
                avg_landmarks[idx]['y'] += lm['y']

        for idx in range(33):
            avg_landmarks[idx]['x'] /= num_frames
            avg_landmarks[idx]['y'] /= num_frames

        return avg_landmarks

    def get_landmark_coordinates(self, results):
        if not results.pose_landmarks:
            return None
        return [{'x': lm.x, 'y': lm.y} for lm in results.pose_landmarks.landmark]

    def select_main_person(self, results, image_shape):
        if not results.pose_landmarks:
            self.person_lost_frames += 1
            if self.person_lost_frames > self.max_lost_frames:
                self.main_person_id = None  # Reset if person is lost for too long
            return None, None

        # Reset lost frame counter
        self.person_lost_frames = 0

        # Get the bounding box of the person
        h, w, _ = image_shape
        landmarks = results.pose_landmarks.landmark

        # Calculate visibility score as the average visibility of key landmarks
        key_landmarks = [0, 11, 12, 23, 24]  # nose, shoulders, hips
        visibility = sum(landmarks[i].visibility for i in key_landmarks) / len(key_landmarks)

        if visibility < self.confidence_threshold:
            return None, None

        # Get bounding box coordinates
        x_coordinates = [landmark.x for landmark in landmarks]
        y_coordinates = [landmark.y for landmark in landmarks]

        x_min = int(min(x_coordinates) * w)
        y_min = int(min(y_coordinates) * h)
        x_max = int(max(x_coordinates) * w)
        y_max = int(max(y_coordinates) * h)

        # Add padding to the bounding box
        padding = int(0.1 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        return (x_min, y_min, x_max, y_max), visibility

    def calculate_angle(self, a, b, c):
        a = np.array([a['x'], a['y']])
        b = np.array([b['x'], b['y']])
        c = np.array([c['x'], c['y']])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle < 180 else 360 - angle

    def compute_reference_distance(self, current_landmarks, reference_landmarks):
        if reference_landmarks is None:
            return 0.0

        total_distance = 0.0
        count = 0

        for idx in self.body_parts.keys():
            current_lm = current_landmarks[idx]
            ref_lm = reference_landmarks[idx]
            dx = current_lm['x'] - ref_lm['x']
            dy = current_lm['y'] - ref_lm['y']
            total_distance += np.sqrt(dx ** 2 + dy ** 2)
            count += 1

        return total_distance / count if count > 0 else 0.0

    def analyze_form(self, current_landmarks):
        if not current_landmarks:
            return "No pose detected", []

        feedback_messages = []
        errors = []

        # Basic form checks
        left_angle = self.calculate_angle(current_landmarks[13], current_landmarks[11], current_landmarks[23])
        right_angle = self.calculate_angle(current_landmarks[14], current_landmarks[12], current_landmarks[24])

        if left_angle < 170 or right_angle < 170:
            feedback_messages.append("Fully extend arms overhead")
            errors.append("incomplete_extension")

        if (current_landmarks[11]['y'] < current_landmarks[13]['y'] or
                current_landmarks[12]['y'] < current_landmarks[14]['y']):
            feedback_messages.append("Keep shoulders down")
            errors.append("raised_shoulders")

        shoulder_center = (current_landmarks[11]['y'] + current_landmarks[12]['y']) / 2
        hip_center = (current_landmarks[23]['y'] + current_landmarks[24]['y']) / 2
        if abs(shoulder_center - hip_center) > 0.40:
            feedback_messages.append("Engage core, don't arch back")
            errors.append("back_arching")

        # Reference-based feedback
        try:
            if self.press_state == "up" and self.avg_up_landmarks:
                distance = self.compute_reference_distance(current_landmarks, self.avg_up_landmarks)
            elif self.press_state == "down" and self.avg_down_landmarks:
                distance = self.compute_reference_distance(current_landmarks, self.avg_down_landmarks)
            else:
                distance = 0.0

            if distance > self.threshold:
                feedback_messages.append("Adjust form to match reference")
                errors.append("reference_mismatch")
        except Exception as e:
            print(f"Reference comparison error: {str(e)}")

        # Track errors for current rep
        if self.tracking_errors and errors:
            for error in errors:
                if error not in self.current_rep_errors:
                    self.current_rep_errors.append(error)

        return ", ".join(feedback_messages) if feedback_messages else "Good form!", errors

    def detect_press_position(self, landmarks):
        if not landmarks:
            return

        left_angle = self.calculate_angle(landmarks[13], landmarks[11], landmarks[23])
        right_angle = self.calculate_angle(landmarks[14], landmarks[12], landmarks[24])
        avg_angle = (left_angle + right_angle) / 2

        previous_state = self.press_state

        if avg_angle > self.state_change_threshold:
            if self.press_state == "down":
                self.press_state = "up"
                self.motion_complete = False
                # Start tracking errors for this rep
                self.tracking_errors = True
                self.current_rep_errors = []
        elif avg_angle < self.down_state_threshold:
            if self.press_state == "up":
                self.press_state = "down"
                if not self.motion_complete:
                    self.rep_count += 1
                    self.motion_complete = True
                    # Calculate accuracy for completed rep
                    self.calculate_rep_accuracy()
                    # Reset tracking
                    self.tracking_errors = False

    def calculate_rep_accuracy(self):
        # Define possible errors and their impact on accuracy
        error_weights = {
            "incomplete_extension": 25,
            "raised_shoulders": 20,
            "back_arching": 30,
            "reference_mismatch": 25
        }

        # Start with perfect score
        score = 100

        # Deduct points for each unique error type
        for error in self.current_rep_errors:
            if error in error_weights:
                score -= error_weights[error]

        # Ensure score doesn't go below 0
        score = max(0, score)

        # Add to scores list
        self.rep_scores.append(score)

        # Reset current rep errors
        self.current_rep_errors = []

    def get_average_accuracy(self):
        if not self.rep_scores:
            return 0
        return sum(self.rep_scores) / len(self.rep_scores)

    def get_last_rep_accuracy(self):
        if not self.rep_scores:
            return 0
        return self.rep_scores[-1]

    def save_landmarks_to_csv(self):
        if self.landmark_data:
            try:
                df = pd.DataFrame(self.landmark_data)
                df.to_csv(self.output_csv, index=False)
                print(f"Data saved to {self.output_csv}")
            except Exception as e:
                print(f"Failed to save data: {str(e)}")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Mirror the image for user feedback
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                # Select main person and get bounding box
                bbox, visibility = self.select_main_person(results, image.shape)

                # Draw green bounding box around main person
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Add "Main Subject" label above the box
                    cv2.putText(image, "Main Subject", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                          circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                # Process landmarks
                current_landmarks = self.get_landmark_coordinates(results)

                # Update feedback periodically
                if time.time() - self.feedback_display_time > self.feedback_duration:
                    if current_landmarks:
                        self.feedback, _ = self.analyze_form(current_landmarks)
                        self.feedback_display_time = time.time()

                # Track exercise state
                if current_landmarks:
                    self.detect_press_position(current_landmarks)

                    # Store landmarks for analysis
                    frame_data = {'frame': len(self.landmark_data)}
                    for idx in self.body_parts.keys():
                        frame_data[f'x_{idx}'] = current_landmarks[idx]['x']
                        frame_data[f'y_{idx}'] = current_landmarks[idx]['y']
                    self.landmark_data.append(frame_data)

                # Display UI elements
                cv2.putText(image, f"Press Count: {self.rep_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (146, 67, 123), 2)
                cv2.putText(image, f"Position: {self.press_state.upper()}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (138,45 , 100), 2)

                # Display accuracy metrics
                last_rep_accuracy = self.get_last_rep_accuracy()
                avg_accuracy = self.get_average_accuracy()
                cv2.putText(image, f"Last Rep: {last_rep_accuracy:.0f}%", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image, f"Avg Accuracy: {avg_accuracy:.0f}%", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (126, 235, 0), 2)

                # Display feedback
                feedback_y_position = 190
                feedback_lines = self.feedback.split(", ")
                for line in feedback_lines:
                    cv2.putText(image, line, (10, feedback_y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 138, 230), 2)
                    feedback_y_position += 30

                # Display help text
                cv2.putText(image, "Press 'q' to quit", (image.shape[1] - 150, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Shoulder Press Form Correction', image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_landmarks_to_csv()


if __name__ == "__main__":
    reference_folder = r"D:\aravind\A 3RD YEAR STUFF\ivp proj sem 6\fitfreak\data\landmarks\dumbbell_overhead_shoulder_press"

    try:
        form_corrector = ShoulderPressFormCorrection(reference_folder)
        form_corrector.run()
    except Exception as e:
        print(f"Application failed: {str(e)}")