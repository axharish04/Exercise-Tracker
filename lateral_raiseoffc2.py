import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from collections import deque


class SideLateralRaiseFormCorrection:
    def __init__(self, reference_folder, output_csv="live_landmarks_lateral_raise2.csv", threshold=0.15):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.reference_landmarks = self.load_all_reference_landmarks(reference_folder)
        self.threshold = threshold
        self.body_parts = {11: "left shoulder", 12: "right shoulder",
                           13: "left elbow", 14: "right elbow",
                           15: "left wrist", 16: "right wrist",
                           23: "left hip", 24: "right hip"}

        self.output_csv = output_csv
        self.landmark_data = []

        # Tracking motion state & reps
        self.rep_count = 0
        self.lateral_state = "down"
        self.state_change_threshold = 0.05  # Threshold for stable state change

        # Added state stability counters to prevent flickering
        self.state_stability_count = 0
        self.min_state_stability = 5  # Frames required to confirm state change

        # Track if a rep has been counted in the current up-down cycle
        self.rep_counted_in_cycle = False

        # Accuracy tracking - IMPROVED
        self.rep_accuracy = []  # Store accuracy for each rep
        self.current_rep_errors = []  # Store errors during current rep
        self.overall_accuracy = 0.0
        self.accuracy_history = deque(maxlen=10)  # Store recent accuracy scores
        self.performance_trend = "stable"  # Options: improving, declining, stable

        # Maximum error threshold removed (was for perfect reps)
        self.error_weights = {  # Different weights for different errors - NEW
            "arm_height": 1.0,
            "shoulder_stability": 1.0,
            "arm_extension": 0.8,  # Slightly less impact on overall score
            "elbow_bend": 0.8,
            "wrist_alignment": 0.7,
            "arm_balance": 0.8
        }

        # Form tracking with improved thresholds - IMPROVED
        self.shoulder_elevation = False
        self.arm_height_issues = False
        self.arm_extension_issues = False
        self.current_form_issues = set()
        self.form_issues_history = {}

        # Tracking continuous frames with good form - NEW
        self.good_form_streak = 0
        self.streak_threshold = 10  # Frames of good form needed for bonus

        # Speed tracking
        self.rep_start_time = None
        self.rep_times = []
        self.ideal_rep_time = 3.0  # Ideal time for one rep in seconds (slightly shorter than bicep curl)
        self.tempo_feedback = ""

        # Improved tempo scoring - NEW
        self.tempo_score = 100.0

        # Feedback management - IMPROVED FOR BETTER VISIBILITY
        self.feedback = ""
        self.form_feedback = ""
        self.accuracy_feedback = ""
        self.feedback_display_time = 0
        self.feedback_duration = 2  # Increased from 1 to 2 seconds for better readability
        self.feedback_priority = ["arm_height", "shoulder_stability", "arm_extension", "elbow_bend", "tempo"]

        # Add feedback queue to display multiple feedback messages
        self.feedback_queue = deque(maxlen=3)
        self.last_rep_feedback = ""  # Store last rep feedback separately

        # FIXED: Make feedback shorter to fit in frame
        self.max_feedback_length = 50  # Maximum length of feedback text

        # Advanced metrics
        self.range_of_motion = 0.0  # Track full range of motion
        self.max_range_in_rep = 0.0  # Track maximum range in current rep - NEW
        self.consistency_score = 0.0  # Track consistency between reps

        # Track the position of the wrist through the raise
        self.wrist_positions = []
        self.position_history = deque(maxlen=20)  # Store recent positions

        # Motion smoothness tracking - NEW
        self.motion_smoothness = 100.0  # Start with perfect score
        self.prev_positions = []

    def load_all_reference_landmarks(self, folder_path):
        reference_landmarks = []
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                path = os.path.join(folder_path, file)
                reference_landmarks.append(self.load_reference_landmarks(path))
        return reference_landmarks

    def load_reference_landmarks(self, csv_path):
        df = pd.read_csv(csv_path)
        frames = []
        for frame_idx in range(len(df)):
            frame_data = df.iloc[frame_idx]
            landmarks = []
            for i in range(33):
                if f'x_{i}' in frame_data and f'y_{i}' in frame_data:
                    landmarks.append({'x': frame_data[f'x_{i}'], 'y': frame_data[f'y_{i}']})
                else:
                    landmarks.append({'x': 0, 'y': 0})
            frames.append(landmarks)
        return frames

    def get_landmark_coordinates(self, results):
        if not results.pose_landmarks:
            return None
        return [{'x': lm.x, 'y': lm.y, 'z': lm.z if hasattr(lm, 'z') else 0} for lm in results.pose_landmarks.landmark]

    def compute_landmark_distances(self, current_landmarks, reference_landmarks):
        distances = {}
        key_landmarks = list(self.body_parts.keys())
        for idx in key_landmarks:
            if idx < len(current_landmarks) and idx < len(reference_landmarks):
                current = np.array([current_landmarks[idx]['x'], current_landmarks[idx]['y']])
                reference = np.array([reference_landmarks[idx]['x'], reference_landmarks[idx]['y']])
                distances[idx] = np.linalg.norm(current - reference)
        return distances

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a['x'], a['y']])
        b = np.array([b['x'], b['y']])
        c = np.array([c['x'], c['y']])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze_form(self, current_landmarks):
        if not current_landmarks:
            return "No pose detected"

        best_match_dist = float('inf')
        best_reference_distances = None

        # Find the closest matching reference frame
        for ref in self.reference_landmarks:
            for frame in ref:  # Iterate through frames in the reference
                distances = self.compute_landmark_distances(current_landmarks, frame)
                total_dist = sum(distances.values())
                if total_dist < best_match_dist:
                    best_match_dist = total_dist
                    best_reference_distances = distances

        # Track form issues
        previous_issues = self.current_form_issues.copy()
        self.current_form_issues = set()

        # Check arm position based on current state
        if self.lateral_state == "up":
            # ARM HEIGHT CHECK
            left_shoulder_y = current_landmarks[11]['y']
            right_shoulder_y = current_landmarks[12]['y']
            left_wrist_y = current_landmarks[15]['y']
            right_wrist_y = current_landmarks[16]['y']

            # More lenient arm height check - IMPROVED
            left_arm_height_diff = left_wrist_y - left_shoulder_y
            right_arm_height_diff = right_wrist_y - right_shoulder_y

            # Track range of motion - IMPROVED
            left_rom = abs(left_shoulder_y - left_wrist_y)
            right_rom = abs(right_shoulder_y - right_wrist_y)
            current_rom = max(left_rom, right_rom)
            self.range_of_motion = current_rom
            self.max_range_in_rep = max(self.max_range_in_rep, current_rom)

            # More lenient threshold
            if left_arm_height_diff > 0.07:  # More lenient (was 0.05)
                self.arm_height_issues = True
                self.current_form_issues.add("arm_height")
                self.form_issues_history["arm_height"] = self.form_issues_history.get("arm_height", 0) + 1
            elif right_arm_height_diff > 0.07:  # More lenient (was 0.05)
                self.arm_height_issues = True
                self.current_form_issues.add("arm_height")
                self.form_issues_history["arm_height"] = self.form_issues_history.get("arm_height", 0) + 1
            else:
                self.arm_height_issues = False

            # ARM EXTENSION CHECK - make sure arms are out to sides properly
            left_shoulder_x = current_landmarks[11]['x']
            right_shoulder_x = current_landmarks[12]['x']
            left_wrist_x = current_landmarks[15]['x']
            right_wrist_x = current_landmarks[16]['x']

            left_extension = abs(left_wrist_x - left_shoulder_x)
            right_extension = abs(right_wrist_x - right_shoulder_x)

            # More lenient threshold
            if min(left_extension, right_extension) < 0.12:  # More lenient (was 0.15)
                self.arm_extension_issues = True
                self.current_form_issues.add("arm_extension")
                self.form_issues_history["arm_extension"] = self.form_issues_history.get("arm_extension", 0) + 1
            else:
                self.arm_extension_issues = False

            # ELBOW BEND CHECK - elbows should be slightly bent
            left_elbow = self.calculate_angle(
                current_landmarks[11],  # left shoulder
                current_landmarks[13],  # left elbow
                current_landmarks[15]  # left wrist
            )
            right_elbow = self.calculate_angle(
                current_landmarks[12],  # right shoulder
                current_landmarks[14],  # right elbow
                current_landmarks[16]  # right wrist
            )

            # More lenient threshold
            if min(left_elbow, right_elbow) > 175:  # More lenient (was 170)
                self.current_form_issues.add("elbow_bend")
                self.form_issues_history["elbow_bend"] = self.form_issues_history.get("elbow_bend", 0) + 1

            # WRIST ALIGNMENT CHECK - wrists should be neutral, not bent
            # More lenient checks for wrist alignment
            if len(self.current_form_issues) < 2:  # Only check if we don't have too many issues already
                left_wrist_alignment = self.calculate_angle(
                    current_landmarks[13],  # left elbow
                    current_landmarks[15],  # left wrist
                    {'x': current_landmarks[15]['x'], 'y': current_landmarks[15]['y'] + 0.1}  # point below wrist
                )
                right_wrist_alignment = self.calculate_angle(
                    current_landmarks[14],  # right elbow
                    current_landmarks[16],  # right wrist
                    {'x': current_landmarks[16]['x'], 'y': current_landmarks[16]['y'] + 0.1}  # point below wrist
                )

                # More lenient threshold
                if min(left_wrist_alignment, right_wrist_alignment) < 155:  # More lenient (was 160)
                    self.current_form_issues.add("wrist_alignment")
                    self.form_issues_history["wrist_alignment"] = self.form_issues_history.get("wrist_alignment", 0) + 1

            # ARM BALANCE CHECK - both arms should be at same level
            if abs(left_wrist_y - right_wrist_y) > 0.07 and len(
                    self.current_form_issues) < 2:  # More lenient (was 0.05)
                self.current_form_issues.add("arm_balance")
                self.form_issues_history["arm_balance"] = self.form_issues_history.get("arm_balance", 0) + 1

        # POSTURE CHECKS - apply to both up and down positions
        # Shoulder shrugging check - only if we don't have too many issues already
        if len(self.current_form_issues) < 2:
            neck_y = current_landmarks[0]['y'] if len(current_landmarks) > 0 else 0
            left_shoulder_y = current_landmarks[11]['y']
            right_shoulder_y = current_landmarks[12]['y']

            # More lenient threshold
            if (abs(neck_y - left_shoulder_y) < 0.07 or abs(
                    neck_y - right_shoulder_y) < 0.07):  # More lenient (was 0.05)
                self.shoulder_elevation = True
                self.current_form_issues.add("shoulder_stability")
                self.form_issues_history["shoulder_stability"] = self.form_issues_history.get("shoulder_stability",
                                                                                              0) + 1
            else:
                self.shoulder_elevation = False

        # Track good form streak - NEW
        if not self.current_form_issues:
            self.good_form_streak += 1
        else:
            self.good_form_streak = 0  # Reset streak on any form issue

        # Calculate weighted error score for this frame - NEW
        weighted_error = 0
        for issue in self.current_form_issues:
            weighted_error += self.error_weights.get(issue, 1.0)

        # Add weighted error to current rep tracking
        self.current_rep_errors.append(weighted_error)

        # IMPROVED: Generate specific form feedback based on issues
        if self.current_form_issues:
            # Sort issues by priority
            prioritized_issues = sorted(self.current_form_issues,
                                        key=lambda x: self.feedback_priority.index(
                                            x) if x in self.feedback_priority else 999)

            # FIXED: Shorter feedback messages
            if "arm_height" in prioritized_issues:
                self.form_feedback = "Raise arms to shoulder level"
            elif "shoulder_stability" in prioritized_issues:
                self.form_feedback = "Keep shoulders down"
            elif "arm_extension" in prioritized_issues:
                self.form_feedback = "Extend arms further out"
            elif "elbow_bend" in prioritized_issues:
                self.form_feedback = "Keep slight bend in elbows"
            elif "wrist_alignment" in prioritized_issues:
                self.form_feedback = "Keep wrists straight"
            elif "arm_balance" in prioritized_issues:
                self.form_feedback = "Keep arms at equal height"

            return self.form_feedback
        else:
            return "Good form!"

    def calculate_rep_accuracy(self):
        """Calculate accuracy for the completed rep - COMPLETELY REWORKED"""
        if not self.current_rep_errors:
            return 100.0

        # Calculate base accuracy using weighted average of form errors
        avg_weighted_error = sum(self.current_rep_errors) / len(self.current_rep_errors)

        # More lenient base accuracy calculation - NEW
        # Max possible weighted error is the sum of all error weights (roughly 5)
        max_possible_error = sum(self.error_weights.values())
        base_accuracy = max(0, 100 - (avg_weighted_error / max_possible_error * 80))  # Only reduce by 80% max

        # Bonus for consistent good form - NEW
        if self.good_form_streak >= self.streak_threshold:
            base_accuracy = min(100, base_accuracy + 5)  # Bonus for good form streak

        # Factor in range of motion - IMPROVED
        # For lateral raises, ideal ROM is less than bicep curls
        ideal_rom = 0.2  # Ideal difference between shoulder and wrist y position
        rom_factor = min(1.0, self.max_range_in_rep / ideal_rom)
        # Apply ROM factor with less penalty - NEW
        rom_accuracy = base_accuracy * (0.9 + 0.1 * rom_factor)  # Only reduce by 10% max

        # Factor in tempo - IMPROVED
        if self.rep_times:
            # Calculate tempo score (100% if perfect tempo, less if too fast/slow)
            rep_time = self.rep_times[-1]
            if rep_time < self.ideal_rep_time * 0.6:  # Too fast
                self.tempo_score = 85  # Less penalty (was multiplying by 0.8)
            elif rep_time > self.ideal_rep_time * 1.6:  # Too slow
                self.tempo_score = 90  # Even less penalty for being slow
            else:
                # Linearly scale tempo score between 90-100% for reasonable speeds
                deviation = abs(rep_time - self.ideal_rep_time) / self.ideal_rep_time
                self.tempo_score = 100 - (10 * min(1, deviation))

            # Apply tempo factor with less penalty
            final_accuracy = rom_accuracy * (0.9 + 0.1 * (self.tempo_score / 100))
        else:
            final_accuracy = rom_accuracy

        # Ensure we don't have artificially low scores
        return max(60, round(final_accuracy))  # Minimum score of 60% - NEW

    def update_performance_metrics(self):
        """Update overall performance metrics - IMPROVED"""
        # Calculate overall accuracy
        if self.rep_accuracy:
            # Use weighted average favoring recent reps - NEW
            if len(self.rep_accuracy) > 3:
                weights = np.linspace(0.5, 1.0, len(self.rep_accuracy))
                self.overall_accuracy = np.average(self.rep_accuracy, weights=weights)
            else:
                self.overall_accuracy = sum(self.rep_accuracy) / len(self.rep_accuracy)

            # Update accuracy history for trend analysis
            self.accuracy_history.append(self.overall_accuracy)

            # Determine performance trend
            if len(self.accuracy_history) >= 3:
                recent = list(self.accuracy_history)[-3:]
                if recent[2] > recent[0] + 5:
                    self.performance_trend = "improving"
                elif recent[2] < recent[0] - 5:
                    self.performance_trend = "declining"
                else:
                    self.performance_trend = "stable"

        # Calculate consistency score with less penalty - IMPROVED
        if len(self.rep_accuracy) >= 2:
            variance = np.std(self.rep_accuracy)
            self.consistency_score = max(0, 100 - variance * 0.8)  # Less penalty for variance

        # Generate accuracy feedback - SHORTENED FOR DISPLAY
        if self.rep_count > 0:
            self.accuracy_feedback = f"Accuracy: {self.overall_accuracy:.1f}%"

            # Add trend information - shortened
            if self.performance_trend == "improving":
                self.accuracy_feedback += " (Improving!)"
            elif self.performance_trend == "declining":
                self.accuracy_feedback += " (Focus on form)"

            # Add consistency information if enough reps - shortened
            if len(self.rep_accuracy) >= 3:
                if self.consistency_score > 85:
                    self.accuracy_feedback += " | Consistent!"
                elif self.consistency_score < 70:
                    self.accuracy_feedback += " | Be more consistent"

    def detect_lateral_raise_position(self, landmarks):
        """Improved lateral raise position detection with stability checking"""
        if not landmarks:
            return

        left_wrist_y = landmarks[15]['y']
        right_wrist_y = landmarks[16]['y']
        left_shoulder_y = landmarks[11]['y']
        right_shoulder_y = landmarks[12]['y']

        # Add current position to history for movement tracking
        self.position_history.append((left_wrist_y, right_wrist_y, left_shoulder_y, right_shoulder_y))

        # Calculate movement smoothness - NEW
        if len(self.position_history) >= 3:
            # Calculate derivatives of movement
            positions = list(self.position_history)[-3:]
            deltas_left = [abs(positions[i + 1][0] - positions[i][0]) for i in range(len(positions) - 1)]
            deltas_right = [abs(positions[i + 1][1] - positions[i][1]) for i in range(len(positions) - 1)]

            # Average of left and right movement
            deltas = [(deltas_left[i] + deltas_right[i]) / 2 for i in range(len(deltas_left))]

            # Check for jerkiness (large changes in deltas)
            if len(deltas) > 1 and max(deltas) > 3 * min(deltas) + 0.05:
                self.motion_smoothness = max(70, self.motion_smoothness - 2)  # Reduce smoothness score
            else:
                self.motion_smoothness = min(100, self.motion_smoothness + 0.5)  # Gradually recover

        # Detect lateral raise position with improved logic
        new_state = self.lateral_state  # Default: keep current state

        # Check for up position - wrists at shoulder level - IMPROVED
        # More lenient threshold allowing wrists to be slightly above or below shoulder level
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2

        # Check if arms are raised to approximately shoulder level
        if abs(avg_wrist_y - avg_shoulder_y) < 0.07:  # More lenient threshold
            potential_state = "up"
        # Check for down position (wrists are significantly below shoulders)
        elif avg_wrist_y > avg_shoulder_y + 0.15:  # More lenient threshold
            potential_state = "down"
        else:
            # In between states, maintain current state
            potential_state = self.lateral_state

        # State change logic with stability check to prevent flickering
        if potential_state != self.lateral_state:
            self.state_stability_count += 1
            if self.state_stability_count >= self.min_state_stability:
                # State change is stable enough to switch
                new_state = potential_state
                self.state_stability_count = 0
        else:
            # Reset stability counter if we're not detecting a change
            self.state_stability_count = 0

        # Detect UP to DOWN transition (completing a rep)
        if self.lateral_state == "up" and new_state == "down" and not self.rep_counted_in_cycle:
            # Count the rep only when returning to down position AND we haven't counted in this cycle
            self.rep_count += 1
            self.rep_counted_in_cycle = True

            # Calculate rep time
            if self.rep_start_time is not None:
                rep_time = time.time() - self.rep_start_time
                self.rep_times.append(rep_time)

                # Generate tempo feedback - SHORTENED
                if rep_time < self.ideal_rep_time * 0.7:
                    self.tempo_feedback = "Slow down"
                elif rep_time > self.ideal_rep_time * 1.5:
                    self.tempo_feedback = "Keep steady pace"
                else:
                    self.tempo_feedback = "Great tempo!"

            # Calculate accuracy for the completed rep
            rep_accuracy = self.calculate_rep_accuracy()
            self.rep_accuracy.append(rep_accuracy)

            # Generate feedback for this rep - SHORTENED
            self.last_rep_feedback = f"Rep #{self.rep_count}: {rep_accuracy:.1f}%"
            if rep_accuracy >= 80:
                self.last_rep_feedback += " | Great!"
            elif rep_accuracy < 70 and self.form_feedback:
                self.last_rep_feedback += f" | {self.form_feedback}"
                self.feedback_queue.append(self.form_feedback)

            # Update overall metrics
            self.update_performance_metrics()

            # Reset for next rep
            self.current_rep_errors = []
            self.max_range_in_rep = 0.0  # Reset maximum range for new rep - NEW

        # Detect DOWN to UP transition (starting a new rep cycle)
        elif self.lateral_state == "down" and new_state == "up":
            # Reset the rep counting flag when starting a new raise
            self.rep_counted_in_cycle = False
            # Start timing when starting a new rep
            self.rep_start_time = time.time()

        # Update state
        self.lateral_state = new_state

    def generate_dynamic_feedback(self):
        """Generate comprehensive dynamic feedback based on performance - SHORTENED"""
        # If we have recently completed a rep, prioritize that feedback
        if self.last_rep_feedback:
            return self.last_rep_feedback

        # If there are queued feedback messages, return the next one
        if self.feedback_queue:
            return self.feedback_queue.popleft()

        # During lateral raise, prioritize form feedback
        if self.form_feedback:
            return self.form_feedback

        # If tempo is an issue
        if self.tempo_feedback and "Great" not in self.tempo_feedback:
            return self.tempo_feedback

        # Default form cues based on lateral raise position - SHORTENED
        if self.lateral_state == "up":
            return "Hold briefly at the top"
        else:
            return "Control the movement"

    def save_landmarks_to_csv(self):
        if self.landmark_data:
            df = pd.DataFrame(self.landmark_data)
            df.to_csv(self.output_csv, index=False)
            print(f"Saved landmark data to {self.output_csv}")

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture video from webcam.")
                break

            # Flip the image horizontally for a more intuitive mirror view
            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # Draw pose landmarks on the image
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Draw bounding box around the person to ensure single person focus
                landmarks = results.pose_landmarks.landmark
                # Get x,y coordinates for all landmarks
                x_coordinates = [landmark.x for landmark in landmarks]
                y_coordinates = [landmark.y for landmark in landmarks]

                # Find min and max to create bounding box
                x_min = int(min(x_coordinates) * image.shape[1])
                x_max = int(max(x_coordinates) * image.shape[1])
                y_min = int(min(y_coordinates) * image.shape[0])
                y_max = int(max(y_coordinates) * image.shape[0])

                # Add some padding around the person
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)

                # Draw green bounding box around the person
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            current_landmarks = self.get_landmark_coordinates(results)

            if time.time() - self.feedback_display_time > self.feedback_duration:
                if current_landmarks:
                    self.analyze_form(current_landmarks)
                    self.feedback = self.generate_dynamic_feedback()
                    self.feedback_display_time = time.time()
                    # Reset last rep feedback after displaying it once
                    if self.feedback == self.last_rep_feedback:
                        self.last_rep_feedback = ""

            if current_landmarks:
                self.detect_lateral_raise_position(current_landmarks)
                frame_data = {'frame': len(self.landmark_data)}
                for idx in range(33):  # Store all landmarks for more complete data
                    if idx < len(current_landmarks):
                        frame_data[f'x_{idx}'] = current_landmarks[idx]['x']
                        frame_data[f'y_{idx}'] = current_landmarks[idx]['y']
                        frame_data[f'z_{idx}'] = current_landmarks[idx].get('z', 0)
                self.landmark_data.append(frame_data)

            # IMPROVED VISUAL DISPLAY - Better organization and visibility
            # Create a semi-transparent overlay for metrics
            overlay = image.copy()
            metrics_bg_color = (0, 0, 0)  # Black background
            metrics_bg_alpha = 0.6  # Semi-transparent

            # Draw background rectangle for metrics
            metrics_height = 120 if self.rep_count > 0 else 80  # Reduced height (was 160)
            cv2.rectangle(overlay, (0, 0), (300, metrics_height), metrics_bg_color, -1)
            cv2.addWeighted(overlay, metrics_bg_alpha, image, 1 - metrics_bg_alpha, 0, image)

            # Draw basic metrics with better visibility - MADE SMALLER
            cv2.putText(image, f"Reps: {self.rep_count}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display lateral raise state with different colors - MADE SMALLER
            state_color = (0, 255, 0) if self.lateral_state == "up" else (0, 200, 200)
            cv2.putText(image, f"Position: {self.lateral_state.upper()}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

            if self.rep_count > 0:
                # Display accuracy information - MADE SMALLER
                cv2.putText(image, f"Accuracy: {self.overall_accuracy:.1f}%", (15, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Display consistency score if we have enough reps
                if len(self.rep_accuracy) >= 3:
                    consistency_color = (0, 255, 0) if self.consistency_score > 85 else (0, 200, 255)
                    cv2.putText(image, f"Consistency: {self.consistency_score:.1f}%", (15, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, consistency_color, 2)

            # Create background for feedback text
            feedback_bg_height = 40
            overlay = image.copy()
            cv2.rectangle(overlay, (0, image.shape[0] - feedback_bg_height),
                          (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # Display dynamic feedback at the bottom of the screen
            if self.feedback:
                # Truncate feedback if too long to fit
                truncated_feedback = self.feedback
                if len(truncated_feedback) > self.max_feedback_length:
                    truncated_feedback = truncated_feedback[:self.max_feedback_length] + "..."

                cv2.putText(image, truncated_feedback, (15, image.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the image
            cv2.imshow('MediaPipe Pose - Side Lateral Raise Form Correction', image)

            # Press 'q' to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Clean up resources
        self.save_landmarks_to_csv()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
        reference_folder = r"D:\aravind\A 3RD YEAR STUFF\ivp proj sem 6\fitfreak\data\landmarks\side_lateral_raise"
        form_corrector = SideLateralRaiseFormCorrection(reference_folder)
        form_corrector.run()

