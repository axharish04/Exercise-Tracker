import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from collections import deque


class BicepCurlFormCorrection:
    def __init__(self, reference_folder, output_csv="live_landmarks2.csv", threshold=0.15):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # Change pose detection configuration to detect multiple people
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Use a more complex model for better accuracy
            smooth_landmarks=True  # Enable smoothing
        )

        self.reference_landmarks = self.load_all_reference_landmarks(reference_folder)
        self.threshold = threshold
        self.body_parts = {11: "left shoulder", 12: "right shoulder",
                           13: "left elbow", 14: "right elbow",
                           15: "left wrist", 16: "right wrist"}

        self.output_csv = output_csv
        self.landmark_data = []

        # Tracking motion state & reps
        self.rep_count = 0
        self.curl_state = "down"
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
        self.perfect_rep_count = 0

        # Maximum error threshold for perfect rep - NEW
        self.perfect_rep_error_threshold = 0.5  # More lenient threshold for perfect reps
        self.error_weights = {  # Different weights for different errors - NEW
            "elbow_position": 1.0,
            "shoulder_stability": 1.0,
            "wrist_alignment": 0.8,  # Slightly less impact on overall score
        }

        # Form tracking with improved thresholds - IMPROVED
        self.elbow_drift = False
        self.shoulder_elevation = False
        self.wrist_alignment = False
        self.current_form_issues = set()
        self.form_issues_history = {}

        # Tracking continuous frames with good form - NEW
        self.good_form_streak = 0
        self.streak_threshold = 10  # Frames of good form needed for bonus

        # Speed tracking
        self.rep_start_time = None
        self.rep_times = []
        self.ideal_rep_time = 4.0  # Ideal time for one rep in seconds
        self.tempo_feedback = ""

        # Improved tempo scoring - NEW
        self.tempo_score = 100.0

        # Feedback management - IMPROVED FOR BETTER VISIBILITY
        self.feedback = ""
        self.form_feedback = ""
        self.accuracy_feedback = ""
        self.feedback_display_time = 0
        self.feedback_duration = 2  # Increased from 1 to 2 seconds for better readability
        self.feedback_priority = ["elbow_position", "shoulder_stability", "wrist_alignment", "tempo"]

        # Add feedback queue to display multiple feedback messages
        self.feedback_queue = deque(maxlen=3)
        self.last_rep_feedback = ""  # Store last rep feedback separately

        # FIXED: Make feedback shorter to fit in frame
        self.max_feedback_length = 50  # Maximum length of feedback text

        # Advanced metrics
        self.range_of_motion = 0.0  # Track full range of motion
        self.max_range_in_rep = 0.0  # Track maximum range in current rep - NEW
        self.consistency_score = 0.0  # Track consistency between reps

        # Track the position of the wrist through the curl
        self.wrist_positions = []
        self.position_history = deque(maxlen=20)  # Store recent wrist positions

        # Motion smoothness tracking - NEW
        self.motion_smoothness = 100.0  # Start with perfect score
        self.prev_positions = []

        # NEW: Person tracking attributes
        self.main_person_initialized = False
        self.main_person_center = None  # Store the center position of the main person
        self.main_person_box = None  # Store the bounding box of the main person
        self.tracking_history = deque(maxlen=30)  # Store recent positions for tracking
        self.tracking_tolerance = 0.15  # Tolerance for movement between frames
        self.frames_since_valid_detection = 0
        self.max_frames_to_keep_tracking = 60  # Maximum frames to keep tracking without valid detection

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

    def calculate_person_center(self, landmarks):
        """Calculate the center position of a person based on their landmarks"""
        if not landmarks:
            return None

        # Use key body points to determine the person's center (shoulders, hips)
        key_points = [11, 12, 23, 24]  # Left/right shoulders and hips
        valid_points = []

        for point in key_points:
            if point < len(landmarks):
                valid_points.append((landmarks[point]['x'], landmarks[point]['y']))

        if not valid_points:
            return None

        # Calculate the average position
        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)

        return (center_x, center_y)

    def calculate_person_box(self, landmarks):
        """Calculate a bounding box around the person"""
        if not landmarks:
            return None

        # Get all x,y coordinates
        x_coords = [landmark['x'] for landmark in landmarks if 'x' in landmark]
        y_coords = [landmark['y'] for landmark in landmarks if 'y' in landmark]

        if not x_coords or not y_coords:
            return None

        # Calculate bounding box with some margin
        margin = 0.05
        min_x = min(x_coords) - margin
        max_x = max(x_coords) + margin
        min_y = min(y_coords) - margin
        max_y = max(y_coords) + margin

        return (min_x, min_y, max_x, max_y)

    def is_same_person(self, current_center, current_box):
        """Determine if the detected person is the same as our main tracked person"""
        if not self.main_person_center or not self.main_person_box or not current_center or not current_box:
            return False

        # Calculate distance between centers
        distance = np.sqrt((current_center[0] - self.main_person_center[0]) ** 2 +
                           (current_center[1] - self.main_person_center[1]) ** 2)

        # If the center has moved too much, it might be a different person
        if distance > self.tracking_tolerance:
            # Check overlap between bounding boxes as a secondary metric
            old_box = self.main_person_box
            new_box = current_box

            # Calculate overlap
            x_overlap = max(0, min(old_box[2], new_box[2]) - max(old_box[0], new_box[0]))
            y_overlap = max(0, min(old_box[3], new_box[3]) - max(old_box[1], new_box[1]))

            overlap_area = x_overlap * y_overlap
            old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
            new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

            # If there's sufficient overlap, still consider it the same person
            overlap_ratio = overlap_area / min(old_area, new_area)
            return overlap_ratio > 0.5

        return True

    def update_tracking(self, landmarks):
        """Update the tracking information for the main person"""
        if not landmarks:
            self.frames_since_valid_detection += 1
            return

        current_center = self.calculate_person_center(landmarks)
        current_box = self.calculate_person_box(landmarks)

        if not current_center or not current_box:
            self.frames_since_valid_detection += 1
            return

        # Initialize tracking if this is the first detection
        if not self.main_person_initialized:
            self.main_person_center = current_center
            self.main_person_box = current_box
            self.main_person_initialized = True
            self.frames_since_valid_detection = 0
            self.tracking_history.append(current_center)
            return True

        # Check if this is the same person
        if self.is_same_person(current_center, current_box):
            # Update tracking info
            self.main_person_center = current_center
            self.main_person_box = current_box
            self.frames_since_valid_detection = 0
            self.tracking_history.append(current_center)
            return True
        else:
            # Not the main person
            self.frames_since_valid_detection += 1
            return False

    def get_landmark_coordinates(self, results):
        if not results.pose_landmarks:
            return None

        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z if hasattr(lm, 'z') else 0}
                     for lm in results.pose_landmarks.landmark]

        # Update person tracking
        is_main_person = self.update_tracking(landmarks)

        # Only return landmarks if this is our main person
        if is_main_person:
            return landmarks
        elif self.frames_since_valid_detection > self.max_frames_to_keep_tracking:
            # If we've lost tracking for too long, reset tracking
            # This allows reacquiring the person if they left and came back
            self.main_person_initialized = False
            return None
        else:
            # Not the main person, don't process these landmarks
            return None

    def compute_landmark_distances(self, current_landmarks, reference_landmarks):
        distances = {}
        key_landmarks = [11, 12, 13, 14, 15, 16]
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
        c = np.array([c['x'], c['y']])  # Fixed: Changed 'c['0']' to 'c['y']'

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

        # Check elbow position - IMPROVED with more lenient threshold
        elbow_z = current_landmarks[13].get('z', 0)
        if abs(elbow_z) > 0.15:  # More lenient threshold (was 0.1)
            self.elbow_drift = True
            self.current_form_issues.add("elbow_position")
            self.form_issues_history["elbow_position"] = self.form_issues_history.get("elbow_position", 0) + 1
        else:
            self.elbow_drift = False

        # Check shoulder elevation - IMPROVED with better detection
        shoulder_wrist_vertical_diff = abs(current_landmarks[11]['y'] - current_landmarks[15]['y'])
        shoulder_elevation_threshold = 0.12  # More lenient (was 0.1)
        if shoulder_wrist_vertical_diff < shoulder_elevation_threshold and self.curl_state == "up":
            # Shoulder is elevated too much
            self.shoulder_elevation = True
            self.current_form_issues.add("shoulder_stability")
            self.form_issues_history["shoulder_stability"] = self.form_issues_history.get("shoulder_stability", 0) + 1
        else:
            self.shoulder_elevation = False

        # Check wrist alignment with forearm - IMPROVED
        wrist = current_landmarks[15]
        elbow = current_landmarks[13]
        shoulder = current_landmarks[11]

        # Calculate elbow angle with correct coordinates
        elbow_angle = self.calculate_angle(
            {'x': shoulder['x'], 'y': shoulder['y']},
            {'x': elbow['x'], 'y': elbow['y']},
            {'x': wrist['x'], 'y': wrist['y']}
        )

        # Track range of motion - IMPROVED
        current_rom = 180 - elbow_angle  # Range increases as elbow bends
        self.range_of_motion = current_rom
        self.max_range_in_rep = max(self.max_range_in_rep, current_rom)

        # Check if wrist is aligned with forearm - MORE LENIENT
        wrist_forearm_angle = abs(90 - elbow_angle)  # Should be approximately 90 degrees
        if wrist_forearm_angle > 20:  # More lenient threshold (was 15)
            self.wrist_alignment = True
            self.current_form_issues.add("wrist_alignment")
            self.form_issues_history["wrist_alignment"] = self.form_issues_history.get("wrist_alignment", 0) + 1
        else:
            self.wrist_alignment = False

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

        # Generate specific form feedback based on issues
        if self.current_form_issues:
            # Sort issues by priority
            prioritized_issues = sorted(self.current_form_issues,
                                        key=lambda x: self.feedback_priority.index(
                                            x) if x in self.feedback_priority else 999)

            # FIXED: Shorter feedback messages
            if "elbow_position" in prioritized_issues:
                self.form_feedback = "Keep elbow close to body"
            elif "shoulder_stability" in prioritized_issues:
                self.form_feedback = "Keep shoulders down"
            elif "wrist_alignment" in prioritized_issues:
                self.form_feedback = "Keep wrists straight"

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
        # Max possible weighted error is the sum of all error weights (roughly 3)
        max_possible_error = sum(self.error_weights.values())
        base_accuracy = max(0, 100 - (avg_weighted_error / max_possible_error * 80))  # Only reduce by 80% max

        # Bonus for consistent good form - NEW
        if self.good_form_streak >= self.streak_threshold:
            base_accuracy = min(100, base_accuracy + 5)  # Bonus for good form streak

        # Factor in range of motion - IMPROVED
        ideal_rom = 120  # More realistic ~120 degrees (was 140)
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
        return max(60, round(final_accuracy, 1))  # Minimum score of 60% - NEW

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

    def detect_curl_position(self, landmarks):
        """Improved curl position detection with fixed perfect rep counting"""
        if not landmarks:
            return

        wrist_y = landmarks[15]['y']
        elbow_y = landmarks[13]['y']
        shoulder_y = landmarks[11]['y']

        # Store wrist position for movement tracking
        self.position_history.append((wrist_y, elbow_y))

        # Calculate movement smoothness - NEW
        if len(self.position_history) >= 3:
            # Calculate derivatives of movement
            positions = list(self.position_history)[-3:]
            deltas = [abs(positions[i + 1][0] - positions[i][0]) for i in range(len(positions) - 1)]
            # Check for jerkiness (large changes in deltas)
            if len(deltas) > 1 and max(deltas) > 3 * min(deltas) + 0.05:
                self.motion_smoothness = max(70, self.motion_smoothness - 2)  # Reduce smoothness score
            else:
                self.motion_smoothness = min(100, self.motion_smoothness + 0.5)  # Gradually recover

        # Detect curl position with improved logic
        new_state = self.curl_state  # Default: keep current state

        # Check for up position (wrist is significantly above elbow) - IMPROVED
        if wrist_y < elbow_y - 0.08:  # More lenient threshold (was 0.1)
            potential_state = "up"
        # Check for down position (wrist is below shoulder) - IMPROVED
        elif wrist_y > shoulder_y - 0.05:  # More lenient threshold (was exact comparison)
            potential_state = "down"
        else:
            # In between states, maintain current state
            potential_state = self.curl_state

        # State change logic with stability check to prevent flickering
        if potential_state != self.curl_state:
            self.state_stability_count += 1
            if self.state_stability_count >= self.min_state_stability:
                # State change is stable enough to switch
                new_state = potential_state
                self.state_stability_count = 0
        else:
            # Reset stability counter if we're not detecting a change
            self.state_stability_count = 0

        # Detect UP to DOWN transition (completing a rep)
        if self.curl_state == "up" and new_state == "down" and not self.rep_counted_in_cycle:
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

            # FIXED: Perfect rep counting with proper 90% threshold
            # This will properly count perfect reps
            if rep_accuracy >= 90:  # More lenient threshold for perfect reps
                self.perfect_rep_count += 1
                # Add feedback about perfect rep - SHORTENED
                perfect_feedback = f"Perfect rep! ({rep_accuracy:.1f}%)"
                self.feedback_queue.append(perfect_feedback)

            # Generate feedback for this rep - SHORTENED
            self.last_rep_feedback = f"Rep #{self.rep_count}: {rep_accuracy:.1f}%"
            if rep_accuracy >= 90:
                self.last_rep_feedback += " | Perfect!"
            elif rep_accuracy >= 80:
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
        elif self.curl_state == "down" and new_state == "up":
            # Reset the rep counting flag when starting a new curl
            self.rep_counted_in_cycle = False
            # Start timing when starting a new rep
            self.rep_start_time = time.time()

        # Update state
        self.curl_state = new_state

    def save_landmarks_to_csv(self):
        if self.landmark_data:
            df = pd.DataFrame(self.landmark_data)
            df.to_csv(self.output_csv, index=False)
            print(f"Saved landmark data to {self.output_csv}")

    def generate_dynamic_feedback(self):
        """Generate comprehensive dynamic feedback based on performance - SHORTENED"""
        # If we have recently completed a rep, prioritize that feedback
        if self.last_rep_feedback:
            return self.last_rep_feedback

        # If there are queued feedback messages, return the next one
        if self.feedback_queue:
            return self.feedback_queue.popleft()

        # During curl, prioritize form feedback
        if self.form_feedback:
            return self.form_feedback

        # If they've been doing well consistently - SHORTENED
        if self.perfect_rep_count > 0 and self.perfect_rep_count >= self.rep_count / 2:
            return f"{self.perfect_rep_count} perfect reps so far!"

        # If tempo is an issue
        if self.tempo_feedback and "Great" not in self.tempo_feedback:
            return self.tempo_feedback

        # Default form cues based on curl position - SHORTENED
        if self.curl_state == "up":
            return "Control the movement"
        else:
            return "Lower weight slowly"

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture video from webcam.")
                break

            # Flip the image horizontally for a selfie-view display
            # CHANGE: Moved the flip to the beginning so all processing is done on the flipped image
            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # Draw landmarks only for the main person
            current_landmarks = None
            if results.pose_landmarks:
                current_landmarks = self.get_landmark_coordinates(results)

                # Only draw landmarks if they belong to our tracked person
                if current_landmarks:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # Visualize person tracking with a bounding box
                    if self.main_person_initialized and self.main_person_box:
                        h, w, _ = image.shape
                        box = self.main_person_box
                        pt1 = (int(box[0] * w), int(box[1] * h))
                        pt2 = (int(box[2] * w), int(box[3] * h))
                        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)  # Green box for tracked person

            # Process feedback and form analysis
            if time.time() - self.feedback_display_time > self.feedback_duration:
                if current_landmarks:
                    self.analyze_form(current_landmarks)
                    self.feedback = self.generate_dynamic_feedback()
                    self.feedback_display_time = time.time()
                    # Reset last rep feedback after displaying it once
                    if self.feedback == self.last_rep_feedback:
                        self.last_rep_feedback = ""

            # Process exercise tracking for the main person only
            if current_landmarks:
                self.detect_curl_position(current_landmarks)
                frame_data = {'frame': len(self.landmark_data)}
                for idx in self.body_parts.keys():
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
            metrics_height = 160 if self.rep_count > 0 else 80
            cv2.rectangle(overlay, (0, 0), (image.shape[1], metrics_height), metrics_bg_color, -1)
            cv2.addWeighted(overlay, metrics_bg_alpha, image, 1 - metrics_bg_alpha, 0, image)

            # Show rep count and form status
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f"Reps: {self.rep_count}", (20, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show current position (up/down)
            position_text = f"Position: {self.curl_state.upper()}"
            cv2.putText(image, position_text, (20, 70), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Show accuracy metrics if we have completed reps
            if self.rep_count > 0:
                cv2.putText(image, self.accuracy_feedback, (20, 110), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                # Show perfect rep count if we have any
                if self.perfect_rep_count > 0:
                    perfect_text = f"Perfect reps: {self.perfect_rep_count}/{self.rep_count}"
                    cv2.putText(image, perfect_text, (20, 150), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Display current feedback in a more visible way - contrasting background
            if self.feedback:
                # Truncate long feedback
                display_feedback = self.feedback
                if len(display_feedback) > self.max_feedback_length:
                    display_feedback = display_feedback[:self.max_feedback_length] + "..."

                # Create feedback background
                feedback_y = image.shape[0] - 60
                feedback_bg = image.copy()
                cv2.rectangle(feedback_bg, (0, feedback_y - 20), (image.shape[1], feedback_y + 40),
                              (0, 0, 100), -1)  # Dark blue background
                cv2.addWeighted(feedback_bg, 0.7, image, 0.3, 0, image)

                # Draw feedback text
                cv2.putText(image, display_feedback, (20, feedback_y + 10), font, 0.8,
                            (255, 255, 0), 2, cv2.LINE_AA)  # Yellow text

            # CHANGE: Removed the flip at the end as we're now flipping at the beginning
            cv2.imshow('Bicep Curl Form Analysis', image)

            # CHANGE: Use 'q' to exit instead of ESC
            if cv2.waitKey(5) & 0xFF == ord('q'):  # Press q to exit
                break

        self.save_landmarks_to_csv()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
        # Change this to the folder containing your reference CSVs
        reference_folder = r"D:\aravind\A 3RD YEAR STUFF\ivp proj sem 6\fitfreak\data\landmarks\dumbbell_biceps_curl"
        app = BicepCurlFormCorrection(reference_folder)
        try:
            app.run()
        except KeyboardInterrupt:
            print("Program terminated by user.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Ensure CSV is saved even if program crashes
            app.save_landmarks_to_csv()
            print("Landmark data saved. Program ended.")