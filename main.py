"""
Using Mediapipe's prebuilt model and functionality hand gestures can be recognized. The limitation of this is by using the default 
pre-built model, not all hand gestures desired are inbuilt. Further work can be done to use Mediapipe's online tools to develop a more in-depth model.

There are three (key) stages of detection occurring. Palm, hand landmarks, and gestures. Both hand landmarks and gestures are thresholded to ensure
an accurate capture is made. 

Another limitation of this solution is that currently, it can only detect one hand in the frame. It is still unclear if the final design will require 
or allow more hands but this is something to consider.
Author: John Elliott
Date:   27/03/2024
"""
import cv2
import numpy as np
import time
from home_assistant import light_service, load_envs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


"""
Given a set of key points for hand landmarks determine if a thumb up, thumb down, open palm, or closed fist is detected. 
This is done by calculating various angles between joints on the hand, and key points. If no gesture is detected None will be returned.

Parameters:
    hand_landmarks:   All key point hand landmark data.
"""
def gesture_calculation(hand_landmarks):
    gesture = "None"

    if len(hand_landmarks) > 0: # Checks if there is a hand detected.
        # Extract hand landmarks.
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks[0]]

        # Key hand landmarks.
        # Thumb.
        thumb_tip = landmarks[4]
        thumb_joint = landmarks[3]
        # Index.
        index_tip = landmarks[8]
        index_joint = landmarks[5]
        # Middle.
        middle_tip = landmarks[12]
        middle_joint = landmarks[9]
        # Ring.
        ring_tip = landmarks[16]
        ring_joint = landmarks[13]
        # Pinky.
        pinky_tip = landmarks[20]
        pinky_joint = landmarks[17]
        
        # Threshold to consider a finger as folded used to detect closed fist gesture.
        folded_threshold = 0.15 
        # Threshold to consider the angle for thumb detection about the palm wide enough to detect a thumb up or down gesture.
        thumb_up_threshold = 1.5
        # Threshold to consider a finger as at a far enough distance apart (from the thumb) used to detect open palm gesture.
        open_hand_threshold = 0.2

        # Calculate distances from tips to corresponding MCP joints
        index_tip_joint_distance = np.linalg.norm(np.array(index_tip) - np.array(index_joint))
        middle_tip_joint_distance = np.linalg.norm(np.array(middle_tip) - np.array(middle_joint))
        ring_tip_joint_distance = np.linalg.norm(np.array(ring_tip) - np.array(ring_joint))
        pinky_tip_joint_distance = np.linalg.norm(np.array(pinky_tip) - np.array(pinky_joint))

        # Thumb up and down gesture detection.
        # Calculate vectors for thumb and palm.
        thumb_vector = np.array(thumb_tip) - np.array(thumb_joint)
        palm_vector = np.array(pinky_joint) - np.array(index_joint)
        # Normalize vectors to ensure a focus on the direction and not magnitude.
        thumb_vector = thumb_vector / np.linalg.norm(thumb_vector)
        palm_vector = palm_vector / np.linalg.norm(palm_vector)
        # Calculate the angle between the thumb vector and the palm vector using the dot product.
        angle = np.arccos(np.dot(thumb_vector, palm_vector))

        # Check if the thumb is pointing up or down by looking at the position of the thumb tip and joint and if the remaining fingers are folded.
        if angle > thumb_up_threshold:
            if thumb_tip[1] < thumb_joint[1] and all(dist < folded_threshold for dist in [index_tip_joint_distance, middle_tip_joint_distance, ring_tip_joint_distance, pinky_tip_joint_distance]):
                gesture = "Thumb_Up"
            elif thumb_tip[1] > thumb_joint[1] and all(dist < folded_threshold for dist in [index_tip_joint_distance, middle_tip_joint_distance, ring_tip_joint_distance, pinky_tip_joint_distance]):
                gesture = "Thumb_Down"

        if gesture == "None":  # No gesture found yet. Close fist gestures can trigger thumb up/down detection so this stops it from overriding it. 
            # Closed fist gesture detection.
            # Must all be below a certain distance to be considered closed.
            if all(dist < folded_threshold for dist in [index_tip_joint_distance, middle_tip_joint_distance, ring_tip_joint_distance, pinky_tip_joint_distance]):
                gesture = "Closed_Fist"

            # Open palm gesture detection.
            # Calculate distances from each tip to the thumb tip.
            thumb_index_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            thumb_middle_distance = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))
            thumb_ring_distance = np.linalg.norm(np.array(thumb_tip) - np.array(ring_tip))
            thumb_pinky_distance = np.linalg.norm(np.array(thumb_tip) - np.array(pinky_tip))
            
            # Must all be over a certain distance to be considered open.
            if (thumb_index_distance > open_hand_threshold and
                thumb_middle_distance > open_hand_threshold and
                thumb_ring_distance > open_hand_threshold and
                thumb_pinky_distance > open_hand_threshold):
                gesture = "Open_Palm"

    return gesture


"""
Detects hand gestures using Mediapipe's prebuilt model using the GestureRecognizer object. This object is called for each frame captured by OpenCV-Python. 
The frame is then drawn on to show hand landmarks and an API call to Home Assistant is made to turn on or of`f a light based on the gesture recognized.

Parameters:
    calculated:   If True, the gesture_calculation() function will be called to determine the gesture. If False, the gesture will be determined by the prebuilt Mediapipe model.
    mock (bool): If True, the API call to Home Assistant will not be made. This is used for testing purposes. By default, this is set to False.
"""
def detection(calculated, mock=False):
    # Creates a gesture recognizer instance and loads the model asset and options. Key options include adding a threshold for hand and gesture classifier detection.
    base_options = python.BaseOptions(model_asset_path = 'gesture_recognizer.task') # Path to the model file.
    cannedGesturesClassifierOption = mp.tasks.components.processors.ClassifierOptions(score_threshold = 0.4)
    options = vision.GestureRecognizerOptions(base_options = base_options, num_hands = 1, min_hand_detection_confidence = 0.4, canned_gesture_classifier_options = cannedGesturesClassifierOption)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Creates Mediapipe tools that will be used to draw detected hand landmarks on a frame.
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Starts the webcam and captures frames.
    capture = cv2.VideoCapture(-1)

    # This value is used to determine the last state the lights were in. It is used to prevent the light from being turned on multiple times in a row.
    last_call = None
    # This value is used to maintain the current brightness of the light.
    brightness = 255
    # This value is used to determine the rate that the brightness changes.
    last_call_made = time.time()
    # This value is used to determine if a gesture was detected and what gesture was detected.
    gesture = "None"

    # The loop will continue to capture frames until the user presses the "q" key.
    while True:
        success, frame = capture.read()
        if success: # Checks to see if we captured a frame from the camera.
            # Converts the frame to RGB format for Mediapipe which is then converted into a Mediapipe image.
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                        image_format = mp.ImageFormat.SRGB,
                        data = RGB_frame
                        )
            # Detects the hand landmarks and gestures in the frame.
            recognition_result = recognizer.recognize(mp_image)
            # Draw all detected hand landmarks on the frame.
            for hand_landmarks in recognition_result.hand_landmarks:
                    # https://developers.google.com/mediapipe/api/solutions/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark
                    hand_landmarks_norm = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_norm.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in hand_landmarks])
                    mp_drawing.draw_landmarks(RGB_frame, hand_landmarks_norm, mp_hands.HAND_CONNECTIONS)

            # Turn gestures into two functions which can be selected by an input variable. 
            if calculated:
                gesture = gesture_calculation(recognition_result.hand_landmarks)
            else:
                if len(recognition_result.gestures) > 0:
                    gesture = recognition_result.gestures[0][0].category_name
                else:
                    gesture = "None"
            
            # Checks if a gesture was detected and then determines if a light should be turned on or off.
            # Also changes brightness, this is a constant value that is maintained between on/off states. 
            # Changing brightness can only be done every 0.2s and only if the light is currently on. Its value range is between 5 and 255.
            if gesture != "None":
                # Thumbs up = Light on, Thumbs down = Light off.
                # Open_Palm = Increase brightness, Closed_Fist = Decrease brightness.
                if (gesture == "Thumb_Up") and ((last_call == "off") or (last_call is None)):
                    last_call = "on"
                    light_service("turn_on", brightness, mock)
                elif (gesture == "Thumb_Down") and ((last_call == "on") or (last_call is None)):
                    last_call = "off"
                    light_service("turn_off", brightness, mock)
                elif (gesture == "Open_Palm") and (time.time() >= last_call_made + 0.2) and (last_call == "on"):
                    last_call_made = time.time() 
                    brightness += 25
                    if brightness > 255:  # Limit of 255.
                        brightness = 255
                    light_service("turn_on", brightness, mock)
                elif (gesture == "Closed_Fist") and (time.time() >= last_call_made + 0.2) and (last_call == "on"):
                    last_call_made = time.time()
                    brightness -= 25
                    if brightness < 5:  # Limit of 5.
                        brightness = 5
                    light_service("turn_on", brightness, mock)
            
            # Displays the frame with the detected hand landmarks drawn on it. Needs to be converted back to BGR format for OpenCV-Python.
            cv2.imshow("Captured frame",  cv2.cvtColor(RGB_frame, cv2.COLOR_RGB2BGR))
            # Checks if the user has pressed the "q" key to exit the loop.
            if cv2.waitKey(1) == ord("q"):
                break
    cv2.destroyAllWindows() # Closes the window displaying the captured frames. Ending the program.


if __name__ == "__main__":
    # Ensures environmental variables are loaded.
    load_envs()
    detection(False, False)

