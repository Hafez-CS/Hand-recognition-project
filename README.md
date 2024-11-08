![0](https://github.com/user-attachments/assets/c3e32bbf-9cae-440b-8116-dadd8d03ebdd)
# Hand recognition project

To detect hands and fingers and draw a box around them, we can use OpenCV and the MediaPipe library. The MediaPipe library provides good tools for hand detection and tracking.

## Download MediaPipe:
* **The `mp.solutions.hands` module is used to detect hands.**
* **`mp_drawing` is used to draw the joints and points of the hand.**
## Hand recognition:
* **The `hands.process` function is used for image processing and hands detection.**
* **We check different points of the hand and get the smallest and largest values ​​of `x` and `y` to draw the box.**
## Frame design:
* **Using the `cv2.rectangle` function, a red box is drawn around the hand.**
## Show image:
* **The processed image from the camera will be displayed and you can exit the program by pressing the 'q' key.**

## coding
```python
import cv2
import mediapipe as mp

# Setting up MediaPipe to detect hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand detection settings
hands = mp_hands.Hands(
    max_num_hands=2,  # The maximum number of hands to detect
    min_detection_confidence=0.7,  # Minimum confidence level for diagnosis
    min_tracking_confidence=0.5  # Minimum trust level for tracking
)

# Launch the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Image color conversion from BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Image processing for hand recognition
    result = hands.process(rgb_frame)
    
    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Find the circle around the hands
            x_min, x_max = float('inf'), float('-inf')
            y_min, y_max = float('inf'), float('-inf')
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            
            # Draw a frame around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red color (BGR): (0, 0, 255)
            
            # Draw connections and hand points
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display image
    cv2.imshow('Hand Detection', frame)
    
    # If the 'q' key is pressed, exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free up resources
cap.release()
cv2.destroyAllWindows()
```
