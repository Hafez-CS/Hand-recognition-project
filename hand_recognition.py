import cv2
import mediapipe as mp

# راه‌اندازی MediaPipe برای تشخیص دست‌ها
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# تنظیمات تشخیص دست
hands = mp_hands.Hands(
    max_num_hands=2,  # حداکثر تعداد دست‌هایی که باید شناسایی شود
    min_detection_confidence=0.7,  # حداقل میزان اعتماد برای تشخیص
    min_tracking_confidence=0.5  # حداقل میزان اعتماد برای ردیابی
)

# راه‌اندازی دوربین
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # تبدیل رنگ تصویر از BGR به RGB برای MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # پردازش تصویر برای تشخیص دست‌ها
    result = hands.process(rgb_frame)
    
    # اگر دست‌ها تشخیص داده شدند
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # پیدا کردن کادر دور دست‌ها
            x_min, x_max = float('inf'), float('-inf')
            y_min, y_max = float('inf'), float('-inf')
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            
            # رسم کادر دور دست
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # رنگ قرمز (BGR): (0, 0, 255)
            
            # رسم اتصالات و نقاط دست
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # نمایش تصویر
    cv2.imshow('Hand Detection', frame)
    
    # اگر کلید 'q' فشرده شود، خروج از برنامه
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزاد کردن منابع
cap.release()
cv2.destroyAllWindows()
