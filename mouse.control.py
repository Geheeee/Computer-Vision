import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import customtkinter as ctk
import threading

# Setup UI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("Gesture Mouse Control")
root.geometry("400x300")

label_status = ctk.CTkLabel(root, text="Status: Idle", font=("Arial", 16))
label_status.pack(pady=20)

btn_start = ctk.CTkButton(root, text="Start Camera")
btn_start.pack(pady=10)

btn_stop = ctk.CTkButton(root, text="Stop Camera", state="disabled")
btn_stop.pack(pady=10)

# Inisialisasi ukuran layar
screen_width, screen_height = pyautogui.size()

# Inisialisasi modul MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mpDraw = mp.solutions.drawing_utils

clicking_left = False
clicking_right = False
scrolling = False
double_clicking = False
prev_scroll_y = None
last_scroll_time = 0
scroll_delay = 0.1

camera_running = False
cap = None

def get_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.hypot(x2 - x1, y2 - y1)

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def find_finger_tip(processed, finger):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[finger]
    return None

def detect_gestures(frame, landmarks_list, processed):
    global clicking_left, clicking_right, scrolling, prev_scroll_y, last_scroll_time

    if len(landmarks_list) < 21:
        return

    now = time.time()

    thumb_tip = (landmarks_list[4][0] * screen_width, landmarks_list[4][1] * screen_height)
    index_tip = (landmarks_list[8][0] * screen_width, landmarks_list[8][1] * screen_height)
    middle_tip = (landmarks_list[12][0] * screen_width, landmarks_list[12][1] * screen_height)
    ring_tip = (landmarks_list[16][0] * screen_width, landmarks_list[16][1] * screen_height)

    dist_thumb_index = get_distance(thumb_tip, index_tip)
    dist_thumb_middle = get_distance(thumb_tip, middle_tip)
    dist_index_middle = get_distance(index_tip, middle_tip)
    dist_index_ring = get_distance(index_tip, ring_tip)

    index_finger_tip = find_finger_tip(processed, mpHands.HandLandmark.INDEX_FINGER_TIP)
    if index_finger_tip is not None:
        move_mouse(index_finger_tip)

    if dist_thumb_index < 40:
        if not clicking_left:
            pyautogui.click(button='left')
            clicking_left = True
            label_status.configure(text="Status: Klik Kiri")
    else:
        clicking_left = False

    if dist_thumb_middle < 40:
        if not clicking_right:
            pyautogui.click(button='right')
            clicking_right = True
            label_status.configure(text="Status: Klik Kanan")
    else:
        clicking_right = False

    if dist_index_middle < 40:
        if not scrolling:
            scrolling = True
            prev_scroll_y = index_tip[1]
            last_scroll_time = now
        else:
            if (now - last_scroll_time) > scroll_delay:
                dy = index_tip[1] - prev_scroll_y
                scroll_amount = int(dy * -10)
                if abs(scroll_amount) > 5:
                    pyautogui.scroll(scroll_amount)
                    prev_scroll_y = index_tip[1]
                    last_scroll_time = now
                    label_status.configure(text="Status: Scroll")
    else:
        scrolling = False
        prev_scroll_y = None

    if dist_index_ring < 40:
        pyautogui.doubleClick()
        label_status.configure(text="Status: Double Klik")
        time.sleep(0.5)

def camera_loop():
    global camera_running, cap
    cap = cv2.VideoCapture(0)
    camera_running = True
    while cap.isOpened() and camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmarks_list = []
        if processed.multi_hand_landmarks:
            for hand_landmarks in processed.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))
            detect_gestures(frame, landmarks_list, processed)

        cv2.imshow('Gesture Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    label_status.configure(text="Status: Kamera Dimatikan")


def start_camera():
    btn_start.configure(state="disabled")
    btn_stop.configure(state="normal")
    label_status.configure(text="Status: Kamera Aktif")
    threading.Thread(target=camera_loop, daemon=True).start()

def stop_camera():
    global camera_running
    camera_running = False
    btn_start.configure(state="normal")
    btn_stop.configure(state="disabled")

btn_start.configure(command=start_camera)
btn_stop.configure(command=stop_camera)

root.mainloop()