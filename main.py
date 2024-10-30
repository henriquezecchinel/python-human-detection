import cv2
import numpy as np
from ultralytics import YOLO
import mss
import os
from datetime import datetime
import time
import pygetwindow as gw  # To manage windows by title
import win32gui
import win32con
import win32ui
from ctypes import windll
from PIL import Image
import telebot
from dotenv import load_dotenv

# Load the YOLO model (pre-trained on COCO dataset, where humans are labeled as 'person')
model = YOLO("yolo11x.pt")
sleep_time = 0.2
confidence_threshold = 0.5
test_save_everything = False

# Load environment variables from .env file
load_dotenv()

# Get the Telegram token and chat ID from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

def send_telegram_image(image_path):
    with open(image_path, 'rb') as image_file:
        bot.send_photo(CHAT_ID, image_file, caption="Humano detectado!")
    print("Imagem enviada pelo Telegram")

def resize_image_if_needed(image_path, max_size=10 * 1024 * 1024):
    img = Image.open(image_path)
    while os.path.getsize(image_path) > max_size:
        # Reduz a resolução da imagem em 10% até que esteja abaixo de 10 MB
        img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
    return image_path

def find_sim_next_windows():
    # Get all windows with "SIM Next" in the title
    windows = [w for w in gw.getWindowsWithTitle("SIM Next") if w.visible]
    return windows

def capture_entire_monitor(monitor_index=1):
    # Capture the entire screen of the specified monitor
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]  # Get monitor by index
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        # Remove alpha channel if present
        if img.shape[2] == 4:
            img = img[..., :3]

        img = img.astype(np.uint8)  # Ensure compatibility with OpenCV
        return img

def save_screenshot(image):
    # Get the current date and time
    now = datetime.now()
    
    # Create the directory structure based on the current date and time
    folder_path = f"detections/{now.year}/{now.month}/{now.day}/{now.hour}"
    os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Create the file name with year, month, day, hour, minute, second, and millisecond
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # Truncate the last 3 digits for millisecond precision
    file_path = os.path.join(folder_path, f"{timestamp}.png")

    # Save the image
    cv2.imwrite(file_path, image)
    print(f"Screenshot saved: {file_path}")

    # Redimensiona a imagem se necessário
    resize_image_if_needed(file_path)

    # Enviar a imagem via Telegram
    send_telegram_image(file_path)

def detect_humans(image):
    if image is None or not isinstance(image, np.ndarray):
        print("Invalid image format.")
        return image

    results = model(image, verbose=False)
    human_detected = False

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            class_id = int(bbox.cls[0])
            confidence = bbox.conf[0]

            if class_id == 0 and confidence > confidence_threshold:  # Adjusted threshold
                # Ensure image is in an acceptable format for drawing
                if not image.flags['WRITEABLE']:
                    image = image.copy()
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'Human: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                human_detected = True

    if human_detected or test_save_everything:
        save_screenshot(image)
    
    # save_screenshot(image)

    return image

def bring_window_to_foreground(window):
    hwnd = window._hWnd  # Get the handle of the window
    
    # Bring the window to the foreground without altering its size
    try:
        win32gui.SetForegroundWindow(hwnd)  # Bring window to foreground
        time.sleep(sleep_time)  # Short delay to ensure the window is fully active
    except Exception as e:
        print(f"Error bringing window to foreground: {e}")
        return None

def main():
    windows = find_sim_next_windows()  # Find all SIM Next instances

    while True:
        for window in windows:
            # frame = capture_window(window)

            bring_window_to_foreground(window)
            frame = capture_entire_monitor(2)
            if frame is not None:
                # Process the frame and save the screenshot if humans are detected
                detect_humans(frame)

            # Delay between processing each window to reduce CPU usage
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
