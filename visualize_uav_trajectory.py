import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button
from PIL import Image, ImageTk

DEFAULT_DIR = os.path.join(os.getcwd(), "images")
DEFAULT_FILE = "default.png"
DISPLAY_HEIGHT_RATIO = 1.8

def overlay_drone_trajectory(video_path, sample_interval, diff_threshold, kernel_size, start_time, end_time, alpha_start, alpha_end):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {video_path}! 경로를 확인해주세요.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    ret, first_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다!")
        return None

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    background_image = first_frame.copy()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 >= end_time:
            break

        if frame_count % sample_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_frame, first_gray)
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(diff_mask, kernel, iterations=1)
            motion_only = cv2.bitwise_and(frame, frame, mask=dilated_mask)

            alpha = alpha_start + (alpha_end - alpha_start) * (frame_count / total_frames)
            beta = 1 - alpha

            for c in range(3):
                background_image[:, :, c] = (
                    (background_image[:, :, c] * beta + motion_only[:, :, c] * alpha) * (dilated_mask == 255)
                    + background_image[:, :, c] * (dilated_mask != 255)
                )

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return background_image

def update_image():
    try:
        sample_interval = int(sample_interval_entry.get())
        diff_threshold = int(diff_threshold_entry.get())
        kernel_size = int(kernel_size_entry.get())
        start_time = float(start_time_entry.get())
        end_time = float(end_time_entry.get())
        alpha_start = float(alpha_start_entry.get())
        alpha_end = float(alpha_end_entry.get())
    except ValueError:
        print("올바른 숫자 값을 입력해주세요!")
        return

    result_image = overlay_drone_trajectory(
        video_path, sample_interval, diff_threshold, kernel_size,
        start_time, end_time, alpha_start, alpha_end
    )
    if result_image is not None:
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(result_image_rgb)

        display_height = int(default_height / DISPLAY_HEIGHT_RATIO)
        aspect_ratio = img.width / img.height
        display_width = int(display_height * aspect_ratio)
        display_img = img.resize((display_width, display_height), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=display_img)
        result_label.config(image=imgtk)
        result_label.image = imgtk

        global saved_image
        saved_image = result_image

def save_image():
    if saved_image is not None:
        if not os.path.exists(DEFAULT_DIR):
            os.makedirs(DEFAULT_DIR)
        file_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_DIR,
            initialfile=DEFAULT_FILE,
            defaultextension=".png",
            filetypes=[("PNG 파일", "*.png"), ("모든 파일", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, saved_image)
            print(f"이미지가 저장되었습니다: {file_path}")

def select_file():
    global video_path
    file_path = filedialog.askopenfilename(filetypes=[("MP4 파일", "*.mp4"), ("모든 파일", "*.*")])
    if file_path:
        video_path = file_path
        file_label.config(text=f"선택한 파일: {file_path}")

video_path = ""
saved_image = None

root = tk.Tk()
root.title("드론 궤적 시각화 설정")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
default_width = int(screen_width * 0.5)
default_height = int(screen_height * 0.7)
root.geometry(f"{default_width}x{default_height}")

file_label = Label(root, text="파일이 선택되지 않았습니다", wraplength=default_width)
file_label.pack()

file_button = Button(root, text="영상 파일 선택", command=select_file)
file_button.pack()

Label(root, text="프레임 샘플 간격").pack()
sample_interval_entry = Entry(root)
sample_interval_entry.pack()
sample_interval_entry.insert(0, "10")

Label(root, text="차이 임계값").pack() # 첫번째 배경 프레임의 픽셀차이를 계산해서 얼마 이상 차이나면 움직인 것으로 간주할 것인가
diff_threshold_entry = Entry(root) # 값 낮으면 미세한 움직임까지 감지, 값 높으면 큰 움직임만 감지
diff_threshold_entry.pack()
diff_threshold_entry.insert(0, "30") 


Label(root, text="팽창 커널 크기").pack() # 노이즈처럼 분리된 점들을 묶어서 하나의 덩어리로 확장 Morphological Dilation라 생각
kernel_size_entry = Entry(root) # 값이 작으면 궤적이 얇게, 뚝뚝 끊겨 보임
kernel_size_entry.pack() # 값이 크면 궤적이 두껍고 선명하지만 너무 커지면 부정확해질 수 있음
kernel_size_entry.insert(0, "15")

Label(root, text="시작 시간 (초)").pack()
start_time_entry = Entry(root)
start_time_entry.pack()
start_time_entry.insert(0, "0")

Label(root, text="종료 시간 (초)").pack()
end_time_entry = Entry(root)
end_time_entry.pack()
end_time_entry.insert(0, "10")

Label(root, text="투명도 시작값").pack()
alpha_start_entry = Entry(root)
alpha_start_entry.pack()
alpha_start_entry.insert(0, "0.2")

Label(root, text="투명도 최종값").pack()
alpha_end_entry = Entry(root)
alpha_end_entry.pack()
alpha_end_entry.insert(0, "1")

update_button = Button(root, text="이미지 업데이트", command=update_image)
update_button.pack()

save_button = Button(root, text="이미지 저장", command=save_image)
save_button.pack()

result_label = Label(root)
result_label.pack()

root.mainloop()
