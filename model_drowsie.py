import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import time
import yaml

class DrowsinessDetector:
    def __init__(self, config_path):
        """
        Khởi tạo module nhận diện trạng thái dáng người với video input từ file cấu hình.
        """
        self.show_frame = True  # Gán mặc định trước khi parse config để tránh AttributeError

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Đọc các tham số cấu hình
        self.display_scale = config.get('display_scale', 0.5)
        self.show_frame = config.get('show_frame', True)
        self.save_output = config.get('save_output', True)
        self.frame_skip = config.get('frame_skip', 1)
        self.log_path = config.get('log_path', None)
        self.yolo_person_class_id = config.get('yolo_person_class_id', 0)
        self.yolo_conf_threshold = config.get('yolo_confidence_threshold', 0.4)
        self.cnn_input_size = config.get('cnn_input_size', 640)

        cnn_model_path = config['cnn_model_path']
        yolo_model_path = config['yolo_model_path']
        video_path = config['video_path']
        self.output_path = config.get('output_path', None)

        # Load mô hình
        self.cnn_session = ort.InferenceSession(cnn_model_path)
        self.cnn_input_name = self.cnn_session.get_inputs()[0].name

        self.yolo_model = YOLO(yolo_model_path, task='pose')

        self.current_state = "normal"
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Không thể mở video: {video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out = None
        if self.save_output and self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def preprocess_image(self, image):
        image = cv2.resize(image, (self.cnn_input_size, self.cnn_input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def detect_person(self, frame):
        results = self.yolo_model(frame, classes=[self.yolo_person_class_id])
        person_boxes = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == self.yolo_person_class_id and box.conf[0] > self.yolo_conf_threshold:
                    x, y, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_boxes.append((x, y, x2 - x, y2 - y))

        return person_boxes

    def detect_state(self, frame, person_box):
        x, y, w, h = person_box
        person_roi = frame[y:y+h, x:x+w]
        input_data = self.preprocess_image(person_roi)
        outputs = self.cnn_session.run(None, {self.cnn_input_name: input_data})
        probs = outputs[0][0]
        state_idx = np.argmax(probs)
        states = ["normal", "play", "sleep"]
        return states[state_idx]

    def run(self):
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("\nĐã xử lý hết video hoặc lỗi khi đọc frame")
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            person_boxes = self.detect_person(frame)

            for idx, person_box in enumerate(person_boxes):
                x, y, w, h = person_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                state = self.detect_state(frame, person_box)
                color = (0, 0, 255) if state == "sleep" else (0, 255, 0)
                cv2.putText(frame, state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if self.log_path:
                    with open(self.log_path, 'a', encoding='utf-8') as logf:
                        logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Frame {frame_count} - Person {idx+1} - State: {state}\n")

            display_frame = cv2.resize(frame, (int(self.width * self.display_scale), int(self.height * self.display_scale)))

            if self.save_output and self.out:
                self.out.write(frame)

            if self.show_frame:
                cv2.imshow('Drowsiness Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        if self.out:
            self.out.release()
        if self.show_frame:
            cv2.destroyAllWindows()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'out') and self.out:
            self.out.release()
        if hasattr(self, 'show_frame') and self.show_frame:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = "./module/Config/configs.yaml"  # Đường dẫn tới file cấu hình
    try:
        detector = DrowsinessDetector(config_path)
        detector.run()
    except Exception as e:
        print(f"Lỗi: {e}")