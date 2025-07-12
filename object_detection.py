import cv2
import math
import time
from ultralytics import YOLO


def detect_objects_yolov12(video_path, model_path="yolov12n.pt", output_path="output.mp4", conf_thresh=0.15,
                           iou_thresh=0.1):
    # Load YOLOv12 Model
    model = YOLO(model_path)

    # Open Video File
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define Video Writer
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # COCO Class Names
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]

    ptime = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        print(f"Frame Number: {count}")

        # Object Detection using YOLOv12
        results = model.predict(frame, conf=conf_thresh, iou=iou_thresh)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)

                class_id = int(box.cls[0])
                classname = cocoClassNames[class_id]
                conf = round(float(box.conf[0]), 2)
                label = f"{classname}: {conf}"

                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Calculate and Display FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Write Frame to Output Video
        output_video.write(frame)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

detect_objects_yolov12("Resources/Videos/video2.mp4")
