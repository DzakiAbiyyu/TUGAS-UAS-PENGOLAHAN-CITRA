from flask import Flask, request, render_template, Response, session, send_from_directory
import os
import cv2
import time
import math
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov12n.pt")
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


def generate_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ptime = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.15, iou=0.1)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)
                classNameInt = int(box.cls[0])
                classname = cocoClassNames[classNameInt]
                conf = round(box.conf[0].item(), 2)
                label = f"{classname}: {conf}"
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

        ctime = time.time()
        fps = 1 / (ctime - ptime + 1e-8)
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        out.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return render_template("index.html", video_path=file.filename)

    return render_template("index.html")


@app.route('/video_feed/<video_filename>')
def video_feed(video_filename):
    input_path = os.path.join(UPLOAD_FOLDER, video_filename)
    output_path = os.path.join(UPLOAD_FOLDER, "output_" + video_filename)
    return Response(generate_frames(input_path, output_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/download/<video_filename>')
def download_video(video_filename):
    return send_from_directory(UPLOAD_FOLDER, video_filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
