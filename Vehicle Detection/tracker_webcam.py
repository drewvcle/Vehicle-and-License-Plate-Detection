import torch
import cv2

# load YOLOv5 model from github (trained weights are local)
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='runs/train/exp23/weights/best.pt',
                       source='github')  # use your trained weights

# webcam (0 = default camera; change to 1 or 2 for external cams)
cap = cv2.VideoCapture(0)

# set resolution (depends on webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("STARTING WEBCAM. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # run YOLOv5 inference
    results = model(frame)

    # render detections directly onto frame
    annotated = results.render()[0]

    # show frame
    cv2.imshow('YOLOv5 Real-Time Detection', annotated)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Stopped webcam.")
