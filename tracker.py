import torch
import cv2

# load YOLOv5 model from github (trained weights are local)
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='runs/train/exp23/weights/best.pt',
                       source='github') 
# change vid path here
cap = cv2.VideoCapture(r"my_test_vids\v1.mp4")

# vid properties
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run YOLOv5 inference
    results = model(frame)

    # render detections directly onto frame
    annotated = results.render()[0]

    # show frame
    cv2.imshow('YOLOv5 Vehicle Detection', annotated)
    out.write(annotated)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done. The processed video is saved as output.avi.")
