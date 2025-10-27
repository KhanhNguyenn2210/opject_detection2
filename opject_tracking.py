"""  
-Opject  Tracking
+ Dùng OD để tìm ra vật thể (yolov9)
+ Gắn ID cho vật thể
+ Traking trong quá trình duyệt frame 

- Tại sao lại cần
+ Khi OD fail dùng tracking đọc tiếp frame
+ Gắn id theo từng frame
- Dùng deepsort (Deep Simple Online and Realtime Tracking)
+ chuyển động
+ ngoại hình đối tượng
- B1: Dùng OD để tìm ra đối tượng
- B2: DeepSORT dùng một mạng CNN nhỏ để trích ra vector đặc trưng
- B3: (Tracking) dùng thuật toán Hungarian Algorithm
- B4: Cập nhật lại ID
+ Nếu khớp thành công → giữ nguyên ID.

+ Nếu không khớp trong vài frame → xóa ID.

+ Nếu có vật thể mới → tạo ID mới.
"""


import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape



#Confix value
video_path = "data_ext/highway.mp4"
conf_threshold = 0.5 # do tin cay
tracking_class = None # car 

#Khoi tao Deepsort

tracker = DeepSort(
    max_age=8,           # số frame mất tín hiệu trước khi xoá ID
    n_init=2,            # số lần phát hiện liên tiếp để xác nhận ID mới
    nn_budget=None       # số lượng feature lưu lại
)
#Khoi tao yolov9
device = "cpu" # co the thanh thanh cpu

model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt",device=device, fuse = True)
model = AutoShape(model)
# load classname tu file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size =(len(class_names),3))
tracks = []

#khoi tao videocapture de toc tu file video

cap = cv2.VideoCapture(video_path)

#Tien hanh doc tung frame
while True:
    #doc
    ret, frame = cap.read()
    if not ret:
        break             
    # Dua ra models de detect
    results = model(frame)
    detect=[]
    for detect_object in results.pred[0]:# 0 chứa tất cả boundingbox
        class_id = int(detect_object[5]) # 5 class_id
        confidence = float(detect_object[4])# 4 độ tin cậy conf
        bbox = detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
# next
        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue
# thêm vào  detect    
        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
        
    # Cap nhat , gan ID = deepsort
    tracks = tracker.update_tracks(detect, frame = frame)

    # Ve len man hinh cac khung chu nhat kem ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
        # lay toa do classid de ve len hinh anh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1,y1, x2,y2 = map(int,ltrb)
            color = colors[class_id]
            B,G,R = map (int,color)

            label_text = f"{class_names[class_id]}-{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label_text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label_text, org=(x1 + 5, y1 - 8),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(255, 255, 255), thickness=2)

    # Show anh len man hinh
    cv2.imshow("OT", frame)

    #exit
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
