"""End-to-end pipeline test (static image instead of webcam + GUI)."""
import os
import time

import cv2
import ultralytics
from ultralytics import YOLO

import webcam_yolo_vlm as app

frame = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
model = YOLO("yolov8n.pt")
model.to("cuda")

worker = app.VLMWorker()
annotator = app.Annotator()
tracker = app.SimpleTracker()

res = model.predict(frame, conf=0.35, verbose=False)[0]
tracked = []  # (tid, zh_name, xyxy, color)
if res.boxes is not None and len(res.boxes):
    boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
    cls = res.boxes.cls.int().cpu().numpy()
    dets = [tuple(b) for b in boxes_xyxy.tolist()]
    id_map = tracker.update(dets)
    for j, tid in id_map.items():
        cn = res.names[int(cls[j])]
        x1, y1, x2, y2 = dets[j]
        crop = app.crop_region(frame, x1, y1, x2, y2)
        worker.submit(tid, cn, crop)
        tracked.append((tid, app.COCO_ZH.get(cn, cn), (x1, y1, x2, y2),
                        app.PALETTE[tid % len(app.PALETTE)]))
else:
    print("no detections")

# wait for descriptions
deadline = time.time() + 60
while time.time() < deadline:
    if tracked and all(worker.get(tid) is not None for tid, *_ in tracked):
        break
    time.sleep(0.3)

print(f"tracked {len(tracked)} objects:")
boxes_out = []
with open("pipeline_zh_result.txt", "w", encoding="utf-8") as fh:
    for tid, zh, (x1, y1, x2, y2), color in tracked:
        desc = worker.get(tid) or "(timeout)"
        print(f"  #{tid} {zh}: {ascii(desc)}")
        fh.write(f"#{tid} {zh}: {desc}\n")
        boxes_out.append({"xyxy": (x1, y1, x2, y2), "label": f"{zh} #{tid}",
                          "desc": desc, "color": color})

out = annotator.draw(frame, boxes_out)
cv2.imwrite("pipeline_result.jpg", out)
worker.stop()
print("saved pipeline_result.jpg and pipeline_zh_result.txt")