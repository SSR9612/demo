"""Enumerate cameras and report which (index, backend) yields a real (non-black) frame."""
import cv2


def probe(index, backend_id, backend_name):
    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = None, None
    for _ in range(6):  # warmup: first frames are often black anyway
        ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return "read failed"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean, std = float(g.mean()), float(g.std())
    verdict = "BLACK" if std < 8 else "OK"
    cap.release()
    return f"{w}x{h}  mean={mean:.1f}  std={std:.1f}  -> {verdict}"


backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
print("扫描摄像头索引 0~5 ...")
found = 0
for i in range(6):
    for name, bid in backends:
        r = probe(i, bid, name)
        if r:
            found += 1
            print(f"  index={i}  backend={name}: {r}")
print(f"共找到 {found} 个可打开的设备。若某行标注 OK, 用它对应的 index 和 backend 即可。")