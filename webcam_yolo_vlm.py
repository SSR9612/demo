#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头智能识别：YOLO 检测 + SmolVLM-500M (llama.cpp) 逐个目标描述。

流程:
   摄像头帧 -> YOLOv8 目标检测 + 跟踪(稳定 ID)
            -> 对每个目标裁剪出局部图 -> 发给本地 llama.cpp 的 SmolVLM-500M
            -> 英文描述经在线接口翻译成中文 -> 叠加回实时画面(框 + 标签 + 描述)

依赖(已在本机安装): ultralytics, opencv, numpy, pillow, requests

运行:
   - 程序会自动启动 llama-server(未运行时); 已在运行则直接复用
   - 由本程序启动的服务在退出时一并关闭; 手动启动的服务不会被关闭
   - 首次运行会自动下载 yolov8n.pt (~6MB)

按键:
   q / ESC   退出
   r         立即重新描述画面中所有当前目标
"""
import base64
import hashlib
import os
import queue
import random
import subprocess
import threading
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# 配置区
# ============================================================
VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
VLM_MODEL = "smolvlm-500m"  # llama-server 会忽略该字段, 仅用于 OpenAI 兼容
VLM_TIMEOUT = 180           # 单次请求超时(秒)
VLM_MAX_TOKENS = 48         # 描述一定要短, 加快速度
VLM_TEMPERATURE = 0.2

# llama-server 自动启停: 启动时若 8080 未开启则自动拉起, 退出时仅关闭本程序启动的那个
LLAMA_SERVER_EXE = r"D:\llama\llama-server.exe"
LLAMA_WORKDIR = r"D:\llama"
LLAMA_MODEL = r"D:\llama\models\SmolVLM-500M-Instruct-f16.gguf"
LLAMA_MMPROJ = r"D:\llama\models\mmproj-SmolVLM-500M-Instruct-f16.gguf"
LLAMA_PORT = 8080
LLAMA_ARGS = ["--jinja", "-c", "8192", "-ngl", "99", "-t", "14"]
LLAMA_STARTUP_TIMEOUT = 120   # 等待服务就绪的最长秒数

YOLO_WEIGHTS = "yolov8n.pt"  # 首次运行自动下载
YOLO_CONF = 0.35             # 检测置信度阈值, 越高越少但越准
YOLO_DEVICE = "cuda"         # 无 N 卡改为 "cpu"
CLASSES_FILTER = None        # 例如 [0] 只检测"人"; None = 全部

DESCRIBE_REFRESH = 6.0       # 同一目标每隔多少秒重新描述一次
CROP_MAX_SIZE = 512          # 裁剪图最长边, 控制 VLM 输入尺寸
OUTPUT_LANG = "zh"           # 最终展示语言: "zh"=中文 / "en"=英文
# 翻译接口优先级(依次尝试, 全部失败则保留原始英文):
#   "mymemory" = 免费在线翻译(免 key, 额度用完返回 429)
#   "baidu"    = 百度翻译标准版(每月 100 万字符免费), 需填下面 BAIDU_APPID / BAIDU_KEY
# 完全不翻译用空列表 []
TRANSLATORS = ["mymemory", "baidu"]
TRANSLATE_URL = "https://api.mymemory.translated.net/get"   # mymemory 用
TRANSLATE_TIMEOUT = 10                                        # mymemory 用

# 百度翻译(通用翻译 API, 文档 https://api.fanyi.baidu.com/doc/23)
BAIDU_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"
BAIDU_APPID = "20210508000819015"               # appid(纯数字 ID)
BAIDU_KEY = "0PKbwro43pxwSWKqlZxj"              # 密钥(secret key)

# 某翻译接口额度/频率受限后进入冷却期(默认 1 小时), 期间不再向其发请求, 防止被拉黑/超额计费
TRANSLATION_COOLDOWN_SECONDS = 3600

CAMERA_ID = 0

# 常见 COCO 类别中文名(其余回退到英文名)
COCO_ZH = {
    "person": "人", "car": "汽车", "bicycle": "自行车", "motorcycle": "摩托车",
    "bus": "公交车", "truck": "卡车", "dog": "狗", "cat": "猫", "bird": "鸟",
    "horse": "马", "sheep": "羊", "cow": "牛", "chair": "椅子", "couch": "沙发",
    "potted plant": "盆栽", "bed": "床", "dining table": "餐桌", "tv": "电视",
    "laptop": "笔记本电脑", "cell phone": "手机", "book": "书", "bottle": "瓶子",
    "cup": "杯子", "bowl": "碗", "banana": "香蕉", "apple": "苹果",
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包", "suitcase": "行李箱",
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "单板滑雪板",
    "sports ball": "球", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "wine glass": "酒杯", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "sink": "水槽", "refrigerator": "冰箱",
    "oven": "烤箱", "toaster": "烤面包机", "microwave": "微波炉",
    "mouse": "鼠标", "keyboard": "键盘", "clock": "时钟", "vase": "花瓶",
    "scissors": "剪刀", "teddy bear": "泰迪熊", "toothbrush": "牙刷",
}

# (SmolVLM-500M 中文生成不可靠: 描述统一走英文, 再由在线翻译转成中文)

# 中文字体(用于 PIL 渲染, OpenCV 自带字体不支持中文)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

PALETTE = [  # RGB 颜色, 按跟踪 ID 循环使用
    (0, 200, 255), (255, 80, 80), (80, 255, 120), (255, 200, 0),
    (200, 120, 255), (120, 220, 255), (255, 150, 200),
]


# ============================================================
# 工具函数
# ============================================================
def load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def encode_jpeg_b64(img_bgr):
    """BGR ndarray -> JPEG base64 字符串。"""
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def crop_region(frame, x1, y1, x2, y2, margin=0.12):
    """裁剪目标区域(带外边距), 并缩放到最长边 <= CROP_MAX_SIZE。"""
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * margin), int(bh * margin)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame[y1:y2, x1:x2]
    crop = frame[cy1:cy2, cx1:cx2]
    scale = CROP_MAX_SIZE / max(crop.shape[0], crop.shape[1])
    if scale < 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
    return crop


def smolvlm_describe(session, class_name, crop):
    """SmolVLM-500M 输出英文描述(它对中文生成不可靠)。"""
    prompt = f"Describe this {class_name} in one short sentence."
    data_url = f"data:image/jpeg;base64,{encode_jpeg_b64(crop)}"
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": VLM_MAX_TOKENS,
        "temperature": VLM_TEMPERATURE,
    }
    r = session.post(VLM_URL, json=payload, timeout=VLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


class _QuotaExceeded(Exception):
    """翻译接口额度/频率受限, 应进入冷却期, 冷却期内不再尝试。"""


_translator_cooldown_until = {}


def _in_cooldown(name):
    return time.time() < _translator_cooldown_until.get(name, 0.0)


def _set_cooldown(name):
    _translator_cooldown_until[name] = time.time() + TRANSLATION_COOLDOWN_SECONDS


def translate_mymemory(session, text):
    """免费在线接口 MyMemory 英译中(免 key, 有额度限制)。"""
    params = {"q": text, "langpair": "en|zh-CN"}
    r = session.get(TRANSLATE_URL, params=params, timeout=TRANSLATE_TIMEOUT)
    if r.status_code == 429:
        raise _QuotaExceeded("HTTP 429 额度/频率受限")
    r.raise_for_status()
    return r.json()["responseData"]["translatedText"].strip()


def translate_baidu(session, text):
    """百度翻译(标准版)英译中：需 BAIDU_APPID + BAIDU_KEY。"""
    if not BAIDU_APPID:
        raise RuntimeError("未配置 BAIDU_APPID")
    salt = str(random.randint(32768, 65536))
    sign = hashlib.md5((BAIDU_APPID + text + salt + BAIDU_KEY).encode("utf-8")).hexdigest()
    params = {
        "q": text,
        "from": "en",
        "to": "zh",
        "appid": BAIDU_APPID,
        "salt": salt,
        "sign": sign,
    }
    r = session.get(BAIDU_URL, params=params, timeout=TRANSLATE_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "trans_result" in data:
        return "".join(x["dst"] for x in data["trans_result"]).strip()
    code = str(data.get("error_code"))
    if code in {"54003", "54004", "54005"}:   # 访问频率受限 / 账户余额不足 / 长query频繁
        raise _QuotaExceeded(f"额度受限(错误码 {code})")
    raise RuntimeError(f"百度翻译错误 {code}: {data.get('error_msg')}")


def describe_crop(session, class_name, crop):
    """先 SmolVLM 英文描述, 再按 TRANSLATORS 优先级翻译成中文(全失败则保留原始英文)。"""
    text = smolvlm_describe(session, class_name, crop)
    if OUTPUT_LANG == "zh":
        for name in TRANSLATORS:
            if _in_cooldown(name):
                continue  # 冷却期内直接跳过, 不再发请求
            try:
                if name == "mymemory":
                    return translate_mymemory(session, text)
                if name == "baidu":
                    return translate_baidu(session, text)
            except _QuotaExceeded as e:
                _set_cooldown(name)
                print(f"[翻译] {name} 额度受限, 冷却 {TRANSLATION_COOLDOWN_SECONDS // 3600} 小时: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"[翻译] {name} 失败: {e}")
    return text


# ============================================================
# VLM 后台工作线程(单线程串行, 避免压垮 llama-server)
# ============================================================
class VLMWorker:
    def __init__(self):
        self.q = queue.Queue(maxsize=32)
        self.results = {}  # track_id -> 描述文本
        self.lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, track_id, class_name, crop):
        """非阻塞提交; 队列满时丢弃, 返回是否成功。"""
        try:
            self.q.put_nowait((track_id, class_name, crop))
            return True
        except queue.Full:
            return False

    def get(self, track_id):
        with self.lock:
            return self.results.get(track_id)

    def reset(self):
        with self.lock:
            self.results.clear()

    def _run(self):
        session = requests.Session()
        while True:
            item = self.q.get()
            if item is None:
                break
            track_id, class_name, crop = item
            try:
                text = describe_crop(session, class_name, crop)
            except Exception as e:  # noqa: BLE001
                text = "(描述失败)"
                print(f"[VLM] track {track_id} 失败: {e}")
            with self.lock:
                self.results[track_id] = text
            self.q.task_done()

    def stop(self):
        self.q.put(None)


# ============================================================
# 轻量 IoU 跟踪器(替代 ultralytics track(), 免去 lap 依赖)
# ============================================================
class SimpleTracker:
    """按帧间 IoU 给每个目标分配稳定 ID。适合摄像头演示场景。"""

    def __init__(self, max_lost=30, iou_thresh=0.25):
        self.tracks = {}      # tid -> {"box": (x1,y1,x2,y2), "lost": int}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        """detections: [(x1,y1,x2,y2), ...] -> {det_idx: tid}"""
        tids = list(self.tracks.keys())
        remaining = list(range(len(detections)))
        matches = {}  # tid -> det_idx

        for tid in tids:
            tb = self.tracks[tid]["box"]
            best_iou, best_j = 0.0, -1
            for j in remaining:
                iou = self._iou(tb, detections[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                matches[tid] = best_j
                remaining.remove(best_j)

        det_to_tid = {}
        for tid, j in matches.items():
            self.tracks[tid]["box"] = detections[j]
            self.tracks[tid]["lost"] = 0
            det_to_tid[j] = tid

        for tid in tids:
            if tid not in matches:
                self.tracks[tid]["lost"] += 1

        for tid in [t for t in self.tracks if self.tracks[t]["lost"] > self.max_lost]:
            del self.tracks[tid]

        for j in remaining:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": detections[j], "lost": 0}
            det_to_tid[j] = tid

        return det_to_tid


# ============================================================
# 画面标注(PIL 渲染, 支持中文)
# ============================================================
class Annotator:
    def __init__(self):
        self.font_label = load_font(20)
        self.font_desc = load_font(17)
        self.max_text_width = 320

    def _textsize(self, draw, text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _wrap(self, draw, text, font):
        lines, cur = [], ""
        for ch in text:
            if self._textsize(draw, cur + ch, font)[0] > self.max_text_width and cur:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
        return lines[:3]

    def _draw_text_block(self, draw, xy, text, font, fill, bg, wrap=True):
        x, y = xy
        if wrap:
            lines = self._wrap(draw, text, font)
        else:
            lines = [text]
        asc, desc = font.getmetrics()
        lh = asc + desc + 5
        widths = [self._textsize(draw, l, font)[0] for l in lines]
        max_w = max(widths) if widths else 0
        draw.rectangle([x - 2, y - 2, x + max_w + 6, y + lh * len(lines) + 3], fill=bg)
        for i, l in enumerate(lines):
            draw.text((x + 2, y - 2 + i * lh), l, font=font, fill=fill)

    def draw(self, frame_bgr, boxes):
        """boxes: [{xyxy, label, desc, color}] -> 返回标注好的 BGR frame。"""
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        H, W = frame_bgr.shape[:2]
        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]
            color = b["color"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = b["label"]
            self._draw_text_block(draw, (x1, max(0, y1 - 28)), label,
                                  self.font_label, (255, 255, 255), color, wrap=False)
            desc = b.get("desc")
            if desc:
                self._draw_text_block(draw, (x1, min(H - 40, y2 + 4)), desc,
                                      self.font_desc, (0, 0, 0), (240, 240, 240), wrap=True)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ============================================================
# llama-server 自动启停
# ============================================================
def llama_health_ok():
    """本机 llama-server 是否已就绪。"""
    try:
        r = requests.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def ensure_llama_server():
    """确保 llama-server 在运行; 返回本程序启动的进程对象(未启动/已存在则为 None)。"""
    if llama_health_ok():
        print("[模型] 检测到 llama-server 已在运行, 直接复用")
        return None
    if not os.path.exists(LLAMA_SERVER_EXE):
        print(f"[警告] 未找到 {LLAMA_SERVER_EXE}, 请先手动启动 llama-server")
        return None
    cmd = [LLAMA_SERVER_EXE,
           "-m", LLAMA_MODEL, "--mmproj", LLAMA_MMPROJ,
           "--host", "127.0.0.1", "--port", str(LLAMA_PORT)] + LLAMA_ARGS
    print("[模型] 未检测到服务, 正在启动 llama-server ...")
    proc = subprocess.Popen(
        cmd, cwd=LLAMA_WORKDIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    deadline = time.time() + LLAMA_STARTUP_TIMEOUT
    while time.time() < deadline:
        if llama_health_ok():
            print("[模型] llama-server 已就绪")
            return proc
        if proc.poll() is not None:
            print(f"[错误] llama-server 启动后提前退出(exit code {proc.returncode}),"
                  " 请检查模型路径/显存/端口")
            return proc
        time.sleep(1.0)
    print(f"[警告] llama-server 未在 {LLAMA_STARTUP_TIMEOUT}s 内就绪, 仍继续(描述可能失败)")
    return proc


def stop_llama_server(proc):
    """关闭由本程序启动的 llama-server(proc 为 None 则不动)。"""
    if proc is None:
        return
    print("[模型] 正在关闭 llama-server ...")
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


# ============================================================
# 主流程
# ============================================================
def main():
    server_proc = ensure_llama_server()

    print("加载 YOLO 模型 ...")
    model = YOLO(YOLO_WEIGHTS)
    model.to(YOLO_DEVICE)

    worker = VLMWorker()
    annotator = Annotator()
    tracker = SimpleTracker()
    last_describe = {}  # track_id -> 上次提交时间

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 ID={CAMERA_ID}")
        return

    print("运行中 ... 按 q/ESC 退出, r 重新描述所有目标")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[错误] 读取摄像头帧失败")
            break

        res = model.predict(frame, conf=YOLO_CONF, classes=CLASSES_FILTER,
                            verbose=False)[0]

        boxes_out = []
        now = time.time()
        if res.boxes is not None and len(res.boxes):
            boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            cls = res.boxes.cls.int().cpu().numpy()
            dets = [tuple(b) for b in boxes_xyxy.tolist()]
            id_map = tracker.update(dets)
            for j, tid in id_map.items():
                class_name = res.names[int(cls[j])]
                zh_name = COCO_ZH.get(class_name, class_name)
                x1, y1, x2, y2 = dets[j]

                # 决定是否(重新)提交描述
                stale = now - last_describe.get(tid, 0) > DESCRIBE_REFRESH
                if stale:
                    crop = crop_region(frame, x1, y1, x2, y2)
                    if worker.submit(tid, class_name, crop):
                        last_describe[tid] = now

                desc = worker.get(tid)
                boxes_out.append({
                    "xyxy": (x1, y1, x2, y2),
                    "label": f"{zh_name} #{tid}",
                    "desc": desc if desc else "…",
                    "color": PALETTE[tid % len(PALETTE)],
                })

        frame = annotator.draw(frame, boxes_out)
        cv2.imshow("YOLO + SmolVLM 智能识别", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            last_describe.clear()
            worker.reset()

    worker.stop()
    stop_llama_server(server_proc)   # 仅关闭本程序启动的 llama-server
    cap.release()
    cv2.destroyAllWindows()
    print("已退出。")


if __name__ == "__main__":
    main()