"""Quick smoke test: send a synthetic image to the local SmolVLM llama-server."""
import base64
import json
import sys

import cv2
import numpy as np
import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "smolvlm-500m"

# Draw a red circle + blue square on a white background.
img = np.full((256, 256, 3), 255, dtype=np.uint8)
cv2.circle(img, (128, 128), 60, (0, 0, 255), -1)
cv2.rectangle(img, (20, 20), (90, 90), (255, 0, 0), -1)

ok, buf = cv2.imencode(".jpg", img)
if not ok:
    sys.exit("encode failed")
b64 = base64.b64encode(buf.tobytes()).decode()

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the main objects in this image in one short sentence.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ],
    "max_tokens": 64,
    "temperature": 0.2,
}

r = requests.post(URL, json=payload, timeout=180)
print("HTTP", r.status_code)
print(r.text)