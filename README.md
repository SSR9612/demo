# 摄像头智能识别（YOLO + SmolVLM-500M）

用 **YOLOv8** 做实时目标检测与跟踪，再把每个目标裁剪出来交给本地 **llama.cpp 跑的 SmolVLM-500M** 生成一句话描述（英文），并经在线翻译接口转成中文（**MyMemory** 免费接口，额度耗尽自动切**百度翻译**，再不行就展示英文），最后把「框 + 标签 + 中文描述」叠加回摄像头画面。

## 环境（本机已具备）

- Python 3.10
- `ultralytics`(YOLO)、`opencv`、`numpy`、`pillow`、`requests`
- `llama-server.exe`（`D:\llama`）
- 模型：`D:\llama\models\SmolVLM-500M-Instruct-f16.gguf` + `mmproj-SmolVLM-500M-Instruct-f16.gguf`
- 显卡：RTX 4060 Laptop（CUDA 可用）

## 运行步骤

直接运行主程序即可 —— 它会**自动启动 llama-server**（若 8080 未运行），退出时**自动关闭由它启动的那个服务**：

```bat
python webcam_yolo_vlm.py
```

- 若想手动管理服务，也可以先双击 `start_smolvlm_server.bat`（或手动执行）：

  ```bat
  cd /d D:\llama
  llama-server.exe --jinja -m ./models/SmolVLM-500M-Instruct-f16.gguf --mmproj ./models/mmproj-SmolVLM-500M-Instruct-f16.gguf --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 -t 14
  ```

  验证：浏览器打开 `http://127.0.0.1:8080/health` 看到 `{"status":"ok"}`。手动启动的服务，程序会**直接复用、退出时不会关闭**（只有程序自己启动的才关闭）。

- 首次运行会自动下载 `yolov8n.pt`（约 6MB）。

## 操作

| 按键 | 作用 |
| ---- | ---- |
| `q` / `Esc` | 退出 |
| `r` | 立即重新描述画面中所有当前目标 |

## 可调参数（都在 `webcam_yolo_vlm.py` 顶部配置区）

| 参数 | 说明 |
| ---- | ---- |
| `YOLO_WEIGHTS` | YOLO 模型；可换 `yolov8s.pt` 更准(更慢)、`yolov11n.pt` 等 |
| `YOLO_CONF` | 检测置信度阈值（默认 0.35，调大更少更准） |
| `CLASSES_FILTER` | 只检测指定类别，如 `[0]`=仅人、`[0,2]`=人+汽车；`None`=全部 |
| `OUTPUT_LANG` | 最终展示语言：`"zh"`=中文（默认）/ `"en"`=英文 |
| `TRANSLATORS` | 翻译接口优先级列表，默认 `["mymemory", "baidu"]`；不翻译用 `[]` |
| `BAIDU_APPID` / `BAIDU_KEY` | 百度翻译标准版凭据（`BAIDU_APPID` 填纯数字 appid，`BAIDU_KEY` 填密钥） |
| `TRANSLATION_COOLDOWN_SECONDS` | 某翻译接口额度受限后的冷却秒数（默认 3600=1 小时），期间不再向该接口发请求 |
| `DESCRIBE_REFRESH` | 同一目标重新描述的间隔秒数 |
| `VLM_MAX_TOKENS` | 描述最大长度，越小越快 |
| `CROP_MAX_SIZE` | 裁剪图最长边，控制 VLM 输入分辨率 |
| `CAMERA_ID` | 摄像头编号，多个摄像头时改 `1` 等 |
| `LLAMA_SERVER_EXE` / `LLAMA_MODEL` / `LLAMA_MMPROJ` | 自动启动 llama-server 所用的路径（文件顶部配置；换机器/换路径时改这里） |

## 工作原理

1. 每帧跑 YOLO 检测，再用**内置轻量 IoU 跟踪器**给每个目标分配稳定 ID（不依赖 `lap`，免去 Windows 上的编译安装）。
2. 对新出现的、以及超过 `DESCRIBE_REFRESH` 秒未更新的目标，裁剪出局部图。
3. 后台单线程把这些裁剪图以 base64 发给 `llama-server` 的 OpenAI 兼容接口，得到一句英文描述。
4. 若 `OUTPUT_LANG="zh"`，按 `TRANSLATORS` 顺序翻译成中文：先免费在线 **MyMemory**，额度耗尽/失败自动切**百度翻译**（每月 100 万字符免费）。任一接口额度/频率受限后进入 1 小时冷却，期间不再请求它；全失败则保留原始英文。
5. 返回的描述缓存到 `track_id`，叠加到画面；请求未返回前显示 `…`。

> 采用**单线程串行**请求是为了不压垮本地 llama-server，同时摄像头画面保持流畅不卡顿。

## 常见问题

- **`无法打开摄像头`**：确认摄像头未被其他程序占用；尝试把 `CAMERA_ID` 改为 `1`。
- **描述一直显示 `…` / `(描述失败)`**：确认 llama-server 在 8080 运行、`/health` 返回 ok；查看终端里的 `[VLM] ... 失败` 日志。
- **中文变成问号**：说明字体目录缺失，程序会自动回退到默认字体，此时可在 `FONT_CANDIDATES` 里加一个可用 TTF 路径。
- **中文/英文怎么切**：改 `OUTPUT_LANG`；翻译优先级在 `TRANSLATORS` 里调。`"mymemory"` 免费但有额度、`"baidu"` 用你的百度翻译标准版（每月 100 万字符免费，需在 `BAIDU_APPID` / `BAIDU_KEY` 填凭据）。接口额度受限后按 `TRANSLATION_COOLDOWN_SECONDS` 冷却（默认 1 小时，期间不再请求它，防止被拉黑/超额计费）。全失败会自动保留原始英文。框里的类别标签始终是中文。