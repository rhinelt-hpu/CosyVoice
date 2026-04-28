# CosyVoice 实时流式 TTS — 从零开始部署指南

本文档面向希望在自有服务器上部署 **CosyVoice 实时流式 TTS WebSocket 服务**的工程师，完整覆盖环境准备、模型下载与服务部署三个阶段，并提供详细的 API 使用说明与常见故障排查。

---

## 目录

1. [架构概览](#1-架构概览)
2. [硬件与操作系统要求](#2-硬件与操作系统要求)
3. [软件环境准备](#3-软件环境准备)
4. [代码获取](#4-代码获取)
5. [Python 依赖安装](#5-python-依赖安装)
6. [模型下载](#6-模型下载)
7. [启动 WebSocket 服务](#7-启动-websocket-服务)
8. [Docker 部署（可选）](#8-docker-部署可选)
9. [WebSocket API 协议参考](#9-websocket-api-协议参考)
10. [客户端调用示例](#10-客户端调用示例)
11. [性能调优建议](#11-性能调优建议)
12. [常见问题排查](#12-常见问题排查)

---

## 1 架构概览

```
客户端 (Python / JavaScript / 任意 WebSocket 实现)
   │  ws://host:8189/v1/realtime
   ▼
FastAPI WebSocket 服务 (runtime/python/websocket/server.py)
   │
   ├─ CosyVoice2 / CosyVoice3 模型
   │     ├─ LLM (Qwen2-0.5B backbone)  → 语音 token 生成（后台线程）
   │     ├─ Flow Matching Decoder       → Mel 频谱生成
   │     └─ HiFi-GAN Vocoder           → 波形合成（24 kHz PCM-16）
   │
   └─ 流式输出：每 ~200 ms 推送一帧 base64 PCM-16 音频块
```

服务兼容 **OpenAI Realtime API** 风格的事件协议，支持四种合成模式：

| 模式 | 说明 |
|------|------|
| `sft` | 使用内置发音人（预置音色，零提示语音） |
| `zero_shot` | 零样本音色克隆（上传 3–10 秒参考音频） |
| `cross_lingual` | 保留参考音色进行跨语言合成 |
| `instruct2` | 通过自然语言指令控制情感、方言、语速等 |

---

## 2 硬件与操作系统要求

### 最低配置（CPU 推理）

| 项目 | 要求 |
|------|------|
| CPU | x86-64，≥ 8 核 |
| 内存 | ≥ 16 GB RAM |
| 磁盘 | ≥ 30 GB 可用空间（模型 + 依赖） |
| OS | Ubuntu 20.04 / 22.04 LTS（推荐）、CentOS 7+、macOS 12+ |

> **注意**：CPU 模式下实时率（RTF）约为 2–5，即合成 1 秒音频需 2–5 秒，不适合低延迟生产场景。

### 推荐配置（GPU 推理）

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 3090 / A10 / A100（≥ 16 GB 显存） |
| CUDA | 12.1 或 12.4 |
| cuDNN | 8.x / 9.x |
| CPU | ≥ 8 核 |
| 内存 | ≥ 32 GB RAM |
| 磁盘 | ≥ 50 GB（含 TensorRT 缓存） |
| OS | Ubuntu 22.04 LTS |

GPU 模式 RTF 约为 0.05–0.2，端到端首帧延迟 ≤ 300 ms。

---

## 3 软件环境准备

### 3.1 安装 CUDA（GPU 部署必须）

```bash
# 以 Ubuntu 22.04 + CUDA 12.4 为例
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4 libcudnn9-cuda-12
```

验证安装：

```bash
nvidia-smi
nvcc --version
```

### 3.2 安装 Miniconda（推荐）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash && source ~/.bashrc
```

### 3.3 安装系统依赖

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y \
    git git-lfs build-essential curl wget \
    ffmpeg sox libsox-dev unzip

git lfs install
```

```bash
# CentOS / RHEL
sudo yum install -y git git-lfs gcc gcc-c++ make curl wget \
    ffmpeg sox sox-devel unzip
```

---

## 4 代码获取

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
# 如果子模块拉取失败，手动重试
git submodule update --init --recursive
```

---

## 5 Python 依赖安装

### 5.1 创建 Conda 虚拟环境

```bash
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# pynini 必须通过 conda-forge 安装
conda install -y -c conda-forge pynini==2.1.5
```

### 5.2 安装 Python 包

```bash
# 使用阿里云镜像加速（国内用户）
pip install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host=mirrors.aliyun.com

# 海外用户可直接使用 PyPI
pip install -r requirements.txt
```

> `requirements.txt` 中已包含 `websockets==12.0`，满足 WebSocket 服务端与客户端的所有依赖。

### 5.3 设置 PYTHONPATH

```bash
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS:$PYTHONPATH"
# 建议写入 ~/.bashrc 永久生效
echo 'export PYTHONPATH="$HOME/CosyVoice:$HOME/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"' >> ~/.bashrc
```

### 5.4（可选）安装 ttsfrd 文本前端

`ttsfrd` 提供更精准的中文文本规范化（数字、符号等），安装后自动生效。

```bash
# 先下载模型（见第 6 节），再执行：
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
cd -
```

---

## 6 模型下载

### 6.1 推荐模型：CosyVoice2-0.5B / Fun-CosyVoice3-0.5B

```python
# 方式一：ModelScope（国内推荐）
from modelscope import snapshot_download

# CosyVoice2（25 Hz 流式，多语言）
snapshot_download('iic/CosyVoice2-0.5B',
                  local_dir='pretrained_models/CosyVoice2-0.5B')

# Fun-CosyVoice3（最新版，推荐）
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# 文本前端资源（可选，提升文本规范化质量）
snapshot_download('iic/CosyVoice-ttsfrd',
                  local_dir='pretrained_models/CosyVoice-ttsfrd')
```

```python
# 方式二：HuggingFace（海外用户）
from huggingface_hub import snapshot_download

snapshot_download('FunAudioLLM/CosyVoice2-0.5B',
                  local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

也可通过命令行下载：

```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
"
```

### 6.2 模型目录结构

下载完成后目录结构示例（CosyVoice2-0.5B）：

```
pretrained_models/CosyVoice2-0.5B/
├── cosyvoice2.yaml         # 模型配置
├── llm.pt                  # LLM 权重（约 1 GB）
├── flow.pt                 # Flow Decoder 权重（约 400 MB）
├── hift.pt                 # HiFi-GAN Vocoder 权重
├── campplus.onnx           # 说话人特征提取器
├── speech_tokenizer_v2.onnx
├── spk2info.pt             # 内置说话人信息
└── CosyVoice-BlankEN/      # Qwen2 分词器资源
```

### 6.3 验证模型

```bash
python example.py   # 执行完整推理示例（需 ~30–60 s 首次加载）
```

---

## 7 启动 WebSocket 服务

### 7.1 基本启动

```bash
cd /path/to/CosyVoice

# 使用 CosyVoice2（推荐）
python runtime/python/websocket/server.py \
    --model_dir pretrained_models/CosyVoice2-0.5B \
    --host 0.0.0.0 \
    --port 8189

# 使用 Fun-CosyVoice3（最新，效果更好）
python runtime/python/websocket/server.py \
    --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
    --host 0.0.0.0 \
    --port 8189
```

启动成功后日志示例：

```
2024-01-01 12:00:00 INFO Model loaded from pretrained_models/CosyVoice2-0.5B, sample_rate=24000
2024-01-01 12:00:00 INFO Available speakers: ['中文女', '中文男', '英文女', '英文男', ...]
2024-01-01 12:00:00 INFO Serving at ws://0.0.0.0:8189/v1/realtime
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8189 (Press CTRL+C to quit)
```

### 7.2 后台持续运行（systemd）

```ini
# /etc/systemd/system/cosyvoice-tts.service
[Unit]
Description=CosyVoice Realtime TTS WebSocket Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/CosyVoice
Environment=PYTHONPATH=/home/ubuntu/CosyVoice:/home/ubuntu/CosyVoice/third_party/Matcha-TTS
ExecStart=/home/ubuntu/miniconda3/envs/cosyvoice/bin/python \
    runtime/python/websocket/server.py \
    --model_dir pretrained_models/CosyVoice2-0.5B \
    --host 0.0.0.0 \
    --port 8189
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable cosyvoice-tts
sudo systemctl start cosyvoice-tts
sudo journalctl -u cosyvoice-tts -f   # 查看日志
```

### 7.3 启动参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址，本地测试可用 `127.0.0.1` |
| `--port` | `8189` | TCP 端口 |
| `--model_dir` | `iic/CosyVoice2-0.5B` | 本地模型路径或 ModelScope repo id |

---

## 8 Docker 部署（可选）

### 8.1 构建镜像

```bash
cd runtime/python
docker build -t cosyvoice:v2.0 .
```

### 8.2 运行容器

```bash
# GPU 模式（需 nvidia-container-toolkit）
docker run -d \
    --runtime=nvidia \
    --gpus all \
    -p 8189:8189 \
    -v $(pwd)/pretrained_models:/workspace/CosyVoice/pretrained_models:ro \
    cosyvoice:v2.0 \
    python runtime/python/websocket/server.py \
        --model_dir pretrained_models/CosyVoice2-0.5B \
        --port 8189

# CPU 模式
docker run -d \
    -p 8189:8189 \
    -v $(pwd)/pretrained_models:/workspace/CosyVoice/pretrained_models:ro \
    cosyvoice:v2.0 \
    python runtime/python/websocket/server.py \
        --model_dir pretrained_models/CosyVoice2-0.5B \
        --port 8189
```

> **提示**：将 `pretrained_models` 目录挂载为只读卷，避免容器重建后重复下载模型。

---

## 9 WebSocket API 协议参考

### 连接地址

```
ws://<host>:<port>/v1/realtime
```

### 9.1 客户端 → 服务器事件

#### `session.update` — 配置合成参数

每次连接后可多次调用，参数持久保存在会话内。

```json
{
  "type": "session.update",
  "session": {
    "mode": "sft",
    "voice": "中文女",
    "output_audio_format": "pcm16",
    "speed": 1.0,
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_audio": "<base64 编码的音频文件内容，支持 WAV/FLAC/MP3>",
    "instruct_text": "用四川话说这句话<|endofprompt|>"
  }
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | `"sft"` | `sft` / `zero_shot` / `cross_lingual` / `instruct2` |
| `voice` | string | 模型首个内置发音人 | SFT 模式使用的内置发音人 ID |
| `output_audio_format` | string | `"pcm16"` | 目前仅支持 `"pcm16"`（有符号 16 位小端整数） |
| `speed` | float | `1.0` | 语速倍率，范围 0.5–2.0 |
| `prompt_text` | string | `""` | `zero_shot` 模式中参考音频的文字内容 |
| `prompt_audio` | string | `null` | base64 编码的音频文件，须为 16 kHz 单声道（服务端自动重采样） |
| `instruct_text` | string | `""` | `instruct2` 模式的自然语言指令 |

#### `response.create` — 发起合成请求

```json
{
  "type": "response.create",
  "response": {
    "modalities": ["audio"],
    "input": [
      {
        "type": "text",
        "text": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
      }
    ]
  }
}
```

| 字段 | 说明 |
|------|------|
| `modalities` | 固定 `["audio"]` |
| `input` | 输入数组；多个 `text` 类型条目会拼接后合成 |

### 9.2 服务器 → 客户端事件

#### `session.created` — 连接建立后立即推送

```json
{
  "type": "session.created",
  "session": {
    "id": "a3f2e1...",
    "mode": "sft",
    "voice": "中文女",
    "output_audio_format": "pcm16",
    "speed": 1.0,
    "prompt_text": "",
    "instruct_text": ""
  }
}
```

#### `session.updated` — `session.update` 处理完毕后推送

```json
{
  "type": "session.updated",
  "session": { "...": "当前完整会话配置" }
}
```

#### `response.created` — 合成任务开始

```json
{
  "type": "response.created",
  "response": { "id": "resp_uuid...", "status": "in_progress" }
}
```

#### `response.audio.delta` — 音频数据块（流式推送）

```json
{
  "type": "response.audio.delta",
  "response_id": "resp_uuid...",
  "delta": "<base64 编码的 PCM-16 LE 音频块>"
}
```

- `delta` 解码后为 **有符号 16 位小端整数（PCM-16 LE）**。
- 采样率：CosyVoice2 为 **24000 Hz**，CosyVoice（v1） 为 **22050 Hz**。
- 单声道（Mono）。

#### `response.audio.done` — 所有音频块已发送完毕

```json
{
  "type": "response.audio.done",
  "response_id": "resp_uuid..."
}
```

#### `response.done` — 合成任务完成

```json
{
  "type": "response.done",
  "response": { "id": "resp_uuid...", "status": "completed" }
}
```

#### `error` — 错误事件

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error | server_error",
    "message": "具体错误描述"
  }
}
```

### 9.3 完整交互时序

```
客户端                                              服务端
  │──── WebSocket 连接 ─────────────────────────────▶│
  │◀─── session.created ───────────────────────────  │
  │
  │──── session.update (mode, voice, ...) ─────────▶│
  │◀─── session.updated ───────────────────────────  │
  │
  │──── response.create (tts_text) ────────────────▶│
  │◀─── response.created ──────────────────────────  │
  │◀─── response.audio.delta (chunk 1) ────────────  │  ← ~200 ms 后首帧
  │◀─── response.audio.delta (chunk 2) ────────────  │
  │◀─── response.audio.delta (chunk N) ────────────  │
  │◀─── response.audio.done ───────────────────────  │
  │◀─── response.done ─────────────────────────────  │
  │
  │  （连接保持，可继续发送 response.create）
```

---

## 10 客户端调用示例

### 10.1 Python 客户端（随附）

```bash
cd /path/to/CosyVoice

# SFT 内置发音人
python runtime/python/websocket/client.py \
    --mode sft \
    --tts_text "你好，我是通义生成式语音大模型" \
    --spk_id "中文女" \
    --output_wav sft_output.wav

# 零样本音色克隆
python runtime/python/websocket/client.py \
    --mode zero_shot \
    --tts_text "收到好友从远方寄来的生日礼物，笑容如花儿般绽放。" \
    --prompt_text "希望你以后能够做的比我还好呦。" \
    --prompt_wav asset/zero_shot_prompt.wav \
    --output_wav zero_shot_output.wav

# 跨语言合成（保留参考音色）
python runtime/python/websocket/client.py \
    --mode cross_lingual \
    --tts_text "<|en|>And then later on, fully acquiring that company." \
    --prompt_wav asset/cross_lingual_prompt.wav \
    --output_wav cross_lingual_output.wav

# Instruct2 指令控制
python runtime/python/websocket/client.py \
    --mode instruct2 \
    --tts_text "收到好友从远方寄来的生日礼物，笑容如花儿般绽放。" \
    --instruct_text "用四川话说这句话<|endofprompt|>" \
    --prompt_wav asset/zero_shot_prompt.wav \
    --output_wav instruct2_output.wav
```

### 10.2 Python 自定义集成示例

```python
import asyncio
import base64
import json
import numpy as np
import torch
import torchaudio
import websockets


async def synthesize(text: str, output_path: str, host="127.0.0.1", port=8189):
    uri = f"ws://{host}:{port}/v1/realtime"
    pcm_buffer = b""

    async with websockets.connect(uri) as ws:
        # 等待会话就绪
        await ws.recv()  # session.created

        # 配置（SFT 模式，使用内置中文女声）
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"mode": "sft", "voice": "中文女"}
        }))
        await ws.recv()  # session.updated

        # 发起合成
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "input": [{"type": "text", "text": text}]
            }
        }))

        # 接收音频流
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "response.audio.delta":
                pcm_buffer += base64.b64decode(msg["delta"])
            elif msg["type"] == "response.done":
                break

    # 保存 WAV
    audio = np.frombuffer(pcm_buffer, dtype=np.int16)
    tensor = torch.from_numpy(audio.copy()).unsqueeze(0).float() / (2**15)
    torchaudio.save(output_path, tensor, 24000)
    print(f"Saved {len(audio)/24000:.2f}s audio to {output_path}")


asyncio.run(synthesize("你好，世界！", "hello.wav"))
```

### 10.3 JavaScript / Node.js 示例

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://127.0.0.1:8189/v1/realtime');
const pcmChunks = [];

ws.on('open', () => {
    // 等待 session.created，然后配置
});

ws.on('message', (data) => {
    const msg = JSON.parse(data);

    if (msg.type === 'session.created') {
        // 配置会话
        ws.send(JSON.stringify({
            type: 'session.update',
            session: { mode: 'sft', voice: '中文女' }
        }));

    } else if (msg.type === 'session.updated') {
        // 发起合成
        ws.send(JSON.stringify({
            type: 'response.create',
            response: {
                modalities: ['audio'],
                input: [{ type: 'text', text: '你好，世界！' }]
            }
        }));

    } else if (msg.type === 'response.audio.delta') {
        // 累积 PCM-16 音频块
        pcmChunks.push(Buffer.from(msg.delta, 'base64'));

    } else if (msg.type === 'response.done') {
        const pcm = Buffer.concat(pcmChunks);
        // pcm 为 PCM-16 LE，采样率 24000 Hz，单声道
        console.log(`Received ${pcm.length} bytes of audio`);
        ws.close();
    }
});
```

---

## 11 性能调优建议

### 11.1 GPU 加速（强烈推荐）

确保 CUDA 可用后，模型自动使用 GPU 推理，无需额外配置：

```bash
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

### 11.2 FP16 推理

如果显存充足，可在代码中指定 `fp16=True`（需在 `server.py` 中调整 `AutoModel` 初始化，或在命令行添加相应参数）：

```python
cosyvoice = AutoModel(model_dir=args.model_dir, fp16=True)
```

FP16 可将显存占用减少约 40%，推理速度提升 20–40%。

### 11.3 JIT 编译

```python
cosyvoice = AutoModel(model_dir=args.model_dir, load_jit=True)
```

首次启动时会预编译 JIT 模型（约 1–3 分钟），后续推理速度提升 15–25%。

### 11.4 TensorRT 加速（最高性能）

适用于生产环境，需要 NVIDIA TensorRT 12.x：

```python
cosyvoice = AutoModel(model_dir=args.model_dir, load_trt=True, fp16=True)
```

首次运行会构建 TensorRT 引擎（约 5–10 分钟），构建完成后缓存到模型目录，后续启动秒级就绪。

### 11.5 并发请求

服务端使用 UUID 隔离每个请求的中间状态，天然支持多连接并发。建议：

- **GPU 部署**：可支持 4–8 路并发（取决于显存和 RTF）。
- 使用反向代理（Nginx / Traefik）+ 多进程 Gunicorn 实现水平扩展。

---

## 12 常见问题排查

### Q1：启动时报 `ModuleNotFoundError: No module named 'cosyvoice'`

```bash
# 确认 PYTHONPATH 包含项目根目录
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS:$PYTHONPATH"
```

### Q2：报 `cosyvoice.yaml not found`

模型目录不完整，重新下载：

```bash
python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
"
```

### Q3：`torch.cuda.is_available()` 返回 False

检查 CUDA 驱动版本是否与 PyTorch 版本匹配：

```bash
nvidia-smi              # 查看驱动支持的最高 CUDA 版本
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

若版本不匹配，重新安装 PyTorch：

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Q4：推理速度慢（RTF > 1）

1. 确认 GPU 推理已启用（见 Q3）。
2. 首次推理会加载模型到 GPU 显存（约 10–30 s），后续请求会快很多。
3. 考虑启用 JIT 或 TensorRT（见第 11 节）。

### Q5：客户端收不到音频 / WebSocket 连接超时

1. 检查防火墙是否放开了对应端口（默认 8189）：
   ```bash
   sudo ufw allow 8189/tcp
   ```
2. 确认服务端使用 `--host 0.0.0.0`（而非 `127.0.0.1`）监听。

### Q6：音频质量差 / 出现杂音

- 确保 `prompt_audio` 为 **16 kHz 单声道** 干净录音（无背景噪音），时长 3–10 秒效果最佳。
- 增大 `speed` 参数测试（默认 1.0）。
- 检查合成文本是否包含大量标点符号或特殊字符，尝试清洗文本后重试。

### Q7：`websockets` 不可用

```bash
pip install websockets==12.0
```

---

## 附录：内置发音人列表（CosyVoice2-0.5B）

启动服务后运行以下代码查询：

```python
from cosyvoice.cli.cosyvoice import AutoModel
m = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')
print(m.list_available_spks())
```

常见内置发音人包括：`中文女`、`中文男`、`英文女`、`英文男`、`粤语女`、`四川女`、`东北女`等（具体以模型实际包含为准）。

---

*本文档基于 CosyVoice 代码库（Apache 2.0 License）编写，对应版本：CosyVoice2-0.5B / Fun-CosyVoice3-0.5B-2512。*
