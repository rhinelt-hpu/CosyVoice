# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CosyVoice Realtime TTS WebSocket Client (reference implementation)
===================================================================
Demonstrates all four synthesis modes against the server in
runtime/python/websocket/server.py.

Requires:
    pip install websockets

Usage examples
--------------
# SFT mode (built-in speaker)
python client.py --mode sft --tts_text "你好，世界" --spk_id "中文女"

# Zero-shot voice cloning
python client.py --mode zero_shot \\
    --tts_text "收到好友从远方寄来的生日礼物，笑容如花儿般绽放。" \\
    --prompt_text "希望你以后能够做的比我还好呦。" \\
    --prompt_wav ../../../asset/zero_shot_prompt.wav

# Cross-lingual synthesis
python client.py --mode cross_lingual \\
    --tts_text "<|en|>And then later on, fully acquiring that company." \\
    --prompt_wav ../../../asset/cross_lingual_prompt.wav

# Instruct2 (naturalness control)
python client.py --mode instruct2 \\
    --tts_text "收到好友从远方寄来的生日礼物，笑容如花儿般绽放。" \\
    --instruct_text "用四川话说这句话<|endofprompt|>" \\
    --prompt_wav ../../../asset/zero_shot_prompt.wav
"""
import asyncio
import argparse
import base64
import io
import json
import logging
import time

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

try:
    import websockets
except ImportError:
    raise SystemExit(
        "The 'websockets' package is required.\n"
        "Install it with:  pip install websockets"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_audio_file(path: str) -> str:
    """Read an audio file and return its base64 representation."""
    with open(path, 'rb') as fh:
        return base64.b64encode(fh.read()).decode('utf-8')


def save_pcm16(pcm_bytes: bytes, sample_rate: int, output_path: str) -> None:
    """Convert raw PCM-16 LE bytes to a WAV file."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    tensor = torch.from_numpy(audio.copy()).unsqueeze(0).float() / (2 ** 15)
    torchaudio.save(output_path, tensor, sample_rate)
    duration = len(audio) / sample_rate
    logging.info("Saved %.2fs of audio (%d samples) → %s",
                 duration, len(audio), output_path)


async def _collect_audio(ws) -> bytes:
    """
    Receive WebSocket frames until *response.done* and accumulate all
    *response.audio.delta* payloads.  Returns raw PCM-16 bytes.
    """
    pcm_buffer = b""
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        t = msg.get("type", "")

        if t == "response.created":
            logging.info("Response started  id=%s", msg["response"]["id"])

        elif t == "response.audio.delta":
            pcm_buffer += base64.b64decode(msg["delta"])

        elif t == "response.audio.done":
            logging.info("All audio chunks received  total=%d bytes", len(pcm_buffer))

        elif t == "response.done":
            logging.info("Response done  id=%s  status=%s",
                         msg["response"]["id"], msg["response"]["status"])
            break

        elif t == "error":
            logging.error("Server error: %s", msg["error"])
            break

    return pcm_buffer


# ---------------------------------------------------------------------------
# Per-mode coroutines
# ---------------------------------------------------------------------------

async def run_sft(uri: str, tts_text: str, spk_id: str,
                  speed: float, output_wav: str, sample_rate: int) -> None:
    async with websockets.connect(uri) as ws:
        # --- session handshake ---
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created", msg
        logging.info("Connected  session_id=%s", msg["session"]["id"])

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"mode": "sft", "voice": spk_id, "speed": speed},
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.updated", msg

        # --- request synthesis ---
        t0 = time.time()
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "input": [{"type": "text", "text": tts_text}],
            },
        }))

        pcm = await _collect_audio(ws)
        logging.info("Total latency: %.2fs", time.time() - t0)

    save_pcm16(pcm, sample_rate, output_wav)


async def run_zero_shot(uri: str, tts_text: str, prompt_text: str,
                        prompt_wav: str, speed: float,
                        output_wav: str, sample_rate: int) -> None:
    prompt_b64 = encode_audio_file(prompt_wav)

    async with websockets.connect(uri) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created", msg
        logging.info("Connected  session_id=%s", msg["session"]["id"])

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "mode": "zero_shot",
                "prompt_text": prompt_text,
                "prompt_audio": prompt_b64,
                "speed": speed,
            },
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.updated", msg

        t0 = time.time()
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "input": [{"type": "text", "text": tts_text}],
            },
        }))

        pcm = await _collect_audio(ws)
        logging.info("Total latency: %.2fs", time.time() - t0)

    save_pcm16(pcm, sample_rate, output_wav)


async def run_cross_lingual(uri: str, tts_text: str, prompt_wav: str,
                             speed: float, output_wav: str,
                             sample_rate: int) -> None:
    prompt_b64 = encode_audio_file(prompt_wav)

    async with websockets.connect(uri) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created", msg
        logging.info("Connected  session_id=%s", msg["session"]["id"])

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "mode": "cross_lingual",
                "prompt_audio": prompt_b64,
                "speed": speed,
            },
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.updated", msg

        t0 = time.time()
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "input": [{"type": "text", "text": tts_text}],
            },
        }))

        pcm = await _collect_audio(ws)
        logging.info("Total latency: %.2fs", time.time() - t0)

    save_pcm16(pcm, sample_rate, output_wav)


async def run_instruct2(uri: str, tts_text: str, instruct_text: str,
                         prompt_wav: str, speed: float,
                         output_wav: str, sample_rate: int) -> None:
    prompt_b64 = encode_audio_file(prompt_wav)

    async with websockets.connect(uri) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created", msg
        logging.info("Connected  session_id=%s", msg["session"]["id"])

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "mode": "instruct2",
                "instruct_text": instruct_text,
                "prompt_audio": prompt_b64,
                "speed": speed,
            },
        }))
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.updated", msg

        t0 = time.time()
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "input": [{"type": "text", "text": tts_text}],
            },
        }))

        pcm = await _collect_audio(ws)
        logging.info("Total latency: %.2fs", time.time() - t0)

    save_pcm16(pcm, sample_rate, output_wav)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CosyVoice Realtime TTS WebSocket client"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8189)
    parser.add_argument(
        "--mode", default="sft",
        choices=["sft", "zero_shot", "cross_lingual", "instruct2"],
        help="Synthesis mode (default: sft)",
    )
    parser.add_argument("--tts_text",
                        default="你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？")
    parser.add_argument("--spk_id", default="中文女",
                        help="Speaker id for sft mode")
    parser.add_argument("--prompt_text", default="希望你以后能够做的比我还好呦。",
                        help="Prompt transcript for zero_shot mode")
    parser.add_argument("--prompt_wav",
                        default="../../../asset/zero_shot_prompt.wav",
                        help="Path to prompt WAV (zero_shot / cross_lingual / instruct2)")
    parser.add_argument("--instruct_text",
                        default="用四川话说这句话<|endofprompt|>",
                        help="Instruction text for instruct2 mode")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier (0.5–2.0, default: 1.0)")
    parser.add_argument("--output_wav", default="output.wav",
                        help="Path to save the synthesised audio (default: output.wav)")
    parser.add_argument("--sample_rate", type=int, default=24000,
                        help="Output sample rate in Hz (default: 24000 for CosyVoice2)")
    args = parser.parse_args()

    uri = "ws://{}:{}/v1/realtime".format(args.host, args.port)
    logging.info("Connecting to %s  mode=%s", uri, args.mode)

    if args.mode == "sft":
        asyncio.run(run_sft(uri, args.tts_text, args.spk_id,
                            args.speed, args.output_wav, args.sample_rate))
    elif args.mode == "zero_shot":
        asyncio.run(run_zero_shot(uri, args.tts_text, args.prompt_text,
                                  args.prompt_wav, args.speed,
                                  args.output_wav, args.sample_rate))
    elif args.mode == "cross_lingual":
        asyncio.run(run_cross_lingual(uri, args.tts_text, args.prompt_wav,
                                      args.speed, args.output_wav, args.sample_rate))
    elif args.mode == "instruct2":
        asyncio.run(run_instruct2(uri, args.tts_text, args.instruct_text,
                                  args.prompt_wav, args.speed,
                                  args.output_wav, args.sample_rate))


if __name__ == "__main__":
    main()
