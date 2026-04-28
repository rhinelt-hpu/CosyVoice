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
CosyVoice Realtime TTS WebSocket Server
========================================
Implements an OpenAI-compatible Realtime API for text-to-speech streaming.

Endpoint : ws://<host>:<port>/v1/realtime

Supported client → server events
  session.update   – configure voice / mode / prompt audio
  response.create  – start a TTS synthesis (returns streaming audio)

Supported server → client events
  session.created        – emitted on new connection
  session.updated        – emitted after session.update
  response.created       – emitted when synthesis starts
  response.audio.delta   – PCM-16 audio chunk, base64-encoded
  response.audio.done    – all audio chunks have been sent
  response.done          – synthesis completed
  error                  – request or server-side error
"""
import os
import sys
import io
import base64
import json
import uuid
import asyncio
import threading
import argparse
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI(title="CosyVoice Realtime TTS", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model handle, initialized in __main__
cosyvoice = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_prompt_audio(audio_b64: str, target_sr: int = 16000) -> torch.Tensor:
    """Decode a base64-encoded audio file (WAV/FLAC/MP3 …) and resample."""
    raw = base64.b64decode(audio_b64)
    buf = io.BytesIO(raw)
    speech, sr = torchaudio.load(buf, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)
    return speech


def _safe_session_dict(session: dict) -> dict:
    """Return a JSON-serializable copy of the session (drop tensors)."""
    return {k: v for k, v in session.items() if k != 'prompt_audio'}


# ---------------------------------------------------------------------------
# Async streaming helper
# ---------------------------------------------------------------------------

async def _stream_model_output(websocket: WebSocket,
                                model_output,
                                response_id: str) -> None:
    """
    Run the synchronous CosyVoice generator in a background thread and forward
    each PCM-16 audio chunk to the WebSocket as a *response.audio.delta* event.

    The generator may block for tens of milliseconds per chunk; running it in
    a thread keeps the asyncio event loop unblocked for other connections.
    """
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def producer() -> None:
        try:
            for chunk in model_output:
                pcm_bytes = (chunk['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                delta = base64.b64encode(pcm_bytes).decode('utf-8')
                asyncio.run_coroutine_threadsafe(q.put(('delta', delta)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(q.put(('error', str(exc))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(('done', None)), loop)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        event_type, data = await q.get()
        if event_type == 'delta':
            await websocket.send_json({
                "type": "response.audio.delta",
                "response_id": response_id,
                "delta": data,
            })
        elif event_type == 'error':
            await websocket.send_json({
                "type": "error",
                "error": {"type": "server_error", "message": data},
            })
            break
        else:  # 'done'
            break

    t.join()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket handler.

    Each connection owns its own *session* dict that persists across multiple
    *response.create* requests during the lifetime of the connection.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    default_voice = (cosyvoice.list_available_spks() or ["中文女"])[0]

    # Per-connection session state
    session = {
        "id": session_id,
        "mode": "sft",               # sft | zero_shot | cross_lingual | instruct2
        "voice": default_voice,      # speaker id (sft mode)
        "output_audio_format": "pcm16",
        "speed": 1.0,
        "prompt_text": "",           # zero_shot / instruct2
        "prompt_audio": None,        # torch.Tensor (16 kHz mono), zero_shot / cross_lingual / instruct2
        "instruct_text": "",         # instruct2 mode
    }

    await websocket.send_json({
        "type": "session.created",
        "session": _safe_session_dict(session),
    })
    logging.info("WebSocket session %s opened", session_id)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "Invalid JSON"},
                })
                continue

            msg_type = msg.get("type", "")

            # ── session.update ──────────────────────────────────────────────
            if msg_type == "session.update":
                updates = msg.get("session", {})

                for key in ("mode", "voice", "speed", "prompt_text",
                            "instruct_text", "output_audio_format"):
                    if key in updates:
                        session[key] = updates[key]

                if updates.get("prompt_audio"):
                    try:
                        session["prompt_audio"] = _decode_prompt_audio(
                            updates["prompt_audio"], target_sr=16000
                        )
                    except Exception as exc:
                        await websocket.send_json({
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Failed to decode prompt_audio: {}".format(exc),
                            },
                        })
                        continue

                await websocket.send_json({
                    "type": "session.updated",
                    "session": _safe_session_dict(session),
                })

            # ── response.create ─────────────────────────────────────────────
            elif msg_type == "response.create":
                response_cfg = msg.get("response", {})

                # Collect TTS text from all text-type input items
                tts_text = "".join(
                    item.get("text", "")
                    for item in response_cfg.get("input", [])
                    if item.get("type") == "text"
                )

                if not tts_text.strip():
                    await websocket.send_json({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "response.create must include at least one text input item",
                        },
                    })
                    continue

                response_id = str(uuid.uuid4())
                await websocket.send_json({
                    "type": "response.created",
                    "response": {"id": response_id, "status": "in_progress"},
                })

                mode = session["mode"]
                speed = float(session.get("speed", 1.0))
                prompt_audio = session["prompt_audio"]

                try:
                    if mode == "sft":
                        model_output = cosyvoice.inference_sft(
                            tts_text, session["voice"],
                            stream=True, speed=speed,
                        )
                    elif mode == "zero_shot":
                        if prompt_audio is None:
                            raise ValueError("prompt_audio must be set before using zero_shot mode")
                        model_output = cosyvoice.inference_zero_shot(
                            tts_text, session["prompt_text"], prompt_audio,
                            stream=True, speed=speed,
                        )
                    elif mode == "cross_lingual":
                        if prompt_audio is None:
                            raise ValueError("prompt_audio must be set before using cross_lingual mode")
                        model_output = cosyvoice.inference_cross_lingual(
                            tts_text, prompt_audio,
                            stream=True, speed=speed,
                        )
                    elif mode == "instruct2":
                        if prompt_audio is None:
                            raise ValueError("prompt_audio must be set before using instruct2 mode")
                        model_output = cosyvoice.inference_instruct2(
                            tts_text, session["instruct_text"], prompt_audio,
                            stream=True, speed=speed,
                        )
                    else:
                        raise ValueError("Unknown mode: '{}'. Must be one of: sft, zero_shot, cross_lingual, instruct2".format(mode))

                except Exception as exc:
                    await websocket.send_json({
                        "type": "error",
                        "error": {"type": "server_error", "message": str(exc)},
                    })
                    continue

                await _stream_model_output(websocket, model_output, response_id)

                await websocket.send_json({
                    "type": "response.audio.done",
                    "response_id": response_id,
                })
                await websocket.send_json({
                    "type": "response.done",
                    "response": {"id": response_id, "status": "completed"},
                })
                logging.info("Session %s response %s completed", session_id, response_id)

            # ── unknown event ────────────────────────────────────────────────
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Unknown event type: '{}'".format(msg_type),
                    },
                })

    except WebSocketDisconnect:
        logging.info("WebSocket session %s disconnected", session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CosyVoice Realtime TTS WebSocket Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Interface to listen on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8189,
                        help="TCP port (default: 8189)")
    parser.add_argument("--model_dir", type=str, default="iic/CosyVoice2-0.5B",
                        help="Local model directory or ModelScope repo id")
    args = parser.parse_args()

    cosyvoice = AutoModel(model_dir=args.model_dir)
    logging.info("Model loaded from %s, sample_rate=%d",
                 args.model_dir, cosyvoice.sample_rate)
    logging.info("Available speakers: %s", cosyvoice.list_available_spks())
    logging.info("Serving at ws://%s:%d/v1/realtime", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)
