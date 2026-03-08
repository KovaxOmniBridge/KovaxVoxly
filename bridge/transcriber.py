import asyncio
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import websockets
from faster_whisper import WhisperModel

# --- CONFIGURAÇÃO ---
PORT = 8000
MODEL_SIZE = "base" # Changed from small for even lower latency on CPU
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MAX_FEED_EVENTS = 500


def _resolve_voxly_root() -> Path:
    env_root = os.environ.get("VOXLY_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "Cargo.toml").exists():
            return p

    here = Path(__file__).resolve().parent.parent
    if (here / "Cargo.toml").exists():
        return here

    cwd = Path.cwd().resolve()
    if (cwd / "Cargo.toml").exists():
        return cwd

    return here


ROOT = _resolve_voxly_root()
TRANSCRIPTION_FEED = ROOT / "transcricoes_feed.json"

_feed_state = {"next_id": 1, "events": []}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flush_feed_file() -> None:
    payload = {"updated_at": _utc_now(), "events": _feed_state["events"]}
    TRANSCRIPTION_FEED.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TRANSCRIPTION_FEED.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(TRANSCRIPTION_FEED)


def _reset_feed_file() -> None:
    _feed_state["next_id"] = 1
    _feed_state["events"] = []
    _flush_feed_file()


def _append_feed_event(status: str, *, text: str = "", error: str = "", elapsed_s: float | None = None, num_samples: int | None = None) -> None:
    event = {
        "id": _feed_state["next_id"],
        "ts": _utc_now(),
        "source": "faster-whisper",
        "status": status,
        "text": text,
        "error": error,
        "elapsed_s": elapsed_s,
        "num_samples": num_samples,
    }
    _feed_state["next_id"] += 1
    _feed_state["events"].append(event)
    if len(_feed_state["events"]) > MAX_FEED_EVENTS:
        _feed_state["events"] = _feed_state["events"][-MAX_FEED_EVENTS:]
    _flush_feed_file()


print(f"[*] Voxly root resolvido em: {ROOT}", flush=True)
print(f"[*] Carregando Whisper ({MODEL_SIZE}) em {DEVICE}...", flush=True)
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"[+] Modelo carregado na memória!", flush=True)


async def process_audio(websocket):
    client_ip = websocket.remote_address[0] if websocket.remote_address else "desconhecido"
    print(f"[CONEXÃO] Rust conectado de {client_ip}")

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                err = "tipo de payload inválido; esperado binário"
                _append_feed_event("error", error=err)
                await websocket.send(json.dumps({"status": "error", "error": err}))
                continue

            try:
                start_time = time.time()
                if len(message) % 4 != 0:
                    raise ValueError("payload de áudio inválido (bytes não múltiplos de 4)")

                num_samples = len(message) // 4
                if num_samples == 0:
                    err = "payload vazio"
                    _append_feed_event("error", error=err, num_samples=num_samples)
                    await websocket.send(json.dumps({"status": "error", "error": err}))
                    continue

                audio_array = np.array(struct.unpack(f"<{num_samples}f", message), dtype=np.float32)
                print(f"[RECEBEU] {num_samples} amostras ({len(message)} bytes). Transcrevendo...")

                # OTIMIZAÇÃO: beam_size reduzido de 5 para 2. A busca gulosa (greedy) é muito mais rápida em CPU.
                segments, _info = model.transcribe(audio_array, language="pt", beam_size=2, vad_filter=True)
                text_result = " ".join(segment.text for segment in segments).strip()
                elapsed = round(time.time() - start_time, 2)

                if not text_result:
                    print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> <vazio>")
                    await websocket.send(json.dumps({"status": "empty", "text": "", "elapsed_s": elapsed}))
                    continue

                print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> '{text_result}'")
                _append_feed_event("success", text=text_result, elapsed_s=elapsed, num_samples=num_samples)
                await websocket.send(json.dumps({"status": "success", "text": text_result, "elapsed_s": elapsed}))

            except Exception as proc_err:
                err = str(proc_err)
                print(f"[ERRO] Falha no processamento do chunk: {err}")
                _append_feed_event("error", error=err)
                try:
                    await websocket.send(json.dumps({"status": "error", "error": err}))
                except Exception as send_err:
                    print(f"[ERRO] Falha ao devolver erro ao cliente: {send_err}")

    except websockets.exceptions.ConnectionClosed as closed:
        print(f"[CONEXÃO] Rust desconectou (code={closed.code}, reason={closed.reason}).")
    except Exception as e:
        print(f"[ERRO] Falha na conexão websocket: {e}")


async def main():
    _reset_feed_file()
    print(f"[*] Feed JSON de transcrição: {TRANSCRIPTION_FEED}", flush=True)
    print(f"[*] Iniciando servidor WebSocket Whisper local na porta {PORT}...", flush=True)
    async with websockets.serve(process_audio, "127.0.0.1", PORT, ping_interval=None, ping_timeout=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
