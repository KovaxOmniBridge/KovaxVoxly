import asyncio
import websockets
import json
from faster_whisper import WhisperModel
import struct
import numpy as np
import time

# --- CONFIGURAÇÃO ---
PORT = 8000
MODEL_SIZE = "large-v3-turbo"
# Projeto configurado para CPU-only
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

print(f"[*] Carregando Whisper ({MODEL_SIZE}) em {DEVICE}...", flush=True)
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"[+] Modelo carregado na memória!", flush=True)

async def process_audio(websocket):
    client_ip = websocket.remote_address[0] if websocket.remote_address else "desconhecido"
    print(f"[CONEXÃO] Rust conectado de {client_ip}")

    try:
        async for message in websocket:
            # Rust envia mensagens binárias contendo array de f32 LE.
            if not isinstance(message, bytes):
                await websocket.send(json.dumps({
                    "status": "error",
                    "error": "tipo de payload inválido; esperado binário"
                }))
                continue

            try:
                start_time = time.time()
                if len(message) % 4 != 0:
                    raise ValueError("payload de áudio inválido (bytes não múltiplos de 4)")

                num_samples = len(message) // 4
                if num_samples == 0:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "error": "payload vazio"
                    }))
                    continue

                audio_array = np.array(struct.unpack(f'<{num_samples}f', message), dtype=np.float32)

                print(f"[RECEBEU] {num_samples} amostras ({len(message)} bytes). Transcrevendo...")

                segments, _info = model.transcribe(
                    audio_array,
                    language="pt",
                    beam_size=5,
                    vad_filter=True,  # Redundante, mas ajuda a limpar ruído final.
                )

                text_result = " ".join(segment.text for segment in segments).strip()
                elapsed = time.time() - start_time

                if not text_result:
                    print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> <vazio>")
                    await websocket.send(json.dumps({
                        "status": "empty",
                        "text": "",
                        "elapsed_s": round(elapsed, 2)
                    }))
                    continue

                print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> '{text_result}'")
                await websocket.send(json.dumps({
                    "status": "success",
                    "text": text_result,
                    "elapsed_s": round(elapsed, 2)
                }))

            except Exception as proc_err:
                print(f"[ERRO] Falha no processamento do chunk: {proc_err}")
                try:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "error": str(proc_err)
                    }))
                except Exception as send_err:
                    print(f"[ERRO] Falha ao devolver erro ao cliente: {send_err}")

    except websockets.exceptions.ConnectionClosed as closed:
        print(f"[CONEXÃO] Rust desconectou (code={closed.code}, reason={closed.reason}).")
    except Exception as e:
        print(f"[ERRO] Falha na conexão websocket: {e}")

async def main():
    print(f"[*] Iniciando servidor WebSocket Whisper local na porta {PORT}...", flush=True)
    # O cliente Rust só lê respostas após enviar áudio, então ele pode ficar
    # vários segundos sem consumir frames de controle. Desabilitar ping/pong
    # automático evita encerramentos prematuros do WS por timeout de keepalive.
    async with websockets.serve(
        process_audio,
        "127.0.0.1",
        PORT,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()  # Roda para sempre

if __name__ == "__main__":
    asyncio.run(main())
