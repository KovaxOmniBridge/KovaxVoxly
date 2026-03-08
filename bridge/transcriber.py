import asyncio
import websockets
import json
import io
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
    client_ip = websocket.remote_address[0]
    print(f"[CONEXÃO] Rust conectado de {client_ip}")
    
    try:
        async for message in websocket:
            # Rust enviará mensagens binárias puras sendo arrays de f32
            if isinstance(message, bytes):
                start_time = time.time()
                
                # O formato do struct depende se a arquitetura é little-endian (padrão)
                # Cada sample é um float de 32 bits (4 bytes)
                num_samples = len(message) // 4
                audio_array = np.array(struct.unpack(f'<{num_samples}f', message), dtype=np.float32)
                
                print(f"[RECEBEU] {num_samples} amostras ({len(message)} bytes). Transcrevendo...")
                
                # A função transcribe aceita numpy arrays diretamente se a SR=16000
                segments, info = model.transcribe(
                    audio_array, 
                    language="pt", 
                    beam_size=5,
                    vad_filter=True # Redundante, mas ajuda a limpar ruído final
                )
                
                text_result = ""
                for segment in segments:
                    text_result += segment.text + " "
                    
                text_result = text_result.strip()
                elapsed = time.time() - start_time
                
                print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> '{text_result}'")
                
                # Devolve a string final para o Rust pintar na tela
                await websocket.send(json.dumps({
                    "status": "success",
                    "text": text_result,
                    "elapsed_s": round(elapsed, 2)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[CONEXÃO] Rust desconectou.")
    except Exception as e:
        print(f"[ERRO] Falha no processamento: {e}")

async def main():
    print(f"[*] Iniciando servidor WebSocket Whisper local na porta {PORT}...", flush=True)
    async with websockets.serve(process_audio, "127.0.0.1", PORT):
        await asyncio.Future()  # Roda para sempre

if __name__ == "__main__":
    asyncio.run(main())
