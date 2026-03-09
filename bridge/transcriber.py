import asyncio
import json
import os
import multiprocessing
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import websockets
from faster_whisper import WhisperModel
import ctranslate2
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer
import queue
import threading
import io
import wave

# --- CONFIGURAÇÃO ---
PORT = 8000
MODEL_SIZE = "base" # Padrão Ouro de Streaming: O modelo precisa ser mais rápido que a janela temporal (1.0s)
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MAX_FEED_EVENTS = 500
CORES = max(multiprocessing.cpu_count() - 1, 2)


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
TRANSCRIPTION_FEED = ROOT / "data" / "transcricoes_feed.json"
VOXLY_CONFIG = ROOT / "voxly_config.json"

_feed_state = {"next_id": 1, "events": []}

# --- MEMÓRIA DE CONTEXTO ---
# Usada para evitar alucinações nas primeiras sílabas e manter coerência semântica.
CALL_CENTER_VOCAB = "Atendimento, protocolo, chamadas, headset, sistema, cliente, operador, transferência, cancelamento."
_context_memory = []
MAX_CONTEXT_PHRASES = 3


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


def _append_feed_event(status: str, *, text: str = "", error: str = "", elapsed_s: float | None = None, num_samples: int | None = None, traduction: str = "", time_traduction: float | None = None) -> int:
    event_id = _feed_state["next_id"]
    event = {
        "id": event_id,
        "ts": _utc_now(),
        "source": "faster-whisper",
        "status": status,
        "text": text,
        "error": error,
        "elapsed_s": elapsed_s,
        "num_samples": num_samples,
    }
    if traduction or time_traduction is not None:
        event["traduction"] = traduction
        event["time_traduction"] = time_traduction

    _feed_state["next_id"] += 1
    _feed_state["events"].append(event)
    if len(_feed_state["events"]) > MAX_FEED_EVENTS:
        _feed_state["events"] = _feed_state["events"][-MAX_FEED_EVENTS:]
    _flush_feed_file()
    return event_id


def _update_feed_event_translation(event_id: int, traduction: str, time_traduction: float) -> None:
    for event in _feed_state["events"]:
        if event.get("id") == event_id:
            event["traduction"] = traduction
            event["time_traduction"] = time_traduction
            _flush_feed_file()
            break


print(f"[*] Voxly root resolvido em: {ROOT}", flush=True)

print(f"[*] Carregando Tradutor NLLB-200 em {DEVICE} (Pode demorar no primeiro uso para download)...", flush=True)
model_path = snapshot_download("entai2965/nllb-200-distilled-600M-ctranslate2")
translator_model = ctranslate2.Translator(
    model_path, 
    device=DEVICE, 
    compute_type=COMPUTE_TYPE,
    inter_threads=2,
    intra_threads=CORES
)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

print(f"[*] Carregando Whisper ({MODEL_SIZE}) com {CORES} threads de CPU...", flush=True)
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=CORES)
print(f"[+] Modelos carregados na memória!", flush=True)


# --- SISTEMA ASSÍNCRONO DE PIPELINING ---
# Desacopla a Tradução (Lenta) da Transcrição (Rápida).
translation_queue = asyncio.Queue()

global_outbound_queue = None
main_loop = None

async def translation_worker():
    """Worker rodando perpetuamente em 2º plano aguardando frases para traduzir."""
    while True:
        try:
            job = await translation_queue.get()
            event_id = job["id"]
            text_result = job["text"]
            config = job["config"]
            websocket = job["websocket"]
            elapsed = job["elapsed"]
            
            start_trans = time.time()
            tokenizer.src_lang = config["input_nllb"]
            source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text_result))
            target_prefix = [config["output_nllb"]]
            
            traduction_result = ""
            time_traduction = None
            
            try:
                # OTIMIZAÇÃO: Roda em executor para não travar o event loop assíncrono com matriz
                loop = asyncio.get_running_loop()
                def run_nllb():
                    res = translator_model.translate_batch(
                        [source_tokens], 
                        target_prefix=[target_prefix],
                        beam_size=2,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )
                    return res[0].hypotheses[0][1:]
                    
                target_tokens = await loop.run_in_executor(None, run_nllb)
                traduction_result = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
                time_traduction = round(time.time() - start_trans, 2)
                print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> '{text_result}' | TRADUÇÃO ({time_traduction}s) -> '{traduction_result}'")
            except Exception as trans_err:
                print(f"[ERRO] Falha na tradução paralela: {trans_err}")

            # 3. Atualiza o evento no Feed JSON com a tradução
            _update_feed_event_translation(
                event_id, 
                traduction=traduction_result, 
                time_traduction=time_traduction
            )
            
            # 4. Avisa a interface gráfica Tkinter para atualizar a tela
            try:
                if main_loop and global_outbound_queue:
                    msg = json.dumps({
                        "status": "translation_update", 
                        "id": event_id,
                        "traduction": traduction_result,
                        "time_traduction": time_traduction
                    })
                    main_loop.call_soon_threadsafe(global_outbound_queue.put_nowait, msg)
            except Exception:
                pass
            
            
            translation_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ERRO] Worker de Tradução falhou: {e}")


async def send_worker(websocket):
    try:
        while True:
            if global_outbound_queue is None:
                break
            msg = await global_outbound_queue.get()
            await websocket.send(msg)
    except Exception as e:
        print(f"[CONEXÃO] Erro no enviador WebSocket: {e}")

async def process_audio(websocket):
    global global_outbound_queue, main_loop
    client_ip = websocket.remote_address[0] if websocket.remote_address else "desconhecido"
    print(f"[CONEXÃO] Rust conectado de {client_ip}")
    
    main_loop = asyncio.get_running_loop()
    global_outbound_queue = asyncio.Queue()
    sender_task = asyncio.create_task(send_worker(websocket))

    try:
        async for message in websocket:
            if not isinstance(message, bytes) or len(message) < 1:
                err = "tipo de payload inválido; esperado binário com cabeçalho"
                _append_feed_event("error", error=err)
                global_outbound_queue.put_nowait(json.dumps({"status": "error", "error": err}))
                continue

            try:
                start_time = time.time()
                
                # FASE 3: Desacopla 1-byte de header (is_final) do array de floats f32
                is_final_byte = message[0]
                is_final = bool(is_final_byte == 1)
                audio_bytes = message[1:]

                if len(audio_bytes) % 4 != 0:
                    raise ValueError("payload de áudio inválido (bytes não múltiplos de 4)")

                num_samples = len(audio_bytes) // 4
                if num_samples == 0:
                    continue

                audio_array = np.array(struct.unpack(f"<{num_samples}f", audio_bytes), dtype=np.float32)
                
                if is_final:
                    print(f"[RECEBEU FINAL] {num_samples} amostras ({len(audio_bytes)} bytes). Consolidando...")
                else:
                    print(f"[RECEBEU PARCIAL] {num_samples} amostras. Atualizando tela...", end="\r")

                # Monta o contexto para o Cérebro Whisper
                prompt_history = " ".join(_context_memory)
                dynamic_prompt = f"{CALL_CENTER_VOCAB} {prompt_history}".strip()

                # OTIMIZAÇÃO: beam_size=2 com multi-threading pesado. Mantém fidelidade do 'small' 
                # e reduz pela metade o tempo de processamento. initial_prompt evita erros bizarros isolados.
                segments, _info = model.transcribe(
                    audio_array, 
                    language="pt", 
                    beam_size=2, 
                    vad_filter=True,
                    initial_prompt=dynamic_prompt
                )
                text_result = " ".join(segment.text for segment in segments).strip()
                elapsed = round(time.time() - start_time, 2)

                if not text_result:
                    if is_final:
                        global_outbound_queue.put_nowait(json.dumps({"status": "empty", "text": "", "elapsed_s": elapsed}))
                    continue
                
                # Se for apenas um vislumbre em tempo real, envia pra interface e descarta.
                if not is_final:
                    global_outbound_queue.put_nowait(json.dumps({
                        "status": "partial", 
                        "text": text_result, 
                        "elapsed_s": elapsed
                    }))
                    continue

                # =======================================================
                # DAQUI PARA BAIXO RODA APENAS QUANDO A FRASE FOI CORTADA
                # =======================================================

                # Salva a frase real na memória rotativa (contexto das próximas)
                _context_memory.append(text_result)
                if len(_context_memory) > MAX_CONTEXT_PHRASES:
                    _context_memory.pop(0)

                # 1. Registra no JSON e avisa que a frase solidificou
                event_id = _append_feed_event(
                    "success", 
                    text=text_result, 
                    elapsed_s=elapsed, 
                    num_samples=num_samples
                )
                
                global_outbound_queue.put_nowait(json.dumps({
                    "status": "success", 
                    "id": event_id,
                    "text": text_result, 
                    "elapsed_s": elapsed
                }))

                # Configuração dinâmica de tradução
                config = {"input_nllb": "por_Latn", "output_nllb": "eng_Latn"}
                if VOXLY_CONFIG.exists():
                    try:
                        config.update(json.loads(VOXLY_CONFIG.read_text("utf-8")))
                    except json.JSONDecodeError:
                        pass
                
                # 2. Despacha para o Worker Assíncrono traduzir no fundo sem travar o microfone
                if config["input_nllb"] != config["output_nllb"]:
                    translation_queue.put_nowait({
                        "id": event_id,
                        "text": text_result,
                        "config": config,
                        "websocket": websocket,
                        "elapsed": elapsed
                    })
                else:
                    print(f"[TRANSCRIÇÃO] ({elapsed:.2f}s) -> '{text_result}'")

            except Exception as proc_err:
                err = str(proc_err)
                print(f"[ERRO] Falha no processamento do chunk: {err}")
                _append_feed_event("error", error=err)
                try:
                    global_outbound_queue.put_nowait(json.dumps({"status": "error", "error": err}))
                except Exception as send_err:
                    print(f"[ERRO] Falha ao devolver erro ao cliente: {send_err}")
                
    except websockets.exceptions.ConnectionClosed as closed:
        print(f"[CONEXÃO] Rust desconectou (code={closed.code}, reason={closed.reason}).")
    except Exception as e:
        print(f"[ERRO] Falha na conexão websocket: {e}")
    finally:
        sender_task.cancel()
        global_outbound_queue = None
        main_loop = None


async def main():
    _reset_feed_file()
    print(f"[*] Feed JSON de transcrição: {TRANSCRIPTION_FEED}", flush=True)
    print(f"[*] Iniciando servidor WebSocket Whisper local na porta {PORT}...", flush=True)
    
    # Inicia o Trator de Fila de Tradução Oculto
    worker_task = asyncio.create_task(translation_worker())
    
    async with websockets.serve(process_audio, "127.0.0.1", PORT, ping_interval=None, ping_timeout=None):
        await asyncio.Future()
        
    worker_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
