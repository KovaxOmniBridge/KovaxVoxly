import os
import asyncio
import sys
import json
import struct
import time
import datetime
import threading
import queue
from pathlib import Path
from datetime import timezone

# Força UTF-8 para conter erros de encoding no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np # type: ignore
import websockets # type: ignore
from faster_whisper import WhisperModel # type: ignore
import ctranslate2 # type: ignore
from huggingface_hub import snapshot_download, hf_hub_download # type: ignore
from transformers import AutoTokenizer # type: ignore
from piper.voice import PiperVoice # type: ignore
from piper.config import SynthesisConfig # type: ignore

# --- CONFIGURAÇÃO ---
PORT = 8000
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MAX_FEED_EVENTS = 500
CORES = 4

def _resolve_voxly_root() -> Path:
    env_root = os.environ.get("VOXLY_ROOT")
    if env_root: return Path(env_root).resolve()
    if getattr(sys, 'frozen', False):
        exe_path = Path(sys.executable).resolve()
        if exe_path.parent.name.lower() == "bridge": return exe_path.parent.parent
        return exe_path.parent
    here = Path(__file__).resolve().parent
    if here.name.lower() == "bridge": return here.parent
    return here

ROOT = _resolve_voxly_root()
TRANSCRIPTION_FEED = ROOT / "data" / "transcricoes_feed.json"
VOXLY_CONFIG = ROOT / "voxly_config.json"

MODEL_SIZE = "base"
NLLB_HF_REPO = "entai2965/nllb-200-distilled-600M-ctranslate2"
NLLB_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"

if VOXLY_CONFIG.exists():
    try:
        conf = json.loads(VOXLY_CONFIG.read_text("utf-8"))
        profile = conf.get("performance_profile", "low")
        # Caminhos Locais
        local_models = ROOT / "models"
        
        if profile == "med":
            MODEL_SIZE = "small"
            local_nllb = local_models / "nllb-1.3B"
            if local_nllb.exists():
                NLLB_HF_REPO = str(local_nllb)
                print(f"[*] Usando modelo local NLLB 1.3B: {NLLB_HF_REPO}", flush=True)
            else:
                NLLB_HF_REPO = "entai2965/nllb-200-distilled-1.3B-ctranslate2"
            NLLB_TOKENIZER_REPO = "facebook/nllb-200-distilled-1.3B"

        elif profile == "high" or profile == "max":
            MODEL_SIZE = "medium"
            local_nllb = local_models / "nllb-3.3B"
            if local_nllb.exists():
                NLLB_HF_REPO = str(local_nllb)
                print(f"[*] Usando modelo local NLLB 3.3B: {NLLB_HF_REPO}", flush=True)
            else:
                NLLB_HF_REPO = "entai2965/nllb-200-3.3B-ctranslate2"
            NLLB_TOKENIZER_REPO = "facebook/nllb-200-3.3B"
        
        # Override se explicitamente definido
        MODEL_SIZE = conf.get("whisper_model", MODEL_SIZE)
    except: pass

from typing import Any
_feed_state: dict[str, Any] = {"next_id": 1, "events": []}

# --- MEMÓRIA DE CONTEXTO ---
_context_memory_me = []
_context_memory_them = []
_last_trad_me = ""
_last_trad_them = ""
MAX_CONTEXT = 5
R_VOCAB = "Atendimento, protocolo, chamadas, headset, sistema, cliente, operador, transferência, cancelamento."

# Mapa de idiomas Whisper (ISO 639-1) para NLLB
WHISPER_TO_NLLB = {
    "pt": "por_Latn", "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "ru": "rus_Cyrl", "ko": "kor_Kore", "ar": "ara_Arab", "hi": "hin_Deva"
}

def _utc_now() -> str:
    return datetime.datetime.now(timezone.utc).isoformat()

def _flush_feed_file() -> None:
    payload = {"updated_at": _utc_now(), "events": _feed_state["events"]}
    TRANSCRIPTION_FEED.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TRANSCRIPTION_FEED.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Retry loop para evitar conflito com a interface no Windows
        for _ in range(5):
            try:
                if TRANSCRIPTION_FEED.exists(): TRANSCRIPTION_FEED.unlink()
                temp_path.rename(TRANSCRIPTION_FEED)
                break
            except Exception:
                time.sleep(0.05)
    except Exception as e:
        print(f"[ERRO FEED] {e}")

def _reset_feed_file() -> None:
    _feed_state["next_id"] = 1
    _feed_state["events"] = []
    _flush_feed_file()

def _append_feed_event(status, text=None, traduction=None, source_id=0):
    event_id = int(time.time() * 1000)
    event = {
        "id": event_id,
        "status": status,
        "source_id": source_id,
        "text": text,
        "traduction": traduction,
        "timestamp": datetime.datetime.now().isoformat()
    }
    _feed_state["events"].append(event)
    if len(_feed_state["events"]) > MAX_FEED_EVENTS:
        _feed_state["events"] = _feed_state["events"][-MAX_FEED_EVENTS:]
    _flush_feed_file()
    return event_id

def _update_feed_event_translation(event_id: int, traduction: str, time_traduction: float | None) -> None:
    for event in _feed_state["events"]:
        if event.get("id") == event_id:
            event["traduction"] = traduction
            event["time_traduction"] = time_traduction
            _flush_feed_file()
            break

print(f"[*] Voxly root: {ROOT}", flush=True)
print(f"[*] Perfil Selecionado: {MODEL_SIZE} | {Path(NLLB_TOKENIZER_REPO).name}", flush=True)
print(f"[*] Carregando Tradutor NLLB...", flush=True)

# Se for um caminho local que existe, evita o snapshot_download
if os.path.isdir(NLLB_HF_REPO):
    model_path = NLLB_HF_REPO
    # Se temos o modelo local, usamos o tokenizer local também
    NLLB_TOKENIZER_REPO = NLLB_HF_REPO
else:
    try:
        model_path = snapshot_download(NLLB_HF_REPO)
    except Exception as e:
        print(f"[AVISO] Falha ao carregar modelo {NLLB_HF_REPO}. Usando fallback 600M. Erro: {e}", flush=True)
        model_path = snapshot_download("entai2965/nllb-200-distilled-600M-ctranslate2")
        NLLB_TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"

translator_model = ctranslate2.Translator(model_path, device=DEVICE, compute_type=COMPUTE_TYPE, inter_threads=2, intra_threads=CORES)
tokenizer = AutoTokenizer.from_pretrained(
    NLLB_TOKENIZER_REPO, 
    trust_remote_code=True, 
    local_files_only=os.path.isdir(NLLB_TOKENIZER_REPO)
)

print(f"[*] Carregando Whisper ({MODEL_SIZE})... (Pode levar alguns minutos se for necessário baixar o modelo)", flush=True)
try:
    model = WhisperModel(
        MODEL_SIZE, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE, 
        cpu_threads=CORES,
        local_files_only=True
    )
except Exception as e:
    print(f"[AVISO] Falha ao carregar Whisper {MODEL_SIZE}. Usando fallback base. Erro: {e}", flush=True)
    model = WhisperModel(
        "base", 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE, 
        cpu_threads=CORES,
        local_files_only=True
    )

print(f"[+] Modelos prontos!", flush=True)

translation_queue = asyncio.Queue()
tts_queue = queue.Queue()
global_outbound_queue = None
global_ws = None
main_loop = None

def download_piper_model(lang_code, voice_name=None):
    PIPER_VOICES_MAP = {
        "Faber (Masculino)": ("pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx", "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json"),
        "Alan (Masculino/Low)": ("pt/pt_BR/alan/low/pt_BR-alan-low.onnx", "pt/pt_BR/alan/low/pt_BR-alan-low.onnx.json"),
        "Lessac (Feminino)": ("en/en_US/lessac/medium/en_US-lessac-medium.onnx", "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"),
        "Ryan (Masculino)": ("en/en_US/ryan/high/en_US-ryan-high.onnx", "en/en_US/ryan/high/en_US-ryan-high.onnx.json")
    }
    LANG_FALLBACKS = {"pt_BR": "Faber (Masculino)", "en_US": "Lessac (Feminino)"}
    key = voice_name if voice_name in PIPER_VOICES_MAP else lang_code
    if key in PIPER_VOICES_MAP: onnx_file, json_file = PIPER_VOICES_MAP[key]
    else: onnx_file, json_file = PIPER_VOICES_MAP["Lessac (Feminino)"]
    try:
        onnx_p = hf_hub_download(repo_id="rhasspy/piper-voices", filename=onnx_file)
        json_p = hf_hub_download(repo_id="rhasspy/piper-voices", filename=json_file)
        return onnx_p, json_p
    except: return download_piper_model("en_US", "Lessac (Feminino)")

def tts_worker():
    current_lang = None
    voice: Any = None
    while True:
        try:
            job = tts_queue.get()
            if job is None: break
            text, lang_code, source_id = job["text"], job["lang_code"], job["source_id"]
            config = {}
            if VOXLY_CONFIG.exists():
                try: config = json.loads(VOXLY_CONFIG.read_text("utf-8"))
                except: pass
            
            voice_name = config.get("tts_voice" if source_id == 0 else "tts_voice_inbound", "Padrão")
            speed = float(config.get("tts_speed", 1.0))
            piper_key = voice_name if voice_name != "Padrão" else lang_code

            if current_lang != piper_key or voice is None:
                onnx_path, json_path = download_piper_model(lang_code, voice_name=piper_key)
                voice = PiperVoice.load(onnx_path, config_path=json_path)
                current_lang = piper_key

            with open(json_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            sample_rate = cfg.get("audio", {}).get("sample_rate", 22050)
            
            for chunk in voice.synthesize(text, syn_config=SynthesisConfig(length_scale=speed)):
                audio_f32 = chunk.audio_float_array
                header = struct.pack('<BI', 2, sample_rate)
                msg = header + audio_f32.tobytes()
                if main_loop and global_outbound_queue:
                    main_loop.call_soon_threadsafe(global_outbound_queue.put_nowait, msg)
            tts_queue.task_done()
        except Exception as e:
            print(f"[ERRO TTS] {e}")

async def translate_item(item):
    global _last_trad_me, _last_trad_them
    try:
        t_id, text, n_src, n_tgt, s_id = item["id"], item["text"], item["nllb_src"], item["nllb_tgt"], item["source_id"]
        context = item.get("context", "").strip()
        last_trad = _last_trad_me if s_id == 0 else _last_trad_them
        start_t = time.time()
        
        if n_src == n_tgt: return

        # Se houver contexto, enviamos o par para inteligência idiomática
        full_text = f"{context}. {text}".strip() if context else text
        source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(full_text))
        
        loop = asyncio.get_running_loop()
        def run_nllb():
            # Usamos beam_size 5 para profundidade
            res = translator_model.translate_batch([source_tokens], target_prefix=[[n_tgt]], beam_size=5, repetition_penalty=1.1)
            return res[0].hypotheses[0][1:]
            
        target_tokens = await loop.run_in_executor(None, run_nllb)
        full_trad = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens)).strip()
        
        # EXTRAÇÃO INTELIGENTE: Remove o "eco" da tradução anterior
        if context and last_trad:
            # Se a tradução completa começa com a tradução anterior (ou similar), removemos
            # Usamos uma margem de segurança para pontuação
            if full_trad.lower().startswith(last_trad.lower()[:10]):
                # Tenta remover a parte antiga
                new_part = full_trad[len(last_trad):].strip()
                # Limpa pontuação residual no início (. , ? !)
                trad = new_part.lstrip(".?!,; ")
                if not trad: trad = full_trad # Fallback se a limpeza removeu tudo
            else:
                # Se mudou muito, tenta quebrar por pontos e pegar a última
                parts = full_trad.split(".")
                trad = parts[-1].strip() if len(parts) > 1 else full_trad
        else:
            trad = full_trad

        # Atualiza a memória da última tradução com a frase atual limpa
        if s_id == 0: _last_trad_me = trad
        else: _last_trad_them = trad

        t_dur = round(time.time() - start_t, 2)
        _update_feed_event_translation(t_id, trad, t_dur)
        _flush_feed_file()
        
        if main_loop and global_outbound_queue:
            msg = f"[TRANS] " + json.dumps({"status": "translation_update", "id": t_id, "source_id": s_id, "traduction": trad})
            main_loop.call_soon_threadsafe(global_outbound_queue.put_nowait, msg)
        
        if trad:
            tts_queue.put({"text": trad, "lang_code": n_tgt, "source_id": s_id})
    except Exception as e: print(f"[ERRO TRAD] {e}")

async def translation_worker():
    while True:
        item = await translation_queue.get()
        # Dispara tarefa de tradução sem bloquear o worker
        asyncio.create_task(translate_item(item))
        translation_queue.task_done()

async def send_worker(websocket):
    while True:
        if global_outbound_queue is None: break
        msg = await global_outbound_queue.get()
        await websocket.send(msg)

async def process_single_audio(message, websocket):
    global global_outbound_queue
    if not isinstance(message, bytes) or len(message) < 2: return
    try:
        source_id, is_final = int(message[0]), bool(message[1] == 1)
        audio_array = np.array(struct.unpack(f"<{len(message[2:])//4}f", message[2:]), dtype=np.float32)
        
        config = {"input_nllb": "por_Latn", "output_nllb": "eng_Latn", "input_whisper": "pt", "output_whisper": "en"}
        if VOXLY_CONFIG.exists():
            try: config.update(json.loads(VOXLY_CONFIG.read_text("utf-8")))
            except: pass

        if source_id == 0:
            w_lang, n_src, n_tgt, mem = config["input_whisper"], config["input_nllb"], config["output_nllb"], _context_memory_me
        else:
            # Reunião: Auto-Detect idioma e traduz SEMPRE para o idioma do usuário
            w_lang, n_src, n_tgt, mem = None, None, config["input_nllb"], _context_memory_them

        prompt = f"{R_VOCAB} {' '.join(mem[-5:])}".strip()
        loop = asyncio.get_running_loop()
        # Transcrição em background para não travar o loop de pings do websocket
        segments, info = await loop.run_in_executor(None, lambda: model.transcribe(audio_array, language=w_lang, beam_size=(5 if is_final else 1), vad_filter=True, initial_prompt=prompt))
        text = " ".join(s.text for s in segments).strip()
        
        # Se auto-detectou na reunião, mapeia para NLLB
        if source_id == 1 and info:
            # Mapeamento simplificado: (ISO 639-1 -> NLLB)
            # Para uma versão comercial, teríamos um mapa completo. Para agora, usamos um fallback seguro.
            det_lang = info.language
            if not n_src:
                n_src = WHISPER_TO_NLLB.get(det_lang, config["output_nllb"])

        if not text:
            if is_final: global_outbound_queue.put_nowait(json.dumps({"status": "empty", "source_id": source_id}))
            return
        
        if not is_final:
            global_outbound_queue.put_nowait(f"[TRANS] " + json.dumps({"status": "partial", "source_id": source_id, "text": text}))
            return

        mem.append(text)
        if len(mem) > MAX_CONTEXT: mem.pop(0)
        eid = _append_feed_event("success", text=text, source_id=source_id)
        global_outbound_queue.put_nowait(f"[TRANS] " + json.dumps({"status": "success", "id": eid, "source_id": source_id, "text": text}))

        if n_src != n_tgt:
            prev_context = mem[-2] if len(mem) > 1 else ""
            translation_queue.put_nowait({
                "id": eid, 
                "text": text, 
                "source_id": source_id, 
                "nllb_src": n_src, 
                "nllb_tgt": n_tgt,
                "context": prev_context
            })
    except Exception as e: print(f"[ERRO PROC] {e}")

async def command_handler():
    """Lê comandos JSON de stdin (vindos da interface) e envia para o motor Rust via WebSocket"""
    global global_ws
    loop = asyncio.get_running_loop()
    cmd_queue = asyncio.Queue()

    def _read_stdin():
        """Thread bloqueante para ler do stdin sem travar o loop do asyncio no Windows."""
        for line in sys.stdin:
            if line:
                loop.call_soon_threadsafe(cmd_queue.put_nowait, line)

    threading.Thread(target=_read_stdin, daemon=True).start()

    while True:
        line = await cmd_queue.get()
        try:
            cmd_str = line.strip()
            if not cmd_str: continue
            c = json.loads(cmd_str)
            if global_ws and not getattr(global_ws, 'closed', False):
                await global_ws.send(cmd_str)
        except Exception: pass

async def heartbeat():
    """Envia um pulso para o Rust para verificar a conexão"""
    while True:
        if global_ws and not getattr(global_ws, 'closed', False):
            try:
                await global_ws.send(json.dumps({"command": "heartbeat", "time": time.time()}))
            except: pass
        await asyncio.sleep(2)

async def process_audio(websocket, path=None):
    global main_loop, global_outbound_queue, global_ws
    main_loop = asyncio.get_running_loop()
    global_ws = websocket
    global_outbound_queue = asyncio.Queue()
    
    sender_task = asyncio.create_task(send_worker(websocket))
    try:
        async for message in websocket:
            # Cada áudio é processado em sua própria tarefa para não travar o socket
            asyncio.create_task(process_single_audio(message, websocket))
    finally:
        sender_task.cancel()
        global_outbound_queue = None

async def main():
    _reset_feed_file()
    threading.Thread(target=tts_worker, daemon=True).start()
    asyncio.create_task(translation_worker())
    asyncio.create_task(command_handler())
    asyncio.create_task(heartbeat())
    print(f"Iniciando servidor WebSocket na porta {PORT}...", flush=True)
    # Aumentamos ping_interval e ping_timeout para evitar desconexões no Windows durante processamento pesado
    async with websockets.serve(process_audio, "127.0.0.1", PORT, ping_interval=20, ping_timeout=60):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
