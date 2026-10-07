import json
import os
import sys
import queue
import re
import subprocess
import threading
import tkinter as tk
import typing
import time
from pathlib import Path
from tkinter import ttk, scrolledtext
import customtkinter as ctk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def _resolve_voxly_root() -> Path:
    # 1. Tenta env var
    env_root = os.environ.get("VOXLY_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        return p

    # 2. Se estiver congelado (PyInstaller)
    if getattr(sys, 'frozen', False):
        exe_path = Path(sys.executable).resolve()
        if exe_path.parent.name.lower() == "bridge":
            return exe_path.parent.parent
        return exe_path.parent

    # 3. Rodando como Script
    here = Path(__file__).resolve().parent
    return here

# --- DEFINIÇÃO DE ROOT E CARREGAMENTO DE DADOS ---
ROOT = _resolve_voxly_root()

LANGUAGES_DATA_PATH = ROOT / "data" / "languages_data.json"

def load_languages_data():
    if not LANGUAGES_DATA_PATH.exists():
        return [], {}, {}
    try:
        data = json.loads(LANGUAGES_DATA_PATH.read_text(encoding="utf-8"))
        lang_list = [l["name"] for l in data]
        lang_map = {l["name"]: {"nllb": l["nllb"], "whisper": l["whisper"], "piper": l["piper"]} for l in data}
        voice_map = {l["name"]: l["voices"] for l in data if l["voices"]}
        return lang_list, lang_map, voice_map
    except Exception as e:
        print(f"Erro ao carregar dados de idiomas: {e}")
        return ["Português (Brasil)"], {"Português (Brasil)": {"nllb": "por_Latn", "whisper": "pt", "piper": "pt_BR"}}, {"Português (Brasil)": ["Faber (Masculino)"]}

LANG_LIST, LANG_MAP_FULL, VOICE_MAP_FULL = load_languages_data()

# --- CAMINHOS ---
IS_FROZEN = getattr(sys, 'frozen', False)

if IS_FROZEN:
    VENV_PYTHON = None
    TRANSCRIBER_EXE = ROOT / "bridge" / "transcriber.exe"
    VOXLY_EXE = ROOT / "Voxly.exe"
else:
    VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
    TRANSCRIBER = ROOT / "bridge" / "transcriber.py"
    VOXLY_EXE = None

TRANSCRIPT_LOG = ROOT / "data" / "transcricoes_voxly.log"
TRANSCRIPTION_FEED = ROOT / "data" / "transcricoes_feed.json"
VOXLY_CONFIG = ROOT / "voxly_config.json"
PYTHON_LOG = ROOT / "data" / "python_last_run.log"
RUST_LOG = ROOT / "data" / "rust_last_run.log"

TRANSCRIPTION_REGEX = re.compile(r"\[TRANS\]\s*(.*)$", re.IGNORECASE)
ANSI_ESCAPE_REGEX = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUST_READY_REGEX = re.compile(r"\[IA\].*Motores ativos", re.IGNORECASE)


class VoxlyLauncherUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voxly Inicializando...")
        
        # Centralizar
        w, h = 900, 600
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f'{w}x{h}+{x}+{y}')

        self.python_proc: typing.Any = None
        self.rust_proc: typing.Any = None
        self.events: "queue.Queue[tuple[str, typing.Any]]" = queue.Queue()
        
        self.running = False
        self._received_transcription = False
        self._feed_last_event_id = 0
        self._feed_last_mtime = 0.0
        self._last_partial_text = ""
        self._main_ui_active = False
        self._hardware_devices = {"inputs": ["Padrão"], "outputs": ["Padrão"]}

        self.status_var = ctk.StringVar(value="Preparando inicialização automática...")
        
        # Lê config atual
        self.config_data = {}
        if VOXLY_CONFIG.exists():
            try:
                self.config_data = json.loads(VOXLY_CONFIG.read_text("utf-8"))
            except: pass

        # UI Elements
        self.btn_exit_splash: typing.Any = None
        self.input_lang_combo: typing.Any = None
        self.output_lang_combo: typing.Any = None
        self.voice_combo: typing.Any = None
        self.speed_scale: typing.Any = None
        self.status_lbl: typing.Any = None
        self.original_text: typing.Any = None
        self.translation_text: typing.Any = None
        self.subtitle_text: typing.Any = None
        self.progress: typing.Any = None
        self.splash_log_py: typing.Any = None
        self.splash_log_rust: typing.Any = None
        
        self.current_frame = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Decide qual tela iniciar
        if "performance_profile" not in self.config_data:
            self._build_setup_ui()
        else:
            self._build_login_ui()
            
    def _clear_window(self):
        if self.current_frame:
            self.current_frame.destroy()
        for widget in self.winfo_children(): 
            widget.destroy()

    def _build_setup_ui(self):
        self._clear_window()
        self.title("Voxly - Perfil de Desempenho")
        
        self.current_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.current_frame.pack(fill=ctk.BOTH, expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(self.current_frame, text="Bem-vindo ao Voxly", font=ctk.CTkFont(size=32, weight="bold")).pack(pady=(20, 10))
        ctk.CTkLabel(self.current_frame, text="Para a melhor experiência, selecione o perfil de hardware do seu computador:", font=ctk.CTkFont(size=14), text_color="gray").pack(pady=(0, 40))
        
        cards_frame = ctk.CTkFrame(self.current_frame, fg_color="transparent")
        cards_frame.pack(fill=ctk.X, pady=10)
        
        def set_profile(level, model):
            self.config_data["performance_profile"] = level
            self.config_data["whisper_model"] = model
            VOXLY_CONFIG.write_text(json.dumps(self.config_data, indent=2, ensure_ascii=False), encoding="utf-8")
            self._build_login_ui()

        # Médio (Agora o novo Mínimo Profissional)
        c2 = ctk.CTkFrame(cards_frame)
        c2.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=20)
        ctk.CTkLabel(c2, text="Médio Desempenho", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(c2, text="Padrão Profissional.\nEquilíbrio entre velocidade e precisão.\n\n• CPU: Core i5 / Ryzen 5\n• RAM: 8GB\n• Tradutor: NLLB 1.3B", text_color="gray").pack(pady=10)
        ctk.CTkButton(c2, text="Selecionar Médio", command=lambda: set_profile("med", "small")).pack(pady=20)

        # Alto
        c3 = ctk.CTkFrame(cards_frame)
        c3.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=20)
        ctk.CTkLabel(c3, text="Alto Desempenho", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(c3, text="Qualidade Máxima.\nPara usuários que exigem perfeição.\n\n• CPU: Core i7 / Ryzen 7+\n• RAM: 16GB+\n• Tradutor: NLLB 3.3B", text_color="gray").pack(pady=10)
        ctk.CTkButton(c3, text="Selecionar Alto", fg_color="#27ae60", hover_color="#2ecc71", command=lambda: set_profile("max", "medium")).pack(pady=20)

    def _build_login_ui(self):
        self._clear_window()
        self.title("Voxly - Login")
        
        self.current_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.current_frame.pack(fill=ctk.BOTH, expand=True, padx=40, pady=40)
        
        login_box = ctk.CTkFrame(self.current_frame, width=400, corner_radius=15)
        login_box.pack(expand=True)
        
        ctk.CTkLabel(login_box, text="Entrar", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(30, 20))
        
        entry_user = ctk.CTkEntry(login_box, placeholder_text="Email de Usuário", width=250)
        entry_user.pack(pady=10)
        
        entry_pass = ctk.CTkEntry(login_box, placeholder_text="Senha", show="*", width=250)
        entry_pass.pack(pady=10)
        
        def do_login():
            # Transição Imediata para não dar sensação de atraso
            self._build_splash_ui()
            self.status_var.set("Iniciando Verificação de Áudio...")
            
            def bg_discovery():
                try:
                    self._discover_hardware()
                except Exception as e:
                    print(f"[!] Erro crítico na descoberta: {e}")
                finally:
                    # Garante que SEMPRE continue a inicialização, mesmo se a busca falhar
                    self.after(100, self.start_all)
                    self.after(200, self._drain_events)
                
            threading.Thread(target=bg_discovery, daemon=True).start()
            
        ctk.CTkButton(login_box, text="Acessar Voxly", width=250, command=do_login).pack(pady=(20, 10))

        def change_hardware():
            if "performance_profile" in self.config_data:
                del self.config_data["performance_profile"]
                if VOXLY_CONFIG.exists():
                    VOXLY_CONFIG.write_text(json.dumps(self.config_data, indent=2, ensure_ascii=False), encoding="utf-8")
            self._build_setup_ui()
            
        ctk.CTkButton(login_box, text="Trocar Perfil de Hardware", fg_color="transparent", text_color="gray", width=250, command=change_hardware).pack(pady=(0, 20))

    def _discover_hardware(self):
        """Executa o motor Rust em modo de listagem para descobrir dispositivos."""
        try:
            # 1. Limpeza de processos zumbis que travam o binário (os error 5)
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/IM", "Voxly.exe", "/T"], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                               creationflags=subprocess.CREATE_NO_WINDOW)
            
            # 2. Caminho do Binário em Desenvolvimento vs Produção
            if IS_FROZEN:
                cmd = [str(VOXLY_EXE), "--list-devices"]
            else:
                debug_exe = ROOT / "target" / "debug" / "Voxly.exe"
                if not debug_exe.exists():
                    self.status_var.set("Compilando motor de áudio (Pode demorar 1-2 min)...")
                    subprocess.run(["cargo", "build", "--quiet"], cwd=ROOT, check=True)
                
                self.status_var.set("Sondando dispositivos de áudio...")
                cmd = [str(debug_exe), "--list-devices"]

            pf = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            out = subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True, encoding="utf-8", creationflags=pf, timeout=5)
            
            # 3. Parse do JSON de hardware
            for line in out.splitlines():
                if line.strip().startswith("{") and "inputs" in line:
                    data = json.loads(line)
                    self._hardware_devices["inputs"] = ["Padrão"] + sorted(list(set(data.get("inputs", []))))
                    self._hardware_devices["outputs"] = ["Padrão"] + sorted(list(set(data.get("outputs", []))))
                    break
        except subprocess.TimeoutExpired:
            print("[!] Timeout ao descobrir hardware (5s). Usando Padrão.")
        except Exception as e:
            print(f"[!] Erro ao descobrir hardware: {e}")
            self._hardware_devices = {"inputs": ["Padrão"], "outputs": ["Padrão"]}

    def _build_splash_ui(self) -> None:
        self._clear_window()
        self.title("Voxly Inicializando Motores...")
        
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(frame, text="VOXLY", font=ctk.CTkFont(size=36, weight="bold"), text_color="#ff9999").pack(pady=(10, 0))
        ctk.CTkLabel(frame, text="IA VOICE CONTROL", font=ctk.CTkFont(size=14, weight="bold"), text_color="gray").pack(pady=(0, 10))
        
        self.progress = ctk.CTkProgressBar(frame, mode='determinate', width=600)
        self.progress.set(0)
        self.progress.pack(pady=(20, 10))
        
        ctk.CTkLabel(frame, textvariable=self.status_var, font=ctk.CTkFont(size=12), text_color="gray").pack()
        
        logs_frame = ctk.CTkFrame(frame, fg_color="transparent")
        logs_frame.pack(fill=ctk.BOTH, expand=True, pady=15)
        
        # Python Log
        py_frame = ctk.CTkFrame(logs_frame)
        py_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 10))
        ctk.CTkLabel(py_frame, text="Servidor Python", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        self.splash_log_py = ctk.CTkTextbox(py_frame, font=ctk.CTkFont(family="Consolas", size=11), text_color="#00fa9a")
        self.splash_log_py.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.splash_log_py.configure(state="disabled")
        
        # Rust Log
        rs_frame = ctk.CTkFrame(logs_frame)
        rs_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(10, 0))
        ctk.CTkLabel(rs_frame, text="Motor de Áudio Rust", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        self.splash_log_rust = ctk.CTkTextbox(rs_frame, font=ctk.CTkFont(family="Consolas", size=11), text_color="#ff79c6")
        self.splash_log_rust.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.splash_log_rust.configure(state="disabled")
        
        self.btn_exit_splash = ctk.CTkButton(frame, text="Cancelar Inicialização", fg_color="transparent", border_width=1, hover_color="#333", command=self.on_close)
        self.btn_exit_splash.pack(side=ctk.BOTTOM, pady=10)

    def _transition_to_main_ui(self):
        if self._main_ui_active: return
        self._main_ui_active = True
        self._clear_window()
        
        self.title("Voxly Voice Control")
        self.minsize(900, 600)
        w, h = 1100, 750
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f'{w}x{h}+{x}+{y}')
        self._build_main_ui()
        
        # Teclas de Atalho (Space para PTT)
        self.bind("<KeyPress-space>", self._on_ptt_press)
        self.bind("<KeyRelease-space>", self._on_ptt_release)

    def _build_main_ui(self) -> None:
        self.current_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.current_frame.pack(fill=ctk.BOTH, expand=True)

        top = ctk.CTkFrame(self.current_frame, fg_color="#1e272e" if ctk.get_appearance_mode() == "Dark" else "#353b48", corner_radius=0)
        top.pack(fill=ctk.X)
        ctk.CTkLabel(top, text="🎙️ Voxly AI", font=ctk.CTkFont(size=20, weight="bold"), text_color="white").pack(side=ctk.LEFT, padx=20, pady=15)
        
        # Botão de Ajuda
        def show_help():
            help_win = ctk.CTkToplevel(self)
            help_win.title("Guia de Configuração Voxly")
            help_win.geometry("600x550")
            help_win.after(200, lambda: help_win.focus()) # Garante foco no Windows
            
            content = ctk.CTkTextbox(help_win, font=ctk.CTkFont(size=14), wrap="word")
            content.pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)
            
            msg = """❓ GUIA DE CONFIGURAÇÃO VOXLY

Este guia explica como configurar o Voxly para uma experiência profissional em reuniões.

--------------------------------------------------
🛠️ SELEÇÃO DE DISPOSITIVOS (No Dashboard):

1. MICROFONE (EU): Escolha o seu microfone físico real (ex: headset, condensador). É por aqui que o Voxly ouve você.

2. FONTE DE REUNIÃO (OUTROS): Escolha por onde o som das outras pessoas está saindo. 
   - Se for 'Padrão', ele tentará captar do Windows.
   - Recomendado: Se estiver usando VB-Cable, selecione 'CABLE Output'.

3. SAÍDA DE TRADUÇÃO: Escolha para onde a voz traduzida da IA deve ir.
   - Recomendado: 'CABLE Input'. Isso envia sua tradução para o cabo virtual.

--------------------------------------------------
🚀 CONFIGURAÇÃO NAS REUNIÕES (Zoom, Meet, Teams):

Para que os outros ouçam sua tradução em vez da sua voz original:
1. Vá nas configurações de ÁUDIO da sua reunião (Zoom/Meet/etc).
2. No campo MICROFONE da reunião, selecione: 'CABLE Output (VB-Audio Virtual Cable)'.
3. No campo ALTO-FALANTE da reunião, selecione seu fone de ouvido normal.

Dessa forma, o Voxly processa seu microfone real e entrega o áudio pronto no Cabo Virtual para a sua reunião!
"""
            content.insert("0.0", msg)
            content.configure(state="disabled")
            
            ctk.CTkButton(help_win, text="Entendi!", command=help_win.destroy).pack(pady=(0, 20))

        self.btn_help = ctk.CTkButton(top, text="?", width=30, height=30, corner_radius=15, 
                                     fg_color="#34495e", hover_color="#3498db", command=show_help)
        self.btn_help.pack(side=ctk.RIGHT, padx=10)
        
        # Theme Switch
        def change_theme():
            if theme_switch.get() == 1: ctk.set_appearance_mode("Dark")
            else: ctk.set_appearance_mode("Light")
            
        theme_switch = ctk.CTkSwitch(top, text="Modo Escuro", command=change_theme, onvalue=1, offvalue=0)
        theme_switch.select() if ctk.get_appearance_mode() == "Dark" else theme_switch.deselect()
        theme_switch.pack(side=ctk.RIGHT, padx=20)
        
        ctk.CTkLabel(top, text="Online e Pronto", font=ctk.CTkFont(size=12, weight="bold"), text_color="#4cd137").pack(side=ctk.RIGHT, padx=10)

        center = ctk.CTkFrame(self.current_frame, fg_color="transparent")
        center.pack(fill=ctk.BOTH, expand=True, padx=20, pady=15)

        ctrl_frame = ctk.CTkFrame(center, corner_radius=10)
        ctrl_frame.pack(fill=ctk.X, pady=(0, 15), ipadx=10, ipady=10)
        
        ctk.CTkLabel(ctrl_frame, text="Configurações de Voz e Idioma", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor=ctk.W, padx=15, pady=(10,0))
        
        grid_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        grid_frame.pack(fill=ctk.X, padx=15, pady=10)

        ctk.CTkLabel(grid_frame, text="Falar em (Entrada):", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky=ctk.W, pady=(0, 10))
        self.input_lang_combo = ctk.CTkOptionMenu(grid_frame, values=LANG_LIST, width=250, command=self._save_config)
        self.input_lang_combo.set(self.config_data.get("input_lang_name", "Português (Brasil)"))
        self.input_lang_combo.grid(row=0, column=1, padx=(10, 30), pady=(0, 10))

        ctk.CTkLabel(grid_frame, text="Traduzir para (Saída):", font=ctk.CTkFont(size=12)).grid(row=0, column=2, sticky=ctk.W, pady=(0, 10))
        self.output_lang_combo = ctk.CTkOptionMenu(grid_frame, values=LANG_LIST, width=250, command=self._on_output_lang_change)
        self.output_lang_combo.set(self.config_data.get("output_lang_name", "Inglês (US)"))
        self.output_lang_combo.grid(row=0, column=3, padx=10, pady=(0, 10))

        ctk.CTkLabel(grid_frame, text="Voz da IA:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky=ctk.W)
        self.voice_combo = ctk.CTkOptionMenu(grid_frame, width=250, command=self._save_config)
        self.voice_combo.grid(row=1, column=1, padx=(10, 30))

        ctk.CTkLabel(grid_frame, text="Velocidade:", font=ctk.CTkFont(size=12)).grid(row=1, column=2, sticky=ctk.W)
        self.speed_scale = ctk.CTkSlider(grid_frame, from_=0.5, to=2.0, number_of_steps=15, command=self._save_config_delayed)
        self.speed_scale.set(self.config_data.get("tts_speed", 1.2))
        self.speed_scale.grid(row=1, column=3, padx=10)

        # Seção de Dispositivos de Áudio (Manual)
        ctk.CTkLabel(ctrl_frame, text="Dispositivos de Áudio (Hardware)", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor=ctk.W, padx=15, pady=(15,0))
        audio_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        audio_frame.pack(fill=ctk.X, padx=15, pady=10)
        
        # Inputs (Mic e Reunião)
        ctk.CTkLabel(audio_frame, text="Seu Microfone:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky=ctk.W)
        self.mic_combo = ctk.CTkOptionMenu(audio_frame, values=self._hardware_devices["inputs"], width=350, command=self._save_config)
        self.mic_combo.set(self.config_data.get("audio_mic", "Padrão"))
        self.mic_combo.grid(row=0, column=1, padx=10, pady=5)

        ctk.CTkLabel(audio_frame, text="Áudio Reunião:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky=ctk.W)
        self.meeting_combo = ctk.CTkOptionMenu(audio_frame, values=self._hardware_devices["inputs"], width=350, command=self._save_config)
        self.meeting_combo.set(self.config_data.get("audio_meeting", "Padrão"))
        self.meeting_combo.grid(row=1, column=1, padx=10, pady=5)

        # Output (Tradução)
        ctk.CTkLabel(audio_frame, text="Saída Tradução:", font=ctk.CTkFont(size=12)).grid(row=0, column=2, sticky=ctk.W, padx=(20, 0))
        self.output_dev_combo = ctk.CTkOptionMenu(audio_frame, values=self._hardware_devices["outputs"], width=350, command=self._save_config)
        self.output_dev_combo.set(self.config_data.get("audio_output", "Padrão"))
        self.output_dev_combo.grid(row=0, column=3, padx=10, pady=5)

        self.status_lbl = ctk.CTkLabel(center, text="🎙️ Microfone em espera (Segure [ESPAÇO])", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3498db")
        self.status_lbl.pack(pady=(5, 5))

        # Switch para tradução de entrada
        self.translate_inbound_var = ctk.BooleanVar(value=self.config_data.get("translate_inbound", True))
        self.inbound_switch = ctk.CTkSwitch(center, text="Traduzir o que os outros dizem (Reunião)", variable=self.translate_inbound_var, command=self._save_config)
        self.inbound_switch.pack(pady=(0, 15))
        
        self.ptt_indicator = ctk.CTkLabel(center, text="SEGURE [ESPAÇO] PARA FALAR", font=ctk.CTkFont(size=12, weight="bold"), text_color="grey")
        self.ptt_indicator.pack(pady=(0, 5))
        
        self._on_output_lang_change()

        # Painel de Chat Único (Timeline)
        chat_panel = ctk.CTkFrame(center)
        chat_panel.pack(fill=ctk.BOTH, expand=True)
        ctk.CTkLabel(chat_panel, text="Linha do Tempo da Conversa", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor=ctk.W, padx=15, pady=(10, 0))
        
        self.chat_feed = ctk.CTkTextbox(chat_panel, font=ctk.CTkFont(size=16), wrap="word")
        self.chat_feed.pack(fill=ctk.BOTH, expand=True, padx=15, pady=15)
        self.chat_feed.insert("0.0", "Aguardando áudio...\n")
        
        # Configurar Cores (Tags do Tkinter interno)
        self.chat_feed._textbox.tag_config("me", foreground="#3498db", font=ctk.CTkFont(size=16, weight="bold"))
        self.chat_feed._textbox.tag_config("them", foreground="#2ecc71", font=ctk.CTkFont(size=16, weight="bold"))
        self.chat_feed._textbox.tag_config("sub", foreground="grey", font=ctk.CTkFont(size=14))
        self.chat_feed._textbox.tag_config("dim", foreground="grey")
        
        self.chat_feed.configure(state="disabled")

        sub_panel = ctk.CTkFrame(center, height=60)
        sub_panel.pack(fill=ctk.X, pady=(15, 0))
        sub_panel.pack_propagate(False)
        self.subtitle_text = ctk.CTkLabel(sub_panel, text="Voxly Ativo.", font=ctk.CTkFont(size=18, weight="bold"), text_color="#e1b12c")
        self.subtitle_text.pack(expand=True)

        self._refresh_transcriptions_display()

    def _save_config_delayed(self, _):
        # Evita salvar intensamente enquanto arrasta o slider
        if hasattr(self, "_save_timer") and self._save_timer:
            self.after_cancel(self._save_timer)
        self._save_timer = self.after(500, self._save_config)

    def _on_output_lang_change(self, event=None) -> None:
        lang = self.output_lang_combo.get() if self.output_lang_combo else "Inglês (US)"
        voices = VOICE_MAP_FULL.get(lang, ["Padrão"])
        if self.voice_combo:
            self.voice_combo.configure(values=voices)
            
            # Se a voz salva não estiver na lista ou o config estiver vazio, usa a primeira
            saved_voice = self.config_data.get("tts_voice")
            if saved_voice in voices:
                self.voice_combo.set(saved_voice)
            else:
                self.voice_combo.set(voices[0] if voices else "Padrão")
                
        self._save_config()

    def _on_ptt_press(self, event):
        if not self._main_ui_active or event.widget.winfo_class() == "Entry": return
        if not hasattr(self, "_ptt_active") or not self._ptt_active:
            self._ptt_active = True
            self.status_lbl.configure(text="🔴 TRANSMITINDO AO VIVO", text_color="#e74c3c")
            self.ptt_indicator.configure(text_color="#e74c3c")
            self._send_bridge_command({"command": "set_ptt", "value": True})

    def _on_ptt_release(self, event):
        if not self._main_ui_active or event.widget.winfo_class() == "Entry": return
        self._ptt_active = False
        self.status_lbl.configure(text="🎙️ Microfone em espera (Segure [ESPAÇO])", text_color="#3498db")
        self.ptt_indicator.configure(text_color="grey")
        self._send_bridge_command({"command": "set_ptt", "value": False})

    def _send_bridge_command(self, cmd_dict):
        if self.python_proc and self.python_proc.poll() is None:
            try:
                msg = json.dumps(cmd_dict) + "\n"
                self.python_proc.stdin.write(msg)
                self.python_proc.stdin.flush()
            except: pass

    def _save_config(self, event=None) -> None:
        try:
            in_lang = self.input_lang_combo.get() if self.input_lang_combo else "Português (Brasil)"
            out_lang = self.output_lang_combo.get() if self.output_lang_combo else "Inglês (US)"
            speed_val = self.speed_scale.get() if self.speed_scale else 1.2
            
            in_data = LANG_MAP_FULL.get(in_lang, {"nllb": "por_Latn", "whisper": "pt", "piper": "pt_BR"})
            out_data = LANG_MAP_FULL.get(out_lang, {"nllb": "eng_Latn", "whisper": "en", "piper": "en_US"})
            
            # Merge com as configurações existentes
            self.config_data.update({
                "input_lang_name": in_lang, "output_lang_name": out_lang,
                "input_nllb": in_data["nllb"], "input_whisper": in_data["whisper"],
                "output_nllb": out_data["nllb"], "output_piper": out_data["piper"],
                "tts_voice": self.voice_combo.get() if self.voice_combo else "Padrão",
                "tts_speed": float(f"{speed_val:.2f}"),
                "translate_inbound": self.translate_inbound_var.get() if hasattr(self, "translate_inbound_var") else True,
                "audio_mic": self.mic_combo.get() if hasattr(self, "mic_combo") else "Padrão",
                "audio_meeting": self.meeting_combo.get() if hasattr(self, "meeting_combo") else "Padrão",
                "audio_output": self.output_dev_combo.get() if hasattr(self, "output_dev_combo") else "Padrão"
            })
            
            VOXLY_CONFIG.write_text(json.dumps(self.config_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e: 
            print("Erro ao salvar config:", e)

    def start_all(self) -> None:
        if self.running: return
        self.running = True
        PYTHON_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PYTHON_LOG, "w", encoding="utf-8") as f: f.write("=== Inicio do Log Python ===\n")
        with open(RUST_LOG, "w", encoding="utf-8") as f: f.write("=== Inicio do Log Rust ===\n")
        threading.Thread(target=self._start_pipeline, daemon=True).start()

    def _start_pipeline(self) -> None:
        try:
            self.events.put(("status", "Limpando conexões anteriores..."))
            self.events.put(("progress", 5))
            env = os.environ.copy()
            env["VOXLY_ROOT"] = str(ROOT)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            if sys.platform == "win32":
                try: # Mata porta 8000
                    out = subprocess.check_output("netstat -ano | findstr :8000", shell=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    for l in out.splitlines():
                        if "LISTENING" in l: subprocess.run(f"taskkill /F /PID {l.split()[-1]}", shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
                except: pass
            
            self.events.put(("status", "Iniciando motores de IA (Python)..."))
            self.events.put(("progress", 10))
            cmd_py = [str(TRANSCRIBER_EXE)] if IS_FROZEN else [str(VENV_PYTHON), "-u", str(TRANSCRIBER)]
            pf = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.python_proc = subprocess.Popen(cmd_py, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=pf)
            if not self._read_until_ready(self.python_proc, "python"): 
                self.events.put(("status", "ERRO: Falha ao iniciar Python"))
                return
            
            self.events.put(("status", "IA Pronta! Iniciando motor de Áudio (Rust)..."))
            self.events.put(("progress", 70))
            
            cmd_rs = [str(VOXLY_EXE)] if IS_FROZEN else [str(ROOT / "target" / "debug" / "Voxly.exe")]
            
            # Adiciona argumentos de áudio manual se não for "Padrão"
            mic = self.config_data.get("audio_mic", "Padrão")
            if mic != "Padrão": cmd_rs.extend(["--mic", mic])
            
            meeting = self.config_data.get("audio_meeting", "Padrão")
            if meeting != "Padrão": cmd_rs.extend(["--meeting", meeting])
            
            out = self.config_data.get("audio_output", "Padrão")
            if out != "Padrão": cmd_rs.extend(["--output", out])
            
            print(f"[*] Iniciando motor Rust: {cmd_rs}")
            self.rust_proc = subprocess.Popen(cmd_rs, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=pf)
            
            self.events.put(("status", "Aguardando inicialização final..."))
            self.events.put(("progress", 90))
            threading.Thread(target=self._stream_logs, args=(self.rust_proc, "rust"), daemon=True).start()
        except Exception as e:
            self.events.put(("python", f"[ERRO] {e}\n"))

    def _read_until_ready(self, proc: typing.Any, origin: str) -> bool:
        while True:
            line = proc.stdout.readline()
            if not line: return proc.poll() is None
            self.events.put((origin, line))
            if "Iniciando servidor WebSocket" in line:
                threading.Thread(target=self._stream_logs, args=(proc, origin), daemon=True).start()
                return True

    def _stream_logs(self, proc: typing.Any, origin: str) -> None:
        for line in proc.stdout: self.events.put((origin, line))
        self.events.put((origin, f"\n[FINALIZADO]\n"))

    def stop_all(self):
        self.running = False
        for p in [self.rust_proc, self.python_proc]:
            if p and p.poll() is None:
                p.terminate()
                try: p.wait(timeout=3)
                except: p.kill()

    def _refresh_transcriptions_display(self):
        if not self._main_ui_active or not TRANSCRIPTION_FEED.exists(): return
        try:
            with open(TRANSCRIPTION_FEED, "r", encoding="utf-8") as f: data = json.load(f)
            
            self.chat_feed.configure(state='normal')
            self.chat_feed.delete('0.0', tk.END)
            
            evs = data.get("events", [])[-30:]
            last_msg = ""
            
            for ev in evs:
                st = ev.get("status")
                t = ev.get("text", "")
                tr = ev.get("traduction", "")
                sid = ev.get("source_id", 0)
                
                if st == "success" and t:
                    icon = "👤 VOCÊ" if sid == 0 else "👥 REUNIÃO"
                    tag = "me" if sid == 0 else "them"
                    
                    self.chat_feed.insert(tk.END, f"{icon}: ", tag)
                    self.chat_feed.insert(tk.END, f"{t}\n")
                    if tr:
                        self.chat_feed.insert(tk.END, f"   ↳ Tradução: {tr}\n", "sub")
                    self.chat_feed.insert(tk.END, "\n")
                    last_msg = tr if tr else f"{icon}: {t}"
            
            if self._last_partial_text:
                self.chat_feed.insert(tk.END, f"⏳ {self._last_partial_text}...\n", "dim")
                self.subtitle_text.configure(text=f"Processando: {self._last_partial_text}")
            else:
                if last_msg: self.subtitle_text.configure(text=last_msg)
                
            self.chat_feed.configure(state='disabled')
            self.chat_feed.see(tk.END)
            
        except Exception as e: 
            print("Erro ao atualizar chat:", e)

    def _drain_events(self) -> None:
        # Limita o processamento de eventos por ciclo para evitar o "travamento" da UI (bolinha azul)
        count = 0 
        while count < 100:
            count += 1
            try: target, payload = self.events.get_nowait()
            except queue.Empty: break
            
            # Se for log ou texto, limpa ANSI
            clean = ""
            if isinstance(payload, str):
                clean = ANSI_ESCAPE_REGEX.sub("", payload).strip()
            
            if target in ["python", "rust"]:
                if not clean: continue
                log = PYTHON_LOG if target == "python" else RUST_LOG
                try:
                    with open(log, "a", encoding="utf-8") as f: f.write(clean + "\n")
                    if not self._main_ui_active:
                        w = self.splash_log_py if target == "python" else self.splash_log_rust
                        if w: 
                            w.configure(state="normal")
                            w.insert(tk.END, clean + "\n")
                            if float(w.index(tk.END)) > 500: w.delete("1.0", "2.0") 
                            w.see(tk.END)
                            w.configure(state="disabled")
                        
                        # Detecção de download de modelos no log para atualizar status visual
                        if target == "python":
                            if "fetching" in clean.lower() or "baixando" in clean.lower():
                                self.status_var.set("📦 Baixando Modelos de IA... (Pode demorar)")
                                if self.progress: self.progress.set(0.15)
                            elif "[+] Modelos prontos!" in clean:
                                self.status_var.set("✅ Modelos Carregados! Iniciando Motor de Áudio...")
                                if self.progress: self.progress.set(0.60)
                            elif "[+] Motor Piper TTS Pronto!" in clean:
                                self.status_var.set("🔊 Áudio TTS Pronto!")
                                if self.progress: self.progress.set(0.85)
                except: pass
                
                # Se for Log de IA Ativa, muda para UI Principal
                if target == "rust" and not self._main_ui_active and RUST_READY_REGEX.search(clean): 
                    self.status_var.set("Motores Prontos! Abrindo interface...")
                    self._transition_to_main_ui()
                
                # Procura por atualizações de transcrição no log
                m = TRANSCRIPTION_REGEX.search(clean)
                if m and self._main_ui_active:
                    try:
                        d = json.loads(m.group(1).strip())
                        if d.get("status") == "partial": self._last_partial_text = d.get("text", "")
                        else: self._last_partial_text = ""
                        self._refresh_transcriptions_display()
                    except: pass
            elif target == "progress" and self.progress: 
                self.progress.set(min(1.0, payload / 100.0))
            elif target == "status": self.status_var.set(payload)
        if self._main_ui_active and TRANSCRIPTION_FEED.exists():
            mt = TRANSCRIPTION_FEED.stat().st_mtime
            if mt > self._feed_last_mtime:
                self._feed_last_mtime = mt
                self._refresh_transcriptions_display()

        self.after(50, self._drain_events)

    def on_close(self): self.stop_all(); self.destroy()

if __name__ == "__main__":
    app = VoxlyLauncherUI()
    app.mainloop()
