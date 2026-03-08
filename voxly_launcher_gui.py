import json
import queue
import re
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRANSCRIBER = ROOT / "bridge" / "transcriber.py"

TRANSCRIPTION_REGEX = re.compile(r"\[TRANSCRIÇÃO -> Locutor\s*(\d+)\]\s*(.+)$")
ANSI_ESCAPE_REGEX = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class VoxlyLauncherUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Voxly Launcher")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.python_proc: subprocess.Popen | None = None
        self.rust_proc: subprocess.Popen | None = None
        self.events: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.running = False

        self.status_var = tk.StringVar(value="Pronto para iniciar")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, padx=12, pady=12)
        top.pack(fill=tk.X)

        tk.Label(top, text="Voxly - Inicialização Automática", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)

        self.start_button = tk.Button(top, text="Iniciar tudo", command=self.start_all, width=20)
        self.start_button.pack(side=tk.RIGHT, padx=6)

        self.stop_button = tk.Button(top, text="Parar", command=self.stop_all, width=20, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT)

        status = tk.Label(self.root, textvariable=self.status_var, anchor="w", fg="#0f5132", padx=12)
        status.pack(fill=tk.X)

        center = tk.LabelFrame(self.root, text="Transcrição ASR (tempo real)", padx=8, pady=8)
        center.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 6))

        self.transcription_text = scrolledtext.ScrolledText(center, height=14, wrap=tk.WORD, font=("Consolas", 12))
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        self.transcription_text.insert(tk.END, "Aguardando inicialização...\n")
        self.transcription_text.config(state=tk.DISABLED)

        bottom = tk.Frame(self.root, padx=12, pady=8)
        bottom.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.LabelFrame(bottom, text="Logs Python (Whisper)", padx=6, pady=6)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        right_panel = tk.LabelFrame(bottom, text="Logs Rust (Voxly)", padx=6, pady=6)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.python_logs = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, font=("Consolas", 10))
        self.python_logs.pack(fill=tk.BOTH, expand=True)

        self.rust_logs = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, font=("Consolas", 10))
        self.rust_logs.pack(fill=tk.BOTH, expand=True)

    def start_all(self) -> None:
        if self.running:
            return

        if not VENV_PYTHON.exists():
            self._set_status("Erro: .venv não encontrado. Crie o ambiente virtual antes.", ok=False)
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._set_status("Iniciando serviço Python...", ok=True)

        thread = threading.Thread(target=self._start_pipeline, daemon=True)
        thread.start()

    def _start_pipeline(self) -> None:
        try:
            self.python_proc = subprocess.Popen(
                [str(VENV_PYTHON), "-u", str(TRANSCRIBER)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self.events.put(("status", "Python iniciado. Aguardando Whisper carregar..."))
            python_ready = self._read_until_ready(self.python_proc, "python")

            if not python_ready:
                self.events.put(("status_error", "Falha ao iniciar Python/Whisper. Veja os logs."))
                self.stop_all()
                return

            self.events.put(("status", "Whisper pronto. Iniciando cargo run..."))
            self.rust_proc = subprocess.Popen(
                ["cargo", "run"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            threading.Thread(target=self._stream_logs, args=(self.rust_proc, "rust"), daemon=True).start()
            self.events.put(("status", "Voxly em execução. Fale no microfone."))

        except Exception as exc:
            self.events.put(("python", f"[ERRO] {exc}\n"))
            self.events.put(("status_error", "Erro ao iniciar pipeline."))
            self.running = False

    def _read_until_ready(self, proc: subprocess.Popen, origin: str) -> bool:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                return proc.poll() is None
            self.events.put((origin, line))
            if "Iniciando servidor WebSocket Whisper local na porta" in line:
                threading.Thread(target=self._stream_logs, args=(proc, origin), daemon=True).start()
                return True

    def _stream_logs(self, proc: subprocess.Popen, origin: str) -> None:
        if not proc.stdout:
            return
        for line in proc.stdout:
            self.events.put((origin, line))

        code = proc.poll()
        self.events.put((origin, f"\n[PROCESSO FINALIZADO] código={code}\n"))

    def stop_all(self) -> None:
        self.running = False
        self._terminate(self.rust_proc)
        self._terminate(self.python_proc)
        self.rust_proc = None
        self.python_proc = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._set_status("Processos parados.", ok=False)

    def _terminate(self, proc: subprocess.Popen | None) -> None:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _append(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.insert(tk.END, text)
        widget.see(tk.END)

    def _append_transcription(self, speaker: str, text: str) -> None:
        self.transcription_text.config(state=tk.NORMAL)
        self._append(self.transcription_text, f"Locutor {speaker}: {text}\n")
        self.transcription_text.config(state=tk.DISABLED)

    def _set_status(self, text: str, ok: bool) -> None:
        self.status_var.set(text)

    def _drain_events(self) -> None:
        while True:
            try:
                target, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if target == "python":
                self._append(self.python_logs, payload)
            elif target == "rust":
                self._append(self.rust_logs, payload)
                clean_payload = ANSI_ESCAPE_REGEX.sub("", payload)
                match = TRANSCRIPTION_REGEX.search(clean_payload)
                if match:
                    speaker, text = match.groups()
                    if text.startswith("{"):
                        try:
                            data = json.loads(text)
                            text = data.get("text", text)
                        except json.JSONDecodeError:
                            pass
                    self._append_transcription(speaker, text)
            elif target == "status":
                self._set_status(payload, ok=True)
            elif target == "status_error":
                self._set_status(payload, ok=False)

        self.root.after(120, self._drain_events)

    def on_close(self) -> None:
        self.stop_all()
        self.root.destroy()


def main() -> None:
    app = tk.Tk()
    VoxlyLauncherUI(app)
    app.mainloop()


if __name__ == "__main__":
    main()
