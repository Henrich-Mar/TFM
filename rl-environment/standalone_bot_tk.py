"""
Tkinter launcher for standalone_bot.py.
"""
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


class StandaloneBotLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Terraforming Mars Standalone Bot")
        self.geometry("980x760")
        self.minsize(860, 620)

        self._script_dir = os.path.abspath(os.path.dirname(__file__))
        self._repo_root = os.path.abspath(os.path.join(self._script_dir, ".."))
        self._bot_script = os.path.join(self._script_dir, "standalone_bot.py")
        self._proc: Optional[subprocess.Popen] = None
        self._line_queue: "queue.Queue[str]" = queue.Queue()
        self._after_id: Optional[str] = None

        self.player_url_var = tk.StringVar(value="")
        # Keep this blank: when a private player URL is pasted, its host must
        # be used. A pre-filled host can silently override that URL.
        self.base_url_var = tk.StringVar(value="")
        self.player_id_var = tk.StringVar(value="")
        self.game_url_var = tk.StringVar(value="")
        self.game_id_var = tk.StringVar(value="")
        self.player_name_var = tk.StringVar(value="")
        default_checkpoint = os.path.join(self._repo_root, "rl-v2", "checkpoints", "candidate_000275219.pth")
        self.checkpoint_var = tk.StringVar(value=default_checkpoint if os.path.isfile(default_checkpoint) else "")
        self.models_var = tk.StringVar(value=os.path.join(self._repo_root, "rl-v2", "checkpoints"))
        self.runtime_var = tk.StringVar(value="Host Python (local)")
        self.min_delay_var = tk.StringVar(value="1000")
        self.poll_interval_var = tk.StringVar(value="1000")
        self.timeout_var = tk.StringVar(value="60")
        self.log_level_var = tk.StringVar(value="INFO")
        self.no_random_fallback_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._set_running_state(False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(1, weight=1)

        row = 0
        row = self._add_entry(top, row, "Player URL", self.player_url_var)
        row = self._add_entry(top, row, "Base URL", self.base_url_var)
        row = self._add_entry(top, row, "Player ID", self.player_id_var)
        row = self._add_entry(top, row, "Game URL (optional)", self.game_url_var)
        row = self._add_entry(top, row, "Game ID (optional)", self.game_id_var)
        row = self._add_entry(top, row, "Player Name (optional)", self.player_name_var)
        ttk.Label(top, text="Runtime").grid(row=row, column=0, sticky="w", pady=(6, 0))
        runtime_combo = ttk.Combobox(
            top,
            textvariable=self.runtime_var,
            values=["Host Python (local)", "Docker (optional)"],
            state="readonly",
        )
        runtime_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        row += 1
        row = self._add_entry_with_button(
            top,
            row,
            "Checkpoint",
            self.checkpoint_var,
            "Browse",
            self._pick_checkpoint,
        )
        row = self._add_entry_with_button(
            top,
            row,
            "Models Folder",
            self.models_var,
            "Browse",
            self._pick_models_dir,
        )

        row = self._add_entry(top, row, "Min Action Delay (ms)", self.min_delay_var)
        row = self._add_entry(top, row, "Poll Interval (ms)", self.poll_interval_var)
        row = self._add_entry(top, row, "Request Timeout (sec)", self.timeout_var)

        ttk.Checkbutton(
            top,
            text="Safe live mode: do not submit random fallback actions after a rejection",
            variable=self.no_random_fallback_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1

        ttk.Label(top, text="Log Level").grid(row=row, column=0, sticky="w", pady=(6, 0))
        level_combo = ttk.Combobox(
            top,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
        )
        level_combo.grid(row=row, column=1, sticky="ew", pady=(6, 0))
        row += 1

        controls = ttk.Frame(top)
        controls.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.start_btn = ttk.Button(controls, text="Start Bot", command=self._start_bot)
        self.start_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        self.stop_btn = ttk.Button(controls, text="Stop Bot", command=self._stop_bot)
        self.stop_btn.grid(row=0, column=1, padx=6, sticky="ew")
        self.clear_btn = ttk.Button(controls, text="Clear Logs", command=self._clear_logs)
        self.clear_btn.grid(row=0, column=2, padx=(6, 0), sticky="ew")

        log_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="none", height=20)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=y_scroll.set)

    def _add_entry(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        return row + 1

    def _add_entry_with_button(self, parent, row, label, variable, button_text, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=(6, 0), padx=(0, 6))
        ttk.Button(parent, text=button_text, command=command).grid(row=row, column=2, sticky="ew", pady=(6, 0))
        return row + 1

    def _pick_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")],
            initialdir=self._script_dir,
        )
        if path:
            self.checkpoint_var.set(path)

    def _pick_models_dir(self):
        path = filedialog.askdirectory(title="Select Models Folder", initialdir=self._script_dir)
        if path:
            self.models_var.set(path)

    def _append_log(self, line: str):
        self.log_text.insert("end", f"{line}\n")
        self.log_text.see("end")

    def _clear_logs(self):
        self.log_text.delete("1.0", "end")

    def _set_running_state(self, running: bool):
        self.start_btn.configure(state=("disabled" if running else "normal"))
        self.stop_btn.configure(state=("normal" if running else "disabled"))

    def _validate_rate_limit(self) -> int:
        try:
            delay_ms = int(float(self.min_delay_var.get().strip()))
        except Exception:
            delay_ms = 1000
        delay_ms = max(1000, delay_ms)
        self.min_delay_var.set(str(delay_ms))
        return delay_ms

    def _container_path(self, host_path: str) -> str:
        """Map a project file to its location in the rl-coordinator container."""
        absolute = os.path.abspath(host_path)
        relative = os.path.relpath(absolute, self._repo_root)
        if relative == ".." or relative.startswith(f"..{os.sep}"):
            raise ValueError("Docker runtime can only use checkpoints inside this project.")
        parts = relative.split(os.sep)
        if parts[0].lower() == "rl-v2":
            return "/app/v2/" + "/".join(parts[1:])
        if parts[0].lower() == "rl-environment":
            return "/app/" + "/".join(parts[1:])
        raise ValueError("Docker runtime checkpoint must be under rl-v2 or rl-environment.")

    @staticmethod
    def _docker_host_url(value: str) -> str:
        """Make a host-local game-server URL reachable from Docker Desktop."""
        raw = value.strip()
        if not raw:
            return raw
        parsed = urlsplit(raw)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            return raw
        hostname = "host.docker.internal"
        netloc = hostname if parsed.port is None else f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    def _build_command(self):
        if not os.path.isfile(self._bot_script):
            raise FileNotFoundError(f"Cannot find bot script: {self._bot_script}")

        player_url = self.player_url_var.get().strip()
        base_url = self.base_url_var.get().strip()
        player_id = self.player_id_var.get().strip()
        if not player_url and not (base_url and player_id):
            raise ValueError("Provide Player URL, or Base URL + Player ID.")

        delay_ms = self._validate_rate_limit()
        bot_args = []

        if player_url:
            bot_args.extend(["--player-url", player_url])
        if base_url:
            bot_args.extend(["--base-url", base_url])
        if player_id:
            bot_args.extend(["--player-id", player_id])

        game_url = self.game_url_var.get().strip()
        if game_url:
            bot_args.extend(["--game-url", game_url])
        game_id = self.game_id_var.get().strip()
        if game_id:
            bot_args.extend(["--game-id", game_id])
        player_name = self.player_name_var.get().strip()
        if player_name:
            bot_args.extend(["--player-name", player_name])

        checkpoint = self.checkpoint_var.get().strip()
        if checkpoint:
            bot_args.extend(["--checkpoint", checkpoint])

        models = self.models_var.get().strip()
        if models:
            bot_args.extend(["--models", models])

        poll = self.poll_interval_var.get().strip() or "1000"
        timeout = self.timeout_var.get().strip() or "60"
        level = self.log_level_var.get().strip() or "INFO"

        bot_args.extend(["--min-action-delay-ms", str(delay_ms)])
        bot_args.extend(["--poll-interval-ms", poll])
        bot_args.extend(["--request-timeout-sec", timeout])
        bot_args.extend(["--log-level", level])
        if self.no_random_fallback_var.get():
            bot_args.append("--no-random-fallback")

        if self.runtime_var.get() == "Host Python (local)":
            try:
                __import__("rust_tfm_rl")
            except Exception as exc:
                raise RuntimeError(
                    "Local inference needs the rust_tfm_rl extension. "
                    "Follow RL-V2-LIVE-INFERENCE.md section 'One-time local setup', "
                    "then restart this launcher."
                ) from exc
            return [sys.executable, self._bot_script, *bot_args]

        if not checkpoint:
            raise ValueError("Choose an RL-v2 checkpoint when using the Docker runtime.")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        docker = shutil.which("docker") or "docker"
        hard_compose = os.path.join(self._repo_root, "docker-compose.rl_hard.yml")
        v2_compose = os.path.join(self._repo_root, "docker-compose.rl_v2.yml")
        for compose_file in (hard_compose, v2_compose):
            if not os.path.isfile(compose_file):
                raise FileNotFoundError(f"Cannot find compose file: {compose_file}")

        translated_args = list(bot_args)
        for flag in ("--player-url", "--base-url", "--game-url"):
            if flag in translated_args:
                index = translated_args.index(flag) + 1
                translated_args[index] = self._docker_host_url(translated_args[index])
        for flag in ("--checkpoint", "--models"):
            if flag in translated_args:
                index = translated_args.index(flag) + 1
                translated_args[index] = self._container_path(translated_args[index])
        return [
            docker, "compose", "-f", hard_compose, "-f", v2_compose,
            "run", "--rm", "--no-deps", "-e", "TFM_RL_V2=1",
            "rl-coordinator", "python", "standalone_bot.py", *translated_args,
        ]

    def _start_bot(self):
        if self._proc and self._proc.poll() is None:
            messagebox.showinfo("Bot Running", "The bot process is already running.")
            return

        try:
            cmd = self._build_command()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        self._append_log(f"[UI] Starting: {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=self._script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror("Failed To Start", str(exc))
            self._proc = None
            return

        self._set_running_state(True)
        self._start_stream_thread(self._proc.stdout, "OUT")
        self._start_stream_thread(self._proc.stderr, "ERR")
        self._schedule_pump()

    def _start_stream_thread(self, stream, prefix: str):
        def _worker():
            try:
                if stream is None:
                    return
                for line in iter(stream.readline, ""):
                    text = line.rstrip("\r\n")
                    if text:
                        self._line_queue.put(f"[{prefix}] {text}")
            except Exception as exc:
                self._line_queue.put(f"[UI] Stream error ({prefix}): {exc}")
            finally:
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def _schedule_pump(self):
        if self._after_id is not None:
            return
        self._after_id = self.after(100, self._pump)

    def _pump(self):
        self._after_id = None
        while True:
            try:
                line = self._line_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(line)

        if self._proc:
            code = self._proc.poll()
            if code is None:
                self._schedule_pump()
            else:
                self._append_log(f"[UI] Bot exited with code {code}")
                self._proc = None
                self._set_running_state(False)

    def _stop_bot(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._set_running_state(False)
            return

        self._append_log("[UI] Stopping bot process...")
        self._set_running_state(False)

        def _terminate():
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
            except Exception as exc:
                self._line_queue.put(f"[UI] Failed to stop process cleanly: {exc}")

        threading.Thread(target=_terminate, daemon=True).start()
        self._schedule_pump()

    def _on_close(self):
        if self._proc and self._proc.poll() is None:
            if not messagebox.askyesno("Exit", "Bot is running. Stop it and exit?"):
                return
            self._stop_bot()
            start = time.time()
            while self._proc and self._proc.poll() is None and (time.time() - start) < 6.0:
                self.update_idletasks()
                self.update()
                time.sleep(0.05)
        self.destroy()


def main():
    app = StandaloneBotLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
