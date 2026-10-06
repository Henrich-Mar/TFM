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

from checkpoint_catalog import default_search_bases, discover_checkpoints


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
        # Left blank on purpose: a hardcoded path goes stale as training moves
        # between stores. Use "Pick Best..." to choose from the ranked list.
        self.checkpoint_var = tk.StringVar(value="")
        self.models_var = tk.StringVar(value="")
        self.search_roots_var = tk.StringVar(value="")
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
        row = self._add_entry_with_buttons(
            top,
            row,
            "Checkpoint",
            self.checkpoint_var,
            [("Pick Best...", self._open_checkpoint_picker), ("Browse...", self._pick_checkpoint)],
        )
        row = self._add_entry(top, row, "Search Folders (optional)", self.search_roots_var)
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

    def _add_entry_with_buttons(self, parent, row, label, variable, buttons):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=(6, 0), padx=(0, 6))
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=2, sticky="ew", pady=(6, 0))
        holder.columnconfigure(list(range(len(buttons))), weight=1)
        for index, (text, command) in enumerate(buttons):
            ttk.Button(holder, text=text, command=command).grid(
                row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0)
            )
        return row + 1

    def _explicit_search_roots(self):
        """Roots the user typed into the Search Folders field."""
        roots = []
        for token in self.search_roots_var.get().replace(";", ",").split(","):
            value = token.strip()
            if value:
                roots.append(value)
        return roots

    def _search_roots(self):
        """Explicit roots if typed, else the models folder plus every known store.

        The models folder is pre-filled with a default that may not exist yet, so
        it must not hide the stores that do contain checkpoints.
        """
        roots = self._explicit_search_roots()
        if roots:
            return roots
        models = self.models_var.get().strip()
        for root in [models, *default_search_bases(self._repo_root)]:
            if root and os.path.isdir(root) and root not in roots:
                roots.append(root)
        return roots

    def _open_checkpoint_picker(self):
        CheckpointPickerDialog(self, self._search_roots())

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

    def _docker_mount_args(self, host_path: str, container_dir: str) -> tuple:
        """Bind-mount the host directory holding ``host_path`` for one ``docker run``.

        The coordinator image only mounts ``rl-environment`` and ``rl-models``, so
        checkpoints from other stores (``rl-alphago``, ``rl-v3``, ``rl-v4``) are
        invisible inside the container. Mounting just the directory that holds the
        selected file keeps every store usable without editing compose files.
        """
        absolute = os.path.abspath(host_path)
        container_dir = container_dir.rstrip("/")
        if os.path.isdir(absolute):
            return ["-v", f"{absolute}:{container_dir}:ro"], container_dir
        host_dir = os.path.dirname(absolute)
        target = f"{container_dir}/{os.path.basename(host_dir) or 'store'}"
        return ["-v", f"{host_dir}:{target}:ro"], f"{target}/{os.path.basename(absolute)}"

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

        explicit_roots = [root for root in self._explicit_search_roots() if os.path.isdir(root)]
        for root in explicit_roots:
            bot_args.extend(["--search-root", root])

        models = self.models_var.get().strip()
        if models and os.path.isdir(models):
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
                    "Build it with 'maturin build --release --skip-auditwheel --interpreter python3' "
                    "from rl-environment, or switch the runtime to Docker."
                ) from exc
            return [sys.executable, self._bot_script, *bot_args]

        if not checkpoint:
            raise ValueError("Pick a checkpoint before starting the Docker runtime.")
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

        mount_args: list = []
        for flag, container_dir in (("--checkpoint", "/app/standalone-checkpoint"), ("--models", "/app/standalone-roots")):
            if flag in translated_args:
                index = translated_args.index(flag) + 1
                mounts, container_path = self._docker_mount_args(translated_args[index], container_dir)
                translated_args[index] = container_path
                mount_args.extend(mounts)
        search_index = 0
        while "--search-root" in translated_args[search_index:]:
            index = translated_args.index("--search-root", search_index) + 1
            search_index = index
            mounts, container_path = self._docker_mount_args(translated_args[index], "/app/standalone-roots")
            translated_args[index] = container_path
            mount_args.extend(mounts)

        return [
            docker, "compose", "-f", hard_compose, "-f", v2_compose,
            "run", "--rm", "--no-deps", "-e", "TFM_RL_V2=1", *mount_args,
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


class CheckpointPickerDialog(tk.Toplevel):
    """Ranked list of every discovered checkpoint, best first.

    The scan runs on a worker thread and results are applied on the Tk thread, so
    a large store with hundreds of benchmark reports never freezes the launcher.
    """

    COLUMNS = ("rank", "verified", "name", "store", "strength", "modified")
    HEADINGS = ("#", "", "Checkpoint", "Store", "Strength evidence", "Modified")
    WIDTHS = (44, 26, 250, 130, 460, 130)

    def __init__(self, parent, search_roots):
        super().__init__(parent)
        self.parent = parent
        self.search_roots = list(search_roots)
        self.candidates = []
        self._by_path = {}
        self._scan_pending = False
        self._scan_queue: "queue.Queue[tuple]" = queue.Queue()
        self._after_id = None

        self.title("Select Checkpoint")
        self.geometry("1120x620")
        self.minsize(880, 480)
        self.transient(parent)

        self.status_var = tk.StringVar(value="Scanning for checkpoints...")
        self.filter_var = tk.StringVar(value="")
        self.only_verified_var = tk.BooleanVar(value=False)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Filter").grid(row=0, column=0, sticky="w")
        filter_entry = ttk.Entry(header, textvariable=self.filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew", padx=6)
        filter_entry.bind("<KeyRelease>", lambda _event: self._render())
        ttk.Checkbutton(
            header,
            text="Strength verified only",
            variable=self.only_verified_var,
            command=self._render,
        ).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(header, text="Rescan", command=self._rescan).grid(row=0, column=3, padx=3)
        ttk.Button(header, text="Use Best", command=self._use_best).grid(row=0, column=4, padx=3)

        table_frame = ttk.Frame(self, padding=(10, 0))
        table_frame.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, columns=self.COLUMNS, show="headings", selectmode="browse", height=6)
        for column, heading, width in zip(self.COLUMNS, self.HEADINGS, self.WIDTHS):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w", stretch=(column == "strength"))
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.grid(row=0, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=y_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._render_details())
        self.tree.bind("<Double-1>", lambda _event: self._use_selected())

        detail_frame = ttk.Frame(self, padding=10)
        detail_frame.grid(row=2, column=0, sticky="nsew")
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self.detail_text = tk.Text(detail_frame, wrap="word", height=7, state="disabled")
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_text.yview)
        detail_scroll.grid(row=0, column=1, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scroll.set, background="#f4f4f4")

        footer = ttk.Frame(self, padding=(10, 0, 10, 10))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Cancel", command=self.destroy).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(footer, text="Use Selected", command=self._use_selected).grid(row=0, column=2, padx=(6, 0))

        self.scan()

    def _visible_rows(self):
        """Filter the ranked list without disturbing its order."""
        needle = self.filter_var.get().strip().lower()
        verified_only = self.only_verified_var.get()
        rows = []
        for index, candidate in enumerate(self.candidates, start=1):
            if verified_only and not candidate.verified:
                continue
            if needle:
                haystack = " ".join(
                    [candidate.name, candidate.store, candidate.summary(), " ".join(candidate.evidence), candidate.path]
                ).lower()
                if needle not in haystack:
                    continue
            rows.append((index, candidate))
        return rows

    def _render(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._by_path = {}
        for index, candidate in self._visible_rows():
            stamp = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(candidate.mtime)) if candidate.mtime else ""
            )
            self._by_path[candidate.path] = candidate
            self.tree.insert(
                "",
                "end",
                iid=candidate.path,
                values=(
                    index,
                    "yes" if candidate.verified else "-",
                    candidate.name,
                    candidate.store,
                    candidate.summary(),
                    stamp,
                ),
            )
        total = len(self.candidates)
        self.status_var.set(
            f"{len(self._visible_rows())} of {total} checkpoints, ranked best first."
            if total
            else f"No checkpoints found in: {', '.join(self.search_roots) or '<no folders>'}"
        )
        children = self.tree.get_children()
        if children and not self.tree.selection():
            self.tree.selection_set(children[0])
        self._render_details()

    def _selected_candidate(self):
        selection = self.tree.selection()
        if not selection:
            return None
        return self._by_path.get(selection[0])

    def _render_details(self):
        candidate = self._selected_candidate()
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        if candidate is not None:
            lines = [f"Path: {candidate.path}", "", candidate.summary(), ""]
            lines.extend(f"- {note}" for note in candidate.evidence)
            self.detail_text.insert("1.0", "\n".join(lines))
        else:
            self.detail_text.insert("1.0", "Select a checkpoint to see its evidence.")
        self.detail_text.configure(state="disabled")

    def _start_scan_thread(self):
        self._scan_pending = True
        self.status_var.set("Scanning for checkpoints...")

        def _worker():
            try:
                result = (discover_checkpoints(self.search_roots, root=self.parent._repo_root), None)
            except Exception as exc:
                result = (None, exc)
            # Never touch Tk from a worker thread; hand the result to the pump.
            self._scan_queue.put(result)

        threading.Thread(target=_worker, daemon=True).start()
        self._pump_scan()

    def _pump_scan(self):
        if self._after_id is None:
            self._after_id = self.after(120, self._check_scan)

    def _check_scan(self):
        self._after_id = None
        try:
            candidates, error = self._scan_queue.get_nowait()
        except queue.Empty:
            candidates, error = None, None
            found = False
        else:
            found = True
        if found:
            self._scan_pending = False
            self._apply_scan(candidates, error)
        if self._scan_pending:
            self._pump_scan()

    def _apply_scan(self, candidates, error):
        if not self.winfo_exists():
            return
        if error is not None:
            self.status_var.set(f"Scan failed: {error}")
            messagebox.showerror("Checkpoint Scan Failed", str(error), parent=self)
            return
        self.candidates = candidates or []
        self._render()
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])

    def scan(self):
        self._start_scan_thread()

    def _rescan(self):
        self.search_roots = self.parent._search_roots()
        self._start_scan_thread()

    def _use_selected(self):
        candidate = self._selected_candidate()
        if candidate is None:
            messagebox.showinfo("No Selection", "Select a checkpoint first.", parent=self)
            return
        self.parent.checkpoint_var.set(candidate.path)
        self.destroy()

    def _use_best(self):
        rows = self._visible_rows()
        if not rows:
            messagebox.showinfo("No Checkpoints", "No checkpoints match the current filter.", parent=self)
            return
        self.parent.checkpoint_var.set(rows[0][1].path)
        self.destroy()


def main():
    app = StandaloneBotLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
