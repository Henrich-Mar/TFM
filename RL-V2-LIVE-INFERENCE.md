# RL-v2: play a checkpoint against human players

The project includes a desktop launcher at `rl-environment/standalone_bot_tk.py`.
It runs the bot directly from your local Python environment; it does not start
or connect to a Docker inference container.

## One-time local setup

The local policy uses the project's `rust_tfm_rl` extension. Install the Rust
toolchain and build that extension once in the same Python environment you
will use to start the UI:

```powershell
# Install Rust if `cargo --version` is not available. Restart PowerShell afterwards.
winget install Rustlang.Rustup

# From the repository root, with your desired virtual/Conda environment active:
python -m pip install maturin
Push-Location rl-environment
python -m maturin develop --release
Pop-Location

# Verify that the local extension can be imported.
python -c "import rust_tfm_rl; print(rust_tfm_rl.backend_info())"
```

If the Rust build reports that the MSVC linker is missing, install **Desktop
development with C++** from Visual Studio Build Tools, reopen PowerShell, and
run `python -m maturin develop --release` again. The bot also needs its usual
local Python dependencies (`torch`, `numpy`, and `aiohttp`); install them with
`python -m pip install -r rl-environment/requirements.txt` when missing.

## Before starting

1. Start Docker Desktop and the intended game server. The current local
   servers use `http://localhost:8081` through `http://localhost:8084`.
2. Create a game containing the human players and a separate bot player seat.
   Copy the bot seat's private player URL. It has the form
   `http://localhost:8081/player?id=PLAYER_ID`.
3. Do not open or submit moves from that bot player URL while the bot is
   running. Run exactly one inference process per player ID.

## Desktop UI (recommended)

From the repository root, launch:

```powershell
python rl-environment/standalone_bot_tk.py
```

Fill in either:

- **Player URL**: paste the bot seat's whole private player URL; or
- **Base URL** and **Player ID**: for example, `http://localhost:8081` and
  the ID after `player?id=`.

Leave **Runtime** set to **Host Python (local)**. The UI defaults to the
newest tested candidate:

```text
rl-v2\checkpoints\candidate_000275219.pth
```

Then press **Start Bot**. The log pane reports the resolved player name,
checkpoint, target, and submitted actions. Use **Stop Bot** only to detach the
bot; it does not change or delete the game.

For a remote server, put its real public HTTP(S) address in Base URL or Player
URL. A local game server can remain `http://localhost:8081`; local Python can
reach it directly.

## Command-line alternative

This runs the same local process as the UI and is useful for long-running
games:

```powershell
python rl-environment/standalone_bot.py `
  --base-url http://localhost:8081 `
  --player-id YOUR_PLAYER_ID `
  --checkpoint rl-v2/checkpoints/candidate_000275219.pth `
  --min-action-delay-ms 1000 --poll-interval-ms 1000 --log-level INFO
```

For a remote server, replace `http://localhost:8081` with its real public
address. To run a different candidate, substitute its path below
`rl-v2/checkpoints/`.

## Checkpoint choice and safeguards

`candidate_000275219.pth` is the latest candidate. Its completed Stage 1
benchmark beat the teacher baseline (50% first-place rate across 120 games),
but did **not** beat the current champion gate, so treat human games as an
evaluation rather than a proven promotion. The bot enforces a minimum
one-second interval between actions; keep this at 1000 ms or higher.

When the game finishes, the process exits after the final state is observed.
The game server remains the source of truth for results.
