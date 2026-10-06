# Running the Standalone Bot

Attach a trained checkpoint to one player slot in a live Terraforming Mars game.

## Playing against 3 humans

1. Start the servers:

   ```bash
   docker compose -f docker-compose.rl_hard.yml -f docker-compose.rl_v2.yml up -d tfm-server-1 tfm-server-2 tfm-server-3 tfm-server-4
   ```

2. Create a 4-player game at <http://localhost:8081> (servers are on `8081`-`8084`).
3. Have the three humans join and open their player links.
4. Copy the **bot's** player link (the empty seat) - it looks like
   `http://localhost:8081/player?id=pXXXX`.
5. Start the bot (below) and paste that link.
6. Keep the process running until the game ends.

## Start the bot

GUI launcher (recommended):

```bash
python rl-environment/standalone_bot_tk.py
```

- **Pick Best...** next to Checkpoint lists every discovered checkpoint ranked best
  first, with the evidence behind each one, and pre-selects the strongest.
- Paste the player URL, leave **Runtime** on `Host Python (local)`, press
  **Start Bot**.
- Keep **Safe live mode** checked so a rejected policy action never turns into a
  random move at a human table.

Command line:

```bash
python rl-environment/standalone_bot.py --player-url "http://localhost:8081/player?id=pXXXX" --min-action-delay-ms 1000
```

Omit `--checkpoint` to auto-load the best ranked checkpoint. To look before you
choose:

```bash
python rl-environment/standalone_bot.py --list-checkpoints
```

## Options worth knowing

| Flag | Notes |
| --- | --- |
| `--min-action-delay-ms` | Clamped to at least `1000`. |
| `--no-random-fallback` | Recommended for human games (on by default in the GUI). |
| `--checkpoint <file.pth>` | Use an exact checkpoint. |
| `--search-root <dir>` | Repeatable. Scans extra folders; overrides `--models`. |
| `--models <dir>` | Single models folder. Defaults to every store in the repo. |
| `--base-url` + `--player-id` | Alternative to pasting the full player URL. |

## Choosing a checkpoint

Ranking uses strength evidence, not the filename: tournament manifests first,
then benchmark reports (preferring the hardest baseline - `teacher` >
`award_teacher` > `champion` > `random`), then legacy Elo, then training
decisions. Only completed, gate-passing benchmark runs count as verified, so a
high win rate against a weak opponent never outranks a real result against the
teacher. `[v]` in the output marks verified checkpoints.

Stores scanned automatically: `rl-v2`, `rl-v3`, `rl-v4`, `rl-alphago`,
`rl-models`, `rl-models-global`.

## Host Python runtime

The `Host Python (local)` runtime needs the Rust inference extension:

```bash
cd rl-environment
maturin build --release --skip-auditwheel --interpreter python3
python -c "import rust_tfm_rl; print(rust_tfm_rl.backend_info())"
```

If that import fails, switch **Runtime** to `Docker (optional)`. The Docker
runtime bind-mounts the selected checkpoint's directory automatically, so
checkpoints from any store work.

## Stopping

Press **Stop Bot** in the GUI, or `Ctrl+C` in the terminal.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Failed to get player state` | Wrong player URL, or the game already started and the slot is gone. |
| `No checkpoints found` | Pass `--checkpoint <file.pth>` or `--search-root <dir>`. |
| Local runtime refuses to start | Build `rust_tfm_rl` (above) or use the Docker runtime. |
| Bot idles | Confirm the humans are all still in the game; the bot only acts on its own turn. |
