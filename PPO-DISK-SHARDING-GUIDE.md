# PPO Disk Sharding Guide

This repo now uses disk-backed PPO rollout storage so self-play rollout data does not need to live mainly in RAM.

## What Changed

- PPO rollouts are written to disk as compressed shards.
- PPO later reads those shards back and trains from them.
- The rollout backlog shown on the dashboard now includes both:
  - in-memory queued rollout steps
  - disk-sharded queued rollout steps

The storage implementation is in:
- `rl-environment/models/rollout_store.py`
- `rl-environment/models/agent.py`

## Practical Flow

1. Games run normally.
2. When a game finishes, the agent builds PPO rollout steps.
3. Those steps are written to disk shards under the rollout shard directory.
4. When PPO starts, the agent loads the oldest queued rollout steps back from disk.
5. PPO trains on those steps.
6. Fully consumed shards are deleted. Partially consumed shards are rewritten with the remainder.

## Important Meaning Of Dashboard Metrics

`Steps available (pre-PPO)` means:
- total queued PPO rollout steps across all agents
- counted before PPO starts
- includes RAM + disk backlog

It does not mean:
- raw JSON state count
- per-game state count
- one-agent count unless you only have one PPO agent

`Rollout steps` means:
- how many rollout steps PPO actually consumed in that PPO cycle

## Current Compose Knobs

These are currently set in `docker-compose.rl_hard.yml`:

```yaml
- AGENT_INFERENCE_DEVICE=cpu
- PPO_DEVICE=cuda
- PPO_ENABLE=true
- PPO_ROLLOUT_STEPS=12288
- PPO_DISK_SHARDING_ENABLE=1
- PPO_DISK_SHARD_MAX_STEPS=2048
- PPO_ROLLOUT_SHARD_DIR=/app/rl-models/ppo-rollouts
- PPO_MIN_STEPS_PER_AGENT=1024
- PPO_EPOCHS=8
- PPO_MINIBATCH_SIZE=512
- PPO_BUFFER_MAX_STEPS=2048
```

## What These Settings Mean

- `PPO_ROLLOUT_STEPS=12288`
  PPO tries to consume up to 12,288 queued rollout steps per PPO cycle.

- `PPO_DISK_SHARD_MAX_STEPS=2048`
  Rollout data is stored in shards of up to 2,048 steps per file.

- `PPO_BUFFER_MAX_STEPS=2048`
  Keep the in-memory queue small. The bulk of queued rollout data should live on disk.

- `PPO_MIN_STEPS_PER_AGENT=1024`
  PPO may raise its effective total budget floor based on agent count. With 4 PPO agents, that floor is about 4,096 total steps if enough queued data exists.

## Where The Shards Live

Inside the container:

```text
/app/rl-models/ppo-rollouts
```

On the host with the current volume mount:

```text
./rl-models/ppo-rollouts
```

Typical layout:

```text
rl-models/
  ppo-rollouts/
    agent_<agent-id>/
      rollout_<timestamp>_<seq>_<count>.pkl.gz
```

## Tuning Rules

If PPO is not keeping up with self-play:
- raise `PPO_ROLLOUT_STEPS`
- reduce game generation speed
- lower `PPO_EPOCHS` if wall-clock PPO time is too high

If VRAM is the problem:
- lower `PPO_MINIBATCH_SIZE`
- move `PPO_DEVICE` to `cpu`

If RAM is the problem:
- keep `PPO_BUFFER_MAX_STEPS` small
- keep disk sharding enabled

If backlog grows too much:
- compare `Steps available (pre-PPO)` to `Rollout steps`
- if available steps stay much higher than rollout steps for many generations, PPO is not draining fast enough

## Caveat

The current shard directories are keyed by agent ID. If agents are recreated with new IDs, old shard directories are not automatically reattached to the new agents.
