# 🚀 Terraforming Mars Reinforcement Learning Training Environment

This project implements a competitive multi-agent reinforcement learning system for training AI players in Terraforming Mars. The system uses evolutionary algorithms where AI agents compete against each other in tournaments, with the best performers breeding to create the next generation.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Game Server   │    │   Game Server   │    │   Game Server   │
│   (Docker)      │    │   (Docker)      │    │   (Docker)      │
│   Port 8081     │    │   Port 8082     │    │   Port 8083     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ RL Coordinator  │
                    │ (Tournament     │
                    │  Manager +      │
                    │  Evolution)     │
                    └─────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                        │
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│   Redis     │        │ PostgreSQL  │        │   Web API   │
│  (Cache &   │        │ (Metrics)   │        │(Monitoring) │
│ Job Queue)  │        │             │        │ Port 5000   │
└─────────────┘        └─────────────┘        └─────────────┘
```

## 🧠 AI Training Strategy

### Evolutionary Tournament System
1. **Population**: 32 AI agents with diverse neural network architectures
2. **Tournaments**: Agents compete in 4-player games (8 agents per tournament)
3. **Selection**: Top 20% survive, rest breed from top 50%
4. **Mutation**: Network weights and hyperparameters evolve
5. **Diversity**: Immigration of new random agents when diversity drops

### Neural Network Architecture
- **Input**: 512-dimensional state vector (game state, board, cards, opponents)
- **Hidden**: 2-4 layers, 128-512 neurons per layer
- **Output**: Policy head (action probabilities) + Value head (position evaluation)

### Reward Function
```python
reward = (ranking_points * 10) + (victory_points * 0.5) + (terraform_rating_gain * 5) + (global_progress * 2)
```

## 🛠️ Practical Migration Plan (High ROI)

This section describes the recommended transition path from the current evolution-first setup to a stronger RL-first setup while keeping existing infrastructure.

### Target Training Stack
1. **Primary optimizer**: PPO (clipped surrogate objective)
2. **Policy architecture**: Recurrent actor-critic (LSTM/GRU)
3. **Action selection**: Strict legal-action masking from API state
4. **Opponent curriculum**: League-style self-play using tournament infrastructure
5. **Evolution usage**: Diversity/initialization only (not primary optimizer)

### What to Change in the Repo

#### 1) Agent training logic (`rl-environment/models/agent.py`)
- Replace one-episode policy update with PPO minibatch updates:
  - Store trajectories with: `(obs, action, logp_old, value_old, reward, done, legal_mask)`
  - Add GAE(\(\lambda\)) advantage estimation
  - Add PPO clipping objective, value clipping, entropy bonus, KL monitoring
- Add recurrent hidden-state handling for rollout and training batches.
- Keep the existing policy/value heads; extend to recurrent backbone.

#### 2) Coordinator / rollout orchestration (`rl-environment/coordinator.py`)
- Separate **rollout collection** from **optimization steps**:
  - Collect N environment steps across tournaments/servers
  - Run K PPO epochs over shuffled minibatches
- Keep tournaments as evaluation and opponent sampling mechanism.
- Add league pools:
  - `main` (current best checkpoints)
  - `historical` (frozen snapshots)
  - `exploiters` (specialized challengers)

#### 3) Evolution manager (`rl-environment/agent_evolution.py`)
- Reduce evolution frequency and scope:
  - Use for occasional re-seeding and hyperparameter diversity
  - Do not overwrite PPO-trained population every generation
- Keep mutation/crossover for a small immigrant subset only.

#### 4) Action masking and observation validation
- Build a canonical legal-action mask directly from API payload (`waitingFor` and related fields).
- Log mask cardinality and rejected-action diagnostics.
- Add state schema versioning so PPO trajectories can be validated across runs.

### Recommended Environment Variables (new)
```bash
# PPO core
PPO_ENABLE=1
PPO_ROLLOUT_STEPS=8192
PPO_EPOCHS=4
PPO_MINIBATCH_SIZE=1024
PPO_CLIP_EPS=0.2
PPO_VALUE_CLIP_EPS=0.2
PPO_GAMMA=0.99
PPO_GAE_LAMBDA=0.95
PPO_ENTROPY_COEF=0.01
PPO_VALUE_COEF=0.5
PPO_MAX_GRAD_NORM=1.0
PPO_TARGET_KL=0.02

# Recurrent policy
PPO_RECURRENT=1
PPO_SEQUENCE_LENGTH=32

# League self-play
LEAGUE_ENABLE=1
LEAGUE_HISTORICAL_RATIO=0.4
LEAGUE_EXPLOITER_RATIO=0.2
LEAGUE_SNAPSHOT_INTERVAL=5
```

## 📏 Metrics to Add (Must-Have)

Add these metrics to PostgreSQL + dashboard so model quality and data integrity can be monitored.

### A) PPO Optimization Metrics
- `ppo/policy_loss`
- `ppo/value_loss`
- `ppo/entropy`
- `ppo/approx_kl`
- `ppo/clip_fraction`
- `ppo/explained_variance`
- `ppo/grad_norm`
- `ppo/learning_rate`

### B) Rollout / Return Metrics
- `rollout/episode_return_mean`
- `rollout/episode_return_p50_p90`
- `rollout/episode_length_mean`
- `rollout/terminal_rank_distribution`
- `rollout/victory_points_mean`

### C) Legal-Action and Reliability Metrics
- `action/legal_count_mean`
- `action/mask_coverage_rate` (fraction of decisions with a non-empty legal set)
- `action/rejected_by_server_rate`
- `action/fallback_rate`
- `action/pass_rate`

### D) Multi-Agent/League Metrics
- `league/winrate_vs_main`
- `league/winrate_vs_historical`
- `league/winrate_vs_exploiter`
- `league/elo_main`
- `league/meta_nash_proxy` (optional advanced)

### E) Data Quality Metrics (API vs Agent Observation)
- `obs/api_fetch_success_rate`
- `obs/state_parse_success_rate`
- `obs/missing_field_rate_by_key`
- `obs/waitingfor_type_distribution`
- `obs/unknown_waitingfor_type_count`
- `obs/state_hash_churn` (detect repeated stale states)

## 🔍 API vs HTML Visibility: Validation Checklist

Your concern is valid: if agents read only API but the UI shows additional information, training may be bottlenecked by missing observations.

### Goal
Verify that all decision-relevant information visible in HTML is available in API payloads consumed by agents.

### Procedure
1. **Capture paired snapshots** per decision step:
   - API JSON used by agent
   - UI-derived snapshot from the same game tick
2. **Define a feature parity spec**:
   - Resources, production, hand/public cards, tags, milestones/awards, board occupancy, global parameters, pending prompts (`waitingFor`), opponent public stats.
3. **Run automatic parity checker**:
   - Compare feature-by-feature availability
   - Emit mismatch report with counts and examples
4. **Block training on severe gaps**:
   - If critical feature missing rate exceeds threshold, flag run as invalid.

### Parity Metrics to Add
- `parity/feature_coverage_rate`
- `parity/critical_feature_missing_rate`
- `parity/html_only_feature_count`
- `parity/api_only_feature_count`
- `parity/blocking_mismatch_events`

### Suggested Critical Feature Set (initial)
- Current player resources and production vectors
- Current legal action family and parameters
- Card costs/tags/effects needed for choice quality
- Board tile ownership and adjacency-relevant context
- Milestones, awards, global parameter status, generation/phase

If any of these are not present in API input, add them to the API-backed state encoder before scaling PPO training.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB+ RAM (recommended)
- 4+ CPU cores (recommended)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd TFM
```

### 2. Start Training Environment
```bash
chmod +x start-rl-training.sh
./start-rl-training.sh
```

This will:
- Start 3 Terraforming Mars game servers
- Launch RL coordinator with evolution system
- Set up Redis for caching
- Set up PostgreSQL for metrics
- Start monitoring API

### 3. Monitor Training
- **Dashboard**: http://localhost:5000/dashboard
- **API Stats**: http://localhost:5000/stats
- **TensorBoard**: http://localhost:6006
- **Logs**: `docker-compose -f docker-compose.rl.yml logs -f rl-coordinator`

### 4. Manual Control
```bash
# View real-time logs
docker-compose -f docker-compose.rl.yml logs -f

# Enter coordinator container
docker exec -it rl-coordinator bash

# Stop everything
docker-compose -f docker-compose.rl.yml down
```

## 📊 Monitoring and Metrics

### Web Dashboard
The dashboard shows:
- Current generation and population size
- Best/average fitness scores
- Game server health
- Active tournaments
- Top performing agents

### API Endpoints
- `GET /stats` - Overall system statistics
- `GET /population` - Current population details
- `GET /tournaments` - Active tournament information
- `GET /servers` - Game server status
- `GET /health` - System health check

### Key Metrics
- **Fitness Score**: Combination of game ranking, victory points, and efficiency
- **Win Rate**: Percentage of games won (1st place)
- **Average VP**: Average victory points per game
- **Diversity**: Genetic diversity of population

## ⚙️ Configuration

### Environment Variables
```bash
# Population settings
POPULATION_SIZE=32          # Number of agents
TOURNAMENT_SIZE=8           # Agents per tournament
GENERATIONS=1000            # Max generations
GAMES_PER_EVAL=20          # Games per evaluation

# Server settings
GAME_SERVERS=tm-server-1:8080,tm-server-2:8080,tm-server-3:8080
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://postgres:password@postgres:5432/rl_metrics

# Self-play policy learning (on top of evolution)
SELF_PLAY_LEARNING=1        # 1=enable in-game policy/value updates, 0=disable
SELF_PLAY_REWARD_SCALE=1.0  # Scales terminal reward before policy update
```

### Agent Hyperparameters
Located in `rl-environment/models/agent.py`:
```python
@dataclass
class AgentConfig:
    state_size: int = 512
    hidden_size: int = 256
    num_layers: int = 3
    learning_rate: float = 3e-4
    epsilon: float = 0.1
    temperature: float = 1.0
```

## 🏆 Training Results

### Expected Timeline
- **Hours 1-4**: Random play, learning basic rules
- **Hours 4-12**: Basic strategy emergence
- **Hours 12-24**: Intermediate play (competitive with random players)
- **Days 2-7**: Advanced strategies (competitive with humans)
- **Week 2+**: Expert-level play optimization

### Performance Targets
- **Beginner**: >60% win rate vs random players
- **Intermediate**: >40% win rate vs rule-based bots
- **Advanced**: >30% win rate vs human players
- **Expert**: Consistent high victory point scores

## 🔧 Development

### Adding New Features

#### Custom Reward Function
Edit `coordinator.py` in the `_calculate_fitness_scores` method:
```python
def _calculate_fitness_scores(self, tournament_results):
    # Modify reward calculation here
    ranking_points = [100, 75, 50, 25][rank - 1]
    # Add your custom metrics
```

#### New Neural Network Architecture
Modify `models/agent.py`:
```python
class TerraformingMarsNetwork(nn.Module):
    def __init__(self, config):
        # Add your custom layers here
```

#### Enhanced State Encoding
Update `models/state_encoder.py`:
```python
def encode(self, player_state):
    # Add new game state features
```

### Testing Individual Agents
```python
# Test a trained agent
agent = RLAgent()
agent.load_model('rl-models/generation_100/agent_0_fitness_85.23.pth')

# Run test games
# (Implementation depends on your testing setup)
```

## 🐛 Troubleshooting

### Common Issues

#### Game Servers Not Starting
```bash
# Check Docker resources
docker system df
docker system prune

# Check ports
netstat -tulpn | grep 808[1-3]
```

#### RL Training Stuck
```bash
# Check logs
docker logs rl-coordinator

# Restart coordinator only
docker-compose -f docker-compose.rl.yml restart rl-coordinator
```

#### Out of Memory
- Reduce population size: `POPULATION_SIZE=16`
- Reduce concurrent games: Modify semaphore in `tournament_manager.py`
- Use smaller neural networks: Reduce `hidden_size` in `AgentConfig`

#### Performance Issues
- Scale down to fewer game servers
- Increase game timeout
- Reduce tournament frequency

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose -f docker-compose.rl.yml up --build
```

## 📈 Advanced Usage

### Distributed Training
To scale across multiple machines:
1. Run game servers on different hosts
2. Update `GAME_SERVERS` environment variable
3. Use external Redis and PostgreSQL

### Custom Opponents
Add rule-based opponents for evaluation:
```python
# In tournament_manager.py
class RuleBasedAgent:
    def play_game(self, game_instance, player_name):
        # Implement rule-based strategy
```

### Model Export
```python
# Export for external use
agent = RLAgent()
agent.load_model('path/to/model.pth')
torch.jit.save(torch.jit.script(agent.network), 'exported_model.pt')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project builds upon the open-source Terraforming Mars implementation and follows the same licensing terms.

## 🙏 Acknowledgments

- Original Terraforming Mars game implementation
- OpenAI Gym for RL environment standards
- PyTorch for deep learning framework
- Docker for containerization
