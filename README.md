# ViZDoom Deep RL Ablation Study

**Reinforcement Learning Course Final Project**

An ablation study of deep reinforcement learning algorithms in ViZDoom environments, focusing on parameter sensitivities and extensions in visual navigation and combat tasks.

## Project Overview

This project implements and compares deep RL algorithms from scratch:

| Algorithm | Type | Description |
|-----------|------|-------------|
| **DQN** | Off-policy TD | Deep Q-Network (Mnih et al., 2015) |
| **Deep SARSA** | On-policy TD | Uses actual next action for updates |
| **DDQN** | Extension | Reduces overestimation bias (Van Hasselt, 2016) |
| **Dueling DQN** | Extension | Separate value/advantage streams (Wang, 2016) |
| **PER** | Extension | Prioritized Experience Replay (Schaul, 2016) |

### ViZDoom Scenarios

- **VizdoomBasic-v0**: Basic shooting task (3 actions, easy)
- **VizdoomTakeCover-v0**: Survival/dodging task (2 actions, medium)
- **VizdoomDeathmatch-v0**: Combat task (6+ actions, hard)

## Quick Start

### Local Setup

```bash
# Clone repository
git clone https://github.com/username/vizdoom-ablation.git
cd vizdoom-ablation

# Create virtual environment (Python 3.12)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python experiments/test_setup.py
```

### Google Colab

Open `notebooks/main_colab.ipynb` in Google Colab for a complete walkthrough.

## Usage

### Single Training Run

```bash
# Train DQN on Basic scenario
python experiments/train.py

# Override parameters
python experiments/train.py agent.type=ddqn env.scenario=VizdoomTakeCover-v0

# Disable WandB for local testing
python experiments/train.py logging.wandb_enabled=false
```

### Ablation Study

```bash
# Run algorithm comparison
python experiments/ablate.py --phase algorithms

# Run learning rate ablation
python experiments/ablate.py --phase lr

# Run all ablations
python experiments/ablate.py --phase all --episodes 1000

# Preview commands (dry run)
python experiments/ablate.py --phase extensions --dry-run
```

## Project Structure

```
vizdoom-ablation/
├── src/
│   ├── agents/           # RL agent implementations
│   │   ├── base.py       # QNetwork CNN + BaseAgent ABC
│   │   ├── dqn.py        # Vanilla DQN
│   │   ├── deep_sarsa.py # On-policy Deep SARSA
│   │   └── extensions.py # DDQN, Dueling DQN
│   ├── envs/             # Environment wrappers
│   │   └── vizdoom_wrapper.py
│   └── utils/            # Utilities
│       ├── replay_buffer.py  # Uniform + PER
│       ├── logging.py        # WandB + CSV
│       ├── plotting.py       # Matplotlib
│       └── factory.py        # Agent/buffer builders
├── configs/              # Hydra YAML configs
│   ├── default.yaml      # Base configuration
│   ├── task_*.yaml       # Scenario-specific
│   ├── ablation_*.yaml   # Parameter ablations
│   └── extension_*.yaml  # DQN extensions
├── experiments/          # Runnable scripts
│   ├── train.py          # Main training script
│   ├── ablate.py         # Grid search runner
│   └── test_setup.py     # Installation test
├── notebooks/
│   └── main_colab.ipynb  # Full Colab demo
└── results/              # Output (gitignored)
```

## Configuration

All experiments are configured via Hydra YAML files. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent.type` | `dqn` | Agent type: dqn, deep_sarsa, ddqn, dueling |
| `agent.learning_rate` | `0.0001` | Adam learning rate |
| `agent.gamma` | `0.99` | Discount factor |
| `agent.n_step` | `1` | N-step returns (1=TD, 3=MC-like) |
| `buffer.prioritized` | `false` | Use PER |
| `training.num_episodes` | `2000` | Training episodes |

See `configs/default.yaml` for full schema.

## Ablation Parameters

The study ablates the following:

| Category | Parameter | Values |
|----------|-----------|--------|
| Learning Rate | `agent.learning_rate` | 0.0001, 0.001, 0.01 |
| Discount Factor | `agent.gamma` | 0.9, 0.99 |
| N-step Returns | `agent.n_step` | 1, 3 |
| Extensions | Agent type | DQN, DDQN, Dueling |
| Replay | `buffer.prioritized` | Uniform, PER |

## Results

Results are saved to `results/`:
- CSV logs for each run
- Learning curve plots (PNG)
- Aggregated summary statistics

### For IEEE Report

1. Export WandB runs to CSV from dashboard
2. Use `results/ablation_summary.csv` for tables
3. Copy learning curve PNGs for figures
4. Generate LaTeX tables with pandas

## Key Dependencies

- Python 3.12+
- PyTorch 2.9.1
- ViZDoom 1.2.4
- Gymnasium 1.2.3
- Hydra 1.3.2
- WandB 0.23.1

## References

1. Mnih et al. (2015) - Human-level control through deep RL
2. Van Hasselt et al. (2016) - Deep RL with Double Q-learning
3. Wang et al. (2016) - Dueling Network Architectures
4. Schaul et al. (2016) - Prioritized Experience Replay
5. Kempka et al. (2016) - ViZDoom: A Doom-based AI Research Platform

## License

MIT License

## Acknowledgments

- ViZDoom team for the research platform

