# RuleSmith: Multi-Agent LLMs for Automated Game Balancing


## Overview

RuleSmith uses Large Language Models to play a simplified strategy game (CivMini) and optimizes game balance parameters through self-play. The system employs Bayesian Optimization with acquisition-based adaptive sampling to efficiently explore the parameter space.

**Key Features:**
- Multi-agent LLMs for rulebook-based game play (InternVL3.5-2B/8B)
- Bayesian Optimization with adaptive sampling
- Multi-GPU parallel game simulation

[Project Page](https://adonis-galaxy.github.io/RuleSmith-website/) ｜ [Paper](https://arxiv.org/abs/2602.06232)


## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, Transformers, scikit-optimize



## Usage

### Training

```bash
# edit parameters in run.sh
bash run.sh
```

### Evaluation

```bash
# edit parameters in run_eval.sh, we provide an example optimzied theta.json for default evaluation
bash run_eval.sh
```

### Visualization

```bash
# Single game
python vis_from_log.py path/to/game.log --output-dir output/

# Batch (all games in iteration)
bash vis_iter.sh -y -g runs/*/logs/run_*/game_logs/iter_*
```

## Scripts

| Script | Description |
|--------|-------------|
| `run.sh` | training |
| `run_eval.sh` | evaluation |
| `eval_theta.py` | Evaluate specific theta parameters |
| `optimize_demo.py` | Main optimization entry point |
| `vis_from_log.py` | Generate game visualizations |
| `vis_iter.sh` | Batch visualization |

## Project Structure

```
RuleSmith/
├── civmini/           # Game environment and agents
├── examples/          # Pre-optimized parameters
├── optimize_demo.py   # Training entry
├── eval_theta.py      # Evaluation script
└── vis_from_log.py    # Visualization
```

## Pre-optimized Parameters

`examples/theta.json` contains parameters optimized for InternVL3.5-8B vs 8B, achieving ~50/50 win rate balance.

## Citation

```bibtex
@article{zeng2026rulesmith,
  title = {Rulesmith: multi-agent llms for automated game balancing},
  author = {Zeng, Ziyao and Liu, Chen and Liu, Tianyu and Wang, Hao and Sun, Xiatao and Yang, Fengyu and Liu, Xiaofeng and Fan, Zhiwen},
  journal={arXiv preprint arXiv:2602.06232},
  year = {2026}
}
```

## License

MIT
