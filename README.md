# RuleSmith: Multi-Agent LLMs for Automated Game Balancing

[Project Page](https://adonis-galaxy.github.io/RuleSmith-website/) | [Paper](https://arxiv.org/abs/2602.06232)

RuleSmith balances an asymmetric game by coupling multi-agent LLM self-play with Bayesian optimization over a parameterized rule space. Two LLM agents play CivMini, a two-faction civilization-style game with 12 tunable parameters, and their win rates define a noisy balance objective. A Gaussian-process Bayesian optimizer with Expected Improvement proposes new rule configurations, and an adaptive budget gives promising candidates more evaluation games.

This repository contains the code used to obtain the results in our NeurIPS 2026 paper. Bugs found after the experiments have been fixed; see the commit history.

## Installation

```bash
pip install -r requirements.txt
pip install optuna   # only needed for the TPE baseline
```

Python 3.10+, PyTorch, Transformers, and scikit-optimize. The agents are [InternVL3.5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B) and [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B), downloaded from Hugging Face on first use. The optimization and evaluation scripts accept `--mock` (with `--num-gpus 1`) to run without a model, which is useful for checking an installation.

## Reproducing the main experiment

The main experiment optimizes the 12 parameters with InternVL3.5-8B playing both factions: 100 Bayesian-optimization iterations with Expected Improvement, 10 random initial points, an adaptive budget of 16 to 64 games per candidate, and a 16-turn cap. Games run in parallel, one model replica per GPU.

**1. Optimize.** `run.sh` contains exactly these settings; set `NUM_GPUS` to the number of visible GPUs.

```bash
bash run.sh
```

or, equivalently,

```bash
python optimize_demo.py --method bayesian --n-iterations 100 --n-games 16 --max-turns 16 \
  --empire-model OpenGVLab/InternVL3_5-8B --nomads-model OpenGVLab/InternVL3_5-8B \
  --adaptive-games --min-games 16 --max-games 64 --adaptive-strategy acquisition \
  --num-gpus 8 --seed 1 --checkpoint-interval 1 \
  --log-dir runs/seed1/logs --checkpoint-dir runs/seed1/checkpoints --output runs/seed1/optimized_theta.json
```

`--seed` sets both the optimizer's random state and the per-game seeds; the paper's repeated runs use seeds 1, 2, and 3. An interrupted run continues with `--resume-from runs/seed1/checkpoints/checkpoint_latest.json`.

**2. Evaluate the selected configuration on held-out games.** The configuration with the lowest observed loss is stored in the checkpoint:

```bash
python eval_theta.py --checkpoint runs/seed1/checkpoints/checkpoint_latest.json --use-best \
  --empire-model 8b --nomads-model 8b --n-games 300 --max-turns 16 --num-gpus 8 --seed 555 \
  --output-dir runs/seed1/eval_300
```

`examples/theta.json` is the optimized configuration reported in the paper (Appendix D.3). To evaluate it, run `bash run_eval.sh` or

```bash
python eval_theta.py --theta examples/theta.json --empire-model 8b --nomads-model 8b \
  --n-games 500 --max-turns 16 --num-gpus 8 --seed 1000 --output-dir runs/theta_eval_500
```

Each evaluation writes `eval_summary.json` (win, loss, and draw counts and the balance loss) and one log per game. Outcomes come from stochastic LLM decoding, so win rates are estimates; the paper reports Wilson intervals and binomial tests for them.

In the paper, a 100-iteration 8B optimization took about 40 hours on 8 NVIDIA A100 80GB GPUs.

## Other experiments

| Experiment | Command |
|---|---|
| LCB acquisition instead of EI | `optimize_demo.py ... --acq-func LCB` |
| Random search, (1+1)-ES | `optimize_demo.py --method random` or `--method evolution` |
| Fixed number of games per candidate | omit `--adaptive-games` and set `--n-games` |
| Other model pairings | `--empire-model` / `--nomads-model` (`2b`, `8b`, or a Hugging Face id) |
| NTBEA baseline | `python optimize_ntbea.py --total-games 1088 --games-per-eval 16 --seed 11 --num-gpus 8 --output-dir runs/ntbea` |
| TPE baseline | `python optimize_tpe.py --total-games 1088 --games-per-eval 16 --seed 12 --num-gpus 8 --output-dir runs/tpe` |
| Trajectory diversity (Appendix I) | `python analysis/trajectory_diversity.py <eval_output_dir>/eval_<timestamp>` |

The NTBEA and TPE scripts write `best_theta.json`, which `eval_theta.py --theta` evaluates.

## Visualization

```bash
python vis_from_log.py path/to/game.log --output-dir output/          # one game
bash vis_iter.sh -y -g runs/seed1/game_logs/iter_*                      # every game of an iteration
```

## Repository structure

```
civmini/              game engine, rulebook and retrieval, LLM agents and prompts, self-play, optimizers
optimize_demo.py      optimization entry point (Bayesian optimization, random search, (1+1)-ES)
eval_theta.py         held-out evaluation of a configuration
optimize_ntbea.py     NTBEA baseline
optimize_tpe.py       TPE baseline (Optuna)
analysis/             trajectory-diversity analysis
examples/theta.json   optimized configuration reported in the paper
run.sh, run_eval.sh   main optimization and evaluation with the paper's settings
```

## Citation

```bibtex
@inproceedings{zeng2026rulesmith,
  title     = {RuleSmith: Multi-Agent {LLMs} for Automated Game Balancing},
  author    = {Zeng, Ziyao and Wang, Hao and Liu, Chen and Yao, Youheng and Ni, Jingcheng and Liu, Tianyu and Sun, Xiatao and Yang, Fengyu and You, Chenyu and Liu, Xiaofeng and Rakita, Daniel and Coifman, Ronald R. and Kluger, Yuval and Fan, Zhiwen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT
