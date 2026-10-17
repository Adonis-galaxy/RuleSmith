#!/usr/bin/env python3
"""
TPE (Tree-structured Parzen Estimator) baseline for CivMini game balance, via Optuna.

Minimizes the noisy balance objective returned by
civmini.selfplay.run_multiple_games:
    stats["balance_score"] = |empire_wr-0.5| + |nomads_wr-0.5| + 0.5*draw_rate
(lower is better; 0.0 == perfect balance).

Optuna is installed into a side directory (NOT the venv) and made importable by
prepending it to sys.path before the optuna import.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np

import optuna
from optuna.samplers import TPESampler

from civmini.config import (
    GameConfig,
    get_theta_bounds,
    get_optimizable_params,
    get_fixed_params,
    get_param_precision,
    discretize_theta,
)
from civmini.llm_client import get_llm_client, MockLLM
from civmini.selfplay import run_multiple_games
from civmini.utils import set_random_seed

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "OpenGVLab/InternVL3_5-8B"


# --------------------------------------------------------------------------- #
# Logging (mirrors optimize_demo.py TeeLogger pattern)
# --------------------------------------------------------------------------- #
class TeeLogger:
    """Logger that writes to both file and stdout."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # line buffered

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if not self.log.closed:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        # Interpreter shutdown flushes stdout after close(); a raise here exits with rc=120
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(output_dir, log_prefix="tpe"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{log_prefix}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    tee = TeeLogger(log_file)
    sys.stdout = tee
    return log_file, tee


# --------------------------------------------------------------------------- #
# Shared evaluation / checkpoint helpers
# --------------------------------------------------------------------------- #
def complete_theta(theta_opt):
    """Merge optimizable theta with fixed params -> full theta for evaluation."""
    complete = dict(theta_opt)
    for k, v in get_fixed_params().items():
        complete.setdefault(k, v)
    return complete


def evaluate_theta(theta_opt, iteration, n_games, seed, config, llm,
                   num_gpus, output_dir):
    """Discretize, run n_games, return (theta_disc, stats). Mirrors optimize.py."""
    theta_disc = discretize_theta(theta_opt)
    complete = complete_theta(theta_disc)

    iter_log_dir = os.path.join(output_dir, "game_logs", f"iter_{iteration}")
    os.makedirs(iter_log_dir, exist_ok=True)

    seed_base = (seed if seed is not None else 0) * 100000 + iteration * 100
    os.environ["CIVMINI_SEED_BASE"] = str(seed_base)

    results, stats = run_multiple_games(
        theta=complete,
        llm=llm,
        config=config,
        n_games=n_games,
        use_checker=False,
        verbose=False,
        multi_action_mode=True,
        game_log_dir=iter_log_dir,
        num_gpus=num_gpus,
    )
    return theta_disc, stats


def save_checkpoint(output_dir, iteration, theta_disc, score, best_theta,
                    best_score, history, total_games_used, method):
    ckpt = {
        "iteration": iteration,
        "theta": theta_disc,
        "score": score,
        "best_theta": best_theta,
        "best_score": best_score,
        "history": history,
        "total_games_used": total_games_used,
        "method": method,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "checkpoint.json"), "w") as f:
        json.dump(ckpt, f, indent=2)


def log_iteration(iteration, n_trials, theta_disc, n_games, stats,
                  best_score, total_games_used, total_games):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration + 1}/{n_trials}")
    print(f"{'='*60}")
    print(f"  theta: {theta_disc}")
    print(f"  n_games: {n_games}")
    print(f"  Empire WR: {stats['empire_winrate']:.2%}  "
          f"Nomads WR: {stats['nomads_winrate']:.2%}  "
          f"Draw rate: {stats['draw_rate']:.2%}")
    print(f"  Balance score: {stats['balance_score']:.4f}")
    print(f"  Running best: {best_score:.4f}")
    print(f"  Games used: {total_games_used}/{total_games}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="TPE (Optuna) baseline for CivMini balance")
    p.add_argument("--total-games", type=int, required=True,
                   help="Total game budget (n_trials = total_games // games_per_eval)")
    p.add_argument("--games-per-eval", type=int, default=16,
                   help="Games per trial (default 16)")
    p.add_argument("--seed", type=int, required=True, help="Random seed (TPESampler seed)")
    p.add_argument("--empire-model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--nomads-model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--mock", action="store_true", help="Use mock LLM (num_gpus=1)")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file, tee = setup_logging(args.output_dir, log_prefix="tpe")

    set_random_seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("TPE (Optuna) Game-Balance Optimization")
    print("=" * 60)
    print(f"  Total games:     {args.total_games}")
    print(f"  Games per eval:  {args.games_per_eval}")
    print(f"  Seed:            {args.seed}")
    print(f"  Optuna version:  {optuna.__version__}")
    print(f"  Empire model:    {args.empire_model}")
    print(f"  Nomads model:    {args.nomads_model}")
    print(f"  Max turns:       {args.max_turns}")
    print(f"  Mock:            {args.mock}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Log file:        {log_file}")

    # ---- config + LLM ----
    config = GameConfig()
    config.max_turns = args.max_turns
    config.empire_model = args.empire_model
    config.nomads_model = args.nomads_model

    if args.mock:
        num_gpus = 1
        llm = get_llm_client(use_mock=True)
    elif args.num_gpus > 1:
        num_gpus = args.num_gpus
        llm = MockLLM()  # placeholder; workers load their own models
    else:
        num_gpus = 1
        llm = get_llm_client(model_name=args.empire_model, use_mock=False)
    print(f"  Num GPUs:        {num_gpus}")

    # ---- search space ----
    optimizable = set(get_optimizable_params())
    all_bounds = get_theta_bounds()
    bounds = {k: v for k, v in all_bounds.items() if k in optimizable}
    names = sorted(bounds.keys())
    print(f"Optimizing {len(names)} params: {names}")

    # Budget -> number of trials (full batches; never exceeds total-games)
    n_trials = args.total_games // args.games_per_eval
    if n_trials < 1:
        n_trials = 1
    print(f"Running {n_trials} trials of {args.games_per_eval} games "
          f"(<= {args.total_games} total games)")
    print("=" * 60)

    # ---- optimization state (closed over by the objective) ----
    state = {
        "history": [],
        "best_score": float("inf"),
        "best_theta_disc": None,
        "total_games_used": 0,
    }

    def suggest_theta(trial):
        theta = {}
        for name in names:
            lo, hi = bounds[name]
            if get_param_precision(name) == 0.1:
                theta[name] = trial.suggest_float(name, float(lo), float(hi), step=0.1)
            else:
                theta[name] = trial.suggest_int(name, int(lo), int(hi))
        return theta

    def objective(trial):
        iteration = trial.number
        theta = suggest_theta(trial)

        n_games = min(args.games_per_eval,
                      args.total_games - state["total_games_used"])
        if n_games <= 0:
            raise optuna.TrialPruned()

        theta_disc, stats = evaluate_theta(
            theta, iteration, n_games, args.seed, config, llm,
            num_gpus, args.output_dir,
        )
        balance = stats["balance_score"]
        state["total_games_used"] += n_games

        state["history"].append([theta_disc, balance])
        if balance < state["best_score"]:
            state["best_score"] = balance
            state["best_theta_disc"] = theta_disc

        log_iteration(iteration, n_trials, theta_disc, n_games, stats,
                      state["best_score"], state["total_games_used"],
                      args.total_games)

        save_checkpoint(
            args.output_dir, iteration, theta_disc, balance,
            complete_theta(state["best_theta_disc"]), state["best_score"],
            state["history"], state["total_games_used"], method="tpe",
        )
        return balance

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=n_trials)

    # ---- final best ----
    best_complete = complete_theta(state["best_theta_disc"])
    best_path = os.path.join(args.output_dir, "best_theta.json")
    with open(best_path, "w") as f:
        json.dump(best_complete, f, indent=2)

    print("\n" + "=" * 60)
    print("TPE complete!")
    print("=" * 60)
    print(f"  Best balance score: {state['best_score']:.4f}")
    print(f"  Best theta: {state['best_theta_disc']}")
    print(f"  Total games used: {state['total_games_used']}/{args.total_games}")
    print(f"  Best theta saved to: {best_path}")
    print(f"  Checkpoint: {os.path.join(args.output_dir, 'checkpoint.json')}")
    print(f"  Log: {log_file}")

    sys.stdout = tee.terminal  # restore before closing to avoid shutdown-flush error
    tee.close()


if __name__ == "__main__":
    main()
