#!/usr/bin/env python3
"""
NTBEA (N-Tuple Bandit Evolutionary Algorithm) baseline for CivMini game balance.

Reference: Lucas et al., "The N-Tuple Bandit Evolutionary Algorithm for Game
Agent Optimisation" (CEC 2018).

Implemented from scratch (pure python + numpy). Minimizes the noisy balance
objective returned by civmini.selfplay.run_multiple_games:
    stats["balance_score"] = |empire_wr-0.5| + |nomads_wr-0.5| + 0.5*draw_rate
(lower is better; 0.0 == perfect balance).

Internally we maximize fitness = -balance_score so that "higher == better" for
the bandit / UCB machinery, while all logging / checkpoints report the raw
balance_score exactly like civmini/optimize.py.

Model over the DISCRETIZED parameter grid:
  * 1-tuples : each of the 12 params alone            (12 patterns)
  * 2-tuples : every pair of params                   (66 patterns)
  * 12-tuple : the full point                         (1 pattern)
For every tuple pattern we keep, keyed by the tuple's value combination, a
running (sum, count) of observed fitness -> mean fitness + visit count.
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np

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
N_NEIGHBORS = 50
UCB_EPSILON = 1e-6


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


def setup_logging(output_dir, log_prefix="ntbea"):
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
# Parameter grid over the discretized search space
# --------------------------------------------------------------------------- #
def build_param_space():
    """Return (names, bounds, grids) for the 12 optimizable params.

    grids[name] is the sorted list of discrete grid values (already discretized).
    """
    optimizable = set(get_optimizable_params())
    all_bounds = get_theta_bounds()
    bounds = {k: v for k, v in all_bounds.items() if k in optimizable}
    names = sorted(bounds.keys())
    grids = {}
    for name in names:
        lo, hi = bounds[name]
        step = get_param_precision(name)  # 1.0 or 0.1
        n = int(round((hi - lo) / step)) + 1
        grids[name] = [round(lo + i * step, 10) for i in range(n)]
    return names, bounds, grids


def build_patterns(n_params):
    """1-tuples + all 2-tuples + the full n-tuple, as tuples of param indices."""
    one = [(i,) for i in range(n_params)]
    two = [(i, j) for i in range(n_params) for j in range(i + 1, n_params)]
    full = tuple(range(n_params))
    return one + two + [full]


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

    # Distinct reproducible per-game seeds (consumed by run_single_game_worker)
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


def log_iteration(iteration, n_iterations, theta_disc, n_games, stats,
                  best_score, total_games_used, total_games):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration + 1}/{n_iterations}")
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
# NTBEA core
# --------------------------------------------------------------------------- #
class NTupleModel:
    """N-tuple bandit statistics over the discrete grid (index space)."""

    def __init__(self, patterns, C, epsilon=UCB_EPSILON):
        self.patterns = patterns
        self.C = C
        self.epsilon = epsilon
        self.stats = {p: {} for p in patterns}  # pattern -> {value_key: [sum, count]}
        self.total_evals = 0

    def update(self, x, fitness):
        """x is a tuple of grid indices (length n_params); fitness higher=better."""
        for p in self.patterns:
            key = tuple(x[i] for i in p)
            entry = self.stats[p].setdefault(key, [0.0, 0])
            entry[0] += fitness
            entry[1] += 1
        self.total_evals += 1

    def ucb(self, x):
        """Mean fitness estimate + exploration bonus, averaged over covering tuples."""
        N = self.total_evals
        log_term = math.log(N + 1)
        total = 0.0
        for p in self.patterns:
            key = tuple(x[i] for i in p)
            entry = self.stats[p].get(key)
            if entry is not None and entry[1] > 0:
                mean = entry[0] / entry[1]
                count = entry[1]
            else:
                mean = 0.0
                count = 0
            total += mean + self.C * math.sqrt(log_term / (count + self.epsilon))
        return total / len(self.patterns)


def make_mutator(rng, grid_sizes):
    """Return mutate(x): flip each param with prob 0.5, always >=1 change."""
    n = len(grid_sizes)

    def mutate(x):
        y = list(x)
        mutated = False
        for i in range(n):
            if rng.random() < 0.5:
                y[i] = int(rng.integers(0, grid_sizes[i]))
                mutated = True
        if not mutated:
            i = int(rng.integers(0, n))
            y[i] = int(rng.integers(0, grid_sizes[i]))
        return tuple(y)

    return mutate


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="NTBEA baseline for CivMini balance")
    p.add_argument("--total-games", type=int, required=True,
                   help="Total game budget (optimization stops when consumed)")
    p.add_argument("--games-per-eval", type=int, default=8,
                   help="Games per point evaluation (default 8, one per GPU)")
    p.add_argument("--seed", type=int, required=True, help="Random seed")
    p.add_argument("--empire-model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--nomads-model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--mock", action="store_true", help="Use mock LLM (num_gpus=1)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("-C", "--ucb-c", type=float, default=1.0,
                   help="UCB exploration constant (default 1.0)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file, tee = setup_logging(args.output_dir, log_prefix="ntbea")

    set_random_seed(args.seed)

    print("=" * 60)
    print("NTBEA Game-Balance Optimization")
    print("=" * 60)
    print(f"  Total games:     {args.total_games}")
    print(f"  Games per eval:  {args.games_per_eval}")
    print(f"  Seed:            {args.seed}")
    print(f"  UCB C:           {args.ucb_c}")
    print(f"  N neighbors:     {N_NEIGHBORS}")
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
    print("=" * 60)

    # ---- search space + model ----
    names, bounds, grids = build_param_space()
    grid_sizes = [len(grids[n]) for n in names]
    print(f"Optimizing {len(names)} params: {names}")
    print(f"Grid sizes: {dict(zip(names, grid_sizes))}")

    patterns = build_patterns(len(names))
    model = NTupleModel(patterns, C=args.ucb_c)

    # Dedicated RNG for proposals (independent of game RNG -> reproducible thetas)
    rng = np.random.default_rng(args.seed)
    mutate = make_mutator(rng, grid_sizes)

    def to_theta(x):
        return {names[i]: grids[names[i]][x[i]] for i in range(len(names))}

    # Budget -> iterations (full batches only, never exceeds total-games)
    n_iterations = args.total_games // args.games_per_eval
    if n_iterations < 1:
        n_iterations = 1
    print(f"Running {n_iterations} iterations of {args.games_per_eval} games "
          f"(<= {args.total_games} total games)\n")

    # Start from a uniformly random grid point (seeded)
    x = tuple(int(rng.integers(0, grid_sizes[i])) for i in range(len(names)))

    history = []
    best_score = float("inf")
    best_theta_disc = None
    total_games_used = 0

    for iteration in range(n_iterations):
        n_games = min(args.games_per_eval, args.total_games - total_games_used)
        if n_games <= 0:
            break

        theta_disc, stats = evaluate_theta(
            to_theta(x), iteration, n_games, args.seed, config, llm,
            num_gpus, args.output_dir,
        )
        balance = stats["balance_score"]
        total_games_used += n_games

        # Bandit update (maximize fitness = -balance_score)
        model.update(x, -balance)

        history.append([theta_disc, balance])
        if balance < best_score:
            best_score = balance
            best_theta_disc = theta_disc

        log_iteration(iteration, n_iterations, theta_disc, n_games, stats,
                      best_score, total_games_used, args.total_games)

        save_checkpoint(
            args.output_dir, iteration, theta_disc, balance,
            complete_theta(best_theta_disc), best_score, history,
            total_games_used, method="ntbea",
        )

        # Propose next point: best-UCB neighbor of current x
        if iteration < n_iterations - 1:
            candidates = [mutate(x) for _ in range(N_NEIGHBORS)]
            x = max(candidates, key=model.ucb)

    # ---- final best ----
    best_complete = complete_theta(best_theta_disc)
    best_path = os.path.join(args.output_dir, "best_theta.json")
    with open(best_path, "w") as f:
        json.dump(best_complete, f, indent=2)

    print("\n" + "=" * 60)
    print("NTBEA complete!")
    print("=" * 60)
    print(f"  Best balance score: {best_score:.4f}")
    print(f"  Best theta: {best_theta_disc}")
    print(f"  Total games used: {total_games_used}/{args.total_games}")
    print(f"  Best theta saved to: {best_path}")
    print(f"  Checkpoint: {os.path.join(args.output_dir, 'checkpoint.json')}")
    print(f"  Log: {log_file}")

    sys.stdout = tee.terminal  # restore before closing to avoid shutdown-flush error
    tee.close()


if __name__ == "__main__":
    main()
