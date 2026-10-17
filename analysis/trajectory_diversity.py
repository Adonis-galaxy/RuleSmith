#!/usr/bin/env python3
"""Trajectory-diversity analysis over CivMini evaluation game logs (Appendix I of the paper).

Parses gameN.log files (selfplay logger format) and reports:
  1. distinct opening sequences (first K turns, both players)
  2. action-type distribution + Shannon entropy per faction
  3. pairwise trajectory divergence (Jaccard on board occupancy at turn T)
  4. win-type distribution (elimination vs score/other) + winner counts
  5. game length mean/std

Usage: python3 analysis/trajectory_diversity.py <eval_dir> [--first-n 100] [--opening-turns 3] [--div-turn 5]
"""
import argparse
import glob
import itertools
import math
import os
import re
import sys
from collections import Counter, defaultdict

TURN_RE = re.compile(r"Turn (\d+), Player: (\w+)")
ACTION_RE = re.compile(r"^\s+(\w+): ([A-Z_]+)(.*)$")
POS_RE = re.compile(r"^\s+(\w+): \((\d+),(\d+)\) HP=([\d.]+)")


def parse_game(path):
    turns = []          # list of (turn_idx, player, [(unit, action, args)])
    states = {}         # (turn_idx, player) -> set of (x, y, unit_prefix)
    section = None
    cur_turn, cur_player = None, None
    cur_actions, cur_state = [], set()
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").replace("INFO:civmini.selfplay:", "")
            m = TURN_RE.match(line)
            if m:
                cur_turn, cur_player = int(m.group(1)), m.group(2)
                section = None
                continue
            if "--- Intended Actions ---" in line:
                section, cur_actions = "actions", []
                continue
            if "--- World State After Actions ---" in line:
                section, cur_state = "state", set()
                continue
            if "--- Executing Actions ---" in line:
                if cur_turn is not None:
                    turns.append((cur_turn, cur_player, cur_actions))
                section = None
                continue
            if "--- End World State ---" in line:
                if cur_turn is not None:
                    states[(cur_turn, cur_player)] = cur_state
                section = None
                continue
            if section == "actions":
                m = ACTION_RE.match(line)
                if m:
                    cur_actions.append((m.group(1), m.group(2), m.group(3).strip()))
            elif section == "state":
                m = POS_RE.match(line)
                if m:
                    unit = m.group(1)
                    prefix = re.sub(r"_\d+$", "", unit)
                    cur_state.add((int(m.group(2)), int(m.group(3)), prefix))
    return turns, states


def analyze(paths, opening_turns, div_turn):
    games = []
    for p in paths:
        turns, states = parse_game(p)
        if not turns:
            continue
        games.append({"path": p, "turns": turns, "states": states})
    n = len(games)

    openings = Counter()
    first_moves = Counter()
    act_by_faction = defaultdict(Counter)
    lengths = []
    end_units = []
    div_states = []

    for g in games:
        opening, first = [], []
        max_turn = 0
        for t, player, actions in g["turns"]:
            max_turn = max(max_turn, t)
            for unit, act, args in actions:
                act_by_faction[player][act] += 1
                if t < opening_turns:
                    opening.append(f"{player}|{unit}|{act}|{args}")
                if t == 0:
                    first.append(f"{player}|{unit}|{act}|{args}")
        openings["\n".join(opening)] += 1
        first_moves["\n".join(first)] += 1
        lengths.append(max_turn + 1)
        # divergence snapshot: after Nomads' block at div_turn (fall back to latest before it)
        snap = None
        for key in [(div_turn, "Nomads"), (div_turn, "Empire")]:
            if key in g["states"]:
                snap = g["states"][key]
                break
        if snap is None and g["states"]:
            snap = g["states"][max(g["states"].keys())]
        div_states.append(snap or set())
        # final state for win-type inference
        last = g["states"][max(g["states"].keys())] if g["states"] else set()
        emp = sum(1 for x, y, u in last if u.startswith("empire") and u != "empire_city")
        nom = sum(1 for x, y, u in last if u.startswith("nomads") and u != "nomads_city")
        end_units.append((emp, nom, max_turn + 1))

    def entropy(counter):
        total = sum(counter.values())
        return -sum((c / total) * math.log2(c / total) for c in counter.values() if c) if total else 0.0

    # pairwise Jaccard distance on div-turn occupancy
    dists = []
    for a, b in itertools.combinations(range(n), 2):
        sa, sb = div_states[a], div_states[b]
        if not sa and not sb:
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        dists.append(1 - inter / union if union else 0.0)

    mean_len = sum(lengths) / n
    std_len = (sum((x - mean_len) ** 2 for x in lengths) / n) ** 0.5
    elim = sum(1 for e, m, t in end_units if (e == 0 or m == 0) and t < 16)
    capped = sum(1 for _, _, t in end_units if t >= 16)

    print(f"games analyzed: {n}")
    print(f"[1] distinct openings (first {opening_turns} turns): {len(openings)}/{n}"
          f"  | most common repeated {openings.most_common(1)[0][1]}x")
    print(f"    distinct first-turn action sets: {len(first_moves)}/{n}")
    for fac in sorted(act_by_faction):
        c = act_by_faction[fac]
        total = sum(c.values())
        dist = ", ".join(f"{k} {v / total:.1%}" for k, v in c.most_common())
        print(f"[2] {fac}: action entropy {entropy(c):.2f} bits over {len(c)} types ({dist})")
    if dists:
        print(f"[3] pairwise trajectory divergence at turn {div_turn}: "
              f"mean Jaccard distance {sum(dists) / len(dists):.3f} (n pairs {len(dists)})")
    print(f"[4] win type: army-eliminated {elim}/{n} ({elim / n:.1%}), reached turn cap {capped}/{n}"
          f" ({capped / n:.1%}), other/score {n - elim - capped}/{n}")
    print(f"[5] game length: {mean_len:.2f} +/- {std_len:.2f} turns (min {min(lengths)}, max {max(lengths)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--first-n", type=int, default=None, help="analyze only game1..gameN")
    ap.add_argument("--opening-turns", type=int, default=3)
    ap.add_argument("--div-turn", type=int, default=5)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.eval_dir, "game*.log")),
                   key=lambda p: int(re.search(r"game(\d+)\.log", p).group(1)))
    if a.first_n:
        paths = [p for p in paths if int(re.search(r"game(\d+)\.log", p).group(1)) <= a.first_n]
    if not paths:
        sys.exit(f"no game logs in {a.eval_dir}")
    analyze(paths, a.opening_turns, a.div_turn)
