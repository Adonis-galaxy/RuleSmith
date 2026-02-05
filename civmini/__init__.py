"""
CivMini - A Multi-Agent Strategy Game with LLM Players and Self-Play Optimization

This package implements a complete turn-based civilization strategy game with:
- LLM-powered player agents (InternVL 2B/8B)
- Rule-based RAG system with TF-IDF retrieval
- LLM referee/checker for action validation
- Parameter optimization via Bayesian Optimization or Evolution Strategies
- Multi-action mode where each unit acts per turn
"""

__version__ = "1.0.0"

from .config import GameConfig, get_default_theta, get_optimizable_params, get_fixed_params
from .env import CivMiniState, CivMiniEnv, Action
from .agents import LLMAgent, MultiAgentManager
from .selfplay import run_single_game, run_multiple_games

__all__ = [
    "GameConfig",
    "get_default_theta",
    "get_optimizable_params",
    "get_fixed_params",
    "CivMiniState",
    "CivMiniEnv",
    "Action",
    "LLMAgent",
    "MultiAgentManager",
    "run_single_game",
    "run_multiple_games",
]

