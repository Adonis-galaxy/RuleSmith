"""
Utility functions for CivMini.
"""

import json
import random
from typing import Dict, List, Any, Optional
import numpy as np


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_json_for_prompt(obj: Any, indent: int = 2) -> str:
    """Format object as JSON string for prompts."""
    return json.dumps(obj, indent=indent)


def parse_json_safely(text: str) -> Optional[Dict]:
    """
    Safely parse JSON from text, handling common errors.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed dict or None if parsing fails
    """
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text
    import re
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_winrate_variance(results: List[str]) -> float:
    """
    Calculate variance in win rates (for balance metric).
    
    Args:
        results: List of winner names (e.g., ["Empire", "Nomads", "Empire", ...])
        
    Returns:
        Variance in win rates (lower is more balanced)
    """
    if not results:
        return float('inf')
    
    # Count wins for each player
    counts = {}
    for winner in results:
        counts[winner] = counts.get(winner, 0) + 1
    
    # Calculate win rates
    total = len(results)
    win_rates = [count / total for count in counts.values()]
    
    # Return variance
    return np.var(win_rates)


def calculate_balance_score(results: List[str]) -> float:
    """
    Calculate balance score for optimization.
    
    Goal: All players should have similar win rates.
    Lower score = better balance.
    
    Args:
        results: List of winner names
        
    Returns:
        Balance score (lower is better)
    """
    if not results:
        return float('inf')
    
    # Count wins
    counts = {"Empire": 0, "Nomads": 0, "Draw": 0}
    for winner in results:
        if winner in counts:
            counts[winner] += 1
    
    total = len(results)
    empire_rate = counts["Empire"] / total
    nomads_rate = counts["Nomads"] / total
    
    # Ideal is 50/50 split (ignoring draws)
    # Penalize deviation from 0.5
    balance_penalty = abs(empire_rate - 0.5) + abs(nomads_rate - 0.5)
    
    # Also penalize too many draws
    draw_penalty = (counts["Draw"] / total) * 0.5
    
    return balance_penalty + draw_penalty


def print_game_summary(state: Any):
    """Print a summary of the game state."""
    print(f"\n{'='*50}")
    print(f"Turn {state.turn}/{state.max_turns} - Current Player: {state.current_player}")
    print(f"{'='*50}")
    
    for civ, player in state.players.items():
        print(f"\n{civ}:")
        print(f"  Resources: {player.resources:.1f}")
        print(f"  Cities: {len(player.cities)}")
        print(f"  Units: {len(player.units)}")
        print(f"  Battles Won: {player.battles_won}")
        print(f"  Score: {player.score:.1f}")


def print_optimization_progress(iteration: int, theta: Dict[str, float], score: float):
    """Print optimization progress."""
    print(f"\n{'='*60}")
    print(f"Optimization Iteration {iteration}")
    print(f"{'='*60}")
    print(f"Balance Score: {score:.4f}")
    print(f"\nKey Parameters:")
    for key in ["empire_battle_bonus", "nomads_battle_bonus", "empire_city_cost", "nomads_city_cost"]:
        if key in theta:
            print(f"  {key}: {theta[key]:.2f}")

