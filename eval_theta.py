#!/usr/bin/env python3
"""
Evaluate game balance with specific theta parameters.

Usage:
    # Basic usage - use default theta
    python eval_theta.py --empire-model 2b --nomads-model 8b
    
    # With custom theta from JSON file
    python eval_theta.py --theta theta.json --empire-model 8b --nomads-model 2b
    
    # From checkpoint
    python eval_theta.py --checkpoint runs/exp/logs/run_xxx/checkpoints/checkpoint_latest.json --empire-model 2b --nomads-model 2b
    
    # With more games
    python eval_theta.py --theta theta.json --empire-model 8b --nomads-model 8b --n-games 32
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from civmini.config import GameConfig, get_default_theta, get_fixed_params
from civmini.llm_client import get_llm_client
from civmini.selfplay import run_multiple_games


def setup_logging(log_dir='logs'):
    """Set up logging to console."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def get_model_path(model_spec: str) -> str:
    """Convert model spec (2b, 8b, or full path) to full model path."""
    model_spec_lower = model_spec.lower()
    
    if model_spec_lower in ['2b', '2b']:
        return "OpenGVLab/InternVL3_5-2B"
    elif model_spec_lower in ['8b', '8b']:
        return "OpenGVLab/InternVL3_5-8B"
    elif model_spec_lower in ['4b', '4b']:
        return "OpenGVLab/InternVL3_5-4B"
    else:
        # Assume it's a full path
        return model_spec


def load_theta(theta_file: str = None, checkpoint_file: str = None, use_best: bool = False) -> dict:
    """
    Load theta parameters from file.
    
    Args:
        theta_file: Path to theta JSON file (direct theta dict)
        checkpoint_file: Path to checkpoint JSON file
        use_best: If using checkpoint, whether to use best_theta instead of current_theta
        
    Returns:
        Complete theta dictionary (merging with default/fixed params if needed)
    """
    theta = None
    
    if theta_file:
        with open(theta_file, 'r') as f:
            data = json.load(f)
        
        # Check if it's a game JSON (has "theta" key) or direct theta
        if "theta" in data:
            theta = data["theta"]
        else:
            theta = data
            
    elif checkpoint_file:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        if use_best and "best_theta" in data:
            theta = data["best_theta"]
        elif "current_theta" in data:
            theta = data["current_theta"]
        else:
            raise ValueError(f"Checkpoint doesn't contain theta: {checkpoint_file}")
    
    # If no theta provided, use default
    if theta is None:
        return get_default_theta()
    
    # Merge with fixed params to get complete theta
    complete_theta = get_default_theta()
    complete_theta.update(theta)
    
    return complete_theta


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate game balance with specific theta parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default theta, Empire 2B vs Nomads 8B
    python eval_theta.py --empire-model 2b --nomads-model 8b
    
    # Load theta from JSON file
    python eval_theta.py --theta optimized_theta.json --empire-model 8b --nomads-model 8b
    
    # Load from checkpoint (use current theta)
    python eval_theta.py --checkpoint checkpoint_latest.json --empire-model 2b --nomads-model 2b
    
    # Load from checkpoint (use best theta)
    python eval_theta.py --checkpoint checkpoint_latest.json --use-best --empire-model 8b --nomads-model 2b
"""
    )
    
    # Model specification
    parser.add_argument('--empire-model', type=str, required=True,
                        help='Empire model: 2b, 4b, 8b, or full model path')
    parser.add_argument('--nomads-model', type=str, required=True,
                        help='Nomads model: 2b, 4b, 8b, or full model path')
    
    # Theta specification (mutually exclusive)
    theta_group = parser.add_mutually_exclusive_group()
    theta_group.add_argument('--theta', type=str, default=None,
                             help='Path to theta JSON file')
    theta_group.add_argument('--checkpoint', type=str, default=None,
                             help='Path to checkpoint JSON file')
    
    parser.add_argument('--use-best', action='store_true',
                        help='When loading from checkpoint, use best_theta instead of current_theta')
    
    # Evaluation settings
    parser.add_argument('--n-games', type=int, default=16,
                        help='Number of games to run (default: 16)')
    parser.add_argument('--max-turns', type=int, default=16,
                        help='Maximum turns per game (default: 16)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs for parallel execution (default: 8)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save game logs (default: no logging)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock LLM for testing')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Get model paths
    empire_model = get_model_path(args.empire_model)
    nomads_model = get_model_path(args.nomads_model)
    
    # Load theta
    theta = load_theta(args.theta, args.checkpoint, args.use_best)
    
    # Print configuration
    print("=" * 70)
    print("CivMini Evaluation")
    print("=" * 70)
    print(f"Empire Model:  {empire_model}")
    print(f"Nomads Model:  {nomads_model}")
    print(f"Number of Games: {args.n_games}")
    print(f"Max Turns:     {args.max_turns}")
    print(f"Num GPUs:      {args.num_gpus}")
    print("-" * 70)
    
    # Print theta source
    if args.theta:
        print(f"Theta from:    {args.theta}")
    elif args.checkpoint:
        print(f"Theta from:    {args.checkpoint} ({'best' if args.use_best else 'current'})")
    else:
        print(f"Theta:         Default parameters")
    
    print("-" * 70)
    print("Key Parameters:")
    key_params = [
        'empire_battle_base_damage', 'nomads_battle_base_damage',
        'empire_soldier_hp', 'nomads_cavalry_hp',
        'empire_unit_production_cost', 'nomads_unit_production_cost',
        'empire_farmer_gather_amount', 'nomads_kill_resource_gain',
        'empire_unit_move_points', 'nomads_cavalry_move_points',
    ]
    for param in key_params:
        if param in theta:
            print(f"  {param}: {theta[param]}")
    print("=" * 70)
    
    # Initialize config
    config = GameConfig()
    config.max_turns = args.max_turns
    config.empire_model = empire_model
    config.nomads_model = nomads_model
    
    # Initialize LLMs
    print("\nInitializing LLMs...")
    
    if empire_model == nomads_model:
        # Same model for both
        llm = get_llm_client(model_name=empire_model, device='cuda', use_mock=args.mock)
        empire_llm = llm
        nomads_llm = llm
        print(f"  Shared LLM: {empire_model}")
    else:
        # Different models
        print(f"  Loading Empire LLM: {empire_model}")
        empire_llm = get_llm_client(model_name=empire_model, device='cuda', use_mock=args.mock)
        
        print(f"  Loading Nomads LLM: {nomads_model}")
        nomads_llm = get_llm_client(model_name=nomads_model, device='cuda', use_mock=args.mock)
        
        llm = empire_llm  # Default
    
    print("LLMs ready!\n")
    
    # Setup output directory
    game_log_dir = None
    if args.output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        game_log_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
        os.makedirs(game_log_dir, exist_ok=True)
        print(f"Game logs will be saved to: {game_log_dir}")
    
    # Run evaluation
    print("=" * 70)
    print("Running Evaluation...")
    print("=" * 70)
    
    results, stats = run_multiple_games(
        theta=theta,
        llm=llm,
        config=config,
        n_games=args.n_games,
        use_checker=False,
        verbose=args.verbose,
        multi_action_mode=True,
        game_log_dir=game_log_dir,
        num_gpus=args.num_gpus,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Count wins
    empire_wins = sum(1 for r in results if r.winner == "Empire")
    nomads_wins = sum(1 for r in results if r.winner == "Nomads")
    draws = sum(1 for r in results if r.winner == "Draw")
    
    print(f"\n  Total Games:     {args.n_games}")
    print(f"  Empire Wins:     {empire_wins} ({empire_wins/args.n_games*100:.1f}%)")
    print(f"  Nomads Wins:     {nomads_wins} ({nomads_wins/args.n_games*100:.1f}%)")
    print(f"  Draws:           {draws} ({draws/args.n_games*100:.1f}%)")
    
    print(f"\n  Empire Win Rate: {stats['empire_winrate']:.2%}")
    print(f"  Nomads Win Rate: {stats['nomads_winrate']:.2%}")
    print(f"  Balance Score:   {stats['balance_score']:.4f}")
    
    # Average scores
    avg_empire_score = sum(r.empire_score for r in results) / len(results)
    avg_nomads_score = sum(r.nomads_score for r in results) / len(results)
    avg_turns = sum(r.turns_played for r in results) / len(results)
    
    print(f"\n  Avg Empire Score: {avg_empire_score:.2f}")
    print(f"  Avg Nomads Score: {avg_nomads_score:.2f}")
    print(f"  Avg Turns Played: {avg_turns:.1f}")
    
    print("\n" + "=" * 70)
    
    # Save results summary
    if args.output_dir:
        summary = {
            "empire_model": empire_model,
            "nomads_model": nomads_model,
            "theta_source": args.theta or args.checkpoint or "default",
            "n_games": args.n_games,
            "max_turns": args.max_turns,
            "results": {
                "empire_wins": empire_wins,
                "nomads_wins": nomads_wins,
                "draws": draws,
                "empire_winrate": stats['empire_winrate'],
                "nomads_winrate": stats['nomads_winrate'],
                "balance_score": stats['balance_score'],
                "avg_empire_score": avg_empire_score,
                "avg_nomads_score": avg_nomads_score,
                "avg_turns": avg_turns,
            },
            "theta": theta,
        }
        
        summary_file = os.path.join(game_log_dir, 'eval_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
    
    return stats


if __name__ == "__main__":
    main()
