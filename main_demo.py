#!/usr/bin/env python3
"""
Main demo script for CivMini.

Runs a few games with LLM agents and displays results.
"""

import sys
import argparse
import logging
from datetime import datetime
import os

from civmini.config import GameConfig, get_default_theta
from civmini.llm_client import get_llm_client
from civmini.selfplay import run_multiple_games
from civmini.utils import set_random_seed


class TeeLogger:
    """Logger that writes to both file and stdout."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', buffering=1)  # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logging(log_dir='logs', log_prefix='demo'):
    """Set up logging to both file and console."""
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{log_prefix}_{timestamp}.log')
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Redirect stdout to also write to file
    tee = TeeLogger(log_file)
    sys.stdout = tee
    
    return log_file, tee

logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='CivMini Demo')
    parser.add_argument('--n-games', type=int, default=3,
                        help='Number of games to play (default: 3)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name or path (default: InternVL3_5-2B)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock LLM for testing (no real model needed)')
    parser.add_argument('--use-checker', action='store_true', default=True,
                        help='Enable LLM rule checker (enabled by default)')
    parser.add_argument('--max-turns', type=int, default=10,
                        help='Maximum turns per game (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None for random)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files (default: logs)')
    parser.add_argument('--single-action', action='store_true',
                        help='Use single action per turn (default: multi-action mode)')
    
    args = parser.parse_args()
    
    # Set up logging
    log_file, tee = setup_logging(log_dir=args.log_dir, log_prefix='demo')
    print(f"Logging to: {log_file}")
    print("="*60)
    
    # Set random seed (if specified)
    if args.seed is not None:
        set_random_seed(args.seed)
        print(f"Using fixed random seed: {args.seed}")
    else:
        import time
        random_seed = int(time.time() * 1000) % 1000000
        set_random_seed(random_seed)
        print(f"Using random seed: {random_seed} (different each run)")
    
    print("CivMini - Multi-Agent Strategy Game Demo")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of games: {args.n_games}")
    print(f"  - Model: {args.model or 'InternVL3_5-2B (or Mock if loading fails)'}")
    print(f"  - Device: {args.device}")
    print(f"  - Mock mode: {args.mock}")
    print(f"  - Rule checker: {args.use_checker}")
    print(f"  - Max turns: {args.max_turns}")
    print(f"  - Random seed: {random_seed if args.seed is None else args.seed} {'(random)' if args.seed is None else '(fixed)'}")
    print(f"  - Action mode: {'Single' if args.single_action else 'Multi (each unit acts)'}")
    print(f"  - Log file: {log_file}")
    print("="*60)
    
    # Initialize configuration
    config = GameConfig()
    config.max_turns = args.max_turns
    
    # Initialize LLM
    print("\nInitializing LLM...")
    llm = get_llm_client(
        model_name=args.model,
        device=args.device,
        use_mock=args.mock,
    )
    print("LLM ready!")
    
    # Get default game parameters
    theta = get_default_theta()
    
    print("\nGame Parameters (θ):")
    for key, value in list(theta.items())[:8]:  # Show first 8 params
        print(f"  - {key}: {value:.2f}")
    print(f"  ... and {len(theta) - 8} more parameters")
    
    # Run games
    print(f"\nStarting {args.n_games} games...")
    print("-"*60)
    
    try:
        results, stats = run_multiple_games(
            theta=theta,
            llm=llm,
            config=config,
            n_games=args.n_games,
            use_checker=args.use_checker,
            verbose=True,
            multi_action_mode=not args.single_action,  # Use multi-action by default
        )
        
        # Display final statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Games Played: {args.n_games}")
        print(f"\nWin Rates:")
        print(f"  - Empire: {stats['empire_winrate']:.1%} ({stats['empire_wins']} wins)")
        print(f"  - Nomads: {stats['nomads_winrate']:.1%} ({stats['nomads_wins']} wins)")
        print(f"  - Draws:  {stats['draw_rate']:.1%} ({stats['draws']} draws)")
        print(f"\nBalance Score: {stats['balance_score']:.4f} (lower = more balanced)")
        print(f"Average Turns: {stats['avg_turns']:.1f}")
        
        # Analyze results
        print(f"\nAnalysis:")
        if abs(stats['empire_winrate'] - stats['nomads_winrate']) < 0.2:
            print("  ✓ Game appears well-balanced!")
        elif stats['empire_winrate'] > stats['nomads_winrate']:
            print("  ! Empire has an advantage (consider rebalancing)")
        else:
            print("  ! Nomads has an advantage (consider rebalancing)")
        
        print("\n" + "="*60)
        print("Demo complete!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        tee.close()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        tee.close()
        sys.exit(1)
    
    # Clean up
    print(f"\nAll output saved to: {log_file}")
    tee.close()


if __name__ == "__main__":
    main()

