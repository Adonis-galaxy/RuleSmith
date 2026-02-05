#!/usr/bin/env python3
"""
Optimization demo for CivMini.

Runs parameter optimization to find balanced game parameters.
"""

import sys
import argparse
import logging
import json
from datetime import datetime
import os
import signal

from civmini.config import GameConfig, get_default_theta
from civmini.llm_client import get_llm_client
from civmini.optimize import optimize_game_balance
from civmini.selfplay import run_multiple_games
from civmini.utils import set_random_seed

# Global flag for graceful shutdown
INTERRUPTED = False

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global INTERRUPTED
    INTERRUPTED = True
    print("\n" + "="*60)
    print("SIGNAL RECEIVED: Graceful shutdown initiated")
    print("Saving checkpoint before exit...")
    print("="*60)
    # Exit gracefully - checkpoint is already saved by optimizer
    import sys
    sys.exit(0)


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


def setup_logging(log_dir='logs', log_prefix='optimize'):
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
    """Main optimization demo function."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='CivMini Optimization Demo')
    parser.add_argument('--n-iterations', type=int, default=20,
                        help='Number of optimization iterations (default: 20)')
    parser.add_argument('--n-games', type=int, default=5,
                        help='Number of games per evaluation (default: 5)')
    parser.add_argument('--method', type=str, default='evolution',
                        choices=['evolution', 'bayesian', 'random'],
                        help='Optimization method (default: evolution)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock LLM for testing')
    parser.add_argument('--max-turns', type=int, default=10,
                        help='Maximum turns per game (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None for random)')
    parser.add_argument('--output', type=str, default='optimized_theta.json',
                        help='Output file for optimized parameters (default: optimized_theta.json)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files (default: logs)')
    parser.add_argument('--single-action', action='store_true',
                        help='Use single action per turn (default: multi-action mode)')
    parser.add_argument('--use-checker', action='store_true',
                        help='Enable LLM rule checker during optimization (slower)')
    parser.add_argument('--verbose-optimize', action='store_true',
                        help='Enable detailed action logging during optimization')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Save checkpoint every N iterations (default: 5)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs for parallel game execution (default: 1)')
    parser.add_argument('--empire-model', type=str, default=None,
                        help='Model for Empire agent (default: use --model)')
    parser.add_argument('--nomads-model', type=str, default=None,
                        help='Model for Nomads agent (default: use --model)')
    parser.add_argument('--adaptive-games', action='store_true',
                        help='Enable adaptive games: fewer early (exploration), more late (exploitation)')
    parser.add_argument('--min-games', type=int, default=None,
                        help='Minimum games per eval (early iterations). Default: n-games // 4')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Maximum games per eval (late iterations). Default: n-games')
    parser.add_argument('--adaptive-strategy', type=str, default='linear',
                        choices=['linear', 'uncertainty', 'acquisition'],
                        help='Adaptive strategy: linear (default), uncertainty (BO only), acquisition (BO only)')
    parser.add_argument('--balance-threshold', type=float, default=0.1,
                        help='Log iterations with balance_score <= threshold (default: 0.1, i.e. 45-55%% win rate)')
    
    args = parser.parse_args()
    
    # Set up logging
    log_file, tee = setup_logging(log_dir=args.log_dir, log_prefix='optimize')
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
    
    print("CivMini - Parameter Optimization Demo")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Optimization method: {args.method}")
    print(f"  - Iterations: {args.n_iterations}")
    if args.adaptive_games:
        min_g = args.min_games if args.min_games else max(2, args.n_games // 4)
        max_g = args.max_games if args.max_games else args.n_games
        print(f"  - Games per evaluation: {min_g} -> {max_g} (adaptive, {args.adaptive_strategy})")
    else:
        print(f"  - Games per evaluation: {args.n_games}")
    print(f"  - Model: {args.model or 'InternVL3_5-2B (or Mock)'}")
    print(f"  - Device: {args.device}")
    print(f"  - Mock mode: {args.mock}")
    print(f"  - Max turns: {args.max_turns}")
    print(f"  - Random seed: {random_seed if args.seed is None else args.seed} {'(random)' if args.seed is None else '(fixed)'}")
    print(f"  - Action mode: {'Single' if args.single_action else 'Multi (each unit acts)'}")
    print(f"  - Checker: {'Enabled' if args.use_checker else 'Disabled'}")
    print(f"  - Verbose optimize: {'Enabled' if args.verbose_optimize else 'Disabled'}")
    print(f"  - Output file: {args.output}")
    print(f"  - Log file: {log_file}")
    print(f"  - Checkpoint dir: {args.checkpoint_dir}")
    print(f"  - Checkpoint interval: {args.checkpoint_interval} iterations")
    if args.resume_from:
        print(f"  - Resuming from: {args.resume_from}")
    print("="*60)
    
    # Initialize configuration
    config = GameConfig()
    config.max_turns = args.max_turns
    
    # Initialize LLM(s)
    print("\nInitializing LLM...")
    
    # Determine models for each civilization
    empire_model = args.empire_model if args.empire_model else args.model
    nomads_model = args.nomads_model if args.nomads_model else args.model
    
    # Skip loading models in main process if using multi-GPU parallel execution
    # (each subprocess will load its own model copy to avoid GPU waste)
    if args.num_gpus > 1:
        from civmini.llm_client import MockLLM
        llm = MockLLM()  # Placeholder LLM (won't be used)
        empire_llm = None
        nomads_llm = None
        print(f"Multi-GPU mode ({args.num_gpus} GPUs): Models will be loaded in each subprocess")
        print(f"  Empire model: {empire_model}")
        print(f"  Nomads model: {nomads_model}")
    elif empire_model == nomads_model:
        # Same model for both - use single LLM instance
        llm = get_llm_client(
            model_name=empire_model,
            device=args.device,
            use_mock=args.mock,
        )
        empire_llm = llm
        nomads_llm = llm
        print(f"LLM ready: {empire_model} (shared)")
    else:
        # Different models - create separate instances
        print(f"Empire model: {empire_model}")
        empire_llm = get_llm_client(
            model_name=empire_model,
            device=args.device,
            use_mock=args.mock,
        )
        
        print(f"Nomads model: {nomads_model}")
        nomads_llm = get_llm_client(
            model_name=nomads_model,
            device=args.device,
            use_mock=args.mock,
        )
        
        llm = empire_llm  # Default for compatibility
        print(f"LLMs ready: Empire={empire_model}, Nomads={nomads_model}")
    
    # Store model info in config for logging
    config.empire_model = empire_model
    config.nomads_model = nomads_model
    
    # Get baseline performance with default parameters (skip if resuming or multi-GPU)
    baseline_stats = None
    if not args.resume_from and args.num_gpus == 1:
        # Only run baseline for single-GPU mode (multi-GPU skips to save time)
        print("\n" + "="*60)
        print("BASELINE EVALUATION (Default Parameters)")
        print("="*60)
        
        default_theta = get_default_theta()
        
        try:
            baseline_results, baseline_stats = run_multiple_games(
                theta=default_theta,
                llm=llm,
                config=config,
                n_games=args.n_games,
                use_checker=args.use_checker,
                verbose=False,
                multi_action_mode=not args.single_action,
                num_gpus=args.num_gpus,
            )
            
            print(f"Baseline Results:")
            print(f"  - Empire Win Rate: {baseline_stats['empire_winrate']:.2%}")
            print(f"  - Nomads Win Rate: {baseline_stats['nomads_winrate']:.2%}")
            print(f"  - Balance Score: {baseline_stats['balance_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during baseline evaluation: {e}")
            baseline_stats = {"balance_score": float('inf')}
    else:
        print("\n" + "="*60)
        print("SKIPPING BASELINE (Resuming from checkpoint)")
        print("="*60)
        baseline_stats = {"balance_score": float('inf')}
    
    # Run optimization
    print("\n" + "="*60)
    print("STARTING OPTIMIZATION")
    print("="*60)
    if args.adaptive_games:
        min_g = args.min_games if args.min_games else max(2, args.n_games // 4)
        max_g = args.max_games if args.max_games else args.n_games
        avg_games = (min_g + max_g) / 2
        print(f"This will run approximately {int(args.n_iterations * avg_games)} games (adaptive: {min_g}-{max_g}/iter).")
    else:
        print(f"This will run approximately {args.n_iterations * args.n_games} games.")
    print("This may take a while...\n")
    
    try:
        optimized_theta = optimize_game_balance(
            llm=llm,
            config=config,
            n_iterations=args.n_iterations,
            n_games_per_eval=args.n_games,
            method=args.method,
            verbose=args.verbose_optimize,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume_from,
            num_gpus=args.num_gpus,
            adaptive_games=args.adaptive_games,
            min_games=args.min_games,
            max_games=args.max_games,
            adaptive_strategy=args.adaptive_strategy,
            balance_threshold=args.balance_threshold,
        )
        
        # Evaluate optimized parameters (skip for multi-GPU to save time)
        if args.num_gpus == 1:
            print("\n" + "="*60)
            print("FINAL EVALUATION (Optimized Parameters)")
            print("="*60)
            
            final_results, final_stats = run_multiple_games(
                theta=optimized_theta,
                llm=llm,
                config=config,
                n_games=args.n_games * 2,  # More games for final eval
                use_checker=args.use_checker,
                verbose=False,
                multi_action_mode=not args.single_action,
                num_gpus=args.num_gpus,
            )
        else:
            # Multi-GPU mode: skip final eval, use eval_theta.py separately
            print("\n" + "="*60)
            print("SKIPPING FINAL EVALUATION (Multi-GPU mode)")
            print("Use eval_theta.py to evaluate the optimized parameters separately.")
            print("="*60)
            final_stats = None
        
        if final_stats:
            print(f"Optimized Results:")
            print(f"  - Empire Win Rate: {final_stats['empire_winrate']:.2%}")
            print(f"  - Nomads Win Rate: {final_stats['nomads_winrate']:.2%}")
            print(f"  - Balance Score: {final_stats['balance_score']:.4f}")
        
        # Compare (only if we ran baseline and final eval)
        if baseline_stats and final_stats:
            print("\n" + "="*60)
            print("COMPARISON")
            print("="*60)
            improvement = baseline_stats['balance_score'] - final_stats['balance_score']
            print(f"Balance Score:")
            print(f"  - Baseline:  {baseline_stats['balance_score']:.4f}")
            print(f"  - Optimized: {final_stats['balance_score']:.4f}")
            print(f"  - Improvement: {improvement:.4f} ({improvement/baseline_stats['balance_score']*100:.1f}%)")
        
        # Show key parameter changes
        print("\nKey Parameter Changes:")
        important_params = [
            "empire_battle_base_damage",
            "nomads_battle_base_damage",
            "empire_battle_bonus",
            "nomads_battle_bonus",
            "empire_city_cost",
            "nomads_city_cost",
            "empire_city_resource_per_turn",
            "nomads_city_resource_per_turn",
            "empire_max_move_points",
            "nomads_max_move_points",
        ]
        
        for param in important_params:
            if param in default_theta and param in optimized_theta:
                default_val = default_theta[param]
                optimized_val = optimized_theta[param]
                change = optimized_val - default_val
                change_pct = (change / default_val) * 100 if default_val != 0 else 0
                print(f"  - {param}:")
                print(f"      {default_val:.2f} → {optimized_val:.2f} ({change_pct:+.1f}%)")
        
        # Save optimized parameters
        print(f"\nSaving optimized parameters to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(optimized_theta, f, indent=2)
        print("Saved!")
        
        print("\n" + "="*60)
        print("Optimization complete!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        tee.close()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        tee.close()
        sys.exit(1)
    
    # Clean up
    print(f"\nAll output saved to: {log_file}")
    tee.close()


if __name__ == "__main__":
    main()

