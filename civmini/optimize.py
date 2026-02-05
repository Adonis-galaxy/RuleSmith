"""
Parameter optimization for game balance.

Uses Bayesian Optimization or Evolution Strategies to optimize θ for balanced gameplay.
"""

from typing import Dict, Callable, List, Tuple, Optional
import logging
import numpy as np
import json
import os
from datetime import datetime

from .config import (
    GameConfig, 
    get_default_theta, 
    get_theta_bounds, 
    theta_to_array, 
    array_to_theta,
    get_optimizable_params,
    get_fixed_params,
    discretize_theta,
)
from .llm_client import InternVLLM
from .selfplay import run_multiple_games

logger = logging.getLogger(__name__)


class GameBalanceOptimizer:
    """
    Optimizer for game balance parameters.
    
    Tries to find parameter values that result in balanced gameplay
    (roughly 50/50 win rate between civilizations).
    
    Supports adaptive sampling: fewer games early (fast exploration),
    more games later (accurate exploitation).
    """
    
    def __init__(
        self,
        llm: InternVLLM,
        config: GameConfig,
        n_games_per_eval: int = 5,
        method: str = "bayesian",  # "bayesian" or "evolution"
        multi_action_mode: bool = True,  # Enable multi-action per turn
        verbose: bool = False,  # Enable verbose logging of actions
        checkpoint_dir: str = "checkpoints",  # Directory to save checkpoints
        checkpoint_interval: int = 5,  # Save checkpoint every N iterations
        num_gpus: int = 1,  # Number of GPUs for parallel execution
        adaptive_games: bool = False,  # Enable adaptive number of games
        min_games: int = None,  # Min games (early iterations)
        max_games: int = None,  # Max games (late iterations)
        n_iterations: int = 100,  # Total iterations (for adaptive schedule)
        adaptive_strategy: str = "linear",  # "linear", "uncertainty", or "acquisition"
        balance_threshold: float = 0.1,  # Log iterations with balance_score <= threshold
    ):
        """
        Initialize optimizer.
        
        Args:
            llm: LLM instance for playing games
            config: Game configuration
            n_games_per_eval: Number of games to evaluate each parameter set
            method: Optimization method ("bayesian" or "evolution")
            multi_action_mode: Enable multi-action per turn
            verbose: Enable verbose logging of actions during optimization
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            num_gpus: Number of GPUs for parallel execution
            adaptive_games: Enable adaptive number of games per iteration
            min_games: Minimum games (early iterations), defaults to n_games_per_eval // 4
            max_games: Maximum games (late iterations), defaults to n_games_per_eval
            n_iterations: Total iterations (used for adaptive schedule)
            adaptive_strategy: Strategy for adaptive games:
                - "linear": Linear interpolation from min to max (default)
                - "uncertainty": Based on GP uncertainty (BO only, more games when confident)
                - "acquisition": Based on Expected Improvement (BO only, more games for promising points)
            balance_threshold: Log iterations with balance_score <= threshold (default: 0.1, i.e., 45-55% win rate)
        """
        self.llm = llm
        self.balance_threshold = balance_threshold
        self.config = config
        self.n_games_per_eval = n_games_per_eval
        self.method = method
        self.multi_action_mode = multi_action_mode
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.num_gpus = num_gpus
        
        # Adaptive games settings
        self.adaptive_games = adaptive_games
        self.adaptive_strategy = adaptive_strategy
        self.n_iterations_total = n_iterations
        self._bo_optimizer = None  # Will be set during BO optimization
        self._acquisition_history = []  # Track acquisition values
        
        if adaptive_games:
            self.min_games = min_games if min_games is not None else max(2, n_games_per_eval // 4)
            self.max_games = max_games if max_games is not None else n_games_per_eval
            logger.info(f"Adaptive sampling enabled: {self.min_games} -> {self.max_games} games")
            logger.info(f"Adaptive strategy: {adaptive_strategy}")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if num_gpus > 1:
            logger.info(f"Parallel execution enabled: {num_gpus} GPUs")
        
        # Get optimizable parameters (excludes fixed params)
        self.optimizable_params = get_optimizable_params()
        self.fixed_params = get_fixed_params()
        
        # Only include bounds for optimizable parameters
        all_bounds = get_theta_bounds()
        self.theta_bounds = {k: v for k, v in all_bounds.items() if k in self.optimizable_params}
        self.param_names = sorted(self.theta_bounds.keys())
        
        logger.info(f"Optimizing {len(self.param_names)} parameters: {self.param_names}")
        logger.info(f"Fixed {len(self.fixed_params)} parameters: {list(self.fixed_params.keys())}")
        
        # History
        self.history: List[Tuple[Dict[str, float], float]] = []
    
    def _get_complete_theta(self, theta: Dict[str, float]) -> Dict[str, float]:
        """
        Merge optimizable theta with fixed parameters to get complete theta.
        
        Args:
            theta: Theta containing optimizable parameters
            
        Returns:
            Complete theta with all parameters
        """
        complete = theta.copy()
        # Add fixed parameters
        for param_name, value in self.fixed_params.items():
            if param_name not in complete:
                complete[param_name] = value
        return complete
    
    def _get_n_games_for_iteration(self, iteration: int, x_next: list = None) -> int:
        """
        Get number of games to run for a given iteration.
        
        Supports multiple strategies:
        - "linear": Linear interpolation from min_games to max_games
        - "uncertainty": Based on GP model uncertainty (BO only)
        - "acquisition": Based on acquisition function value (BO only)
        
        Args:
            iteration: Current iteration number (0-indexed)
            x_next: Next point to evaluate (for BO-based strategies)
            
        Returns:
            Number of games to run
        """
        if not self.adaptive_games:
            return self.n_games_per_eval
        
        if self.adaptive_strategy == "linear" or self._bo_optimizer is None:
            # Linear schedule: min_games at start, max_games at end
            progress = iteration / max(1, self.n_iterations_total - 1)
            n_games = int(self.min_games + (self.max_games - self.min_games) * progress)
        
        elif self.adaptive_strategy == "uncertainty" and x_next is not None:
            # Use GP posterior uncertainty
            # High uncertainty → exploring → fewer games
            # Low uncertainty → exploiting → more games
            n_games = self._get_n_games_from_uncertainty(x_next)
        
        elif self.adaptive_strategy == "acquisition" and x_next is not None:
            # Use acquisition function value  
            # High acquisition → promising → more games
            # Low acquisition → exploring → fewer games
            n_games = self._get_n_games_from_acquisition(x_next)
        
        else:
            # Fallback to linear
            progress = iteration / max(1, self.n_iterations_total - 1)
            n_games = int(self.min_games + (self.max_games - self.min_games) * progress)
        
        # Ensure bounds
        n_games = max(self.min_games, min(self.max_games, n_games))
        
        return n_games
    
    def _get_n_games_from_uncertainty(self, x_next: list) -> int:
        """
        Determine n_games based on GP model uncertainty.
        
        Low uncertainty (confident) → more games for accurate evaluation
        High uncertainty (exploring) → fewer games for fast exploration
        """
        try:
            if self._bo_optimizer is None or len(self._bo_optimizer.models) == 0:
                return self.min_games
            
            model = self._bo_optimizer.models[-1]
            _, y_std = model.predict([x_next], return_std=True)
            
            # Get historical std values for normalization
            if len(self._acquisition_history) > 0:
                max_std = max(h.get('std', 1.0) for h in self._acquisition_history)
                max_std = max(max_std, y_std[0], 0.01)  # Avoid division by zero
            else:
                max_std = max(y_std[0], 0.01)
            
            # Store for history
            self._acquisition_history.append({'std': y_std[0]})
            
            # Low uncertainty → high confidence → more games
            # normalized_std ∈ [0, 1], then invert
            normalized_std = min(y_std[0] / max_std, 1.0)
            confidence = 1.0 - normalized_std
            
            n_games = int(self.min_games + (self.max_games - self.min_games) * confidence)
            
            logger.info(f"  Uncertainty-based: std={y_std[0]:.4f}, confidence={confidence:.2f} -> {n_games} games")
            return n_games
            
        except Exception as e:
            logger.warning(f"Failed to compute uncertainty-based n_games: {e}")
            return self.n_games_per_eval
    
    def _get_n_games_from_acquisition(self, x_next: list) -> int:
        """
        Determine n_games based on acquisition function value.
        
        High acquisition (promising point) → more games for accurate evaluation
        Low acquisition (less promising) → fewer games
        """
        try:
            if self._bo_optimizer is None:
                return self.min_games
            
            # Compute Expected Improvement manually
            from scipy.stats import norm
            
            model = self._bo_optimizer.models[-1]
            y_pred, y_std = model.predict([x_next], return_std=True)
            
            # Best observed value so far
            y_best = min(self._bo_optimizer.yi) if self._bo_optimizer.yi else 0
            
            # Expected Improvement
            if y_std[0] > 0:
                z = (y_best - y_pred[0]) / y_std[0]
                ei = (y_best - y_pred[0]) * norm.cdf(z) + y_std[0] * norm.pdf(z)
            else:
                ei = 0
            
            # Store for history
            self._acquisition_history.append({'ei': ei, 'std': y_std[0]})
            
            # Normalize EI by max historical EI
            if len(self._acquisition_history) > 1:
                max_ei = max(h.get('ei', 0) for h in self._acquisition_history)
                max_ei = max(max_ei, 0.001)
            else:
                max_ei = max(ei, 0.001)
            
            # High EI → promising → more games
            normalized_ei = min(ei / max_ei, 1.0)
            
            n_games = int(self.min_games + (self.max_games - self.min_games) * normalized_ei)
            
            logger.info(f"  Acquisition-based: EI={ei:.4f}, normalized={normalized_ei:.2f} -> {n_games} games")
            return n_games
            
        except Exception as e:
            logger.warning(f"Failed to compute acquisition-based n_games: {e}")
            return self.n_games_per_eval
    
    def objective_function(self, theta: Dict[str, float], x_next: list = None) -> float:
        """
        Objective function to minimize.
        
        Lower score = better balance.
        
        Args:
            theta: Parameter dict (may contain only optimizable params)
            x_next: Next point in BO space (for BO-based adaptive strategies)
            
        Returns:
            Balance score (lower is better)
        """
        iteration = len(self.history)
        
        # Get adaptive number of games for this iteration
        n_games = self._get_n_games_for_iteration(iteration, x_next=x_next)
        
        if self.adaptive_games:
            logger.info(f"Evaluating theta... (Iteration {iteration}, {n_games} games)")
        else:
            logger.info(f"Evaluating theta... (Iteration {iteration})")
        
        # Discretize theta before evaluation
        theta_discrete = discretize_theta(theta)
        
        # Merge with fixed parameters to get complete theta
        complete_theta = self._get_complete_theta(theta_discrete)
        
        # Create iteration-specific log directory
        iter_log_dir = os.path.join(self.checkpoint_dir, f'../game_logs/iter_{iteration}')
        os.makedirs(iter_log_dir, exist_ok=True)
        
        # Run games with this theta and capture detailed logs
        results, stats = run_multiple_games(
            theta=complete_theta,
            llm=self.llm,
            config=self.config,
            n_games=n_games,  # Use adaptive n_games
            use_checker=False,  # Disable checker for speed during optimization
            verbose=self.verbose,  # Use verbose setting from optimizer
            multi_action_mode=self.multi_action_mode,
            game_log_dir=iter_log_dir,  # Save individual game logs
            num_gpus=self.num_gpus,  # Parallel execution
        )
        
        # Save detailed game logs
        self._save_game_logs(results, iteration, iter_log_dir, complete_theta, stats)
        
        # Get balance score
        balance_score = stats["balance_score"]
        
        logger.info(f"  Empire WR: {stats['empire_winrate']:.2%}, "
                   f"Nomads WR: {stats['nomads_winrate']:.2%}, "
                   f"Balance: {balance_score:.4f}")
        logger.info(f"  Game logs saved to: {iter_log_dir}")
        
        # Log good balance iterations (balance_score <= threshold)
        if balance_score <= self.balance_threshold:
            good_balance_log = os.path.join(self.checkpoint_dir, '../good_balance_iters.txt')
            with open(good_balance_log, 'a') as f:
                f.write(f"Iteration {iteration}: balance_score = {balance_score:.4f}\n")
                f.write(f"  Empire WR: {stats['empire_winrate']:.2%}, Nomads WR: {stats['nomads_winrate']:.2%}\n")
                f.write(f"  Theta: {theta_discrete}\n\n")
            if balance_score == 0.0:
                logger.info(f"  🎯 PERFECT BALANCE logged! (threshold: {self.balance_threshold})")
            else:
                logger.info(f"  ✓ Good balance logged! (score={balance_score:.4f} <= threshold={self.balance_threshold})")
        
        # Store discretized theta in history
        self.history.append((theta_discrete.copy(), balance_score))
        
        return balance_score
    
    def _save_game_logs(self, results, iteration, log_dir, theta, stats):
        """Save detailed logs for each game in this iteration."""
        # Save iteration summary
        summary = {
            'iteration': iteration,
            'theta': theta,
            'stats': stats,
            'games': []
        }
        
        for i, result in enumerate(results):
            game_info = {
                'game_number': i + 1,
                'winner': result.winner,
                'turns_played': result.turns_played,
                'empire_score': result.empire_score,
                'nomads_score': result.nomads_score,
                'empire_units': result.empire_units,
                'nomads_units': result.nomads_units,
                'empire_cities': result.empire_cities,
                'nomads_cities': result.nomads_cities,
                'empire_resources': result.empire_resources,
                'nomads_resources': result.nomads_resources,
            }
            summary['games'].append(game_info)
        
        # Save summary
        summary_file = os.path.join(log_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"    Saved iteration {iteration} summary with {len(results)} games")
    
    def optimize(self, n_iterations: int = 20, resume_from: str = None) -> Dict[str, float]:
        """
        Run optimization.
        
        Args:
            n_iterations: Number of optimization iterations
            resume_from: Path to checkpoint file to resume from (optional)
            
        Returns:
            Best theta found
        """
        if self.method == "bayesian":
            return self._optimize_bayesian(n_iterations, resume_from=resume_from)
        elif self.method == "evolution":
            return self._optimize_evolution(n_iterations, resume_from=resume_from)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _optimize_bayesian(self, n_iterations: int, resume_from: str = None) -> Dict[str, float]:
        """
        Bayesian optimization using scikit-optimize with checkpoint support.
        
        Supports adaptive sampling strategies:
        - "linear": Linear schedule (default)
        - "uncertainty": Based on GP posterior uncertainty
        - "acquisition": Based on Expected Improvement
        
        Args:
            n_iterations: Number of iterations
            resume_from: Path to checkpoint file to resume from
            
        Returns:
            Best theta
        """
        try:
            from skopt import Optimizer
            from skopt.space import Real
            
            logger.info("Starting Bayesian Optimization...")
            if self.adaptive_games:
                logger.info(f"Adaptive strategy: {self.adaptive_strategy}")
            
            # Define search space
            space = [
                Real(self.theta_bounds[name][0], self.theta_bounds[name][1], name=name)
                for name in self.param_names
            ]
            
            # Initialize or restore state
            start_iteration = 0
            optimizer = Optimizer(space, random_state=42)
            self._bo_optimizer = optimizer  # Store for adaptive games
            
            if resume_from and os.path.exists(resume_from):
                checkpoint = self.load_checkpoint(resume_from)
                start_iteration = checkpoint['iteration'] + 1
                logger.info(f"Resuming from iteration {start_iteration}")
                
                # Restore optimizer state (tell it about previous evaluations)
                for theta_dict, score in checkpoint['history']:
                    x = [theta_dict[name] for name in self.param_names]
                    optimizer.tell(x, score)
            
            # Run optimization incrementally
            for iteration in range(start_iteration, n_iterations):
                # Ask for next point to evaluate
                x_next = optimizer.ask()
                
                # Convert to theta dict
                theta = {name: float(val) for name, val in zip(self.param_names, x_next)}
                
                # Discretize before evaluation
                theta_discrete = discretize_theta(theta)
                
                # Evaluate with discretized values (pass x_next for BO-based adaptive)
                score = self.objective_function(theta_discrete, x_next=x_next)
                
                # Tell optimizer the result (use discretized values)
                x_discrete = [theta_discrete[name] for name in self.param_names]
                optimizer.tell(x_discrete, score)
                
                # Save checkpoint periodically
                if (iteration + 1) % self.checkpoint_interval == 0 or iteration == n_iterations - 1:
                    # Get current best
                    best_theta_opt, best_score = self.get_best_from_history()
                    self.save_checkpoint(iteration, theta, score, best_theta_opt, best_score, sigma=None)
                    logger.info(f"Checkpoint saved at iteration {iteration + 1}")
            
            # Get best theta
            best_theta, best_score = self.get_best_from_history()
            
            logger.info(f"Optimization complete! Best balance score: {best_score:.4f}")
            
            # Ensure best_theta has all parameters (including fixed ones)
            best_theta = self._get_complete_theta(best_theta)
            return best_theta
            
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to random search")
            return self._optimize_random(n_iterations)
    
    def _optimize_evolution(self, n_iterations: int, resume_from: str = None) -> Dict[str, float]:
        """
        Evolution strategies optimization using simple (1+1)-ES.
        
        Args:
            n_iterations: Number of iterations
            resume_from: Path to checkpoint file to resume from (optional)
            
        Returns:
            Best theta (complete, including fixed params)
        """
        logger.info("Starting Evolution Strategies Optimization...")
        
        # Try to resume from checkpoint
        start_iteration = 0
        checkpoint = None
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
        
        if checkpoint:
            # Resume from checkpoint
            start_iteration = checkpoint['iteration'] + 1
            current_theta = checkpoint['current_theta']
            current_score = checkpoint['current_score']
            best_theta = checkpoint['best_theta']
            best_score = checkpoint['best_score']
            sigma = checkpoint.get('sigma', 0.2)
            logger.info(f"Resuming from iteration {start_iteration}/{n_iterations}")
        else:
            # Start from scratch with discretized default
            current_theta = discretize_theta(get_default_theta())
            current_score = self.objective_function(current_theta)
            best_theta = current_theta.copy()
            best_score = current_score
            sigma = 0.2  # Mutation strength
            
            # Save initial checkpoint
            self.save_checkpoint(0, current_theta, current_score, best_theta, best_score, sigma)
        
        for iteration in range(start_iteration, n_iterations):
            if iteration == 0 and not checkpoint:
                continue  # Already evaluated iteration 0
            
            logger.info(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # Mutate
            mutant_theta = self._mutate_theta(current_theta, sigma)
            mutant_score = self.objective_function(mutant_theta)
            
            # Selection: accept if better
            if mutant_score < current_score:
                current_theta = mutant_theta
                current_score = mutant_score
                logger.info(f"  ✓ Accepted (score improved to {current_score:.4f})")
                
                if current_score < best_score:
                    best_theta = current_theta.copy()
                    best_score = current_score
            else:
                logger.info(f"  ✗ Rejected (score: {mutant_score:.4f} vs {current_score:.4f})")
            
            # Adapt sigma (optional)
            # Increase sigma if we're stuck, decrease if we're making progress
            if iteration % 5 == 0 and iteration > 0:
                recent_improvements = sum(
                    1 for i in range(max(0, len(self.history) - 5), len(self.history))
                    if i > 0 and self.history[i][1] < self.history[i-1][1]
                )
                if recent_improvements < 2:
                    sigma = min(sigma * 1.5, 0.5)  # Increase exploration
                else:
                    sigma = max(sigma * 0.8, 0.05)  # Increase exploitation
            
            # Save checkpoint periodically
            if (iteration + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(iteration, current_theta, current_score, 
                                   best_theta, best_score, sigma)
        
        # Save final checkpoint
        self.save_checkpoint(n_iterations - 1, current_theta, current_score, 
                           best_theta, best_score, sigma)
        
        logger.info(f"\nOptimization complete! Best balance score: {best_score:.4f}")
        
        # Ensure best_theta has all parameters (including fixed ones)
        best_theta = self._get_complete_theta(best_theta)
        return best_theta
    
    def _mutate_theta(self, theta: Dict[str, float], sigma: float) -> Dict[str, float]:
        """
        Mutate theta with Gaussian noise and discretize.
        Only mutates optimizable parameters, keeps fixed parameters unchanged.
        
        Args:
            theta: Current parameters
            sigma: Mutation strength
            
        Returns:
            Mutated and discretized parameters
        """
        mutant = theta.copy()
        
        # Only mutate optimizable parameters
        for param_name in self.param_names:
            # Get bounds
            lower, upper = self.theta_bounds[param_name]
            
            # Add Gaussian noise
            current_value = theta[param_name]
            noise = np.random.normal(0, sigma * (upper - lower))
            new_value = current_value + noise
            
            # Clip to bounds
            new_value = np.clip(new_value, lower, upper)
            
            mutant[param_name] = new_value
        
        # Discretize the mutated values
        mutant = discretize_theta(mutant)
        
        # Keep fixed parameters unchanged
        for param_name, value in self.fixed_params.items():
            mutant[param_name] = value
        
        return mutant
    
    def _optimize_random(self, n_iterations: int) -> Dict[str, float]:
        """
        Random search fallback.
        Only searches over optimizable parameters.
        
        Args:
            n_iterations: Number of iterations
            
        Returns:
            Best theta found
        """
        logger.info("Starting Random Search...")
        
        best_theta = discretize_theta(get_default_theta())
        best_score = self.objective_function(best_theta)
        
        for iteration in range(n_iterations - 1):
            logger.info(f"\nIteration {iteration + 2}/{n_iterations}")
            
            # Start with default values
            random_theta = get_default_theta()
            
            # Random sample only for optimizable parameters
            for param_name in self.param_names:
                lower, upper = self.theta_bounds[param_name]
                random_theta[param_name] = np.random.uniform(lower, upper)
            
            # Keep fixed parameters at default
            for param_name, value in self.fixed_params.items():
                random_theta[param_name] = value
            
            score = self.objective_function(random_theta)
            
            if score < best_score:
                best_theta = random_theta
                best_score = score
                logger.info(f"  ✓ New best! Score: {best_score:.4f}")
            else:
                logger.info(f"  Score: {score:.4f} (best: {best_score:.4f})")
        
        logger.info(f"\nRandom search complete! Best balance score: {best_score:.4f}")
        
        # Ensure best_theta has all parameters (including fixed ones)
        best_theta = self._get_complete_theta(best_theta)
        return best_theta
    
    def get_best_from_history(self) -> Tuple[Dict[str, float], float]:
        """
        Get best theta from optimization history.
        
        Returns:
            Tuple of (best_theta, best_score)
        """
        if not self.history:
            return get_default_theta(), float('inf')
        
        best_idx = min(range(len(self.history)), key=lambda i: self.history[i][1])
        return self.history[best_idx]
    
    def save_checkpoint(self, iteration: int, current_theta: Dict[str, float], 
                       current_score: float, best_theta: Dict[str, float], 
                       best_score: float, sigma: float = None):
        """
        Save optimization checkpoint.
        
        Args:
            iteration: Current iteration number
            current_theta: Current parameters
            current_score: Current score
            best_theta: Best parameters so far
            best_score: Best score so far
            sigma: Evolution strategy sigma (optional)
        """
        checkpoint = {
            'iteration': iteration,
            'current_theta': current_theta,
            'current_score': current_score,
            'best_theta': best_theta,
            'best_score': best_score,
            'history': self.history,
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
        }
        
        if sigma is not None:
            checkpoint['sigma'] = sigma
        
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Also save a backup with iteration number
        backup_path = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.json')
        with open(backup_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved at iteration {iteration}")
    
    def load_checkpoint(self, checkpoint_path: str = None) -> Optional[Dict]:
        """
        Load optimization checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest checkpoint.
            
        Returns:
            Checkpoint dict if found, None otherwise
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.json')
        
        if not os.path.exists(checkpoint_path):
            logger.info(f"No checkpoint found at {checkpoint_path}")
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore history
            self.history = checkpoint['history']
            
            logger.info(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
            logger.info(f"Best score so far: {checkpoint['best_score']:.4f}")
            
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


def optimize_game_balance(
    llm: InternVLLM,
    config: GameConfig,
    n_iterations: int = 20,
    n_games_per_eval: int = 5,
    method: str = "evolution",
    multi_action_mode: bool = True,
    verbose: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5,
    resume_from: str = None,
    num_gpus: int = 1,
    adaptive_games: bool = False,
    min_games: int = None,
    max_games: int = None,
    adaptive_strategy: str = "linear",
    balance_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Convenience function to optimize game balance.
    
    Args:
        llm: LLM instance
        config: Game configuration
        n_iterations: Number of optimization iterations
        n_games_per_eval: Games per evaluation (or max_games if adaptive)
        method: Optimization method
        multi_action_mode: Enable multi-action per turn
        verbose: Enable verbose logging of actions during optimization
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N iterations
        resume_from: Path to checkpoint file to resume from (optional)
        num_gpus: Number of GPUs for parallel execution
        adaptive_games: Enable adaptive number of games (fewer early, more late)
        min_games: Minimum games for early iterations (default: n_games_per_eval // 4)
        max_games: Maximum games for late iterations (default: n_games_per_eval)
        adaptive_strategy: Strategy for adaptive games:
            - "linear": Linear interpolation min->max (default)
            - "uncertainty": Based on GP uncertainty (BO only)
            - "acquisition": Based on Expected Improvement (BO only)
        balance_threshold: Log iterations with balance_score <= threshold (default: 0.1)
        
    Returns:
        Optimized theta parameters
    """
    optimizer = GameBalanceOptimizer(
        llm=llm,
        config=config,
        n_games_per_eval=n_games_per_eval,
        method=method,
        multi_action_mode=multi_action_mode,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        num_gpus=num_gpus,
        adaptive_games=adaptive_games,
        min_games=min_games,
        max_games=max_games,
        n_iterations=n_iterations,
        adaptive_strategy=adaptive_strategy,
        balance_threshold=balance_threshold,
    )
    
    best_theta = optimizer.optimize(n_iterations=n_iterations, resume_from=resume_from)
    
    return best_theta

