"""
Self-play system for CivMini.

Runs games between LLM agents and collects statistics.
"""

from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import os
import json
import sys
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import multiprocessing as mp

from .env import CivMiniEnv, CivMiniState, Action
from .agents import MultiAgentManager
from .checker import LLMRuleChecker
from .llm_client import InternVLLM
from .config import GameConfig

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before creating any CUDA contexts
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game."""
    winner: str  # "Empire", "Nomads", or "Draw"
    turns_played: int
    empire_score: float
    nomads_score: float
    empire_units: int
    nomads_units: int
    empire_cities: int
    nomads_cities: int
    empire_resources: float
    nomads_resources: float


def run_single_game(
    theta: Dict[str, float],
    llm: InternVLLM,
    config: GameConfig,
    use_checker: bool = False,
    verbose: bool = False,
    multi_action_mode: bool = True,  # NEW: Enable multi-action per turn
    game_log_file: str = None,  # NEW: File to save detailed game log
    empire_llm: InternVLLM = None,  # NEW: Optional separate LLM for Empire
    nomads_llm: InternVLLM = None,  # NEW: Optional separate LLM for Nomads
) -> GameResult:
    """
    Run a single game with given parameters.
    
    Args:
        theta: Game balance parameters
        llm: LLM instance for agents
        config: Game configuration
        use_checker: Whether to use LLM rule checker
        verbose: Whether to print detailed progress
        
    Returns:
        GameResult object
    """
    # Initialize environment
    env = CivMiniEnv(theta=theta, map_size=config.map_size, max_turns=config.max_turns)
    state = env.state
    
    # Initialize agents (with optional separate LLMs per civilization)
    agent_manager = MultiAgentManager(
        llm=llm, 
        rag_top_k=config.rag_top_k,
        empire_llm=empire_llm,
        nomads_llm=nomads_llm
    )
    
    # Initialize checker if needed
    checker = LLMRuleChecker(llm=llm, rag_top_k=config.rag_top_k) if use_checker else None
    
    # Setup game log file if specified
    log_handler = None
    env_log_handler = None
    if game_log_file:
        # Create file handler for this specific game
        os.makedirs(os.path.dirname(game_log_file), exist_ok=True)
        log_handler = logging.FileHandler(game_log_file, mode='w')
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        logger.addHandler(log_handler)
        
        # IMPORTANT: Also add handler to env logger for execution logs (PRODUCED, MOVE, etc.)
        env_logger = logging.getLogger('civmini.env')
        env_log_handler = logging.FileHandler(game_log_file, mode='a')  # Append mode
        env_log_handler.setLevel(logging.INFO)
        env_log_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        env_logger.addHandler(env_log_handler)
        
        # Force verbose for game log
        verbose = True
    
    if verbose:
        logger.info(f"{'='*60}")
        logger.info(f"Starting new game")
        logger.info(f"{'='*60}")
    
    # Game loop
    max_steps = config.max_turns * 2 + 10  # Safety limit
    step_count = 0
    
    while not state.game_over and step_count < max_steps:
        step_count += 1
        
        if verbose:
            logger.info(f"Turn {state.turn}, Player: {state.current_player}")
        
            # LOG ALL UNIT POSITIONS at start of turn (for visualization)
            player = state.players[state.current_player]
            logger.info(f"  --- Unit Positions ---")
            for unit_id, unit in player.units.items():
                logger.info(f"    {unit_id}: ({unit.x},{unit.y}) HP={unit.hp:.1f}")
            logger.info(f"  --- End Positions ---")
        
        try:
            if multi_action_mode:
                # NEW: Each unit gets to perform an action
                player = state.players[state.current_player]
                num_units = len(player.units)
                
                if verbose:
                    logger.info(f"  Units: {num_units}")
                
                # Get actions for all units (with optional checker)
                actions = agent_manager.get_actions_for_all_units(
                    state, 
                    env,
                    checker=checker if use_checker else None,
                    max_retries=2,
                )
                
                # Log intended actions (for debugging, marked as INTENDED)
                if verbose:
                    logger.info(f"  --- Intended Actions ---")
                    for unit_id, action in actions.items():
                        action_str = f"    {unit_id}: {action.action_type}"
                        if action.action_type == "PRODUCE_UNIT":
                            action_str += f" {action.produce_unit_type}"
                            if action.to_x is not None:
                                action_str += f" at ({action.to_x},{action.to_y})"
                        elif action.to_x is not None:
                            action_str += f" -> ({action.to_x},{action.to_y})"
                        logger.info(action_str)
                    logger.info(f"  --- Executing Actions ---")
                
                # Execute all actions (logging happens inside env.step_all_units)
                state, done = env.step_all_units(state, actions)
                
                # LOG WORLD STATE AFTER ACTIONS (for visualization)
                if verbose:
                    logger.info(f"  --- World State After Actions ---")
                    for player_name in ["Empire", "Nomads"]:
                        player = state.players[player_name]
                        for unit_id, unit in player.units.items():
                            logger.info(f"    {unit_id}: ({unit.x},{unit.y}) HP={unit.hp:.1f}")
                    logger.info(f"  --- End World State ---")
                
            else:
                # OLD: Single action per player turn (original behavior)
                action = agent_manager.get_action_for_current_player(state, env)
                
                if verbose:
                    logger.info(f"  Action: {action.action_type}" + 
                          (f" (unit: {action.unit_id})" if action.unit_id else ""))
                
                # Validate action with checker if enabled
                if checker and action.action_type != "PASS":
                    is_valid, reason = checker.check_action(state, action, env)
                    
                    if not is_valid:
                        if verbose:
                            logger.info(f"  Action invalid: {reason}")
                            logger.info(f"  Attempting correction...")
                        
                        # Try to get a corrected action
                        corrected_action = checker.suggest_correction(state, action, env)
                        if corrected_action:
                            action = corrected_action
                            if verbose:
                                logger.info(f"  Corrected to: {action.action_type}")
                        else:
                            action = Action(action_type="PASS")
                            if verbose:
                                logger.info(f"  No correction found, using PASS")
                
                # Execute action
                state, done = env.step(state, action)
            
        except Exception as e:
            logger.error(f"Error during game step: {e}")
            # Try to recover with PASS action
            if multi_action_mode:
                player = state.players[state.current_player]
                pass_actions = {uid: Action(action_type="PASS") for uid in player.units.keys()}
                state, done = env.step_all_units(state, pass_actions)
            else:
                state, done = env.step(state, Action(action_type="PASS"))
        
        if done:
            break
    
    # Collect results
    result = GameResult(
        winner=state.winner or "Draw",
        turns_played=state.turn,
        empire_score=state.players["Empire"].score,
        nomads_score=state.players["Nomads"].score,
        empire_units=len(state.players["Empire"].units),
        nomads_units=len(state.players["Nomads"].units),
        empire_cities=len(state.players["Empire"].cities),
        nomads_cities=len(state.players["Nomads"].cities),
        empire_resources=state.players["Empire"].resources,
        nomads_resources=state.players["Nomads"].resources,
    )
    
    if verbose:
        logger.info(f"{'='*60}")
        logger.info(f"Game Over! Winner: {result.winner}")
        logger.info(f"{'='*60}")
        logger.info(f"Empire Score: {result.empire_score:.1f} | Nomads Score: {result.nomads_score:.1f}")
        logger.info(f"Turns Played: {result.turns_played}")
        logger.info(f"Empire: {result.empire_units} units, {result.empire_cities} cities, {result.empire_resources:.1f} resources")
        logger.info(f"Nomads: {result.nomads_units} units, {result.nomads_cities} cities, {result.nomads_resources:.1f} resources")
    
    # Clean up log handlers if we created them
    if log_handler:
        logger.removeHandler(log_handler)
        log_handler.close()
    if env_log_handler:
        env_logger = logging.getLogger('civmini.env')
        env_logger.removeHandler(env_log_handler)
        env_log_handler.close()
    
    return result


def run_single_game_worker(args):
    """Worker function for parallel game execution."""
    theta, game_idx, config, use_checker, verbose, multi_action_mode, game_log_file, gpu_id = args
    
    # Set GPU for this worker
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import here to avoid issues with multiprocessing
    from .llm_client import get_llm_client
    
    # Create LLM instance(s) for this worker
    empire_model = getattr(config, 'empire_model', None)
    nomads_model = getattr(config, 'nomads_model', None)
    
    if empire_model and nomads_model and empire_model != nomads_model:
        # Different models
        empire_llm = get_llm_client(model_name=empire_model, use_mock=False)
        nomads_llm = get_llm_client(model_name=nomads_model, use_mock=False)
        llm = empire_llm  # Default
    else:
        # Same model or not specified
        llm = get_llm_client(use_mock=False)
        empire_llm = None
        nomads_llm = None
    
    # Run the game
    result = run_single_game(
        theta=theta,
        llm=llm,
        config=config,
        use_checker=use_checker,
        verbose=verbose,
        multi_action_mode=multi_action_mode,
        game_log_file=game_log_file,
        empire_llm=empire_llm,
        nomads_llm=nomads_llm,
    )
    
    return game_idx, result


def run_multiple_games(
    theta: Dict[str, float],
    llm: InternVLLM,
    config: GameConfig,
    n_games: int = 5,
    use_checker: bool = False,
    verbose: bool = False,
    multi_action_mode: bool = True,  # NEW: Enable multi-action per turn
    game_log_dir: str = None,  # NEW: Directory to save individual game logs
    num_gpus: int = 1,  # NEW: Number of GPUs for parallel execution
) -> Tuple[List[GameResult], Dict[str, float]]:
    """
    Run multiple games and collect statistics.
    
    Supports parallel execution across multiple GPUs.
    
    Args:
        theta: Game balance parameters
        llm: LLM instance (not used if num_gpus > 1)
        config: Game configuration
        n_games: Number of games to run
        use_checker: Whether to use LLM checker
        verbose: Whether to print progress
        multi_action_mode: Multi-action per turn
        game_log_dir: Directory to save game logs
        num_gpus: Number of GPUs for parallel execution
        
    Returns:
        Tuple of (list of results, statistics dict)
    """
    results = [None] * n_games  # Preallocate to preserve order
    
    # Parallel execution if multiple GPUs
    if num_gpus > 1 and n_games > 1:
        logger.info(f"Running {n_games} games in parallel on {num_gpus} GPUs")
        
        # Prepare worker arguments
        worker_args = []
        for i in range(n_games):
            detailed_log_file = None
            if game_log_dir:
                detailed_log_file = os.path.join(game_log_dir, f'game{i+1}.log')
            
            gpu_id = i % num_gpus  # Distribute games across GPUs
            worker_args.append((theta, i, config, use_checker, verbose, 
                              multi_action_mode, detailed_log_file, gpu_id))
        
        # Run in parallel
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(run_single_game_worker, args): args[1] 
                      for args in worker_args}
            
            for future in as_completed(futures):
                game_idx = futures[future]
                try:
                    idx, result = future.result()
                    results[idx] = result
                    logger.info(f"Completed game {idx+1}/{n_games}")
                except Exception as e:
                    logger.error(f"Game {game_idx+1} failed: {e}")
                    # Create dummy result
                    results[game_idx] = GameResult(
                        winner="Draw", turns_played=0,
                        empire_score=0, nomads_score=0,
                        empire_units=0, nomads_units=0,
                        empire_cities=0, nomads_cities=0,
                        empire_resources=0, nomads_resources=0
                    )
        
        # Save JSON summaries
        if game_log_dir:
            for i, result in enumerate(results):
                json_file = os.path.join(game_log_dir, f'game{i+1}.json')
                game_data = {
                    'game_number': i + 1,
                    'theta': theta,
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
                with open(json_file, 'w') as f:
                    json.dump(game_data, f, indent=2)
    
    else:
        # Sequential execution (single GPU or single game)
        for i in range(n_games):
            if verbose:
                logger.info(f"{'#'*60}")
                logger.info(f"Game {i+1}/{n_games}")
                logger.info(f"{'#'*60}")
            
            # Determine game log file path
            detailed_log_file = None
            if game_log_dir:
                detailed_log_file = os.path.join(game_log_dir, f'game{i+1}.log')
            
            result = run_single_game(
                theta=theta,
                llm=llm,
                config=config,
                use_checker=use_checker,
                verbose=verbose,
                multi_action_mode=multi_action_mode,
                game_log_file=detailed_log_file,
            )
            results[i] = result
            
            # Save JSON summary
            if game_log_dir:
                json_file = os.path.join(game_log_dir, f'game{i+1}.json')
                game_data = {
                    'game_number': i + 1,
                    'theta': theta,
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
                with open(json_file, 'w') as f:
                    json.dump(game_data, f, indent=2)
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    if verbose:
        logger.info(f"{'='*60}")
        logger.info(f"Statistics over {n_games} games:")
        logger.info(f"{'='*60}")
        logger.info(f"Empire Win Rate: {stats['empire_winrate']:.2%}")
        logger.info(f"Nomads Win Rate: {stats['nomads_winrate']:.2%}")
        logger.info(f"Draw Rate: {stats['draw_rate']:.2%}")
        logger.info(f"Average Turns: {stats['avg_turns']:.1f}")
        logger.info(f"Balance Score: {stats['balance_score']:.4f}")
    
    return results, stats


def calculate_statistics(results: List[GameResult]) -> Dict[str, float]:
    """
    Calculate statistics from game results.
    
    Args:
        results: List of game results
        
    Returns:
        Dictionary of statistics
    """
    if not results:
        return {
            "empire_winrate": 0.0,
            "nomads_winrate": 0.0,
            "draw_rate": 0.0,
            "avg_turns": 0.0,
            "balance_score": float('inf'),
        }
    
    n_games = len(results)
    
    # Count outcomes
    empire_wins = sum(1 for r in results if r.winner == "Empire")
    nomads_wins = sum(1 for r in results if r.winner == "Nomads")
    draws = sum(1 for r in results if r.winner == "Draw")
    
    # Win rates
    empire_winrate = empire_wins / n_games
    nomads_winrate = nomads_wins / n_games
    draw_rate = draws / n_games
    
    # Average turns
    avg_turns = sum(r.turns_played for r in results) / n_games
    
    # Balance score: deviation from 50/50 split
    # Lower is better (more balanced)
    balance_score = abs(empire_winrate - 0.5) + abs(nomads_winrate - 0.5)
    
    # Also consider draws (penalize high draw rate slightly)
    balance_score += draw_rate * 0.5
    
    return {
        "empire_winrate": empire_winrate,
        "nomads_winrate": nomads_winrate,
        "draw_rate": draw_rate,
        "avg_turns": avg_turns,
        "balance_score": balance_score,
        "empire_wins": empire_wins,
        "nomads_wins": nomads_wins,
        "draws": draws,
    }

