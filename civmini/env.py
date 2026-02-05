"""
CivMini game environment implementation.

Defines the game state, actions, and core game logic.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import random
import copy
import json


@dataclass
class Cell:
    """A single cell on the game map."""
    x: int
    y: int
    city_owner: Optional[str] = None  # None or civilization name
    units: List[str] = field(default_factory=list)  # List of unit IDs in this cell


@dataclass
class Unit:
    """A game unit or city.
    
    Types:
    - farmer: Empire unit, can only gather resources
    - soldier: Empire unit, can only battle
    - cavalry: Nomads unit, can battle (gains resources by killing)
    - city: Immobile, can produce resources or units, can be attacked
    """
    id: str
    owner: str  # Civilization name
    unit_type: str  # "farmer", "soldier", "cavalry", "city"
    x: int
    y: int
    hp: float = 10.0
    move_points: float = 0.0  # Remaining movement this turn (0 for cities)


@dataclass
class Player:
    """Player state."""
    civ: str  # "Empire" or "Nomads"
    resources: float = 0.0
    units: Dict[str, Unit] = field(default_factory=dict)  # unit_id -> Unit
    cities: List[Tuple[int, int]] = field(default_factory=list)  # List of (x, y)
    battles_won: int = 0
    score: float = 0.0


@dataclass
class CivMiniState:
    """Complete game state."""
    turn: int = 0
    max_turns: int = 10
    current_player: str = "Empire"  # "Empire" or "Nomads"
    theta: Dict[str, float] = field(default_factory=dict)
    map_grid: List[List[Cell]] = field(default_factory=list)  # 3x3 grid
    players: Dict[str, Player] = field(default_factory=dict)  # civ name -> Player
    game_over: bool = False
    winner: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "current_player": self.current_player,
            "theta": self.theta,
            "map_size": len(self.map_grid),
            "players": {
                civ: {
                    "civ": p.civ,
                    "resources": p.resources,
                    "num_units": len(p.units),
                    "num_cities": len(p.cities),
                    "battles_won": p.battles_won,
                    "score": p.score,
                }
                for civ, p in self.players.items()
            },
            "game_over": self.game_over,
            "winner": self.winner,
        }


@dataclass
class Action:
    """Player action.
    
    Action types:
    - GATHER: Unit gathers resources (farmer/cavalry only)
    - MOVE: Unit moves to new position
    - BATTLE: Unit attacks enemy (soldier/cavalry only, or city)
    - PRODUCE_RESOURCE: City produces resources
    - PRODUCE_UNIT: City produces a unit (farmer/soldier/cavalry)
    - PASS: Do nothing
    """
    action_type: str  # "GATHER", "MOVE", "BATTLE", "PRODUCE_RESOURCE", "PRODUCE_UNIT", "PASS"
    unit_id: Optional[str] = None
    to_x: Optional[int] = None
    to_y: Optional[int] = None
    target_civ: Optional[str] = None
    produce_unit_type: Optional[str] = None  # For PRODUCE_UNIT: "farmer", "soldier", "cavalry"
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Action':
        """Create Action from dictionary."""
        to_dict = d.get("to", {})
        return Action(
            action_type=d.get("action_type", "PASS"),
            unit_id=d.get("unit_id"),
            to_x=to_dict.get("x") if to_dict else d.get("to_x"),
            to_y=to_dict.get("y") if to_dict else d.get("to_y"),
            target_civ=d.get("target_civ"),
            produce_unit_type=d.get("produce_unit_type"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"action_type": self.action_type}
        if self.unit_id is not None:
            result["unit_id"] = self.unit_id
        if self.to_x is not None and self.to_y is not None:
            result["to"] = {"x": self.to_x, "y": self.to_y}
        if self.target_civ is not None:
            result["target_civ"] = self.target_civ
        if self.produce_unit_type is not None:
            result["produce_unit_type"] = self.produce_unit_type
        return result


class CivMiniEnv:
    """CivMini game environment."""
    
    def __init__(self, theta: Dict[str, float], map_size: int = 7, max_turns: int = 10):
        """
        Initialize game environment.
        
        Args:
            theta: Game balance parameters
            map_size: Size of the square map (default 3x3)
            max_turns: Maximum number of turns
        """
        self.theta = theta
        self.map_size = map_size
        self.max_turns = max_turns
        self.state = self.init_state()
    
    def init_state(self) -> CivMiniState:
        """Initialize a new game state."""
        state = CivMiniState(
            turn=0,
            max_turns=self.max_turns,
            current_player="Empire",
            theta=self.theta.copy(),
            map_grid=[],
            players={},
        )
        
        # Initialize map
        state.map_grid = self._create_map()
        
        # Initialize players
        state.players = {
            "Empire": Player(civ="Empire", resources=self.theta["initial_resources"]),
            "Nomads": Player(civ="Nomads", resources=self.theta["initial_resources"]),
        }
        
        # Place initial units
        self._place_initial_units(state)
        
        return state
    
    def _create_map(self) -> List[List[Cell]]:
        """Create the game map (simple grid with no terrain)."""
        grid = []
        
        for y in range(self.map_size):
            row = []
            for x in range(self.map_size):
                cell = Cell(x=x, y=y)
                row.append(cell)
            grid.append(row)
        
        return grid
    
    def _place_initial_units(self, state: CivMiniState):
        """Place initial cities and units for both players.
        
        New rules:
        - Each side starts with 1 city (cannot build more)
        - Empire city at (0,0), Nomads city at (2,2)
        - Cities are immobile units with HP, can be attacked
        - Empire gets farmers and soldiers
        - Nomads gets cavalry
        """
        
        # Place Empire city at (1, 1)
        empire_city = Unit(
            id="empire_city",
            owner="Empire",
            unit_type="city",
            x=1,
            y=1,
            hp=self.theta["empire_soldier_hp"],  # City HP same as soldier
            move_points=0.0,  # Cities cannot move
        )
        state.players["Empire"].units["empire_city"] = empire_city
        state.map_grid[1][1].units.append("empire_city")
        state.map_grid[1][1].city_owner = "Empire"  # Mark cell as having a city
        
        # Place Nomads city at (5, 5)
        nomads_city = Unit(
            id="nomads_city",
            owner="Nomads",
            unit_type="city",
            x=5,
            y=5,
            hp=self.theta["nomads_cavalry_hp"],  # City HP same as cavalry
            move_points=0.0,  # Cities cannot move
        )
        state.players["Nomads"].units["nomads_city"] = nomads_city
        state.map_grid[5][5].units.append("nomads_city")
        state.map_grid[5][5].city_owner = "Nomads"  # Mark cell as having a city
        
        # Place Empire farmers around city at (1,1)
        num_farmers = int(self.theta["empire_initial_farmers"])
        farmer_positions = [(2, 1), (1, 2), (0, 1), (1, 0)]  # Adjacent to city
        for i in range(min(num_farmers, len(farmer_positions))):
            x, y = farmer_positions[i]
            unit_id = f"empire_farmer_{i}"
            unit = Unit(
                id=unit_id,
                owner="Empire",
                unit_type="farmer",
                x=x,
                y=y,
                hp=5.0,  # Farmers are weak
                move_points=self.theta["empire_unit_move_points"],
            )
            state.players["Empire"].units[unit_id] = unit
            state.map_grid[y][x].units.append(unit_id)
        
        # Place Empire soldiers around city at (1,1)
        num_soldiers = int(self.theta["empire_initial_soldiers"])
        soldier_positions = [(0, 0), (2, 2), (0, 2), (2, 0)]  # Diagonal and other positions
        for i in range(min(num_soldiers, len(soldier_positions))):
            x, y = soldier_positions[i]
            unit_id = f"empire_soldier_{i}"
            unit = Unit(
                id=unit_id,
                owner="Empire",
                unit_type="soldier",
                x=x,
                y=y,
                hp=self.theta["empire_soldier_hp"],
                move_points=self.theta["empire_unit_move_points"],
            )
            state.players["Empire"].units[unit_id] = unit
            state.map_grid[y][x].units.append(unit_id)
        
        # Place Nomads cavalry around city at (5,5)
        num_cavalry = int(self.theta["nomads_initial_cavalry"])
        cavalry_positions = [(4, 5), (5, 4), (6, 5), (5, 6), (4, 4), (6, 6)]  # Around city
        for i in range(min(num_cavalry, len(cavalry_positions))):
            x, y = cavalry_positions[i]
            unit_id = f"nomads_cavalry_{i}"
            unit = Unit(
                id=unit_id,
                owner="Nomads",
                unit_type="cavalry",
                x=x,
                y=y,
                hp=self.theta["nomads_cavalry_hp"],
                move_points=self.theta["nomads_cavalry_move_points"],
            )
            state.players["Nomads"].units[unit_id] = unit
            state.map_grid[y][x].units.append(unit_id)
    
    def get_legal_actions(self, state: CivMiniState) -> List[Action]:
        """Get all legal actions for the current player.
        
        Action rules by unit type:
        - farmer: can GATHER, MOVE
        - soldier: can BATTLE, MOVE
        - cavalry: can GATHER, BATTLE, MOVE
        - city: can PRODUCE_RESOURCE, PRODUCE_UNIT
        """
        player = state.players[state.current_player]
        actions = [Action(action_type="PASS")]
        enemy_civ = "Nomads" if state.current_player == "Empire" else "Empire"
        
        for unit_id, unit in player.units.items():
            unit_type = unit.unit_type
            
            # City actions
            if unit_type == "city":
                # PRODUCE_RESOURCE: City generates resources
                actions.append(Action(
                    action_type="PRODUCE_RESOURCE",
                    unit_id=unit_id,
                ))
                
                # PRODUCE_UNIT: City produces units at adjacent empty cells
                if player.civ == "Empire":
                    # Empire can produce farmers or soldiers
                    cost = self.theta["empire_unit_production_cost"]
                    if player.resources >= cost:
                        # Find empty adjacent cells
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = unit.x + dx, unit.y + dy
                            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                                cell = state.map_grid[ny][nx]
                                if len(cell.units) == 0:
                                    actions.append(Action(
                                        action_type="PRODUCE_UNIT",
                                        unit_id=unit_id,
                                        produce_unit_type="farmer",
                                        to_x=nx,
                                        to_y=ny,
                                    ))
                                    actions.append(Action(
                                        action_type="PRODUCE_UNIT",
                                        unit_id=unit_id,
                                        produce_unit_type="soldier",
                                        to_x=nx,
                                        to_y=ny,
                                    ))
                else:  # Nomads
                    # Nomads can only produce cavalry
                    cost = self.theta["nomads_unit_production_cost"]
                    if player.resources >= cost:
                        # Find empty adjacent cells
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = unit.x + dx, unit.y + dy
                            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                                cell = state.map_grid[ny][nx]
                                if len(cell.units) == 0:
                                    actions.append(Action(
                                        action_type="PRODUCE_UNIT",
                                        unit_id=unit_id,
                                        produce_unit_type="cavalry",
                                        to_x=nx,
                                        to_y=ny,
                                    ))
                
                # Cities can also be attacked (BATTLE action checked below)
            
            # GATHER action (for farmers only - cavalry cannot gather)
            if unit_type == "farmer":
                actions.append(Action(action_type="GATHER", unit_id=unit_id))
            
            # MOVE actions (all non-city units)
            # Generate moves up to unit's move_points distance (Manhattan distance)
            if unit_type != "city" and unit.move_points > 0:
                max_distance = int(unit.move_points)
                for dist in range(1, max_distance + 1):
                    # Generate all cells at exactly this Manhattan distance
                    for dx in range(-dist, dist + 1):
                        dy_abs = dist - abs(dx)
                        for dy in ([-dy_abs, dy_abs] if dy_abs > 0 else [0]):
                            new_x, new_y = unit.x + dx, unit.y + dy
                            if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                                # Check if destination cell is empty (no units at all)
                                dest_cell = state.map_grid[new_y][new_x]
                                if len(dest_cell.units) == 0:
                                    actions.append(Action(
                                        action_type="MOVE",
                                        unit_id=unit_id,
                                        to_x=new_x,
                                        to_y=new_y,
                                    ))
            
            # BATTLE action (for soldiers, cavalry, and cities) - attack ADJACENT cells
            if unit_type in ["soldier", "cavalry", "city"]:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    target_x, target_y = unit.x + dx, unit.y + dy
                    if 0 <= target_x < self.map_size and 0 <= target_y < self.map_size:
                        target_cell = state.map_grid[target_y][target_x]
                        # Check if there's an enemy unit in adjacent cell
                        enemy_units_there = [
                            uid for uid in target_cell.units
                            if uid in state.players[enemy_civ].units
                        ]
                        if enemy_units_there:
                            actions.append(Action(
                                action_type="BATTLE",
                                unit_id=unit_id,
                                to_x=target_x,  # Target position
                                to_y=target_y,
                                target_civ=enemy_civ,
                            ))
        
        return actions
    
    def step(self, state: CivMiniState, action: Action) -> Tuple[CivMiniState, bool]:
        """
        Execute one game step with the given action (single action mode).
        
        Args:
            state: Current game state
            action: Action to execute
            
        Returns:
            Tuple of (new_state, done)
        """
        # Deep copy state to avoid mutations
        new_state = copy.deepcopy(state)
        
        # Execute action
        self._execute_action(new_state, action)
        
        # City production phase (cities generate resources)
        self._city_production(new_state)
        
        # Switch player
        new_state.current_player = "Nomads" if new_state.current_player == "Empire" else "Empire"
        
        # If we're back to Empire, increment turn counter
        if new_state.current_player == "Empire":
            new_state.turn += 1
            # Reset movement points for all units
            self._reset_movement_points(new_state)
        
        # Check win conditions
        done = self._check_game_over(new_state)
        
        if done:
            self._compute_final_scores(new_state)
        
        return new_state, done
    
    def step_all_units(self, state: CivMiniState, actions: Dict[str, Action]) -> Tuple[CivMiniState, bool]:
        """
        Execute actions for all units of the current player (multi-action mode).
        
        Args:
            state: Current game state
            actions: Dictionary mapping unit_id to Action
            
        Returns:
            Tuple of (new_state, done)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Deep copy state to avoid mutations
        new_state = copy.deepcopy(state)
        
        # Execute all actions for current player's units
        # Logging happens inside _execute_action and _do_* methods
        player = new_state.players[new_state.current_player]
        for unit_id in list(player.units.keys()):  # Use list() to avoid dict change during iteration
            if unit_id in actions:
                action = actions[unit_id]
                # Execute action - all logging happens inside
                self._execute_action(new_state, action)
        
        # City production phase (cities generate resources)
        self._city_production(new_state)
        
        # Switch player
        new_state.current_player = "Nomads" if new_state.current_player == "Empire" else "Empire"
        
        # If we're back to Empire, increment turn counter
        if new_state.current_player == "Empire":
            new_state.turn += 1
            # Reset movement points for all units
            self._reset_movement_points(new_state)
        
        # Check win conditions
        done = self._check_game_over(new_state)
        
        if done:
            self._compute_final_scores(new_state)
        
        return new_state, done
    
    def _execute_action(self, state: CivMiniState, action: Action) -> bool:
        """Execute the given action and modify state.
        
        All illegal actions are converted to PASS with logging.
        
        Returns:
            True if action was executed successfully, False if converted to PASS
        """
        import logging
        logger = logging.getLogger(__name__)
        
        player = state.players[state.current_player]
        
        if action.action_type == "PASS":
            return True  # PASS is always valid
        
        if action.unit_id is None or action.unit_id not in player.units:
            logger.info(f"ILLEGAL ACTION: Invalid unit_id '{action.unit_id}'")
            logger.info(f"  → PASS")
            return False  # Invalid unit
        
        unit = player.units[action.unit_id]
        
        # Validate and execute actions (illegal → PASS)
        
        if action.action_type == "GATHER":
            # Check if unit can gather
            if unit.unit_type != "farmer":
                logger.info(f"{unit.id} ({unit.unit_type}): ILLEGAL GATHER - only farmers can gather")
                logger.info(f"  → PASS")
                return False  # Convert to PASS
            self._do_gather(state, player, unit)
            return True
        
        elif action.action_type == "MOVE":
            # Validate coordinates are provided
            if action.to_x is None or action.to_y is None:
                logger.info(f"{unit.id}: ILLEGAL MOVE - missing coordinates")
                logger.info(f"  → PASS")
                return False
            
            # Validate coordinates are within bounds
            if not (0 <= action.to_x < self.map_size and 0 <= action.to_y < self.map_size):
                logger.info(f"{unit.id}: ILLEGAL MOVE to ({action.to_x},{action.to_y}) - out of bounds")
                logger.info(f"  → PASS")
                return False
            
            # Validate unit is not a city (cities cannot move)
            if unit.unit_type == "city":
                logger.info(f"{unit.id}: ILLEGAL MOVE - cities cannot move")
                logger.info(f"  → PASS")
                return False
            
            # Validate movement distance (must be within unit's move_points, orthogonal/Manhattan)
            distance = abs(unit.x - action.to_x) + abs(unit.y - action.to_y)
            max_move = int(unit.move_points)
            if distance < 1 or distance > max_move:
                logger.info(f"{unit.id}: ILLEGAL MOVE to ({action.to_x},{action.to_y}) - distance {distance} exceeds max {max_move}")
                logger.info(f"  → PASS")
                return False
            
            # Check if unit has movement points
            if unit.move_points <= 0:
                logger.info(f"{unit.id}: ILLEGAL MOVE - no movement points remaining")
                logger.info(f"  → PASS")
                return False
            
            # Re-check if destination is empty (concurrent action protection)
            dest_cell = state.map_grid[action.to_y][action.to_x]
            if len(dest_cell.units) > 0:
                logger.info(f"{unit.id}: ILLEGAL MOVE to ({action.to_x},{action.to_y}) - occupied by {dest_cell.units}")
                logger.info(f"  → PASS")
                return False  # Convert to PASS
            
            success = self._do_move(state, player, unit, action.to_x, action.to_y)
            return success
        
        elif action.action_type == "PRODUCE_RESOURCE":
            if unit.unit_type != "city":
                logger.info(f"{unit.id}: ILLEGAL PRODUCE_RESOURCE - only cities can produce")
                logger.info(f"  → PASS")
                return False
            self._do_produce_resource(state, player, unit)
            return True
        
        elif action.action_type == "PRODUCE_UNIT":
            if unit.unit_type != "city":
                logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - only cities can produce units")
                logger.info(f"  → PASS")
                return False
            # _do_produce_unit will validate target position (if specified) or find first available
            success = self._do_produce_unit(state, player, unit, action.produce_unit_type, 
                                           action.to_x, action.to_y)
            return success
        
        elif action.action_type == "BATTLE":
            if unit.unit_type not in ["soldier", "cavalry", "city"]:
                logger.info(f"{unit.id} ({unit.unit_type}): ILLEGAL BATTLE - cannot battle")
                logger.info(f"  → PASS")
                return False
            
            # Validate coordinates
            if action.to_x is None or action.to_y is None:
                logger.info(f"{unit.id}: ILLEGAL BATTLE - missing target coordinates")
                logger.info(f"  → PASS")
                return False
            
            # Validate adjacency for battle
            distance = abs(unit.x - action.to_x) + abs(unit.y - action.to_y)
            if distance != 1:
                logger.info(f"{unit.id}: ILLEGAL BATTLE at ({action.to_x},{action.to_y}) - not adjacent (distance {distance})")
                logger.info(f"  → PASS")
                return False
            
            success = self._do_battle(state, player, unit, action.target_civ, action.to_x, action.to_y)
            return success
        
        else:
            logger.info(f"{unit.id}: ILLEGAL ACTION - unknown action type '{action.action_type}'")
            logger.info(f"  → PASS")
            return False
    
    def _do_gather(self, state: CivMiniState, player: Player, unit: Unit):
        """Execute GATHER action.
        
        Only farmers can gather (validation done in _execute_action).
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Farmers gather resources
        amount = self.theta["empire_farmer_gather_amount"]
        player.resources += amount
        logger.info(f"{unit.id}: GATHER -> +{amount:.1f} resources (total: {player.resources:.1f})")
    
    def _do_move(self, state: CivMiniState, player: Player, unit: Unit, to_x: int, to_y: int) -> bool:
        """Execute MOVE action (validation done in _execute_action).
        
        Returns:
            True if move succeeded, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if unit.move_points <= 0:
            logger.info(f"{unit.id}: MOVE FAILED - no movement points")
            return False
        
        # Check distance is within move_points
        distance = abs(unit.x - to_x) + abs(unit.y - to_y)
        max_move = int(unit.move_points)
        if distance < 1 or distance > max_move:
            logger.info(f"{unit.id}: MOVE FAILED - distance {distance} exceeds max {max_move}")
            return False
        
        # Final safety check (should have been validated in _execute_action)
        new_cell = state.map_grid[to_y][to_x]
        if len(new_cell.units) > 0:
            logger.error(f"CRITICAL: {unit.id} moving to occupied cell despite validation!")
            return False
        
        # Store old position for logging
        old_x, old_y = unit.x, unit.y
        
        # Remove from old cell
        old_cell = state.map_grid[unit.y][unit.x]
        if unit.id in old_cell.units:
            old_cell.units.remove(unit.id)
        
        # Double-check destination is still empty (safety)
        if len(new_cell.units) > 0:
            logger.error(f"CRITICAL: Cell ({to_x},{to_y}) became occupied during move!")
            # Restore to old cell
            old_cell.units.append(unit.id)
            return False
        
        # Move unit
        unit.x = to_x
        unit.y = to_y
        unit.move_points -= 1
        
        # Add to new cell
        new_cell.units.append(unit.id)
        
        logger.info(f"{unit.id}: MOVE ({old_x},{old_y}) -> ({to_x},{to_y}) [SUCCESS]")
        return True
    
    def _do_produce_resource(self, state: CivMiniState, player: Player, unit: Unit):
        """Execute PRODUCE_RESOURCE action (city generates resources)."""
        import logging
        logger = logging.getLogger(__name__)
        
        if unit.unit_type != "city":
            return
        
        # Get civilization-specific production rate
        if player.civ == "Empire":
            amount = self.theta["empire_city_resource_production"]
        else:  # Nomads
            amount = self.theta["nomads_city_resource_production"]
        
        player.resources += amount
        logger.info(f"{unit.id}: PRODUCE_RESOURCE -> +{amount:.1f} resources (total: {player.resources:.1f})")
    
    def _do_produce_unit(self, state: CivMiniState, player: Player, unit: Unit, produce_unit_type: str, target_x: int = None, target_y: int = None) -> bool:
        """Execute PRODUCE_UNIT action (city produces a unit at specified location).
        
        Args:
            target_x, target_y: Target position for new unit (must be adjacent to city and empty)
        
        Returns:
            True if unit was produced successfully, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if unit.unit_type != "city":
            return False
        
        if produce_unit_type is None:
            logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - no unit type specified")
            logger.info(f"  → PASS")
            return False
        
        # Validate unit type is allowed for this civilization
        if player.civ == "Empire" and produce_unit_type not in ["farmer", "soldier"]:
            logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - Empire cannot produce {produce_unit_type}")
            logger.info(f"  → PASS")
            return False
        if player.civ == "Nomads" and produce_unit_type != "cavalry":
            logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - Nomads can only produce cavalry")
            logger.info(f"  → PASS")
            return False
        
        # Get cost based on civilization
        if player.civ == "Empire":
            cost = self.theta["empire_unit_production_cost"]
        else:  # Nomads
            cost = self.theta["nomads_unit_production_cost"]
        
        # Get unit stats based on unit type
        if produce_unit_type == "farmer":
            hp = 5.0
            move_points = self.theta["empire_unit_move_points"]
        elif produce_unit_type == "soldier":
            hp = self.theta["empire_soldier_hp"]
            move_points = self.theta["empire_unit_move_points"]
        elif produce_unit_type == "cavalry":
            hp = self.theta["nomads_cavalry_hp"]
            move_points = self.theta["nomads_cavalry_move_points"]
        else:
            logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - invalid unit type '{produce_unit_type}'")
            logger.info(f"  → PASS")
            return False  # Invalid unit type
        
        # Check resources
        if player.resources < cost:
            logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - insufficient resources ({player.resources:.1f} < {cost})")
            logger.info(f"  → PASS")
            return False
        
        # If target position is specified, validate and use it
        if target_x is not None and target_y is not None:
            # Check if target is adjacent to city (distance == 1)
            distance = abs(unit.x - target_x) + abs(unit.y - target_y)
            if distance != 1:
                logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT at ({target_x},{target_y}) - not adjacent to city (distance {distance})")
                logger.info(f"  → PASS")
                return False
            
            # Check if target is within bounds
            if not (0 <= target_x < self.map_size and 0 <= target_y < self.map_size):
                logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT at ({target_x},{target_y}) - out of bounds")
                logger.info(f"  → PASS")
                return False
            
            # Check if target cell is empty
            cell = state.map_grid[target_y][target_x]
            if len(cell.units) > 0:
                logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT at ({target_x},{target_y}) - cell occupied by {cell.units}")
                logger.info(f"  → PASS")
                return False
            
            # Valid target - create unit at specified position
            new_x, new_y = target_x, target_y
        else:
            # No target specified - find first available adjacent cell (legacy behavior)
            candidates = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            new_x, new_y = None, None
            
            for dx, dy in candidates:
                nx, ny = unit.x + dx, unit.y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    cell = state.map_grid[ny][nx]
                    if len(cell.units) == 0:
                        new_x, new_y = nx, ny
                        break
            
            if new_x is None:
                logger.info(f"{unit.id}: ILLEGAL PRODUCE_UNIT - no empty adjacent cells")
                logger.info(f"  → PASS")
                return False
        
        # Create new unit
        cell = state.map_grid[new_y][new_x]
        unit_counter = len([u for u in player.units if produce_unit_type in u])
        new_unit_id = f"{player.civ.lower()}_{produce_unit_type}_{unit_counter}"
        new_unit = Unit(
            id=new_unit_id,
            owner=player.civ,
            unit_type=produce_unit_type,
            x=new_x,
            y=new_y,
            hp=hp,
            move_points=move_points,
        )
        
        # Final safety check - cell still empty
        if len(cell.units) > 0:
            logger.warning(f"{unit.id}: Cell ({new_x},{new_y}) became occupied")
            logger.info(f"  → PASS")
            return False
        
        # Deduct cost and add unit
        player.resources -= cost
        player.units[new_unit_id] = new_unit
        cell.units.append(new_unit_id)
        
        # Verify no overlap occurred
        if len(cell.units) > 1:
            logger.error(f"CRITICAL OVERLAP: Cell ({new_x},{new_y}) now has {cell.units}!")
            cell.units.remove(new_unit_id)
            del player.units[new_unit_id]
            player.resources += cost
            logger.info(f"  → Production aborted, refunded {cost} resources")
            return False
        
        logger.info(f"{unit.id}: PRODUCED {new_unit_id} at ({new_x},{new_y})")
        return True
    
    def _do_battle(self, state: CivMiniState, player: Player, unit: Unit, target_civ: str, to_x: int = None, to_y: int = None) -> bool:
        """Execute BATTLE action - attack unit in adjacent cell.
        
        Args:
            to_x, to_y: Target cell coordinates (must be adjacent)
            
        Returns:
            True if battle was executed, False if invalid
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # If to_x/to_y not specified, this is legacy code - skip
        if to_x is None or to_y is None:
            logger.info(f"{unit.id}: ILLEGAL BATTLE - missing target coordinates")
            logger.info(f"  → PASS")
            return False
        
        # Validate target is within bounds
        if not (0 <= to_x < self.map_size and 0 <= to_y < self.map_size):
            logger.info(f"{unit.id}: ILLEGAL BATTLE at ({to_x},{to_y}) - out of bounds")
            logger.info(f"  → PASS")
            return False
        
        # Check adjacency
        if abs(unit.x - to_x) + abs(unit.y - to_y) != 1:
            logger.info(f"{unit.id}: ILLEGAL BATTLE at ({to_x},{to_y}) - not adjacent")
            logger.info(f"  → PASS")
            return False  # Target must be adjacent
        
        target_cell = state.map_grid[to_y][to_x]
        
        # Validate target_civ
        if target_civ not in state.players:
            logger.info(f"{unit.id}: ILLEGAL BATTLE - invalid target civilization '{target_civ}'")
            logger.info(f"  → PASS")
            return False
        
        # Find enemy units in target cell
        enemy_player = state.players[target_civ]
        enemy_units_there = [
            uid for uid in target_cell.units
            if uid in enemy_player.units
        ]
        
        if not enemy_units_there:
            logger.info(f"{unit.id}: ILLEGAL BATTLE at ({to_x},{to_y}) - no enemy units there")
            logger.info(f"  → PASS")
            return False
        
        # Pick first enemy unit to attack
        target_unit_id = enemy_units_there[0]
        target_unit = enemy_player.units[target_unit_id]
        
        # Calculate damage (based on attacker's civilization and unified battle bonus)
        if player.civ == "Empire":
            base_damage = self.theta["empire_battle_base_damage"]
        else:
            base_damage = self.theta["nomads_battle_base_damage"]
        
        damage = base_damage * self.theta["battle_bonus"]
        
        # Deal damage
        target_unit.hp -= damage
        logger.info(f"{unit.id}: BATTLE -> {target_unit_id} at ({to_x},{to_y}), dealt {damage:.1f} damage, HP now {target_unit.hp:.1f}")
        
        # Remove unit if destroyed (including cities!)
        if target_unit.hp <= 0:
            target_cell.units.remove(target_unit_id)
            del enemy_player.units[target_unit_id]
            player.battles_won += 1
            logger.info(f"  {target_unit_id} DESTROYED!")
            
            # Nomads gain resources from kills
            if player.civ == "Nomads":
                resource_gain = self.theta.get("nomads_kill_resource_gain", 0)
                player.resources += resource_gain
                logger.info(f"  Nomads killed {target_unit_id}, gained {resource_gain} resources")
            
            # If destroyed unit was a city, mark cell as no longer having a city
            if target_unit.unit_type == "city":
                target_cell.city_owner = None
        
        return True
    
    def _city_production(self, state: CivMiniState):
        """Placeholder for city production (now handled via PRODUCE_RESOURCE action)."""
        # Cities no longer automatically produce resources each turn
        # Production is now an explicit action choice
        pass
    
    def _reset_movement_points(self, state: CivMiniState):
        """Reset movement points for all units at turn start based on unit type."""
        for civ, player in state.players.items():
            for unit in player.units.values():
                # Set movement points based on unit type
                if unit.unit_type == "city":
                    unit.move_points = 0.0  # Cities cannot move
                elif unit.unit_type in ["farmer", "soldier"]:
                    unit.move_points = self.theta["empire_unit_move_points"]
                elif unit.unit_type == "cavalry":
                    unit.move_points = self.theta["nomads_cavalry_move_points"]
    
    def _check_game_over(self, state: CivMiniState) -> bool:
        """Check if the game is over.
        
        Win conditions:
        1. Enemy city destroyed -> immediate victory
        2. Max turns reached -> calculate scores
        """
        # Check if any player's city has been destroyed
        for civ, player in state.players.items():
            city_id = f"{civ.lower()}_city"
            if city_id not in player.units:
                # This player's city has been destroyed!
                state.game_over = True
                state.winner = "Nomads" if civ == "Empire" else "Empire"
                return True
        
        # Max turns reached
        if state.turn >= state.max_turns:
            state.game_over = True
            return True
        
        # Legacy check: one player has no units at all (shouldn't happen with cities)
        for civ, player in state.players.items():
            if len(player.units) == 0:
                state.game_over = True
                # Winner is the other player
                state.winner = "Nomads" if civ == "Empire" else "Empire"
                return True
        
        return False
    
    def _compute_final_scores(self, state: CivMiniState):
        """Compute final scores for all players.
        
        Score components:
        - Resources accumulated
        - Battles won
        - Surviving units (including city if still alive)
        
        Note: Cities are no longer a score component since each side has exactly 1.
        """
        for player in state.players.values():
            score = 0.0
            
            # Resources
            score += player.resources * self.theta["score_per_resource"]
            
            # Battles won
            score += player.battles_won * self.theta["score_per_battle_won"]
            
            # Surviving units (including city if still alive)
            score += len(player.units) * self.theta["score_per_surviving_unit"]
            
            player.score = score
        
        # Determine winner by score if not already set
        if state.winner is None:
            empire_score = state.players["Empire"].score
            nomads_score = state.players["Nomads"].score
            
            if empire_score > nomads_score:
                state.winner = "Empire"
            elif nomads_score > empire_score:
                state.winner = "Nomads"
            else:
                state.winner = "Draw"
    
    def get_state_description(self, state: CivMiniState) -> str:
        """Get a text description of the game state for LLM consumption."""
        lines = []
        lines.append(f"=== CivMini Game State ===")
        lines.append(f"Turn: {state.turn}/{state.max_turns}")
        lines.append(f"Current Player: {state.current_player}")
        lines.append("")
        
        # Player info
        for civ, player in state.players.items():
            lines.append(f"--- {civ} ---")
            lines.append(f"Resources: {player.resources:.1f}")
            lines.append(f"Cities: {len(player.cities)}")
            lines.append(f"Units: {len(player.units)}")
            lines.append(f"Battles Won: {player.battles_won}")
            
            # Unit details
            for unit_id, unit in list(player.units.items())[:3]:  # Show first 3 units
                lines.append(f"  - {unit_id}: pos=({unit.x},{unit.y}), hp={unit.hp:.1f}, move={unit.move_points:.1f}")
            
            lines.append("")
        
        # Map summary
        lines.append("--- Map ---")
        for y in range(len(state.map_grid)):
            row_str = ""
            for x in range(len(state.map_grid[y])):
                cell = state.map_grid[y][x]
                symbol = "."
                if cell.city_owner:
                    symbol = "E" if cell.city_owner == "Empire" else "N"
                elif len(cell.units) > 0:
                    # Get owner of first unit
                    first_unit_id = cell.units[0]
                    if first_unit_id.startswith("empire"):
                        symbol = "e"
                    else:
                        symbol = "n"
                row_str += symbol + " "
            lines.append(f"  {row_str}")
        
        return "\n".join(lines)

