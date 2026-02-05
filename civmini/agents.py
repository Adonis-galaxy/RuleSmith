"""
LLM-based agents for CivMini.

Implements player agents that use LLM for decision making.
"""

from typing import Dict, List, Optional, Any
import logging
import json

from .llm_client import InternVLLM
from .env import CivMiniState, CivMiniEnv, Action
from .rulebook import retrieve_rules, format_rules_for_prompt
from .utils import parse_json_safely

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    An LLM-powered agent that plays CivMini.
    
    Uses prompts with rules, game state, and role to make decisions.
    """
    
    def __init__(
        self,
        civilization: str,
        llm: InternVLLM,
        rag_top_k: int = 3,
    ):
        """
        Initialize LLM agent.
        
        Args:
            civilization: "Empire" or "Nomads"
            llm: LLM client instance
            rag_top_k: Number of rules to retrieve for context
        """
        self.civilization = civilization
        self.llm = llm
        self.rag_top_k = rag_top_k
    
    def get_system_prompt(self) -> str:
        """Get the system prompt defining the agent's role."""
        return f"""You are playing CivMini as the {self.civilization} civilization.

Your goal is to WIN the game by DEFEATING your opponent!

CRITICAL: Try your best to attack the units and city of the opponent to win the game! 
The enemy city is your PRIMARY TARGET - destroying it means INSTANT VICTORY!

Strategy:
- ATTACK enemy units and cities aggressively
- Gather resources to build an army
- Expand your forces to overwhelm the enemy
- Protect your own city from enemy attacks

You will be given the current game state and legal actions.
Choose actions based on your civilization's strengths:

{self._get_civ_traits()}

Respond with a valid JSON action object. Prioritize military aggression!"""
    
    def _get_civ_traits(self) -> str:
        """Get description of civilization traits."""
        if self.civilization == "Empire":
            return """Empire Traits:
- STRONGER in combat (battle bonus)
- Economy: GATHER resources with farmers
- Your city at (1,1) TOP-LEFT. Enemies come from (5,5) BOTTOM-RIGHT.

ACTIONS:
- SOLDIERS: MOVE right/down (X+1 or Y+1) to intercept enemies
- FARMERS: GATHER resources when safe
- If enemy adjacent: BATTLE immediately!"""
        else:
            return """Nomads Traits:
- FAST cavalry (mobile raiders) - CAN MOVE 2 CELLS PER TURN!
- Economy: KILL enemies for resources
- Your city at (5,5) BOTTOM-RIGHT. Empire city at (1,1) TOP-LEFT.

STRATEGY:
- PRIMARY GOAL: Destroy Empire city at (1,1)!
- CAVALRY: Move towards Empire CITY (not just left!)
- Use your SPEED ADVANTAGE - Empire units only move 1 cell!
- If enemy adjacent: BATTLE immediately!
- DEFEND: If Empire soldiers approach (5,5), intercept them!
- Target priority: City > Farmers > Soldiers"""
    
    def choose_action(
        self,
        state: CivMiniState,
        env: CivMiniEnv,
    ) -> Action:
        """
        Choose an action for the current game state.
        
        Args:
            state: Current game state
            env: Game environment
            
        Returns:
            Chosen action
        """
        # Get legal actions
        legal_actions = env.get_legal_actions(state)
        
        if not legal_actions:
            return Action(action_type="PASS")
        
        # Retrieve relevant rules
        query = f"What actions can {self.civilization} take? How to {self._get_strategy_query(state)}?"
        relevant_rules = retrieve_rules(query, top_k=self.rag_top_k)
        
        # Build prompt
        user_prompt = self._build_decision_prompt(state, legal_actions, relevant_rules)
        system_prompt = self.get_system_prompt()
        
        # Query LLM
        try:
            response = self.llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=200,
                temperature=0.7,
            )
            
            # Parse action from response
            action = self._parse_action_response(response, legal_actions)
            return action
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            # Fallback: random legal action
            import random
            return random.choice(legal_actions)
    
    def _get_strategy_query(self, state: CivMiniState) -> str:
        """Generate a query based on current state for RAG."""
        player = state.players[self.civilization]
        
        if len(player.cities) == 0:
            return "build cities"
        elif player.resources < 5:
            return "gather resources"
        else:
            return "win battles and expand"
    
    def _build_decision_prompt(
        self,
        state: CivMiniState,
        legal_actions: List[Action],
        relevant_rules: List[str],
    ) -> str:
        """Build the prompt for action decision."""
        lines = []
        
        # Add relevant rules
        lines.append(format_rules_for_prompt(relevant_rules))
        lines.append("")
        
        # Add game state
        lines.append("CURRENT GAME STATE:")
        lines.append(f"Turn: {state.turn}/{state.max_turns}")
        lines.append(f"Your Civilization: {self.civilization}")
        lines.append("")
        
        # Player status
        player = state.players[self.civilization]
        lines.append(f"Your Status:")
        lines.append(f"- Resources: {player.resources:.1f}")
        lines.append(f"- Cities: {len(player.cities)}")
        lines.append(f"- Units: {len(player.units)}")
        lines.append(f"- Battles Won: {player.battles_won}")
        lines.append("")
        
        # Units detail (limited to avoid token overflow)
        lines.append("Your Units:")
        for i, (unit_id, unit) in enumerate(list(player.units.items())[:3]):
            lines.append(f"  {unit_id}: position=({unit.x},{unit.y}), hp={unit.hp:.1f}, moves={unit.move_points:.1f}")
        lines.append("")
        
        # Enemy status
        enemy_civ = "Nomads" if self.civilization == "Empire" else "Empire"
        enemy = state.players[enemy_civ]
        lines.append(f"Enemy ({enemy_civ}) Status:")
        lines.append(f"- Resources: {enemy.resources:.1f}")
        lines.append(f"- Cities: {len(enemy.cities)}")
        lines.append(f"- Units: {len(enemy.units)}")
        lines.append("")
        
        # Legal actions (show a few examples)
        lines.append("LEGAL ACTIONS (choose one):")
        for i, action in enumerate(legal_actions[:10]):  # Limit to 10 to save tokens
            lines.append(f"{i+1}. {json.dumps(action.to_dict())}")
        if len(legal_actions) > 10:
            lines.append(f"... and {len(legal_actions) - 10} more actions")
        lines.append("")
        
        lines.append("OUTPUT FORMAT:")
        lines.append("Respond with a JSON object representing your chosen action.")
        lines.append('Example: {"action_type": "GATHER", "unit_id": "empire_unit_0"}')
        lines.append("")
        lines.append("YOUR DECISION:")
        
        return "\n".join(lines)
    
    def _parse_action_response(
        self,
        response: str,
        legal_actions: List[Action],
    ) -> Action:
        """
        Parse LLM response into an Action.
        
        Args:
            response: Raw LLM output
            legal_actions: List of legal actions
            
        Returns:
            Parsed action or fallback action
        """
        # Try to extract JSON
        action_dict = parse_json_safely(response)
        
        if action_dict is None:
            # Try to use LLM's built-in JSON extraction
            action_dict = self.llm.extract_json(response)
        
        if action_dict:
            try:
                action = Action.from_dict(action_dict)
                
                # Validate action is legal
                if self._is_action_legal(action, legal_actions):
                    return action
                    
            except Exception as e:
                logger.warning(f"Failed to parse action: {e}")
        
        # Fallback: choose a reasonable default action
        logger.warning("Using fallback action selection")
        return self._choose_fallback_action(legal_actions)
    
    def _is_action_legal(self, action: Action, legal_actions: List[Action]) -> bool:
        """Check if an action is in the legal actions list."""
        for legal in legal_actions:
            if (action.action_type == legal.action_type and
                action.unit_id == legal.unit_id and
                action.to_x == legal.to_x and
                action.to_y == legal.to_y and
                action.target_civ == legal.target_civ):
                return True
        return False
    
    def _choose_fallback_action(self, legal_actions: List[Action]) -> Action:
        """Choose a reasonable fallback action using heuristics."""
        import random
        
        # Priority: More balanced priorities to encourage diverse actions
        priorities = {
            "BUILD_CITY": 5,  # Prioritize city building
            "BATTLE": 4,      # Then combat
            "GATHER": 3,      # Then gathering
            "MOVE": 2,        # Then movement
            "PASS": 1,        # Last resort
        }
        
        # Group actions by priority
        priority_groups = {}
        for action in legal_actions:
            priority = priorities.get(action.action_type, 0)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(action)
        
        # Get highest priority group
        if priority_groups:
            max_priority = max(priority_groups.keys())
            # Randomly choose from highest priority actions
            return random.choice(priority_groups[max_priority])
        
        return legal_actions[0] if legal_actions else Action(action_type="PASS")


class MultiAgentManager:
    """
    Manages multiple LLM agents playing against each other.
    """
    
    def __init__(
        self,
        llm: InternVLLM = None,
        rag_top_k: int = 3,
        empire_llm: InternVLLM = None,
        nomads_llm: InternVLLM = None,
    ):
        """
        Initialize multi-agent manager.
        
        Args:
            llm: Shared LLM instance (used if empire_llm/nomads_llm not specified)
            rag_top_k: Number of rules to retrieve
            empire_llm: Optional separate LLM for Empire
            nomads_llm: Optional separate LLM for Nomads
        """
        self.llm = llm
        self.rag_top_k = rag_top_k
        
        # Use separate LLMs if provided, otherwise use shared
        empire_model = empire_llm if empire_llm is not None else llm
        nomads_model = nomads_llm if nomads_llm is not None else llm
        
        # Create agents for both civilizations
        self.agents = {
            "Empire": LLMAgent("Empire", empire_model, rag_top_k),
            "Nomads": LLMAgent("Nomads", nomads_model, rag_top_k),
        }
    
    def get_agent(self, civilization: str) -> LLMAgent:
        """Get agent for a civilization."""
        return self.agents[civilization]
    
    def get_action_for_current_player(
        self,
        state: CivMiniState,
        env: CivMiniEnv,
    ) -> Action:
        """
        Get action for the current player (single action mode).
        
        Args:
            state: Current game state
            env: Game environment
            
        Returns:
            Chosen action
        """
        agent = self.get_agent(state.current_player)
        return agent.choose_action(state, env)
    
    def get_actions_for_all_units(
        self,
        state: CivMiniState,
        env: CivMiniEnv,
        checker=None,
        max_retries: int = 2,
    ) -> Dict[str, Action]:
        """
        Get actions for all units of the current player (multi-action mode).
        Uses a SINGLE LLM forward pass to generate all actions.
        With optional checker to validate and regenerate illegal actions.
        
        Args:
            state: Current game state
            env: Game environment
            checker: Optional LLM rule checker
            max_retries: Maximum number of regeneration attempts
            
        Returns:
            Dictionary mapping unit_id to chosen Action
        """
        from .env import Action
        
        agent = self.get_agent(state.current_player)
        player = state.players[state.current_player]
        
        if not player.units:
            return {}
        
        # Get legal actions for all units
        all_legal_actions = env.get_legal_actions(state)
        
        # Build a comprehensive prompt for all units at once
        user_prompt = self._build_multi_unit_prompt(state, player, all_legal_actions, env)
        system_prompt = agent.get_system_prompt()
        
        actions = None
        previous_response = None
        
        # Try up to max_retries times
        for attempt in range(max_retries + 1):
            try:
                # Build prompt (with feedback if retrying)
                if attempt > 0 and previous_response:
                    # Add feedback about invalid actions
                    retry_prompt = self._build_retry_prompt(
                        user_prompt, 
                        previous_response, 
                        invalid_actions_feedback
                    )
                    current_prompt = retry_prompt
                else:
                    current_prompt = user_prompt
                
                # LLM call
                response = agent.llm.chat(
                    system_prompt=system_prompt,
                    user_prompt=current_prompt,
                    max_new_tokens=300,
                    temperature=0.7,
                )
                
                previous_response = response
                
                # Parse actions for all units from response
                actions = self._parse_multi_unit_response(
                    response, 
                    player.units.keys(), 
                    all_legal_actions, 
                    agent
                )
                
                # Validate with checker if provided
                if checker:
                    invalid_actions_feedback = []
                    all_valid = True
                    
                    for unit_id, action in actions.items():
                        if action.action_type != "PASS":
                            is_valid, reason = checker.check_action(state, action, env)
                            
                            if not is_valid:
                                all_valid = False
                                invalid_actions_feedback.append({
                                    'unit_id': unit_id,
                                    'action': action.to_dict(),
                                    'reason': reason
                                })
                                logger.warning(f"Invalid action for {unit_id}: {reason}")
                    
                    # If all valid, return
                    if all_valid:
                        logger.info(f"All actions validated successfully (attempt {attempt + 1})")
                        return actions
                    
                    # If not all valid and we have retries left, continue loop
                    if attempt < max_retries:
                        logger.info(f"Retrying action generation (attempt {attempt + 2}/{max_retries + 1})")
                        continue
                    else:
                        # Max retries reached, use corrected actions
                        logger.warning(f"Max retries reached, using corrected actions")
                        actions = self._correct_invalid_actions(
                            actions, 
                            invalid_actions_feedback, 
                            all_legal_actions,
                            agent
                        )
                        return actions
                else:
                    # No checker, return actions as is
                    return actions
                    
            except Exception as e:
                logger.warning(f"Error in multi-unit action generation (attempt {attempt + 1}): {e}")
                
                if attempt == max_retries:
                    # Last attempt failed, use fallback
                    logger.warning("All attempts failed, using fallback actions")
                    actions = {}
                    for unit_id in player.units.keys():
                        unit_actions = [a for a in all_legal_actions if a.unit_id == unit_id]
                        if unit_actions:
                            actions[unit_id] = agent._choose_fallback_action(unit_actions)
                        else:
                            actions[unit_id] = Action(action_type="PASS")
                    return actions
        
        return actions if actions else {}
    
    def _build_multi_unit_prompt(
        self,
        state: CivMiniState,
        player: Any,
        legal_actions: List[Action],
        env: CivMiniEnv,
    ) -> str:
        """Build prompt for generating actions for all units in one LLM call."""
        from .rulebook import retrieve_rules, format_rules_for_prompt
        
        lines = []
        
        # Retrieve relevant rules
        rules = retrieve_rules("multi-unit actions turn structure", top_k=3)
        lines.append(format_rules_for_prompt(rules))
        lines.append("")
        
        # Game state
        lines.append("CURRENT GAME STATE:")
        lines.append(f"Turn: {state.turn}/{state.max_turns}")
        lines.append(f"Your Civilization: {player.civ}")
        lines.append(f"Resources: {player.resources:.1f}")
        lines.append(f"Cities: {len(player.cities)}")
        lines.append(f"Battles Won: {player.battles_won}")
        lines.append("")
        
        # Get enemy information
        enemy_civ = "Nomads" if player.civ == "Empire" else "Empire"
        enemy = state.players[enemy_civ]
        
        # Add enemy unit positions for strategic planning
        lines.append(f"ENEMY ({enemy_civ}) UNITS:")
        for unit_id, unit in list(enemy.units.items())[:8]:
            lines.append(f"  {unit_id}: ({unit.x},{unit.y}) Type={unit.unit_type} HP={unit.hp:.1f}")
        lines.append("")
        
        # Add civilization-specific STRATEGY GUIDE
        lines.append("="*50)
        lines.append("STRATEGY GUIDE (FOLLOW THIS!):")
        lines.append("="*50)
        if player.civ == "Nomads":
            lines.append(self._get_nomads_strategy(state, player, enemy))
        else:  # Empire
            lines.append(self._get_empire_strategy(state, player, enemy))
        lines.append("="*50)
        lines.append("")
        
        # Unit information with legal actions for each
        lines.append("YOUR UNITS (each needs an action):")
        for unit_id, unit in list(player.units.items())[:5]:  # Limit to 5 units to avoid token overflow
            lines.append(f"\n{unit_id}:")
            lines.append(f"  Position: ({unit.x}, {unit.y})")
            lines.append(f"  HP: {unit.hp:.1f}, Move Points: {unit.move_points:.1f}")
            
            # Show legal actions for this unit
            unit_legal = [a for a in legal_actions if a.unit_id == unit_id]
            lines.append(f"  Legal actions: {', '.join(set(a.action_type for a in unit_legal[:5]))}")
        
            # For cities, show available production positions
            if unit.unit_type == "city":
                # Find empty adjacent cells
                available_positions = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = unit.x + dx, unit.y + dy
                    if 0 <= nx < len(state.map_grid) and 0 <= ny < len(state.map_grid):
                        cell = state.map_grid[ny][nx]
                        if len(cell.units) == 0:
                            available_positions.append(f"({nx},{ny})")
                if available_positions:
                    lines.append(f"  Available production positions: {', '.join(available_positions)}")
        
        lines.append("")
        lines.append("OUTPUT FORMAT:")
        lines.append("Return a JSON object with actions for ALL units:")
        lines.append('{')
        lines.append('  "unit_id_1": {"action_type": "GATHER"},')
        lines.append('  "unit_id_2": {"action_type": "MOVE", "to": {"x": 1, "y": 1}},')
        lines.append('  "unit_id_3": {"action_type": "BATTLE", "to": {"x": 2, "y": 2}, "target_civ": "Nomads"},')
        lines.append('  "empire_city": {"action_type": "PRODUCE_UNIT", "produce_unit_type": "soldier", "to": {"x": 1, "y": 2}}')
        lines.append('}')
        lines.append("")
        lines.append("NOTE: For PRODUCE_UNIT, specify the target position with 'to': {'x': X, 'y': Y}")
        lines.append("      The position must be adjacent to the city and empty!")
        lines.append("")
        lines.append("YOUR ACTIONS:")
        
        return "\n".join(lines)
    
    def _get_nomads_strategy(self, state: CivMiniState, player: Any, enemy: Any) -> str:
        """Get aggressive strategy guide for Nomads with exact move targets."""
        # Find enemy units
        enemy_city_pos = (1, 1)  # Empire city is always at (1,1)
        our_city_pos = (5, 5)    # Nomads city is always at (5,5)
        enemy_farmers = []
        enemy_soldiers = []
        for uid, u in enemy.units.items():
            if u.unit_type == "city":
                enemy_city_pos = (u.x, u.y)
            elif u.unit_type == "farmer":
                enemy_farmers.append((uid, u))
            elif u.unit_type == "soldier":
                enemy_soldiers.append((uid, u))
        
        # Build list of all enemy positions
        enemy_positions = set()
        for uid, u in enemy.units.items():
            enemy_positions.add((u.x, u.y))
        
        # Check if any enemy units are threatening our city (within 3 cells)
        enemies_near_our_city = []
        for uid, u in enemy.units.items():
            if u.unit_type in ["soldier"]:
                dist_to_our_city = abs(u.x - our_city_pos[0]) + abs(u.y - our_city_pos[1])
                if dist_to_our_city <= 3:
                    enemies_near_our_city.append((uid, u, dist_to_our_city))
        
        # Sort cavalry by distance to enemy city - closest ones attack, furthest defend if needed
        cavalry_list = [(uid, u) for uid, u in player.units.items() if u.unit_type == "cavalry"]
        cavalry_list.sort(key=lambda x: abs(x[1].x - enemy_city_pos[0]) + abs(x[1].y - enemy_city_pos[1]))
        
        # Decide how many cavalry to defend (if enemies near our city)
        num_defenders = min(len(enemies_near_our_city), len(cavalry_list) // 2)  # At most half defend
        defender_ids = set()
        if num_defenders > 0:
            # Assign the cavalry furthest from enemy city as defenders
            for i in range(num_defenders):
                if i < len(cavalry_list):
                    defender_ids.add(cavalry_list[-(i+1)][0])
        
        strategy_lines = []
        strategy_lines.append("🔥🔥🔥 NOMADS CAVALRY - ATTACK EMPIRE CITY! 🔥🔥🔥")
        strategy_lines.append("")
        strategy_lines.append(f"PRIMARY TARGET: Empire CITY at ({enemy_city_pos[0]},{enemy_city_pos[1]})")
        strategy_lines.append("⚡ CAVALRY CAN MOVE 2 CELLS PER TURN! Use this speed advantage! ⚡")
        strategy_lines.append("")
        
        if enemies_near_our_city:
            strategy_lines.append(f"⚠️ WARNING: {len(enemies_near_our_city)} enemy units approaching our city!")
            strategy_lines.append(f"   Defenders assigned: {defender_ids}")
        
        strategy_lines.append("")
        strategy_lines.append("COPY THESE EXACT ACTIONS:")
        strategy_lines.append("")
        
        for uid, unit in player.units.items():
            if unit.unit_type == "cavalry":
                # Check for adjacent enemies (for BATTLE)
                adjacent_enemy = None
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    check_pos = (unit.x + dx, unit.y + dy)
                    if check_pos in enemy_positions:
                        for euid, eu in enemy.units.items():
                            if (eu.x, eu.y) == check_pos:
                                adjacent_enemy = (euid, eu, check_pos)
                                break
                        break
                
                if adjacent_enemy:
                    euid, eu, pos = adjacent_enemy
                    strategy_lines.append(f'"{uid}": {{"action_type": "BATTLE", "to": {{"x": {pos[0]}, "y": {pos[1]}}}, "target_civ": "Empire"}}')
                    strategy_lines.append(f"  ^ ATTACK {euid} at ({pos[0]},{pos[1]})!")
                elif uid in defender_ids and enemies_near_our_city:
                    # This cavalry should defend - move towards nearest threatening enemy
                    nearest_enemy = min(enemies_near_our_city, key=lambda e: abs(e[1].x - unit.x) + abs(e[1].y - unit.y))
                    target_enemy = nearest_enemy[1]
                    
                    # Move towards the threatening enemy
                    move_points = int(unit.move_points)
                    dx = 1 if target_enemy.x > unit.x else (-1 if target_enemy.x < unit.x else 0)
                    dy = 1 if target_enemy.y > unit.y else (-1 if target_enemy.y < unit.y else 0)
                    
                    # Move up to 2 cells towards enemy
                    steps_x = min(abs(target_enemy.x - unit.x), move_points) * (1 if dx > 0 else -1 if dx < 0 else 0)
                    remaining = move_points - abs(steps_x)
                    steps_y = min(abs(target_enemy.y - unit.y), remaining) * (1 if dy > 0 else -1 if dy < 0 else 0)
                    
                    target_x = max(0, min(6, unit.x + steps_x))
                    target_y = max(0, min(6, unit.y + steps_y))
                    
                    if target_x == unit.x and target_y == unit.y:
                        target_x = max(0, min(6, unit.x + dx))
                    
                    strategy_lines.append(f'"{uid}": {{"action_type": "MOVE", "to": {{"x": {target_x}, "y": {target_y}}}}}')
                    strategy_lines.append(f"  ^ DEFEND: Intercept {nearest_enemy[0]} approaching our city!")
                else:
                    # Attack mode - move towards enemy city
                    move_points = int(unit.move_points)  # Should be 2 for cavalry
                    
                    # Calculate direction to enemy city
                    dx_to_city = enemy_city_pos[0] - unit.x  # Positive = go right, negative = go left
                    dy_to_city = enemy_city_pos[1] - unit.y  # Positive = go down, negative = go up
                    
                    # Move up to move_points cells towards enemy city
                    steps_x = min(abs(dx_to_city), move_points) * (1 if dx_to_city > 0 else -1 if dx_to_city < 0 else 0)
                    remaining = move_points - abs(steps_x)
                    steps_y = min(abs(dy_to_city), remaining) * (1 if dy_to_city > 0 else -1 if dy_to_city < 0 else 0)
                    
                    target_x = max(0, min(6, unit.x + steps_x))
                    target_y = max(0, min(6, unit.y + steps_y))
                    
                    # Make sure we're actually moving
                    if target_x == unit.x and target_y == unit.y:
                        # Already at city location!
                        pass
                    
                    strategy_lines.append(f'"{uid}": {{"action_type": "MOVE", "to": {{"x": {target_x}, "y": {target_y}}}}}')
                    dist = abs(target_x - enemy_city_pos[0]) + abs(target_y - enemy_city_pos[1])
                    strategy_lines.append(f"  ^ ATTACK: Move towards Empire city! ({dist} cells away after move)")
            elif unit.unit_type == "city":
                # Check if we should produce more cavalry or save resources
                if player.resources >= 4 and len(cavalry_list) < 4:
                    strategy_lines.append(f'"{uid}": {{"action_type": "PRODUCE_UNIT", "produce_unit_type": "cavalry", "to": {{"x": 5, "y": 4}}}}')
                    strategy_lines.append(f"  ^ Produce cavalry for attack!")
                else:
                    strategy_lines.append(f'"{uid}": {{"action_type": "PRODUCE_RESOURCE"}}')
                    strategy_lines.append(f"  ^ Build economy")
        
        strategy_lines.append("")
        strategy_lines.append("REMEMBER: COPY THE EXACT JSON ABOVE!")
        strategy_lines.append(f"⚡ PRIMARY GOAL: Destroy Empire city at ({enemy_city_pos[0]},{enemy_city_pos[1]})!")
        strategy_lines.append("⚡ Cavalry moves 2 cells per turn - use this speed advantage!")
        
        return "\n".join(strategy_lines)
    
    def _get_empire_strategy(self, state: CivMiniState, player: Any, enemy: Any) -> str:
        """Get defensive/offensive strategy guide for Empire with exact move targets."""
        # Find positions
        own_city_pos = (1, 1)   # Empire city at (1,1)
        enemy_city_pos = (5, 5)  # Nomads city at (5,5)
        
        own_farmers = []
        own_soldiers = []
        for uid, u in player.units.items():
            if u.unit_type == "city":
                own_city_pos = (u.x, u.y)
            elif u.unit_type == "farmer":
                own_farmers.append((uid, u))
            elif u.unit_type == "soldier":
                own_soldiers.append((uid, u))
        
        # Find enemy units
        enemy_positions = set()
        enemy_cavalry = []
        for uid, u in enemy.units.items():
            enemy_positions.add((u.x, u.y))
            if u.unit_type == "cavalry":
                enemy_cavalry.append((uid, u))
            if u.unit_type == "city":
                enemy_city_pos = (u.x, u.y)
        
        # Check if enemy cavalry is threatening our city
        enemies_near_city = []
        for uid, u in enemy_cavalry:
            dist = abs(u.x - own_city_pos[0]) + abs(u.y - own_city_pos[1])
            if dist <= 4:
                enemies_near_city.append((uid, u, dist))
        
        # Sort soldiers by distance to our city - closest ones defend, furthest attack
        soldiers_sorted = sorted(own_soldiers, key=lambda x: abs(x[1].x - own_city_pos[0]) + abs(x[1].y - own_city_pos[1]))
        
        # Decide roles: if enemies near, assign some soldiers to defend
        num_defenders = min(len(enemies_near_city), max(1, len(soldiers_sorted) // 2))
        defender_ids = set(s[0] for s in soldiers_sorted[:num_defenders]) if enemies_near_city else set()
        
        strategy_lines = []
        strategy_lines.append("🛡️🛡️🛡️ EMPIRE STRATEGY 🛡️🛡️🛡️")
        strategy_lines.append("")
        strategy_lines.append(f"YOUR CITY: ({own_city_pos[0]},{own_city_pos[1]}) - PROTECT IT!")
        strategy_lines.append(f"ENEMY CITY: ({enemy_city_pos[0]},{enemy_city_pos[1]}) - ATTACK TARGET!")
        strategy_lines.append("")
        
        if enemies_near_city:
            strategy_lines.append(f"⚠️ WARNING: {len(enemies_near_city)} enemy cavalry approaching!")
            strategy_lines.append(f"   Defenders assigned: {defender_ids}")
        else:
            strategy_lines.append("✅ No immediate threats - push towards enemy city!")
        
        strategy_lines.append("")
        strategy_lines.append("COPY THESE EXACT ACTIONS:")
        strategy_lines.append("")
        
        for uid, unit in player.units.items():
            if unit.unit_type == "soldier":
                # Check for adjacent enemies (for BATTLE)
                adjacent_enemy = None
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    check_pos = (unit.x + dx, unit.y + dy)
                    if check_pos in enemy_positions:
                        for euid, eu in enemy.units.items():
                            if (eu.x, eu.y) == check_pos:
                                adjacent_enemy = (euid, eu, check_pos)
                                break
                        break
                
                if adjacent_enemy:
                    euid, eu, pos = adjacent_enemy
                    strategy_lines.append(f'"{uid}": {{"action_type": "BATTLE", "to": {{"x": {pos[0]}, "y": {pos[1]}}}, "target_civ": "Nomads"}}')
                    strategy_lines.append(f"  ^ ATTACK {euid}!")
                elif uid in defender_ids and enemies_near_city:
                    # Defend - intercept nearest threatening cavalry
                    nearest = min(enemies_near_city, key=lambda e: abs(e[1].x - unit.x) + abs(e[1].y - unit.y))
                    target_cav = nearest[1]
                    
                    # Move towards threatening cavalry
                    dx = 1 if target_cav.x > unit.x else (-1 if target_cav.x < unit.x else 0)
                    dy = 1 if target_cav.y > unit.y else (-1 if target_cav.y < unit.y else 0)
                    target_x = max(0, min(6, unit.x + dx))
                    target_y = unit.y if dx != 0 else max(0, min(6, unit.y + dy))
                    
                    strategy_lines.append(f'"{uid}": {{"action_type": "MOVE", "to": {{"x": {target_x}, "y": {target_y}}}}}')
                    strategy_lines.append(f"  ^ DEFEND: Intercept {nearest[0]}!")
                else:
                    # Attack mode - move towards enemy city
                    dx = 1 if enemy_city_pos[0] > unit.x else (-1 if enemy_city_pos[0] < unit.x else 0)
                    dy = 1 if enemy_city_pos[1] > unit.y else (-1 if enemy_city_pos[1] < unit.y else 0)
                    target_x = max(0, min(6, unit.x + dx))
                    target_y = unit.y if dx != 0 else max(0, min(6, unit.y + dy))
                    
                    strategy_lines.append(f'"{uid}": {{"action_type": "MOVE", "to": {{"x": {target_x}, "y": {target_y}}}}}')
                    dist = abs(target_x - enemy_city_pos[0]) + abs(target_y - enemy_city_pos[1])
                    strategy_lines.append(f"  ^ ATTACK: Move towards Nomads city! ({dist} cells away)")
            elif unit.unit_type == "farmer":
                # Check if enemy is nearby (within 2 squares)
                enemy_nearby = None
                for euid, eu in enemy_cavalry:
                    dist = abs(eu.x - unit.x) + abs(eu.y - unit.y)
                    if dist <= 2:
                        enemy_nearby = (euid, eu)
                        break
                
                if enemy_nearby:
                    # Flee towards city
                    dx = -1 if unit.x > own_city_pos[0] else (1 if unit.x < own_city_pos[0] else 0)
                    dy = -1 if unit.y > own_city_pos[1] else (1 if unit.y < own_city_pos[1] else 0)
                    flee_x = max(0, min(6, unit.x + dx))
                    flee_y = unit.y if dx != 0 else max(0, min(6, unit.y + dy))
                    strategy_lines.append(f'"{uid}": {{"action_type": "MOVE", "to": {{"x": {flee_x}, "y": {flee_y}}}}}')
                    strategy_lines.append(f"  ^ FLEE from {enemy_nearby[0]}!")
                else:
                    strategy_lines.append(f'"{uid}": {{"action_type": "GATHER"}}')
                    strategy_lines.append(f"  ^ Gather resources safely")
            elif unit.unit_type == "city":
                # Produce soldiers if we have enough resources and enemies are near
                if player.resources >= 4 and len(own_soldiers) < 3:
                    # Find empty adjacent cell
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        prod_x, prod_y = unit.x + dx, unit.y + dy
                        if 0 <= prod_x < 7 and 0 <= prod_y < 7:
                            cell = state.map_grid[prod_y][prod_x]
                            if len(cell.units) == 0:
                                strategy_lines.append(f'"{uid}": {{"action_type": "PRODUCE_UNIT", "produce_unit_type": "soldier", "to": {{"x": {prod_x}, "y": {prod_y}}}}}')
                                strategy_lines.append(f"  ^ Produce soldier for defense!")
                                break
                    else:
                        strategy_lines.append(f'"{uid}": {{"action_type": "PRODUCE_RESOURCE"}}')
                        strategy_lines.append(f"  ^ Build economy")
                else:
                    strategy_lines.append(f'"{uid}": {{"action_type": "PRODUCE_RESOURCE"}}')
                    strategy_lines.append(f"  ^ Build economy")
        
        strategy_lines.append("")
        strategy_lines.append("REMEMBER: COPY THE EXACT JSON ABOVE!")
        strategy_lines.append(f"GOAL: Protect city at ({own_city_pos[0]},{own_city_pos[1]}) AND attack enemy city at ({enemy_city_pos[0]},{enemy_city_pos[1]})!")
        
        return "\n".join(strategy_lines)
    
    def _parse_multi_unit_response(
        self,
        response: str,
        unit_ids: List[str],
        legal_actions: List[Action],
        agent: 'LLMAgent',
    ) -> Dict[str, Action]:
        """Parse LLM response containing actions for multiple units."""
        from .env import Action
        from .utils import parse_json_safely
        
        actions = {}
        
        # Try to parse JSON response
        parsed = parse_json_safely(response)
        if not parsed:
            parsed = agent.llm.extract_json(response)
        
        if parsed and isinstance(parsed, dict):
            # Parse each unit's action
            for unit_id in unit_ids:
                if unit_id in parsed:
                    try:
                        action_dict = parsed[unit_id]
                        action = Action.from_dict(action_dict)
                        action.unit_id = unit_id  # Ensure unit_id is set
                        
                        # Validate it's a legal action
                        unit_legal = [a for a in legal_actions if a.unit_id == unit_id]
                        if self._is_action_legal_for_unit(action, unit_legal):
                            actions[unit_id] = action
                        else:
                            # Use fallback
                            actions[unit_id] = self._choose_fallback_for_unit(unit_id, legal_actions)
                    except:
                        actions[unit_id] = self._choose_fallback_for_unit(unit_id, legal_actions)
                else:
                    # Unit not in response, use fallback
                    actions[unit_id] = self._choose_fallback_for_unit(unit_id, legal_actions)
        else:
            # Failed to parse, use fallback for all
            for unit_id in unit_ids:
                actions[unit_id] = self._choose_fallback_for_unit(unit_id, legal_actions)
        
        return actions
    
    def _is_action_legal_for_unit(self, action: Action, legal_actions: List[Action]) -> bool:
        """Check if action is legal for the unit."""
        for legal in legal_actions:
            if (action.action_type == legal.action_type and
                action.unit_id == legal.unit_id and
                action.to_x == legal.to_x and
                action.to_y == legal.to_y and
                action.target_civ == legal.target_civ):
                return True
        return False
    
    def _choose_fallback_for_unit(self, unit_id: str, all_legal_actions: List[Action]) -> Action:
        """Choose fallback action for a specific unit."""
        from .env import Action
        
        unit_actions = [a for a in all_legal_actions if a.unit_id == unit_id]
        if unit_actions:
            agent = self.agents.get(list(self.agents.keys())[0])  # Get any agent for fallback logic
            return agent._choose_fallback_action(unit_actions)
        return Action(action_type="PASS", unit_id=unit_id)
    
    def _build_retry_prompt(
        self,
        original_prompt: str,
        previous_response: str,
        invalid_feedback: List[Dict],
    ) -> str:
        """Build a retry prompt with feedback about invalid actions."""
        lines = [original_prompt]
        lines.append("\n" + "="*50)
        lines.append("PREVIOUS ATTEMPT HAD INVALID ACTIONS:")
        lines.append("="*50)
        
        for feedback in invalid_feedback:
            lines.append(f"\n❌ {feedback['unit_id']}:")
            lines.append(f"   Action: {feedback['action']}")
            lines.append(f"   Problem: {feedback['reason']}")
        
        lines.append("\n" + "="*50)
        lines.append("Please generate VALID actions for ALL units.")
        lines.append("Make sure to follow the game rules carefully.")
        lines.append("="*50)
        lines.append("\nYOUR CORRECTED ACTIONS:")
        
        return "\n".join(lines)
    
    def _correct_invalid_actions(
        self,
        actions: Dict[str, Action],
        invalid_feedback: List[Dict],
        all_legal_actions: List[Action],
        agent: 'LLMAgent',
    ) -> Dict[str, Action]:
        """Correct invalid actions using fallback logic."""
        from .env import Action
        
        invalid_unit_ids = {f['unit_id'] for f in invalid_feedback}
        
        # Keep valid actions, replace invalid ones with fallback
        corrected_actions = {}
        for unit_id, action in actions.items():
            if unit_id in invalid_unit_ids:
                # Use fallback for invalid action
                unit_legal = [a for a in all_legal_actions if a.unit_id == unit_id]
                if unit_legal:
                    corrected_actions[unit_id] = agent._choose_fallback_action(unit_legal)
                    logger.info(f"Corrected {unit_id}: {action.action_type} -> {corrected_actions[unit_id].action_type}")
                else:
                    corrected_actions[unit_id] = Action(action_type="PASS", unit_id=unit_id)
            else:
                # Keep valid action
                corrected_actions[unit_id] = action
        
        return corrected_actions

