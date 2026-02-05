"""
LLM-based rule checker/referee for CivMini.

Validates actions using RAG + LLM to ensure rule compliance.
"""

from typing import Optional, Tuple
import logging
import json

from .llm_client import InternVLLM
from .env import CivMiniState, CivMiniEnv, Action
from .rulebook import retrieve_rules, format_rules_for_prompt

logger = logging.getLogger(__name__)


class LLMRuleChecker:
    """
    LLM-based rule checker that validates actions.
    
    Uses RAG to retrieve relevant rules and LLM to judge legality.
    """
    
    def __init__(
        self,
        llm: InternVLLM,
        rag_top_k: int = 3,
    ):
        """
        Initialize rule checker.
        
        Args:
            llm: LLM client instance
            rag_top_k: Number of rules to retrieve
        """
        self.llm = llm
        self.rag_top_k = rag_top_k
    
    def check_action(
        self,
        state: CivMiniState,
        action: Action,
        env: CivMiniEnv,
        use_llm_check: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if an action is legal according to game rules.
        
        Args:
            state: Current game state
            action: Action to validate
            env: Game environment
            use_llm_check: Whether to use LLM validation (slower)
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # First, check against programmatic legal actions
        legal_actions = env.get_legal_actions(state)
        is_programmatically_legal = self._is_in_legal_actions(action, legal_actions)
        
        if not is_programmatically_legal:
            return False, "Action is not in the list of programmatically legal actions"
        
        # Optional: Query LLM for additional validation
        if use_llm_check:
            try:
                is_valid_llm, reason_llm = self._check_with_llm(state, action)
                return is_valid_llm, reason_llm
            except Exception as e:
                logger.warning(f"LLM checker failed: {e}, falling back to programmatic check")
                return is_programmatically_legal, "Programmatic validation passed"
        else:
            return True, "Programmatic validation passed"
    
    def _is_in_legal_actions(self, action: Action, legal_actions: list) -> bool:
        """Check if action matches any legal action."""
        for legal in legal_actions:
            if (action.action_type == legal.action_type and
                action.unit_id == legal.unit_id and
                action.to_x == legal.to_x and
                action.to_y == legal.to_y and
                action.target_civ == legal.target_civ):
                return True
        return False
    
    def _check_with_llm(
        self,
        state: CivMiniState,
        action: Action,
    ) -> Tuple[bool, str]:
        """
        Use LLM to validate action.
        
        Args:
            state: Game state
            action: Action to check
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Retrieve relevant rules
        query = f"rules for {action.action_type} action in CivMini"
        relevant_rules = retrieve_rules(query, top_k=self.rag_top_k)
        
        # Build validation prompt
        system_prompt = """You are a rule checker for CivMini game.
Your job is to determine if a proposed action is VALID according to the game rules.

Respond with:
- "YES - [reason]" if the action is valid
- "NO - [reason]" if the action is invalid

Be strict and ensure all rules are followed."""
        
        user_prompt = self._build_validation_prompt(state, action, relevant_rules)
        
        # Query LLM
        response = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=150,
            temperature=0.3,  # Lower temperature for more consistent validation
        )
        
        # Parse response
        is_valid = self.llm.extract_yes_no(response)
        reason = response.strip()
        
        return is_valid, reason
    
    def _build_validation_prompt(
        self,
        state: CivMiniState,
        action: Action,
        relevant_rules: list,
    ) -> str:
        """Build prompt for action validation."""
        lines = []
        
        # Rules
        lines.append(format_rules_for_prompt(relevant_rules))
        lines.append("")
        
        # Current state
        lines.append("CURRENT STATE:")
        lines.append(f"Turn: {state.turn}/{state.max_turns}")
        lines.append(f"Current Player: {state.current_player}")
        
        player = state.players[state.current_player]
        lines.append(f"Player Resources: {player.resources:.1f}")
        lines.append(f"Player Units: {len(player.units)}")
        lines.append(f"Player Cities: {len(player.cities)}")
        lines.append("")
        
        # Action to validate
        lines.append("PROPOSED ACTION:")
        lines.append(json.dumps(action.to_dict(), indent=2))
        lines.append("")
        
        # Add relevant unit info if applicable
        if action.unit_id and action.unit_id in player.units:
            unit = player.units[action.unit_id]
            lines.append(f"Unit Details:")
            lines.append(f"  - Position: ({unit.x}, {unit.y})")
            lines.append(f"  - HP: {unit.hp:.1f}")
            lines.append(f"  - Move Points: {unit.move_points:.1f}")
            lines.append("")
        
        lines.append("QUESTION:")
        lines.append("Is this action VALID according to the rules? Answer YES or NO with a brief reason.")
        
        return "\n".join(lines)
    
    def suggest_correction(
        self,
        state: CivMiniState,
        invalid_action: Action,
        env: CivMiniEnv,
    ) -> Optional[Action]:
        """
        Suggest a corrected action if the original was invalid.
        
        Args:
            state: Game state
            invalid_action: The invalid action
            env: Game environment
            
        Returns:
            Corrected action or None
        """
        # Get legal actions
        legal_actions = env.get_legal_actions(state)
        
        if not legal_actions:
            return None
        
        # Try to find a similar legal action
        # Priority: same type > same unit > any legal action
        
        # Same type and unit
        for action in legal_actions:
            if (action.action_type == invalid_action.action_type and
                action.unit_id == invalid_action.unit_id):
                return action
        
        # Same type
        for action in legal_actions:
            if action.action_type == invalid_action.action_type:
                return action
        
        # Same unit
        for action in legal_actions:
            if action.unit_id == invalid_action.unit_id:
                return action
        
        # Any legal action (prioritize non-PASS)
        for action in legal_actions:
            if action.action_type != "PASS":
                return action
        
        # Fallback: PASS
        return legal_actions[0]

