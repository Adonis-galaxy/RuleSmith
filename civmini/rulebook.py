"""
CivMini Rulebook and RAG retrieval system.

Defines the game rules as natural language text and provides simple
RAG-based retrieval for rule lookup.
"""

from typing import List, Tuple
import re
from collections import Counter
import math


# CivMini Game Rules (Updated)
CIVMINI_RULES = [
    # Rule 1: Game Setup
    """Rule 1 - Game Setup: CivMini is played on a 7x7 grid map between two civilizations: 
    Empire and Nomads. Empire starts with one city at position (1,1), Nomads starts with one 
    city at position (5,5). Both cities are immobile units that can be attacked. The game lasts 
    for a maximum number of turns (typically 10). Players take turns alternating. IMPORTANT: 
    Only ONE unit can occupy a cell at any time - no two units (friendly or enemy) can share 
    the same cell.""",
    
    # Rule 2: Unit Types
    """Rule 2 - Unit Types: Empire has THREE unit types: (1) City - immobile, can produce 
    resources or units, can be attacked; (2) Farmer - can only GATHER resources and MOVE, 
    cannot battle; (3) Soldier - can only BATTLE and MOVE, cannot gather. Nomads has TWO unit 
    types: (1) City - same as Empire city; (2) Cavalry - versatile units that can GATHER, 
    BATTLE, and MOVE. All units (including cities) have hit points and can be destroyed.""",
    
    # Rule 3: Map
    """Rule 3 - Map: The 7x7 map is a simple grid with no terrain or resource types. All cells 
    are identical. IMPORTANT: Each cell can contain AT MOST ONE unit (no stacking). Units cannot 
    move to occupied cells.""",
    
    # Rule 4: Turn Structure
    """Rule 4 - Turn Structure: Each turn consists of: (1) Player action phase where EACH UNIT 
    (including cities) of the current player performs ONE action (multi-action mode), (2) Switch 
    to next player. After both players have acted, the turn counter increments and all units 
    reset their movement points. In single-action mode, only one unit acts per turn.""",
    
    # Rule 5: City Actions
    """Rule 5 - City Actions: Cities are special immobile units that can perform two types of 
    actions: (1) PRODUCE_RESOURCE - generate resources (empire_city_resource_production for Empire, 
    nomads_city_resource_production for Nomads), (2) PRODUCE_UNIT - spend resources to create a 
    new unit. Unit costs differ by civilization (empire_unit_production_cost and nomads_unit_production_cost). 
    Empire cities can produce Farmers (gather only) or Soldiers (combat only). Nomads cities can 
    only produce Cavalry (both gather and combat). New units appear in an adjacent empty cell.""",
    
    # Rule 6: Resource Economy
    """Rule 6 - Resource Economy: Empire gains resources through GATHER action (Farmers only). 
    Farmers have empire_farmer_gather_amount gathering rate. Nomads CANNOT gather - instead, 
    Nomads gain resources by KILLING enemy units. When Nomads destroy an enemy, they receive 
    nomads_kill_resource_gain resources. This creates asymmetric economies: Empire is peaceful 
    gatherers, Nomads are aggressive raiders.""",
    
    # Rule 7: Movement Rules
    """Rule 7 - Movement Rules: MOVE action allows a unit to move up to its move_points in Manhattan 
    distance (up, down, left, right - no diagonals). Empire units (Farmers and Soldiers) have 
    empire_unit_move_points (typically 1). Cavalry have nomads_cavalry_move_points (typically 2, 
    meaning they can move UP TO 2 cells per turn for greater mobility). Cities have 0 movement 
    points (immobile). CRITICAL: A cell can only contain ONE unit - moves are blocked if ANY unit 
    (friendly or enemy) is at the destination. Units cannot stack.""",
    
    # Rule 8: Combat System
    """Rule 8 - Combat System: BATTLE action attacks an enemy unit in an ADJACENT cell (range 1, 
    up/down/left/right). Only Soldiers, Cavalry, and Cities can battle. Farmers cannot battle. 
    Damage calculation: base_damage (empire_battle_base_damage for Empire, nomads_battle_base_damage 
    for Nomads) multiplied by battle_bonus (unified for both sides). When a unit's HP reaches 0, 
    it is destroyed and removed from the map. Cities have the same HP as their civilization's 
    combat units (empire_soldier_hp or nomads_cavalry_hp). The attacker gains a battle victory count.""",
    
    # Rule 9: Victory Conditions
    """Rule 9 - Victory Conditions: The game ends immediately if one player's CITY is destroyed - 
    the other player wins instantly. If both cities survive to max_turns, the winner is determined 
    by final score. Final score = resources * score_per_resource + battles_won * score_per_battle_won 
    + surviving_units * score_per_surviving_unit. Cities are not scored separately since each 
    side has exactly one.""",
    
    # Rule 10: Strategic Considerations
    """Rule 10 - Strategic Considerations: Empire has peaceful economy (Farmers gather resources) 
    but must protect Farmers with Soldiers. Nomads have aggressive economy (gain resources ONLY 
    by killing enemies) forcing them into combat from the start. Cavalry can move 2 cells per turn, 
    allowing rapid strikes and retreats. Empire can choose defensive play, but Nomads MUST attack 
    to gain resources for army expansion. This asymmetry creates distinct playstyles: Empire builds 
    economy then army, Nomads use mobility to raid immediately. Destroying enemy city is instant 
    victory. Cities produce resources OR units each turn.""",
]


def get_full_rulebook() -> List[str]:
    """
    Get the complete rulebook as a list of rule strings.
    
    Returns:
        List of rule strings
    """
    return CIVMINI_RULES.copy()


class SimpleRAG:
    """
    Simple RAG (Retrieval-Augmented Generation) system using TF-IDF and cosine similarity.
    
    This is a lightweight implementation that doesn't require heavy ML libraries.
    """
    
    def __init__(self, documents: List[str]):
        """
        Initialize RAG with a corpus of documents.
        
        Args:
            documents: List of text documents (rules)
        """
        self.documents = documents
        self.processed_docs = [self._preprocess(doc) for doc in documents]
        self.vocab = self._build_vocab()
        self.doc_vectors = [self._vectorize(doc) for doc in self.processed_docs]
    
    def _preprocess(self, text: str) -> List[str]:
        """Preprocess text: lowercase and tokenize."""
        text = text.lower()
        # Simple tokenization: split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_vocab(self) -> List[str]:
        """Build vocabulary from all documents."""
        all_tokens = set()
        for doc in self.processed_docs:
            all_tokens.update(doc)
        return sorted(list(all_tokens))
    
    def _vectorize(self, tokens: List[str]) -> dict:
        """Convert tokens to TF-IDF vector (as dict for efficiency)."""
        # Term frequency
        tf = Counter(tokens)
        total_tokens = len(tokens)
        tf = {term: count / total_tokens for term, count in tf.items()}
        
        # IDF (inverse document frequency)
        vector = {}
        for term in tf:
            # Count documents containing this term
            df = sum(1 for doc in self.processed_docs if term in doc)
            idf = math.log(len(self.documents) / (1 + df))
            vector[term] = tf[term] * idf
        
        return vector
    
    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """Compute cosine similarity between two sparse vectors."""
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, similarity_score) tuples, sorted by relevance
        """
        # Preprocess and vectorize query
        query_tokens = self._preprocess(query)
        query_vector = self._vectorize(query_tokens)
        
        # Compute similarities
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            sim = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((self.documents[i], sim))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Global RAG instance
_rag_instance = None


def get_rag_instance() -> SimpleRAG:
    """Get or create the global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SimpleRAG(CIVMINI_RULES)
    return _rag_instance


def retrieve_rules(query: str, top_k: int = 3) -> List[str]:
    """
    Retrieve relevant rules for a query.
    
    Args:
        query: Query text (e.g., "how do I build a city?")
        top_k: Number of rules to retrieve
        
    Returns:
        List of relevant rule strings
    """
    rag = get_rag_instance()
    results = rag.retrieve(query, top_k=top_k)
    return [doc for doc, score in results]


def format_rules_for_prompt(rules: List[str]) -> str:
    """
    Format a list of rules for inclusion in an LLM prompt.
    
    Args:
        rules: List of rule strings
        
    Returns:
        Formatted string suitable for prompts
    """
    if not rules:
        return "No specific rules retrieved."
    
    formatted = "RELEVANT RULES:\n"
    for i, rule in enumerate(rules, 1):
        formatted += f"{i}. {rule}\n\n"
    
    return formatted.strip()

