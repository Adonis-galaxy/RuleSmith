#!/usr/bin/env python3
"""
Visualize game directly from log file.
Parses log to extract unit positions and creates visualizations.
"""

import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import argparse
from collections import defaultdict

class LogGameParser:
    """Parse game from log file by reading World State After Actions blocks."""
    
    def __init__(self, map_size=7):
        self.map_size = map_size
    
    def parse_game(self, log_file, game_number=1):
        """Parse a specific game from log.
        
        Uses two sources:
        1. "--- Unit Positions ---" block: State BEFORE current player acts
        2. "--- World State After Actions ---" block: State AFTER current player acts
        
        Frame semantics:
        - Frame for (Turn X, Player Y) shows world state BEFORE Player Y acts
        - This means: state AFTER the previous player acted
        """
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Check if new format exists (World State After Actions)
        has_world_state = "--- World State After Actions ---" in content
        
        if has_world_state:
            return self._parse_with_world_state(content, game_number)
        else:
            return self._parse_legacy(content, game_number)
    
    def _parse_with_world_state(self, content, game_number):
        """Parse using World State After Actions blocks (new format)."""
        states = []
        
        # Split content by turns
        # Pattern to find each turn block
        turn_pattern = r'Turn (\d+), Player: (\w+)(.*?)(?=Turn \d+, Player:|Game Over!|$)'
        turn_matches = re.findall(turn_pattern, content, re.DOTALL)
        
        # Track the world state (all units)
        # Start with initial positions
        world_state = {
            'empire_city': (1, 1),
            'empire_farmer_0': (2, 1),
            'empire_soldier_0': (0, 0),
            'nomads_city': (5, 5),
            'nomads_cavalry_0': (4, 5),
            'nomads_cavalry_1': (5, 4),
        }
        
        for turn_str, player, block in turn_matches:
            turn = int(turn_str)
            
            # Extract Unit Positions (state before this player acts)
            unit_pos_match = re.search(r'--- Unit Positions ---\n(.*?)--- End Positions ---', block, re.DOTALL)
            if unit_pos_match:
                # Parse this player's current positions
                pos_pattern = r'(\w+): \((\d+),(\d+)\)'
                for unit_id, x, y in re.findall(pos_pattern, unit_pos_match.group(1)):
                    world_state[unit_id] = (int(x), int(y))
            
            # Save state BEFORE this player acts
            state = {
                'turn': turn,
                'player': player,
                'positions': world_state.copy(),
                'alive': set(world_state.keys()),
                'resources': {'Empire': 5, 'Nomads': 5},
            }
            states.append(state)
            
            # Extract World State After Actions (state after this player acts)
            world_state_match = re.search(r'--- World State After Actions ---\n(.*?)--- End World State ---', block, re.DOTALL)
            if world_state_match:
                # Parse ALL units' positions after actions
                new_world_state = {}
                pos_pattern = r'(\w+): \((\d+),(\d+)\)'
                for unit_id, x, y in re.findall(pos_pattern, world_state_match.group(1)):
                    new_world_state[unit_id] = (int(x), int(y))
                
                # Update world state with new positions
                # Units not in new state are dead
                world_state = new_world_state
        
        # Parse game over info
        game_over_match = re.search(r'Game Over! Winner: (\w+)', content)
        if game_over_match:
            winner = game_over_match.group(1)
            
            score_match = re.search(r'Empire Score: ([\d.]+) \| Nomads Score: ([\d.]+)', content)
            empire_score, nomads_score = 0, 0
            if score_match:
                empire_score = float(score_match.group(1))
                nomads_score = float(score_match.group(2))
            
            if states:
                final_state = {
                    'turn': states[-1]['turn'],
                    'player': 'GameOver',
                    'positions': world_state.copy(),
                    'alive': set(world_state.keys()),
                    'resources': {'Empire': 5, 'Nomads': 5},
                    'winner': winner,
                    'scores': {'Empire': empire_score, 'Nomads': nomads_score},
                }
                states.append(final_state)
        
        print(f"✓ Extracted {len(states)} game states (using World State After Actions)")
        return {'states': states, 'game_number': game_number} if states else None
    
    def _parse_legacy(self, content, game_number):
        """Parse using only Unit Positions blocks (legacy format)."""
        # Extract all Turn blocks with their Unit Positions
        pattern = r'Turn (\d+), Player: (\w+).*?--- Unit Positions ---\n(.*?)--- End Positions ---'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            print("❌ No Turn blocks found in log")
            return None
        
        # Build position data for each (turn, player) pair
        turn_positions = {}
        
        for turn_str, player, positions_block in matches:
            turn = int(turn_str)
            key = (turn, player)
            
            pos_pattern = r'(\w+): \((\d+),(\d+)\)'
            units = {uid: (int(x), int(y)) for uid, x, y in re.findall(pos_pattern, positions_block)}
            turn_positions[key] = units
        
        # Build states
        states = []
        all_turns = sorted(set(turn for turn, player in turn_positions.keys()))
        max_turn = max(all_turns) if all_turns else 0
        
        for turn in all_turns:
            for player in ['Empire', 'Nomads']:
                key = (turn, player)
                if key not in turn_positions:
                    continue
                
                current_player_units = turn_positions[key]
                
                if player == 'Empire':
                    empire_units = {k: v for k, v in current_player_units.items() if 'empire' in k}
                    
                    nomads_key = (turn, 'Nomads')
                    if nomads_key in turn_positions:
                        nomads_units = {k: v for k, v in turn_positions[nomads_key].items() if 'nomads' in k}
                    else:
                        for prev_turn in range(turn - 1, -1, -1):
                            prev_key = (prev_turn, 'Nomads')
                            if prev_key in turn_positions:
                                nomads_units = {k: v for k, v in turn_positions[prev_key].items() if 'nomads' in k}
                                break
                        else:
                            nomads_units = {}
                else:
                    nomads_units = {k: v for k, v in current_player_units.items() if 'nomads' in k}
                    
                    next_empire_key = (turn + 1, 'Empire')
                    if next_empire_key in turn_positions:
                        empire_units = {k: v for k, v in turn_positions[next_empire_key].items() if 'empire' in k}
                    else:
                        this_empire_key = (turn, 'Empire')
                        if this_empire_key in turn_positions:
                            empire_units = {k: v for k, v in turn_positions[this_empire_key].items() if 'empire' in k}
                        else:
                            for prev_turn in range(turn - 1, -1, -1):
                                prev_key = (prev_turn + 1, 'Empire')
                                if prev_key in turn_positions:
                                    empire_units = {k: v for k, v in turn_positions[prev_key].items() if 'empire' in k}
                                    break
                            else:
                                empire_units = {}
                
                all_positions = {}
                all_positions.update(empire_units)
                all_positions.update(nomads_units)
                
                alive = set(all_positions.keys())
                
                state = {
                    'turn': turn,
                    'player': player,
                    'positions': all_positions,
                    'alive': alive,
                    'resources': {'Empire': 5, 'Nomads': 5},
                }
                states.append(state)
        
        # Parse game over
        game_over_match = re.search(r'Game Over! Winner: (\w+)', content)
        if game_over_match:
            winner = game_over_match.group(1)
            
            score_match = re.search(r'Empire Score: ([\d.]+) \| Nomads Score: ([\d.]+)', content)
            empire_score, nomads_score = 0, 0
            if score_match:
                empire_score = float(score_match.group(1))
                nomads_score = float(score_match.group(2))
            
            if states:
                final_state = states[-1].copy()
                final_state['player'] = 'GameOver'
                final_state['winner'] = winner
                final_state['scores'] = {'Empire': empire_score, 'Nomads': nomads_score}
                states.append(final_state)
        
        print(f"✓ Extracted {len(states)} game states (legacy format)")
        return {'states': states, 'game_number': game_number} if states else None


class GameVisualizer:
    """Visualize game states."""
    
    def __init__(self, map_size=7):
        self.map_size = map_size
        self.colors = {
            'empire_city': '#0066cc',
            'empire_farmer': '#66b3ff',
            'empire_soldier': '#ff6600',
            'nomads_city': '#009900',
            'nomads_cavalry': '#66cc66',
        }
    
    def draw_state(self, state, ax, show_legend=False):
        """Draw a single game state."""
        ax.clear()
        
        # Draw grid
        for i in range(self.map_size + 1):
            ax.plot([0, self.map_size], [i, i], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([i, i], [0, self.map_size], 'k-', alpha=0.2, linewidth=0.5)
        
        def get_unit_type(unit_id):
            if 'city' in unit_id:
                return 'city'
            elif 'farmer' in unit_id:
                return 'farmer'
            elif 'soldier' in unit_id:
                return 'soldier'
            elif 'cavalry' in unit_id:
                return 'cavalry'
            return 'unknown'
        
        positions = state['positions']
        alive = state.get('alive', set(positions.keys()))
        
        for unit_id, (x, y) in positions.items():
            if unit_id not in alive:
                continue
            
            unit_type = get_unit_type(unit_id)
            owner = 'empire' if 'empire' in unit_id else 'nomads'
            
            color = self.colors.get(f'{owner}_{unit_type}', 'gray')
            
            if unit_type == 'city':
                marker, size = 's', 1200
            elif unit_type == 'farmer':
                marker, size = 'o', 600
            elif unit_type == 'soldier':
                marker, size = '^', 600
            elif unit_type == 'cavalry':
                marker, size = 'D', 600
            else:
                marker, size = 'o', 400
            
            ax.scatter(x + 0.5, y + 0.5, s=size, c=color, marker=marker,
                      alpha=0.85, edgecolors='black', linewidth=2, zorder=10)
            
            label = unit_id.replace('empire_', 'E-').replace('nomads_', 'N-')
            label = label.replace('city', 'C').replace('farmer', 'F')
            label = label.replace('soldier', 'S').replace('cavalry', 'V')
            ax.text(x + 0.5, y + 0.25, label, ha='center', va='center',
                   fontsize=7, weight='bold', zorder=11,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-0.1, self.map_size + 0.1)
        ax.set_ylim(-0.1, self.map_size + 0.1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        ax.set_xticks(range(self.map_size + 1))
        ax.set_yticks(range(self.map_size + 1))
        
        if state.get('player') == 'GameOver':
            title = f"Final - {state.get('winner', 'Unknown')} Wins!"
            if 'scores' in state:
                title += f"\nEmpire: {state['scores']['Empire']:.1f} | Nomads: {state['scores']['Nomads']:.1f}"
        else:
            title = f"Turn {state['turn']} - {state['player']}"
        
        ax.set_title(title, fontsize=11, weight='bold')
        
        return ax
    
    def create_frames(self, states, output_dir='frames'):
        """Save individual frames."""
        os.makedirs(output_dir, exist_ok=True)
        frame_files = []
        
        print(f"Creating {len(states)} frames...")
        for i, state in enumerate(states):
            fig, ax = plt.subplots(figsize=(8, 7))
            self.draw_state(state, ax)
            
            legend_elements = [
                mpatches.Patch(color=self.colors['empire_city'], label='Empire City'),
                mpatches.Patch(color=self.colors['empire_farmer'], label='Farmer'),
                mpatches.Patch(color=self.colors['empire_soldier'], label='Soldier'),
                mpatches.Patch(color=self.colors['nomads_city'], label='Nomads City'),
                mpatches.Patch(color=self.colors['nomads_cavalry'], label='Cavalry'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
            
            filename = os.path.join(output_dir, f'frame_{i:04d}.png')
            plt.tight_layout()
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            frame_files.append(filename)
            
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(states)} frames created")
        
        print(f"✓ Created {len(frame_files)} frames")
        return frame_files
    
    def create_gif(self, frame_files, output_file='game.gif', duration=300):
        """Create GIF from frames."""
        print(f"\nCreating GIF from {len(frame_files)} frames...")
        
        images = [Image.open(f) for f in frame_files]
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✓ Created GIF: {output_file} ({size_mb:.1f} MB)")
        return output_file
    
    def create_grid(self, states, output_file='grid.png', max_cols=4, max_rows=8):
        """Create grid showing all frames."""
        n_cols = max_cols
        n_rows = max_rows
        max_panels = n_rows * n_cols
        
        print(f"\nCreating {n_rows}x{n_cols} grid figure (showing all {len(states)} states)...")
        
        sampled = states[:max_panels]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        axes_flat = [ax for row in axes for ax in row]
        
        for i, state in enumerate(sampled):
            self.draw_state(state, axes_flat[i])
        
        for i in range(len(sampled), len(axes_flat)):
            axes_flat[i].axis('off')
        
        legend_elements = [
            mpatches.Patch(color=self.colors['empire_city'], label='Empire City'),
            mpatches.Patch(color=self.colors['empire_farmer'], label='Farmer'),
            mpatches.Patch(color=self.colors['empire_soldier'], label='Soldier'),
            mpatches.Patch(color=self.colors['nomads_city'], label='Nomads City'),
            mpatches.Patch(color=self.colors['nomads_cavalry'], label='Cavalry'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                  fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, -0.01))
        
        winner = states[-1].get('winner', 'Unknown')
        plt.suptitle(f'CivMini Game Progression - {winner} Wins', 
                    fontsize=16, weight='bold', y=0.99)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Created grid: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Visualize game from log file')
    parser.add_argument('log_file', type=str, help='Path to log file')
    parser.add_argument('--game', type=int, default=1, help='Game number (1-10)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same dir as log, named <logname>_vis)')
    parser.add_argument('--gif-duration', type=int, default=300,
                       help='Duration per frame in GIF (ms)')
    parser.add_argument('--skip-frames', action='store_true',
                       help='Skip individual frames (only create GIF and grid)')
    parser.add_argument('--skip-gif', action='store_true',
                       help='Skip GIF creation')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        log_name = os.path.splitext(os.path.basename(args.log_file))[0]
        args.output_dir = os.path.join(log_dir, f"{log_name}_vis")
        print(f"Auto-determined output directory: {args.output_dir}")
    
    print("="*60)
    print("CivMini Game Visualizer (from Log)")
    print("="*60)
    print(f"Log file: {args.log_file}")
    print(f"Game number: {args.game}")
    print(f"Output: {args.output_dir}/")
    print()
    
    parser_obj = LogGameParser()
    print("Parsing log file...")
    game_data = parser_obj.parse_game(args.log_file, args.game)
    
    if game_data is None:
        print("\n❌ Failed to parse game from log")
        return 1
    
    states = game_data['states']
    
    if states[-1].get('winner'):
        print(f"  Winner: {states[-1]['winner']}")
        if 'scores' in states[-1]:
            print(f"  Empire: {states[-1]['scores']['Empire']:.1f}")
            print(f"  Nomads: {states[-1]['scores']['Nomads']:.1f}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    viz = GameVisualizer()
    
    frame_files = []
    if not args.skip_frames:
        frames_dir = os.path.join(args.output_dir, 'frames')
        frame_files = viz.create_frames(states, frames_dir)
    
    if not args.skip_gif and not args.skip_frames:
        gif_file = os.path.join(args.output_dir, f'game{args.game}.gif')
        viz.create_gif(frame_files, gif_file, duration=args.gif_duration)
    
    grid_file = os.path.join(args.output_dir, f'game{args.game}_grid.png')
    viz.create_grid(states, grid_file, max_cols=4, max_rows=8)
    
    print("\n" + "="*60)
    print("✓ Visualization Complete!")
    print("="*60)
    print(f"Output in: {args.output_dir}/")
    if not args.skip_frames:
        print(f"  - frames/ ({len(frame_files)} images)")
    if not args.skip_gif and not args.skip_frames:
        print(f"  - game{args.game}.gif")
    print(f"  - game{args.game}_grid.png")
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
