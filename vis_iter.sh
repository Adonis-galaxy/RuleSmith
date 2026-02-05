#!/bin/bash
# Visualize all games in an iteration
# bash vis_iter.sh -y -g runs/exp_nums/logs/run_*/game_logs/iter_xx

# Usage
show_usage() {
    echo "Usage:"
    echo "  bash vis_iter.sh [options] [iter_dir]"
    echo ""
    echo "Options:"
    echo "  -y, --yes       Non-interactive mode (auto-confirm)"
    echo "  -g, --gif       Generate GIFs (default: grid only in non-interactive mode)"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  bash vis_iter.sh                                    # Auto-find latest iteration (interactive)"
    echo "  bash vis_iter.sh -y runs/RUNNAME/logs/run_*/game_logs/iter_5  # Non-interactive, grid only"
    echo "  bash vis_iter.sh -y -g runs/RUNNAME/logs/run_*/game_logs/iter_5  # Non-interactive, with GIFs"
    echo ""
}

# Parse options
AUTO_YES=false
WANT_GIF=false
ITER_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        -g|--gif)
            WANT_GIF=true
            shift
            ;;
        *)
            ITER_ARG="$1"
            shift
            ;;
    esac
done

echo "========================================="
echo "Batch Visualization - All Games in Iteration"
echo "========================================="

# Determine iteration directory
if [ -n "$ITER_ARG" ]; then
    # Expand wildcards first
    EXPANDED=($(eval echo "$ITER_ARG"))

    if [ ${#EXPANDED[@]} -eq 0 ]; then
        echo "❌ Path not found: $1"
        exit 1
    elif [ ${#EXPANDED[@]} -gt 1 ]; then
        echo "⚠️  Multiple matches found, using first:"
        for path in "${EXPANDED[@]}"; do
            echo "    - $path"
        done | head -5
        ITER_DIR="${EXPANDED[0]}"
    else
        ITER_DIR="${EXPANDED[0]}"
    fi

    # Convert to absolute path
    if [ -d "$ITER_DIR" ]; then
        ITER_DIR="$(cd "$ITER_DIR" && pwd)"
        echo "Using specified iteration: $ITER_DIR"
    else
        echo "❌ Directory not found: $ITER_DIR"
        exit 1
    fi
else
    # Auto-find latest iteration
    ITER_DIR=$(find runs/*/logs/run_*/game_logs/iter_* -type d 2>/dev/null | sort -r | head -1)

    if [ -z "$ITER_DIR" ]; then
        echo "❌ No iteration directories found"
        echo ""
        show_usage
        exit 1
    fi

    ITER_DIR="$(cd "$ITER_DIR" && pwd)"
    echo "Auto-detected latest iteration: $ITER_DIR"
fi

# Find all game log files
GAME_LOGS=($(ls $ITER_DIR/game*.log 2>/dev/null | sort -V))

if [ ${#GAME_LOGS[@]} -eq 0 ]; then
    echo "❌ No game*.log files found in $ITER_DIR"
    exit 1
fi

echo "Found ${#GAME_LOGS[@]} game log files"
echo ""

# Determine GIF option
if [ "$AUTO_YES" = true ]; then
    # Non-interactive mode
    if [ "$WANT_GIF" = true ]; then
        SKIP_GIF=""
        echo "Mode: Non-interactive, generating grids AND GIFs"
    else
        SKIP_GIF="--skip-gif"
        echo "Mode: Non-interactive, generating grids only"
    fi
else
    # Interactive mode - ask for options
    echo "Options:"
    echo "  1) Grid only (fast)"
    echo "  2) Grid + GIF (slower but complete)"
    read -p "Select option (1/2): " -n 1 -r
    echo

    SKIP_GIF="--skip-gif"
    if [[ $REPLY == "2" ]]; then
        SKIP_GIF=""
        echo "Will generate grids AND GIFs"
    else
        echo "Will generate grids only (skipping GIFs)"
    fi

    read -p "Proceed with ${#GAME_LOGS[@]} games? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "Starting batch visualization..."
echo "========================================="

# Process each game
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!GAME_LOGS[@]}"; do
    GAME_LOG="${GAME_LOGS[$i]}"
    GAME_NUM=$((i + 1))
    GAME_NAME=$(basename "$GAME_LOG" .log)

    echo ""
    echo "[$GAME_NUM/${#GAME_LOGS[@]}] Visualizing $GAME_NAME..."

    # Output directory: same location as log, named game1_vis, game2_vis, etc.
    OUTPUT_DIR="${ITER_DIR}/${GAME_NAME}_vis"

    # Find the vis_from_log.py script - first check in experiment directory
    # Traverse up from ITER_DIR to find the experiment root (where vis_from_log.py lives)
    # ITER_DIR = runs/experiment/logs/run_xxx/game_logs/iter_44
    # We need: runs/experiment
    EXPERIMENT_DIR="$ITER_DIR"
    for _ in 1 2 3 4; do
        EXPERIMENT_DIR=$(dirname "$EXPERIMENT_DIR")
    done

    if [ -f "$EXPERIMENT_DIR/vis_from_log.py" ]; then
        VIS_SCRIPT="$EXPERIMENT_DIR/vis_from_log.py"
    elif [ -f "vis_from_log.py" ]; then
        VIS_SCRIPT="vis_from_log.py"
    else
        echo "  ✗ vis_from_log.py not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Run visualization from the experiment directory
    pushd "$EXPERIMENT_DIR" > /dev/null 2>&1

    python vis_from_log.py "$GAME_LOG" \
        --game 1 \
        --output-dir "$OUTPUT_DIR" \
        $SKIP_GIF \
        2>&1 | tail -5

    popd > /dev/null 2>&1

    if [ -f "$OUTPUT_DIR"/*_grid.png ] 2>/dev/null; then
        echo "  ✓ Success: $OUTPUT_DIR"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ✗ Failed: $GAME_NAME"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "========================================="
echo "Batch Visualization Complete!"
echo "========================================="
echo "Success: $SUCCESS_COUNT / ${#GAME_LOGS[@]}"
echo "Failed:  $FAIL_COUNT / ${#GAME_LOGS[@]}"
echo ""
echo "Results saved in: $ITER_DIR"
echo ""
echo "View outputs:"
ls -d $ITER_DIR/game*_vis 2>/dev/null | while read dir; do
    echo "  - $(basename $dir)"
done | head -10

if [ ${#GAME_LOGS[@]} -gt 10 ]; then
    echo "  ... and $((${#GAME_LOGS[@]} - 10)) more"
fi

echo ""
echo "Generated files:"
echo "  Grid images:  ls $ITER_DIR/game*_vis/*_grid.png"
if [ -z "$SKIP_GIF" ]; then
    echo "  GIF animations: ls $ITER_DIR/game*_vis/*.gif"
fi
