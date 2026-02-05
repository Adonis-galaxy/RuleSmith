#!/bin/bash
# ============================================================================
# RuleSmith Local Evaluation Script
# ============================================================================
#
# Evaluates game balance with specified models and theta parameters.
#
# Usage:
#   ./run_eval.sh                    # Run with settings below
#   ./run_eval.sh --mock             # Test with mock LLM (no GPU)
#
# For SLURM cluster, use: sbatch run_submit.sh with --eval flag
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Models: 2b, 4b, 8b, or full HuggingFace path
EMPIRE_MODEL="8b"
NOMADS_MODEL="8b"

# Game settings
N_GAMES=100
MAX_TURNS=16
NUM_GPUS=8  # Number of GPUs to use

# Output directory for game logs (leave empty to skip logging)
OUTPUT_DIR="logs/eval_results"

# ============================================================================
# THETA FILE - Path to theta JSON file (optional)
# ============================================================================
# Set to empty string "" to use default theta parameters
# Example: THETA_FILE="examples/theta.json"
THETA_FILE="examples/theta.json"
# ============================================================================
# END CONFIGURATION
# ============================================================================

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)
            EXTRA_ARGS="$EXTRA_ARGS --mock"
            echo "Running in MOCK mode (no real LLM)"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p logs
[ -n "$OUTPUT_DIR" ] && mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "RuleSmith Evaluation"
echo "============================================================"
echo "Date: $(date)"
echo ""
echo "Configuration:"
echo "  Empire Model:  $EMPIRE_MODEL"
echo "  Nomads Model:  $NOMADS_MODEL"
echo "  N Games:       $N_GAMES"
echo "  Max Turns:     $MAX_TURNS"
echo "  Num GPUs:      $NUM_GPUS"
echo "  Output Dir:    ${OUTPUT_DIR:-"(no logging)"}"

THETA_ARG=""
if [ -n "$THETA_FILE" ]; then
    if [ ! -f "$THETA_FILE" ]; then
        echo "ERROR: Theta file not found: $THETA_FILE"
        exit 1
    fi
    echo "  Theta File:    $THETA_FILE"
    THETA_ARG="--theta $THETA_FILE"
else
    echo "  Theta:         Default"
fi

echo "============================================================"
echo ""

# Build command
CMD="python eval_theta.py"
CMD="$CMD --empire-model $EMPIRE_MODEL"
CMD="$CMD --nomads-model $NOMADS_MODEL"
CMD="$CMD --n-games $N_GAMES"
CMD="$CMD --max-turns $MAX_TURNS"
CMD="$CMD --num-gpus $NUM_GPUS"
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD $THETA_ARG"
CMD="$CMD $EXTRA_ARGS"

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "Evaluation completed!"
[ -n "$OUTPUT_DIR" ] && echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
