#!/bin/bash
# ============================================================================
# RuleSmith Local Run Script
# ============================================================================
#
# Usage:
#   ./run.sh                    # Run with default settings
#   ./run.sh --mock             # Quick test with mock LLM
#   ./run.sh --eval             # Run evaluation only
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# PARAMETERS - Edit these for your experiment
# ============================================================================
RUN_NAME="exp_8b_vs_8b"                           # Experiment name (outputs to runs/$RUN_NAME/)
EMPIRE_MODEL="OpenGVLab/InternVL3_5-8B"           # Model for Empire
NOMADS_MODEL="OpenGVLab/InternVL3_5-8B"           # Model for Nomads
N_ITERATIONS=100                                   # Optimization iterations
N_GAMES=16                                         # Games per evaluation (or max if adaptive)
METHOD=bayesian                                    # bayesian, evolution, or random
MAX_TURNS=16                                       # Max turns per game
NUM_GPUS=8                                         # Number of GPUs to use
CHECKPOINT_INTERVAL=1                              # Save checkpoint every N iterations

# Adaptive sampling: fewer games early (exploration), more games late (exploitation)
ADAPTIVE_GAMES=true                                # Enable adaptive games
MIN_GAMES=16                                       # Early iterations: fewer games
MAX_GAMES=64                                       # Late iterations: more games
ADAPTIVE_STRATEGY=acquisition                      # linear, uncertainty, or acquisition (BO only)
BALANCE_THRESHOLD=0.1                              # Log iterations with balance_score <= threshold
# ============================================================================

# Parse command line arguments
MOCK_MODE=""
EVAL_MODE=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)
            MOCK_MODE="--mock"
            echo "Running in MOCK mode (no real LLM)"
            shift
            ;;
        --eval)
            EVAL_MODE=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Setup output directory
OUTPUT_DIR="runs/$RUN_NAME"
mkdir -p "$OUTPUT_DIR/logs"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/logs/run_${TIMESTAMP}.log"

echo "========================================="
echo "RuleSmith Local Run"
echo "========================================="
echo "Start Time: $(date)"
echo "Run Name: $RUN_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo ""

if [ "$EVAL_MODE" = true ]; then
    # Run evaluation
    echo "Running EVALUATION mode"
    echo "========================================="
    echo "  Empire: $EMPIRE_MODEL"
    echo "  Nomads: $NOMADS_MODEL"
    echo "  Games: $N_GAMES"
    echo "  Max Turns: $MAX_TURNS"
    echo "========================================="

    python eval_theta.py \
        --empire-model "$EMPIRE_MODEL" \
        --nomads-model "$NOMADS_MODEL" \
        --n-games $N_GAMES \
        --max-turns $MAX_TURNS \
        --num-gpus $NUM_GPUS \
        --output-dir "$OUTPUT_DIR/eval" \
        $MOCK_MODE \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
else
    # Run optimization
    echo "Running OPTIMIZATION mode"
    echo "========================================="
    echo "  Method: $METHOD"
    echo "  Iterations: $N_ITERATIONS"
    if [ "$ADAPTIVE_GAMES" = true ]; then
        echo "  Games/eval: $MIN_GAMES -> $MAX_GAMES (adaptive, $ADAPTIVE_STRATEGY)"
    else
        echo "  Games/eval: $N_GAMES"
    fi
    echo "  Empire: $EMPIRE_MODEL"
    echo "  Nomads: $NOMADS_MODEL"
    echo "  GPUs: $NUM_GPUS"
    echo "========================================="

    # Build adaptive games flags
    ADAPTIVE_FLAGS=""
    if [ "$ADAPTIVE_GAMES" = true ]; then
        ADAPTIVE_FLAGS="--adaptive-games --min-games $MIN_GAMES --max-games $MAX_GAMES --adaptive-strategy $ADAPTIVE_STRATEGY"
    fi

    python optimize_demo.py \
        --verbose-optimize \
        --method $METHOD \
        --n-iterations $N_ITERATIONS \
        --n-games $N_GAMES \
        --max-turns $MAX_TURNS \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --num-gpus $NUM_GPUS \
        --empire-model "$EMPIRE_MODEL" \
        --nomads-model "$NOMADS_MODEL" \
        --balance-threshold $BALANCE_THRESHOLD \
        --log-dir "$OUTPUT_DIR/logs" \
        --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
        --output "$OUTPUT_DIR/optimized_theta.json" \
        $ADAPTIVE_FLAGS \
        $MOCK_MODE \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "========================================="
echo "Run Complete"
echo "========================================="
echo "End Time: $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "========================================="
