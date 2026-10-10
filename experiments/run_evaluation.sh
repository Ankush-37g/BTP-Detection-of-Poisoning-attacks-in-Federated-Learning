#!/bin/bash

# run_evaluation.sh
# Runs a full evaluation suite for the BTP project.
# Compares FedAvg, Krum, FedCVG, FedCC, and the Proposed Defense under Scaling Attack.

set -e

echo "==========================================================="
echo " Starting Full BTP Evaluation Suite"
echo "==========================================================="

# Attack configuration
CONFIG="experiments/configs/attack_scaling.yaml"
ROUNDS=10

# Defenses to test
DEFENSES=("fedavg" "krum" "fedcvg" "fedcc" "proposed")

for DEFENSE in "${DEFENSES[@]}"; do
    echo ""
    echo ">>> Running $DEFENSE defense..."
    .venv/bin/python experiments/run_fl.py \
        --config $CONFIG \
        --defense $DEFENSE \
        --num_rounds $ROUNDS \
        --seed 42
    echo "<<< Finished $DEFENSE"
done

echo ""
echo "==========================================================="
echo " Generating Comparison Plots"
echo "==========================================================="

.venv/bin/python experiments/generate_plots.py --attack scaling

echo "All evaluations complete! Check the experiments/plots directory."
