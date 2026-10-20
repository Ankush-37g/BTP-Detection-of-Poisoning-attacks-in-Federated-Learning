#!/bin/bash
CONFIG="experiments/configs/attack_sign_flip.yaml"
# Create a config for sign flip
mkdir -p experiments/configs
cat << 'CONFIG_EOF' > $CONFIG
dataset: mnist
model: lenet5
num_clients: 10
clients_per_round: 10
local_epochs: 2
batch_size: 32
lr: 0.01
momentum: 0.9
weight_decay: 0.0001

# Partitioning
partition: noniid
alpha: 0.5

# Attack setup
attack: sign_flip
malicious_fraction: 0.3

# Experiment output
results_dir: experiments/results
CONFIG_EOF

DEFENSES=("fedcvg" "fedcc" "proposed")
for DEFENSE in "${DEFENSES[@]}"; do
    echo "Running $DEFENSE under sign-flip attack..."
    .venv/bin/python experiments/run_fl.py --config $CONFIG --defense $DEFENSE --num_rounds 10 --seed 42
done
.venv/bin/python experiments/generate_plots.py --attack sign_flip
