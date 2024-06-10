#!/bin/bash


ENV_NAME="mne"


# Update conda
conda update --name=base conda

# Create a new environment
conda create --channel=conda-forge --strict-channel-priority --name=$ENV_NAME python=3.12 mne

# Activate the environment
conda activate $ENV_NAME