#!/usr/bin/env bash

rsync -avz\
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'wandb' \
  --exclude 'runs' \
  ./ KHJ@143.248.147.70:/home/KHJ/AI611/PreferenceTransformer/
