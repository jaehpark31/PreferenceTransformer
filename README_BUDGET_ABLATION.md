# Human Feedback Budget Ablation Add-on

This archive keeps the original PreferenceTransformer project layout and adds Section 8 to `AI611_team16.ipynb`.

Added files:

- `AI611_team16.ipynb`: original Sections 1--7 are preserved; Section 8 adds the AntMaze human feedback budget ablation.
- `scripts/train_budget_ablation.py`: portable training/evaluation script used by Section 8.
- `runs/budget_ablation_3seed/summary_3seed_mean_std.csv`: 3-seed mean/std summary.
- `runs/budget_ablation_3seed/merged_3seed_results.csv`: all seed-level results.
- `runs/budget_ablation_3seed/figures/budget_ablation_score.png`: budget-ablation score plot.
- `runs/budget_ablation_3seed/figures/seed1_rollouts.png`: seed-1 rollout visualization.
- `human_label/antmaze-medium-play-v2/`: human preference label files used by the experiment.

The notebook uses relative paths from the project root. Open the notebook after extracting this archive and setting the working directory to the extracted `PreferenceTransformer` folder.

By default, the notebook loads the included results and figures. To re-run training, set `RUN_TRAINING = True` in Section 8. Re-training requires a compatible Python 3.8 environment with PyTorch, D4RL, MuJoCo, and mujoco-py installed.
