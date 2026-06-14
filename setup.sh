#!/usr/bin/env bash
set -euo pipefail

# PreferenceTransformer + D4RL + MuJoCo setup script
# Tested flow:
#   Python 3.8 + CUDA 11.1 + cuDNN 8.2.1
#   JAX/JAXLIB installed first
#   D4RL installed with --no-deps
#   MuJoCo 2.1 binary installed separately
#   mjrl installed separately for D4RL MuJoCo locomotion env registration

ENV_NAME="${ENV_NAME:-offline}"
REPO_DIR="${REPO_DIR:-$HOME/AI611/PreferenceTransformer}"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco}"
MUJOCO_PATH="${MUJOCO_PATH:-$MUJOCO_DIR/mujoco210}"
D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-}"
D4RL_TEST_ENV="${D4RL_TEST_ENV:-antmaze-medium-play-v2}"

if [[ -n "${RUN_SUDO_APT:-}" ]]; then
  run_sudo_apt="$RUN_SUDO_APT"
elif [[ -t 0 ]]; then
  run_sudo_apt="1"
else
  run_sudo_apt="0"
fi

echo "==== PreferenceTransformer setup ===="
echo "ENV_NAME      = $ENV_NAME"
echo "REPO_DIR      = $REPO_DIR"
echo "MUJOCO_PATH   = $MUJOCO_PATH"
echo "D4RL_TEST_ENV = $D4RL_TEST_ENV"
echo "RUN_SUDO_APT  = $run_sudo_apt"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found. Install Miniconda/Anaconda first."
  exit 1
fi

# Make conda activate work in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==== 1. Create conda env if needed ===="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists. Reusing it."
else
  conda create -y -n "$ENV_NAME" python=3.8
fi

conda activate "$ENV_NAME"

echo "==== 2. Install system libraries for mujoco-py ===="
if [[ "$run_sudo_apt" == "1" ]]; then
  sudo apt update
  sudo apt install -y \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    patchelf

  # Ubuntu version compatibility: libgl1-mesa-glx may not exist on some newer distros.
  sudo apt install -y libgl1-mesa-glx || sudo apt install -y libgl1
else
  echo "Skipping apt install because RUN_SUDO_APT=$run_sudo_apt"
fi

echo "==== 3. Install CUDA toolkit and cuDNN inside conda env ===="
conda install -y -c conda-forge cudatoolkit=11.1 cudnn=8.2.1

echo "==== 4. Upgrade base Python packaging tools ===="
python -m pip install --upgrade "pip<24.1" setuptools wheel

echo "==== 5. Install JAX/JAXLIB first ===="
python -m pip uninstall -y jax jaxlib distrax chex optax flax || true
python -m pip install "jax==0.3.25" \
  "jaxlib==0.3.25+cuda11.cudnn82" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "==== 6. Install general Python dependencies ===="
python -m pip install \
  "numpy==1.23.5" \
  "scipy==1.9.3" \
  "absl-py" \
  "gdown" \
  "tqdm" \
  "ml_collections==0.1.1" \
  "tensorboardX==2.1" \
  "tensorflow-probability==0.17.0" \
  "imageio" \
  "imageio-ffmpeg" \
  "pandas==1.5.3" \
  "protobuf==3.20.1" \
  "gym==0.23.1" \
  "ujson==5.8.0" \
  "wandb==0.15.12" \
  "transformers==4.30.2"

echo "==== 7. Install JAX ecosystem dependencies with pinned versions ===="
python -m pip install \
  "chex==0.1.5" \
  "optax==0.1.4" \
  "flax==0.5.3"

# distrax sometimes tries to resolve jaxlib again; --no-deps avoids that
# after jax and jaxlib have already been installed explicitly.
python -m pip install "distrax==0.1.2" --no-deps

echo "==== 8. Install MuJoCo 2.1 binary if missing ===="
mkdir -p "$MUJOCO_DIR"

if [[ -d "$MUJOCO_PATH" ]]; then
  echo "MuJoCo already exists at $MUJOCO_PATH"
else
  cd "$MUJOCO_DIR"
  if [[ ! -f "mujoco210-linux-x86_64.tar.gz" ]]; then
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  fi
  tar -xzf mujoco210-linux-x86_64.tar.gz
fi

if [[ ! -d "$MUJOCO_PATH/bin" ]]; then
  echo "ERROR: MuJoCo bin directory not found at $MUJOCO_PATH/bin"
  exit 1
fi

echo "==== 9. Detect NVIDIA library path ===="
NVIDIA_LIB_DIR=""

if [[ -d /usr/lib/nvidia ]]; then
  NVIDIA_LIB_DIR="/usr/lib/nvidia"
elif compgen -G "/usr/lib/nvidia-*" >/dev/null; then
  NVIDIA_LIB_DIR="$(ls -d /usr/lib/nvidia-* | sort -V | tail -n 1)"
elif [[ -d /usr/lib/wsl/lib ]]; then
  NVIDIA_LIB_DIR="/usr/lib/wsl/lib"
elif [[ -d /usr/lib/x86_64-linux-gnu ]]; then
  NVIDIA_LIB_DIR="/usr/lib/x86_64-linux-gnu"
else
  echo "WARNING: Could not detect NVIDIA lib dir automatically."
  echo "If mujoco_py import fails, add your NVIDIA driver lib path to LD_LIBRARY_PATH."
fi

echo "NVIDIA_LIB_DIR = ${NVIDIA_LIB_DIR:-not_detected}"

if [[ -n "$NVIDIA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$NVIDIA_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
fi
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_PATH"

if [[ -n "$D4RL_DATASET_DIR" ]]; then
  mkdir -p "$D4RL_DATASET_DIR"
  export D4RL_DATASET_DIR="$D4RL_DATASET_DIR"
fi

echo "==== 10. Add MuJoCo env vars to ~/.bashrc if missing ===="
append_once() {
  local line="$1"
  local file="$2"
  grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

append_once "export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210" "$HOME/.bashrc"

if [[ -n "$NVIDIA_LIB_DIR" ]]; then
  # Use the detected concrete path. This avoids the specific mujoco_py error asking for /usr/lib/nvidia.
  append_once "export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:$NVIDIA_LIB_DIR:\$LD_LIBRARY_PATH" "$HOME/.bashrc"
else
  append_once "export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:\$LD_LIBRARY_PATH" "$HOME/.bashrc"
fi

echo "==== 11. Install mujoco-py and related packages ===="
python -m pip install "cython<3" "mujoco-py==2.1.2.14" "h5py==3.8.0" "pybullet==3.2.5" termcolor click

echo "==== 12. Build/test mujoco_py ===="
rm -rf "$HOME/.cache/mujoco_py"
python - <<'PY'
import mujoco_py
print("mujoco_py import ok")
print("mujoco path:", mujoco_py.utils.discover_mujoco())
PY

echo "==== 13. Install D4RL from the repository with --no-deps ===="
if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR not found: $REPO_DIR"
  echo "Clone PreferenceTransformer first or run with REPO_DIR=/path/to/PreferenceTransformer"
  exit 1
fi

cd "$REPO_DIR/d4rl"
python -m pip install -e . --no-deps
cd "$REPO_DIR"

echo "==== 14. Install mjrl for D4RL MuJoCo locomotion env registration ===="
python -m pip install "git+https://github.com/aravindr93/mjrl.git@master#egg=mjrl" --no-deps

echo "==== 15. Final import and dataset test ===="
python - <<'PY'
import jax
print("jax:", jax.__version__)
print("jax devices:", jax.devices())

import mujoco_py
print("mujoco_py ok:", mujoco_py.utils.discover_mujoco())

import mjrl
print("mjrl import ok")

import os

import gym
import d4rl

env_name = os.environ.get("D4RL_TEST_ENV", "antmaze-medium-play-v2")
env = gym.make(env_name)
dataset = env.get_dataset()

print("D4RL env ok:", env_name)
print("dataset keys:", dataset.keys())
print("observations:", dataset["observations"].shape)
print("actions:", dataset["actions"].shape)
PY

echo
echo "==== Setup complete ===="
echo "Activate later with:"
echo "  conda activate $ENV_NAME"
echo
echo "Run PreferenceTransformer example:"
echo "  cd \"$REPO_DIR\""
cat <<'CMD'
  CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main \
    --use_human_label True \
    --comment test_antmaze_medium_play_pt \
    --transformer.embd_dim 256 \
    --transformer.n_layer 1 \
    --transformer.n_head 4 \
    --env antmaze-medium-play-v2 \
    --logging.output_dir './logs/pref_reward' \
    --batch_size 256 \
    --num_query 1000 \
    --query_len 100 \
    --n_epochs 10000 \
    --skip_flag 0 \
    --seed 42 \
    --model_type PrefTransformer
CMD
