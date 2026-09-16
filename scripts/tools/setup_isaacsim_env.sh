#!/usr/bin/env bash
# Install an external, self-contained IsaacSim + IsaacLab runtime for UniLab.
#
# IsaacSim 6 / IsaacLab 3 require Python 3.12 while the main UniLab uv
# environment uses its own interpreter, so this runtime lives outside the repo
# under UNISIM_ISAACSIM_HOME (default: $HOME/.cache/unisim/isaacsim) and is
# located purely through environment variables; this repo never hard-codes
# machine-local paths.
#
# Layout produced by this script:
#   $UNISIM_ISAACSIM_HOME/venv        dedicated Python 3.12 venv
#   $UNISIM_ISAACSIM_HOME/IsaacLab    IsaacLab source tree at ISAACLAB_REF
#   $UNISIM_ISAACSIM_HOME/.markers    per-step completion markers
#   $UNISIM_ISAACSIM_HOME/install.log full install log
#
# The IsaacLab tree pins its own Isaac Sim, torch and Newton versions; the
# install runs IsaacLab's installer (`isaaclab.sh -i isaacsim`) inside the venv
# so those pins are the single source of truth.  Every completed step drops a
# marker, and uv caches downloads, so re-running after a failure resumes.
#
# Usage:
#   bash scripts/tools/setup_isaacsim_env.sh [--verify-sim]
#
# Environment overrides:
#   UNISIM_ISAACSIM_HOME  install root              (default: ~/.cache/unisim/isaacsim)
#   ISAACLAB_REF          IsaacLab commit           (default: develop 4ecd0b036da1)
#   ISAACLAB_SOURCE       local IsaacLab git clone  (optional; avoids the tarball download)
set -euo pipefail

ISAACSIM_HOME=${UNISIM_ISAACSIM_HOME:-$HOME/.cache/unisim/isaacsim}
ISAACLAB_REF=${ISAACLAB_REF:-4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8}
PYTHON_VERSION=3.12
VENV_DIR=$ISAACSIM_HOME/venv
ISAACLAB_DIR=$ISAACSIM_HOME/IsaacLab
MARKER_DIR=$ISAACSIM_HOME/.markers
LOG_FILE=$ISAACSIM_HOME/install.log

VERIFY_SIM=0
for arg in "$@"; do
  case "$arg" in
    --verify-sim) VERIFY_SIM=1 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[setup_isaacsim_env $(date +%H:%M:%S)] $*"; }

run_step() {
  local marker=$1; shift
  if [[ -f $MARKER_DIR/$marker ]]; then
    log "SKIP  $marker (already done)"
    return 0
  fi
  log "START $marker: $*"
  "$@"
  touch "$MARKER_DIR/$marker"
  log "DONE  $marker"
}

mkdir -p "$MARKER_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
log "UNISIM_ISAACSIM_HOME=$ISAACSIM_HOME isaaclab=$ISAACLAB_REF python=$PYTHON_VERSION"

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
# Isaac Sim wheels are multi-GB; uv's default 30 s read timeout aborts slow mirrors.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}

step_venv() {
  command -v uv >/dev/null || { echo "uv is required to create the Python $PYTHON_VERSION venv" >&2; return 1; }
  rm -rf "$VENV_DIR"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
}
run_step 01_venv step_venv

step_isaaclab_source() {
  rm -rf "$ISAACLAB_DIR"
  if [[ -n ${ISAACLAB_SOURCE:-} ]]; then
    git -C "$ISAACLAB_SOURCE" archive --format=tar --prefix=IsaacLab/ "$ISAACLAB_REF" \
      | tar -x -C "$ISAACSIM_HOME"
  else
    local tarball=$ISAACSIM_HOME/isaaclab-$ISAACLAB_REF.tar.gz
    curl -fL -C - --retry 5 --retry-delay 10 -o "$tarball" \
      "https://github.com/isaac-sim/IsaacLab/archive/$ISAACLAB_REF.tar.gz"
    tar -xzf "$tarball" -C "$ISAACSIM_HOME"
    mv "$ISAACSIM_HOME/IsaacLab-$ISAACLAB_REF" "$ISAACLAB_DIR"
    rm -f "$tarball"
  fi
}
run_step 02_isaaclab_source step_isaaclab_source

step_install() {
  (
    cd "$ISAACLAB_DIR"
    env -u PYTHONPATH VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$PATH" TERM=xterm \
      ./isaaclab.sh -i isaacsim
  )
}
run_step 03_isaaclab_install step_install

step_verify() {
  "$VENV_DIR/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda)
from importlib.metadata import version
print("isaacsim", version("isaacsim"), "isaaclab", version("isaaclab"))
EOF
}
run_step 04_verify step_verify

if [[ $VERIFY_SIM -eq 1 ]]; then
  log "Running a bounded headless PhysX smoke test (first Kit launch compiles shaders and caches extensions)"
  "$VENV_DIR/bin/python" - <<'EOF'
from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True}).app

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0", physics=PhysxCfg()))
sim.reset()
for _ in range(10):
    sim.step(render=False)
print("SMOKE TEST OK: PhysX stepped 10 times", flush=True)
app.close()
EOF
fi

cat <<EOF

[setup_isaacsim_env] Done. Add to your shell rc if the runtime is not in the default location:

export UNISIM_ISAACSIM_HOME="$ISAACSIM_HOME"

  venv:     $VENV_DIR
  IsaacLab: $ISAACLAB_DIR
  log:      $LOG_FILE
EOF
