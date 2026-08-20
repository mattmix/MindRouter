#!/usr/bin/env bash
############################################################
# mindrouter - standalone GPU DLP microservice installer
#
# Idempotent. Creates a NODE-LOCAL venv + HuggingFace cache
# (never on shared ceph), installs the Python deps, renders
# and installs the systemd unit, and enables it.
#
# The CUDA torch build is NOT installed automatically — the
# correct wheel depends on the node's CUDA/driver. See the
# "MANUAL STEP" note below; run it once before first start.
#
# Usage:
#   sudo DLP_SERVICE_KEY=... ./install.sh
# or edit the values below and run as a user with sudo.
############################################################
set -euo pipefail

# ---- values to adjust (node-local paths!) --------------------------------
USER_NAME="${USER_NAME:-sheneman}"
INSTALL_DIR="${INSTALL_DIR:-/opt/dlp_service}"      # node-local disk
VENV_DIR="${VENV_DIR:-${INSTALL_DIR}/venv}"          # node-local disk
HF_HOME_DIR="${HF_HOME_DIR:-${INSTALL_DIR}/hf_cache}"  # node-local disk (NOT ceph)
ENV_DIR="${ENV_DIR:-/etc/dlp_service}"
ENV_FILE="${ENV_FILE:-${ENV_DIR}/dlp.env}"
SERVICE_NAME="dlp-service"
# Fleet convention (MindRouter / sidecars / vLLM): a python3.11 -m venv.
# Rocky 8 ships 3.6 as python3; install python3.11 first (dnf install python3.11)
# and this uses it. Override PYTHON_BIN for other layouts.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

# torch CUDA wheel index (adjust to the node's CUDA runtime; cu121 for CUDA 12.x)
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

# Directory containing this install.sh (…/dlp_service/deploy)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "${SCRIPT_DIR}")"                 # …/dlp_service
REPO_DIR="$(dirname "${PKG_DIR}")"                   # repo root (has dlp_service/)
UNIT_TEMPLATE="${SCRIPT_DIR}/dlp-service.service"

echo "==> install dir : ${INSTALL_DIR}"
echo "==> venv        : ${VENV_DIR}"
echo "==> hf cache    : ${HF_HOME_DIR}"
echo "==> env file    : ${ENV_FILE}"
echo "==> repo source : ${REPO_DIR}"

# ---- node-local dirs -----------------------------------------------------
sudo mkdir -p "${INSTALL_DIR}" "${HF_HOME_DIR}" "${ENV_DIR}"
sudo chown -R "${USER_NAME}:${USER_NAME}" "${INSTALL_DIR}"

# ---- sync the package onto node-local disk (idempotent) ------------------
# Ship the whole repo so `python -m dlp_service` resolves the package.
if command -v rsync >/dev/null 2>&1; then
  sudo -u "${USER_NAME}" rsync -a --delete \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    "${REPO_DIR}/dlp_service" "${INSTALL_DIR}/"
else
  sudo -u "${USER_NAME}" cp -a "${REPO_DIR}/dlp_service" "${INSTALL_DIR}/"
fi

# ---- venv (node-local) ---------------------------------------------------
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "==> creating venv"
  sudo -u "${USER_NAME}" "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
sudo -u "${USER_NAME}" "${VENV_DIR}/bin/python" -m pip install --upgrade pip

# ---- MANUAL STEP: CUDA torch ---------------------------------------------
# The default PyPI torch wheel is CPU-only. Install the CUDA build FIRST so
# gliner binds against it. This is intentionally explicit — pick the index
# URL that matches the node's CUDA runtime (cu121 for CUDA 12.x, driver 565):
#
#   sudo -u ${USER_NAME} ${VENV_DIR}/bin/pip install torch \
#       --index-url ${TORCH_CUDA_INDEX}
#
if ! sudo -u "${USER_NAME}" "${VENV_DIR}/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "==> installing CUDA torch from ${TORCH_CUDA_INDEX}"
  sudo -u "${USER_NAME}" "${VENV_DIR}/bin/pip" install torch --index-url "${TORCH_CUDA_INDEX}"
else
  echo "==> torch already present, leaving it as-is (not forcing the CUDA wheel)"
fi

# ---- remaining deps ------------------------------------------------------
sudo -u "${USER_NAME}" "${VENV_DIR}/bin/pip" install -r "${INSTALL_DIR}/dlp_service/requirements.txt"

# ---- env file (create a template once; never overwrite an existing one) --
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "==> writing env template ${ENV_FILE} (fill in DLP_SERVICE_KEY)"
  sudo tee "${ENV_FILE}" >/dev/null <<EOF
# DLP GPU service environment — chmod 600, holds the shared secret.
DLP_SERVICE_KEY=${DLP_SERVICE_KEY:-CHANGE_ME}
DLP_SERVICE_HOST=127.0.0.1
DLP_SERVICE_PORT=8710
DLP_DEVICE=cuda:0
DLP_REPLICAS=2
DLP_MAX_BATCH=32
DLP_BATCH_WINDOW_MS=8
DLP_MAX_QUEUE=512
DLP_MAX_CHARS_CAP=10000
DLP_TORCH_THREADS=4
DLP_HF_HOME=${HF_HOME_DIR}
EOF
  sudo chown "${USER_NAME}:${USER_NAME}" "${ENV_FILE}"
  sudo chmod 600 "${ENV_FILE}"
else
  echo "==> env file ${ENV_FILE} already exists, leaving it untouched"
fi

# ---- render + install systemd unit ---------------------------------------
RENDERED="$(mktemp)"
sed -e "s#__INSTALL_DIR__#${INSTALL_DIR}#g" \
    -e "s#__VENV_DIR__#${VENV_DIR}#g" \
    -e "s#__ENV_FILE__#${ENV_FILE}#g" \
    -e "s#__USER__#${USER_NAME}#g" \
    "${UNIT_TEMPLATE}" > "${RENDERED}"
sudo cp "${RENDERED}" "/etc/systemd/system/${SERVICE_NAME}.service"
rm -f "${RENDERED}"

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"

echo "==> installed. Set DLP_SERVICE_KEY in ${ENV_FILE}, then:"
echo "    sudo systemctl restart ${SERVICE_NAME} && systemctl status ${SERVICE_NAME}"
echo "    curl -s http://127.0.0.1:8710/healthz"
