#!/usr/bin/env bash
# Open a shell in the onnxim image with THIS checkout bind-mounted over
# /workspace/ONNXim, so edits on the host are what gets compiled and run.
#
#   ./scripts/devshell.sh              # interactive shell
#   ./scripts/devshell.sh <command>    # run one command and exit
#
# The build tree lives in build-docker/ (not build/) so a host-side manual
# build and the container build can coexist without clobbering each other.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the conan cache on the host so rebuilds don't re-download every time.
mkdir -p "${REPO_ROOT}/.conan-cache"

# Only allocate a TTY when we actually have one (so CI / piped use works).
TTY_FLAGS=(-i)
[ -t 0 ] && TTY_FLAGS+=(-t)

exec docker run --rm "${TTY_FLAGS[@]}" \
  -v "${REPO_ROOT}:/workspace/ONNXim" \
  -v "${REPO_ROOT}/.conan-cache:/root/.conan/data" \
  -w /workspace/ONNXim \
  -e ONNXIM_HOME=/workspace/ONNXim \
  onnxim "$@"
