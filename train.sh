#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. The canonical single-GPU ImageNet script is
# train_caspr_imagenet.sh.
exec bash "$(dirname "$0")/train_caspr_imagenet.sh" "$@"
