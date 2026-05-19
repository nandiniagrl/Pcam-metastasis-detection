#!/usr/bin/env bash
# Helper for downloading PCam .h5 files.
#
# The official PCam release is hosted on Google Drive:
#   https://github.com/basveeling/pcam
# Google Drive does not support clean wget/curl downloads for large files;
# we therefore document the expected layout and let the user place the files.
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

cat <<EOF

PCam dataset setup
==================

Place the following six files in $DATA_DIR/ :

  camelyonpatch_level_2_split_train_x.h5
  camelyonpatch_level_2_split_train_y.h5
  camelyonpatch_level_2_split_valid_x.h5
  camelyonpatch_level_2_split_valid_y.h5
  camelyonpatch_level_2_split_test_x.h5
  camelyonpatch_level_2_split_test_y.h5

You can obtain them from:
  https://github.com/basveeling/pcam

After downloading, verify with:
  python -m src.data.check_dataset --data-root $DATA_DIR

EOF
