#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


set -e
set -u
set -o pipefail

FOUND_CHANGED_FILES=0
CHANGE_IN_SKIP_DIR=0
CHANGE_IN_OTHER_DIR=0
SKIP_DIRS="docs/ docker/"

changed_files=`git diff --no-commit-id --name-only -r origin/main`

for file in $changed_files; do
    FOUND_CHANGED_FILES=1
    CHANGE_IN_SKIP_DIR=0
    for dir in $SKIP_DIRS; do
        if grep -q "$dir" <<< "$file"; then
            >&2 echo "[Skip] $file"
            CHANGE_IN_SKIP_DIR=1
            break
        fi
    done
    if [ ${CHANGE_IN_SKIP_DIR} -eq 0 ]; then
        >&2 echo "[Non-Skip] $file...break"
        CHANGE_IN_OTHER_DIR=1
        break
    fi
done

>&2 echo "Change? ${FOUND_CHANGED_FILES}, Non-skip? ${CHANGE_IN_OTHER_DIR}"
if [ ${FOUND_CHANGED_FILES} -eq 0 -o ${CHANGE_IN_OTHER_DIR} -eq 1 ]; then
    # Cannot skip testing if
    # 1) No file change against main branch. This happens when merging to main.
    # 2) One or more changes are in non-skip dirs.
    echo "0"
else
    # Skip testing if all changes are in skip dirs
    >&2 echo "Rest tests can be skipped"
    echo "1"
fi

