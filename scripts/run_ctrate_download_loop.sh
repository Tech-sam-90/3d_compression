#!/bin/bash
# Auto-retry wrapper for download_ctrate.py — restarts on any non-zero exit
# (e.g. the transient Lustre "Bus error"/[Errno 19] No such device crash seen
# at 2026-07-30 22:11:10) instead of sitting dead until someone notices.
cd ~/3d_compression || exit 1

while true; do
    # train_fixed finished (47149/47149, 0 failed) as of 2026-07-31 ~12:53.
    # Only valid_fixed remains -- --skip-space-check because the flat 14TB
    # threshold was sized for a from-scratch download; post-train_fixed free
    # space (~12.7TB) is far more than the ~450GB valid_fixed still needs.
    python scripts/download_ctrate.py --split valid_fixed --workers 8 --skip-space-check
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "DOWNLOAD FINISHED SUCCESSFULLY"
        break
    fi
    echo "Exited with status $status -- retrying in 60s..."
    sleep 60
done
