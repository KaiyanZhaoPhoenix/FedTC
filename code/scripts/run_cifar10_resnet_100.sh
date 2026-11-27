#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin=${PYTHON:-python}
cfg=${CONFIG:-configs/cifar10_resnet_100r.json}
data=${DATA_ROOT:-"$PWD/dataset"}
raw=${RAW_ROOT:-"$PWD/dataset/Cifar10-raw/rawdata"}
out=${RUN_ROOT:-"$PWD/runs/cifar10_resnet_$(date +%Y%m%d_%H%M%S)_$$"}
base=(main.py --config "$cfg" --data-root "$data")
smoke=(-m diagnostics.smoke_cifar10_entry --config "$cfg")
if [[ -n "${DEVICE:-}" ]]; then
    base+=(--device "$DEVICE")
    smoke+=(--device "$DEVICE")
fi

mode=${1:-check}
case "$mode" in
    check)
        if [[ -d "$data/Cifar10" ]]; then
            "$python_bin" "${base[@]}" --phase check --output "$out/check"
        else
            "$python_bin" "${smoke[@]}" --output "$out/check"
        fi
        exit 0
        ;;
    full) ;;
    *) echo "Usage: bash scripts/run_cifar10_resnet_100.sh [check|full]" >&2; exit 2 ;;
esac

if [[ -d "$data/Cifar10" ]]; then
    "$python_bin" -m dataset.prepare_cifar10 --config "$cfg" --data-root "$data" --verify-only
else
    "$python_bin" -m dataset.prepare_cifar10 --config "$cfg" --data-root "$data" --raw-root "$raw" --download
fi
"$python_bin" "${base[@]}" --phase check --output "$out/check"
"$python_bin" "${base[@]}" --phase dev --output "$out/dev"
"$python_bin" "${base[@]}" --phase held --policy "$out/dev/policy.json" --output "$out/held"
echo "Held results: $out/held/summary.json"
echo "Per-method means and sample variances: $out/held/seed_statistics.csv"
