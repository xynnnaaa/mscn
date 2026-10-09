#!/usr/bin/env bash
# Dataset -> batch size: serial; all learning rates within a batch: parallel.
# Usage: bash run_test_mixture.sh [--dry-run]
# Set PYTHON to the interpreter containing the training dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"
LRS=(0.00005 0.0001 0.0002 0.0005 0.001 0.002)
BS=(1024 128)
DB=("imdb" "genome")
GPUS=(2 3 0 1)

DRY_RUN=0
if [[ $# -eq 1 && $1 == --dry-run ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

# Check every output before launching any training, to avoid overwriting runs.
for db in "${DB[@]}"; do
    root="$SCRIPT_DIR/data/mixure_model/$db"
    [[ -f "$root/config_single_mixture.json" ]] || { echo "Missing config: $root/config_single_mixture.json" >&2; exit 1; }
    for bs in "${BS[@]}"; do
        for lr in "${LRS[@]}"; do
            for output in "$root/runfile/single-lr${lr}-bs${bs}.log" \
                          "$root/model/single-${lr}-${bs}.pt" \
                          "$root/model/single-${lr}-${bs}.pt.config.json"; do
                if [[ $DRY_RUN -eq 0 && -e "$output" ]]; then
                    echo "Output already exists; move it or change output names before rerunning: $output" >&2
                    exit 1
                fi
            done
        done
    done
done

pids=()
stop_jobs() {
    trap - INT TERM
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    fi
    exit 130
}
trap stop_jobs INT TERM

for db in "${DB[@]}"; do
    root="$SCRIPT_DIR/data/mixure_model/$db"
    if [[ $DRY_RUN -eq 0 ]]; then
        mkdir -p "$root/runfile" "$root/model"
    fi
    for bs in "${BS[@]}"; do
        echo "Starting dataset=$db batch=$bs; learning rates run in parallel."
        pids=()
        names=()
        for i in "${!LRS[@]}"; do
            lr="${LRS[$i]}"
            gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
            log="$root/runfile/single-lr${lr}-bs${bs}.log"
            cmd=("$PYTHON" -u "$SCRIPT_DIR/train.py"
                 --config "$root/config_single_mixture.json"
                 --lr "$lr" --batch "$bs"
                 --model-output-path "$root/model/single-${lr}-${bs}.pt")
            printf '  GPU=%s ' "$gpu"
            printf '%q ' "${cmd[@]}"
            printf '> %q 2>&1\n' "$log"
            if [[ $DRY_RUN -eq 0 ]]; then
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1 &
                pids+=("$!")
                names+=("$db lr=$lr batch=$bs (log: $log)")
            fi
        done

        failed=0
        for i in "${!pids[@]}"; do
            if wait "${pids[$i]}"; then
                echo "Finished: ${names[$i]}"
            else
                status=$?
                echo "FAILED (exit $status): ${names[$i]}" >&2
                failed=1
            fi
        done
        pids=()
        if [[ $failed -ne 0 ]]; then
            echo "Batch group failed; stopping before the next group." >&2
            exit 1
        fi
        echo "Completed dataset=$db batch=$bs"
    done
done
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete; no training was started."
else
    echo "All 48 training runs completed."
fi
