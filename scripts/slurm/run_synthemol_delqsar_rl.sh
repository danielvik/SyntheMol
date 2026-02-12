#!/bin/bash

#SBATCH --job-name=synthemol-delqsar-rl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=normal
#SBATCH --nodelist=dkcn-papp-nudk2
####### --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

source ~/.bashrc
conda activate synthemol-v2

echo "Job: $SLURM_JOB_NAME (ID: ${SLURM_JOB_ID:-manual})"

base_dir=/compchem/arc/users/dvik/repos/SyntheMol
workdir=${base_dir}
JOB_ID=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}

tmp_dir=/scratch/arc/${JOB_ID}
run_root=${RUN_ROOT:-${base_dir}/runs}
save_dir=${SAVE_DIR:-${run_root}/delqsar_real_rl_${JOB_ID}}

mkdir -p "${tmp_dir}" "${run_root}" "${save_dir}" "${workdir}/logs"
cd "${workdir}"

# Defaults (override via env vars)
SCORE_MODEL_PATH=${SCORE_MODEL_PATH:-/compchem/arc/users/dvik/repos/del-deseq2/delqsar_outputs/model_checkpoints/155486_6653_fine_tuning/best-epoch=5-val_loss=0.45.ckpt}
BUILDING_BLOCKS_PATH=${BUILDING_BLOCKS_PATH:-/compchem/arc/users/dvik/repos/SyntheMol/delqsar_real_preds.csv}
BUILDING_BLOCKS_SCORE_COLUMN=${BUILDING_BLOCKS_SCORE_COLUMN:-pred_0}
N_ROLLOUT=${N_ROLLOUT:-1000}
CHEMICAL_SPACES=${CHEMICAL_SPACES:-real}
RL_PREDICTION_TYPES=${RL_PREDICTION_TYPES:-regression}
STATUS_LOG_FREQUENCY=${STATUS_LOG_FREQUENCY:-10}

if [[ ! -f "${SCORE_MODEL_PATH}" ]]; then
  echo "Missing SCORE_MODEL_PATH: ${SCORE_MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${BUILDING_BLOCKS_PATH}" ]]; then
  echo "Missing BUILDING_BLOCKS_PATH: ${BUILDING_BLOCKS_PATH}"
  exit 1
fi

echo "Parameters:"
echo "SCORE_MODEL_PATH: ${SCORE_MODEL_PATH}"
echo "BUILDING_BLOCKS_PATH: ${BUILDING_BLOCKS_PATH}"
echo "BUILDING_BLOCKS_SCORE_COLUMN: ${BUILDING_BLOCKS_SCORE_COLUMN}"
echo "N_ROLLOUT: ${N_ROLLOUT}"
echo "CHEMICAL_SPACES: ${CHEMICAL_SPACES}"
echo "RL_PREDICTION_TYPES: ${RL_PREDICTION_TYPES}"
echo "STATUS_LOG_FREQUENCY: ${STATUS_LOG_FREQUENCY}"
echo "SAVE_DIR: ${save_dir}"

start_time=$(date +%s)

synthemol \
  --search_type rl \
  --score_types chemprop \
  --score_model_paths "${SCORE_MODEL_PATH}" \
  --chemical_spaces "${CHEMICAL_SPACES}" \
  --building_blocks_paths "${BUILDING_BLOCKS_PATH}" \
  --building_blocks_score_columns "${BUILDING_BLOCKS_SCORE_COLUMN}" \
  --save_dir "${save_dir}" \
  --n_rollout "${N_ROLLOUT}" \
  --rl_model_type chemprop \
  --rl_prediction_types "${RL_PREDICTION_TYPES}" \
  --chemprop_version v2 \
  --use_gpu \
  --status_log_frequency "${STATUS_LOG_FREQUENCY}"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds"

# Archive slurm output and script
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  slurm_out=${workdir}/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
  if [[ -f "${slurm_out}" ]]; then
    cp "${slurm_out}" "${tmp_dir}/"
  fi
fi
cp "$0" "${tmp_dir}/"
cp -r "${tmp_dir}" "${save_dir}/"

echo "Run complete. Outputs in: ${save_dir}"

# Submit with:
# sbatch scripts/slurm/run_synthemol_delqsar_rl.sh
