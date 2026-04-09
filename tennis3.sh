#!/bin/bash
#SBATCH --job-name=lioness_rtaware_eval
#SBATCH --account=PAS2942
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=31
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -----------------------
# USER-CONFIGURABLE PATHS
# -----------------------
PROJECT_ROOT=/fs/scratch/PAS2942/Alejandro/networks
OUTDIR=${OUTDIR:-${PROJECT_ROOT}/out/lioness_rtaware_eval}

# Inputs
EXPR=${EXPR:-/fs/scratch/PAS2942/Alejandro/RNA/TCGA-PAAD_expression_matrix.tsv}
RT=${RT:-${PROJECT_ROOT}/gene_rt_case_matrix.tsv}

# Optional labels / gene sets
SUBTYPES=${SUBTYPES:-${PROJECT_ROOT}/subtypes_case_mapped.tsv}
MOFFITT_BASAL=${MOFFITT_BASAL:-${PROJECT_ROOT}/Basal.txt}
MOFFITT_CLASSICAL=${MOFFITT_CLASSICAL:-${PROJECT_ROOT}/Classical.txt}

# Optional clinical (leave empty to disable)
CLIN=${CLIN:-}
CLIN_TIME=${CLIN_TIME:-OS_time}
CLIN_EVENT=${CLIN_EVENT:-OS_event}
CLIN_SAMPLE_COL=${CLIN_SAMPLE_COL:-sample}

# Script path
LIONESS_PY=${LIONESS_PY:-${PROJECT_ROOT}/lioness5.py}

# -----------------------
# PARAMETERS
# -----------------------
ABS_MIN=${ABS_MIN:-0.2}
TOPK_GLOBAL=${TOPK_GLOBAL:-5000}
TOPK_PLOT=${TOPK_PLOT:-500}
MIN_NON_NA=${MIN_NON_NA:-5}
SEED=${SEED:-1337}
RT_MIX=${RT_MIX:-1.0,0.5,0.5}

RT_SHUFFLES=${RT_SHUFFLES:-100}
GRID_MIX=${GRID_MIX:-0.5:1.5:0.25,0:1:0.25,0:1:0.25}
PARTIAL_CORR=${PARTIAL_CORR:-1}
STAB_BOOT=${STAB_BOOT:-50}
STAB_FRAC=${STAB_FRAC:-0.8}
STAB_TOPK=${STAB_TOPK:-10000}
RT_THRESHOLD=${RT_THRESHOLD:-0.0}

# Subtype column configuration
SUBTYPE_COL=${SUBTYPE_COL:-Subtype}
SUBTYPE_POS=${SUBTYPE_POS:-Classical}
SUBTYPE_NEG=${SUBTYPE_NEG:-Basal-like}
SUBTYPE_SAMPLE_COL=${SUBTYPE_SAMPLE_COL:-Case_ID}

# -----------------------
# ENV SETUP
# -----------------------
module purge
module load miniconda3/24.1.2-py310

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnaseq

PY=$(which python)
echo "[PYTHON] ${PY}"

python - <<'PYTEST'
import matplotlib, numpy, pandas, networkx
print("Imports OK")
PYTEST

need() {
  [[ -s "$1" ]] || { echo "Missing or empty: $1" >&2; exit 2; }
}

need "${LIONESS_PY}"
need "${EXPR}"
need "${RT}"
need "${SUBTYPES}"

mkdir -p "${OUTDIR}" logs

# -----------------------
# ARGUMENTS
# -----------------------
args=(
  --expr "${EXPR}"
  --rt "${RT}"
  --outdir "${OUTDIR}"
  --top_k_edges_global "${TOPK_GLOBAL}"
  --top_k_edges_plot "${TOPK_PLOT}"
  --min_non_na "${MIN_NON_NA}"
  --seed "${SEED}"
  --rt_mix "${RT_MIX}"
  --rt_shuffles "${RT_SHUFFLES}"
  --partial_corr "${PARTIAL_CORR}"
  --stability_boot "${STAB_BOOT}"
  --stability_frac "${STAB_FRAC}"
  --stability_topk "${STAB_TOPK}"
  --rt_threshold "${RT_THRESHOLD}"
  --subtype_tsv "${SUBTYPES}"
  --subtype_col "${SUBTYPE_COL}"
  --subtype_pos "${SUBTYPE_POS}"
  --subtype_neg "${SUBTYPE_NEG}"
  --subtype_sample_col "${SUBTYPE_SAMPLE_COL}"
)

# Moffitt gene sets
if [[ -s "${MOFFITT_BASAL}" && -s "${MOFFITT_CLASSICAL}" ]]; then
  args+=( --moffitt_basal "${MOFFITT_BASAL}" --moffitt_classical "${MOFFITT_CLASSICAL}" )
fi

# Thresholding / grid search
[[ -n "${ABS_MIN}" ]] && args+=( --abs_min "${ABS_MIN}" )
[[ -n "${GRID_MIX}" ]] && args+=( --grid_mix "${GRID_MIX}" )

# Clinical
if [[ -n "${CLIN}" && -s "${CLIN}" ]]; then
  args+=( --clinical_tsv "${CLIN}" --clinical_time "${CLIN_TIME}" --clinical_event "${CLIN_EVENT}" )
  [[ -n "${CLIN_SAMPLE_COL}" ]] && args+=( --clinical_sample_col "${CLIN_SAMPLE_COL}" )
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-31}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-31}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-31}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-31}"
export PYTHONUNBUFFERED=1

echo "[EXPR] ${EXPR}"
echo "[RT] ${RT}"
echo "[SUBTYPES] ${SUBTYPES}"
echo "[OUTDIR] ${OUTDIR}"

set -x
srun -u "${PY}" "${LIONESS_PY}" "${args[@]}"
set +x

echo "[OK] Outputs (first 200):"
find "${OUTDIR}" -maxdepth 2 -type f | sed "s|${OUTDIR}/||" | sort | head -n 200
