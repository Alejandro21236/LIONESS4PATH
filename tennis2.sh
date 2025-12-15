#!/bin/bash
#SBATCH --job-name=lioness_rtaware_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -----------------------
# USER-CONFIGURABLE PATHS
# -----------------------
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
OUTDIR=${OUTDIR:-${PROJECT_ROOT}/out/lioness_rtaware_eval}

# Inputs (provide your own paths)
EXPR=${EXPR:-${DATA_ROOT}/expr.gene_x_sample.tsv}
RT=${RT:-${DATA_ROOT}/rt.gene_x_sample.tsv}

# Optional labels / gene sets
SUBTYPES=${SUBTYPES:-${DATA_ROOT}/subtypes.csv}
MOFFITT_BASAL=${MOFFITT_BASAL:-${DATA_ROOT}/Basal_ENSG.txt}
MOFFITT_CLASSICAL=${MOFFITT_CLASSICAL:-${DATA_ROOT}/Classical_ENSG.txt}

# Optional clinical (leave empty to disable)
CLIN=${CLIN:-}
CLIN_TIME=${CLIN_TIME:-OS_time}
CLIN_EVENT=${CLIN_EVENT:-OS_event}
CLIN_SAMPLE_COL=${CLIN_SAMPLE_COL:-sample}

# Script path (repo-local by default)
LIONESS_PY=${LIONESS_PY:-${PROJECT_ROOT}/lioness4.py}

# -----------------------
# PARAMETERS
# -----------------------
ABS_MIN=${ABS_MIN:-.5}
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

# Subtype column configuration (edit to match your file)
SUBTYPE_COL=${SUBTYPE_COL:-subtype}
SUBTYPE_POS=${SUBTYPE_POS:-classical}
SUBTYPE_NEG=${SUBTYPE_NEG:-basal}
SUBTYPE_SAMPLE_COL=${SUBTYPE_SAMPLE_COL:-sample_id}

# -----------------------
# ENV SETUP (generic)
# -----------------------
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load miniconda3 || true
fi

if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
  conda activate rnaseq || conda activate base || true
fi

PY="${PYTHON:-$(command -v python3 || command -v python)}"

need(){ [[ -s "$1" ]] || { echo "Missing or empty: $1" >&2; exit 2; }; }

need "${LIONESS_PY}"
need "${EXPR}"
need "${RT}"

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
)

# Moffitt gene sets (optional but expected in your original)
if [[ -s "${MOFFITT_BASAL}" && -s "${MOFFITT_CLASSICAL}" ]]; then
  args+=( --moffitt_basal "${MOFFITT_BASAL}" --moffitt_classical "${MOFFITT_CLASSICAL}" )
fi

# Subtypes (optional)
if [[ -s "${SUBTYPES}" ]]; then
  args+=( --subtype_tsv "${SUBTYPES}"
          --subtype_col "${SUBTYPE_COL}"
          --subtype_pos "${SUBTYPE_POS}"
          --subtype_neg "${SUBTYPE_NEG}" )
  [[ -n "${SUBTYPE_SAMPLE_COL}" ]] && args+=( --subtype_sample_col "${SUBTYPE_SAMPLE_COL}" )
fi

# Thresholding / grid search knobs
[[ -n "${ABS_MIN}" ]] && args+=( --abs_min "${ABS_MIN}" )
[[ -n "${GRID_MIX}" ]] && args+=( --grid_mix "${GRID_MIX}" )

# Clinical (optional)
if [[ -n "${CLIN}" && -s "${CLIN}" ]]; then
  args+=( --clinical_tsv "${CLIN}" --clinical_time "${CLIN_TIME}" --clinical_event "${CLIN_EVENT}" )
  [[ -n "${CLIN_SAMPLE_COL}" ]] && args+=( --clinical_sample_col "${CLIN_SAMPLE_COL}" )
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

set -x
srun -u "${PY}" "${LIONESS_PY}" "${args[@]}"
set +x

echo "[OK] Outputs (first 200):"
find "${OUTDIR}" -maxdepth 2 -type f | sed "s|${OUTDIR}/||" | sort | head -n 200

