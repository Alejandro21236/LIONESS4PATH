#!/bin/bash
#SBATCH --job-name=lioness_morph_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=64:00:00
#SBATCH --output=job.out
#SBATCH --error=job.err

set -euo pipefail

# =========================
# USER-CONFIGURABLE PATHS
# =========================
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}

# Network / output roots
NETWORKS=${NETWORKS:-${PROJECT_ROOT}/networks}
PRODIR=${PRODIR:-${NETWORKS}/pro}
OUTDIR=${OUTDIR:-${NETWORKS}/outputs/morph_eval}

# RNA expression (genes x samples)
EXPR=${EXPR:-${PRODIR}/rna_tpm.gene_x_sample.tsv}

# Morphology embeddings directory (.npy files)
MORPH_DIR=${MORPH_DIR:-${PROJECT_ROOT}/morph_embeddings}

# Optional mapping TSV (embed_id, sample_id, skip_flag)
EMBED_MAP=${EMBED_MAP:-${NETWORKS}/embed_to_sample_map.tsv}
SKIP_LABEL_COL=${SKIP_LABEL_COL:-skip_flag}

# =========================
# CORE PARAMETERS
# =========================
ABS_MIN=".5"
TOPK_GLOBAL="5000"
TOPK_PLOT="500"
MIN_NON_NA="3"
SEED="1337"

# Integrated mixture: w_ij = α*z_RNA + β*morph + γ*connectivity
MIX="1.0,0.5,0.5"
GRID_MIX="0.5:1.5:0.25,0:1:0.25,0:1:0.25"

TAU="1.0"
TOPK_ALPHA="50"

STAB_BOOT="50"
STAB_FRAC="0.8"
STAB_TOPK="10000"
MORPH_SHUFFLES="100"

CASE_REGEX=""

# =========================
# SUBTYPING (OPTIONAL)
# =========================
SUBTYPES=${SUBTYPES:-${NETWORKS}/subtypes.csv}
SUBTYPE_COL=${SUBTYPE_COL:-subtype}
SUBTYPE_POS=${SUBTYPE_POS:-classical}
SUBTYPE_NEG=${SUBTYPE_NEG:-basal}
SUBTYPE_SAMPLE_COL=${SUBTYPE_SAMPLE_COL:-sample_id}

# =========================
# SCRIPT PATH
# =========================
LIONESS_PY=${LIONESS_PY:-${PROJECT_ROOT}/lioness_morph.py}

# =========================
# ENVIRONMENT
# =========================
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load miniconda3 || true
fi

if [[ -f "${HOME}/.conda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.conda/etc/profile.d/conda.sh"
  conda activate rnaseq || conda activate base || true
fi

PY="${PYTHON:-$(command -v python3 || command -v python)}"

# =========================
# CHECKS
# =========================
need(){ [[ -e "$1" ]] || { echo "Missing: $1" >&2; exit 2; }; }
need "${LIONESS_PY}"
need "${EXPR}"
[[ -d "${MORPH_DIR}" ]] || { echo "Missing dir: ${MORPH_DIR}" >&2; exit 2; }

shopt -s nullglob
npys=( "${MORPH_DIR}"/*.npy )
(( ${#npys[@]} > 0 )) || { echo "No .npy files in ${MORPH_DIR}" >&2; exit 2; }
shopt -u nullglob

mkdir -p "${OUTDIR}"

# =========================
# ARGUMENTS
# =========================
args=(
  --expr "${EXPR}"
  --morph_dir "${MORPH_DIR}"
  --outdir "${OUTDIR}"
  --top_k_edges_global "${TOPK_GLOBAL}"
  --top_k_edges_plot "${TOPK_PLOT}"
  --min_non_na "${MIN_NON_NA}"
  --seed "${SEED}"
  --mix "${MIX}"
  --grid_mix "${GRID_MIX}"
  --tau "${TAU}"
  --topk_alpha "${TOPK_ALPHA}"
  --stability_boot "${STAB_BOOT}"
  --stability_frac "${STAB_FRAC}"
  --stability_topk "${STAB_TOPK}"
  --morph_shuffles "${MORPH_SHUFFLES}"
  --lioness 1
)

[[ -n "${ABS_MIN}" ]] && args+=( --abs_min "${ABS_MIN}" )
[[ -n "${CASE_REGEX}" ]] && args+=( --case_regex "${CASE_REGEX}" )

# Mapping TSV (optional)
if [[ -s "${EMBED_MAP}" ]]; then
  args+=( --embed_map "${EMBED_MAP}" --skip_label_col "${SKIP_LABEL_COL}" )
fi

# Subtyping (optional)
if [[ -s "${SUBTYPES}" ]]; then
  args+=( --subtype_tsv "${SUBTYPES}"
          --subtype_col "${SUBTYPE_COL}"
          --subtype_pos "${SUBTYPE_POS}"
          --subtype_neg "${SUBTYPE_NEG}" )
  [[ -n "${SUBTYPE_SAMPLE_COL}" ]] && args+=( --subtype_sample_col "${SUBTYPE_SAMPLE_COL}" )
fi

# =========================
# THREAD LIMITS
# =========================
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

# =========================
# RUN
# =========================
set -x
"${PY}" "${LIONESS_PY}" "${args[@]}"
set +x

echo "[OK] Outputs (top 200 files):"
find "${OUTDIR}" -maxdepth 3 -type f | sed "s|${OUTDIR}/||" | sort | head -n 200

