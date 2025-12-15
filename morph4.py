#!/usr/bin/env python3
import argparse, os, sys, re, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Utils -----------------

def set_seed(s=1337):
    random.seed(s); np.random.seed(s)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")

def dedupe_index(df: pd.DataFrame) -> pd.DataFrame:
    return df

def read_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    return dedupe_index(df)

def is_case_like(x: str, regex: Optional[str]) -> bool:
    if regex:
        return bool(re.match(regex, str(x)))
    return str(x).startswith("TCGA-") and (len(str(x).split("-")) >= 3)

def filter_case_columns(df: pd.DataFrame, case_regex: Optional[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if is_case_like(c, case_regex)]
    return df[cols]

def save_edges(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def save_df(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def plot_graph(G: nx.Graph, out_png: Path, top_k_plot=500, seed=1337, title=""):
    if G.number_of_edges() == 0 or (top_k_plot is not None and top_k_plot <= 0):
        return
    H = G.copy()
    if H.number_of_edges() > top_k_plot:
        e_sorted = sorted(H.edges(data=True), key=lambda e: abs(e[2].get("weight",0.0)), reverse=True)[:top_k_plot]
        H = nx.Graph(); H.add_nodes_from(G.nodes()); H.add_edges_from(e_sorted)
    pos = nx.spring_layout(H, seed=seed, dim=2)
    wabs = np.array([abs(H[u][v].get("weight",0.0)) for u,v in H.edges()], float)
    widths = 1.0 if wabs.size==0 or np.ptp(wabs)==0 else 0.5 + 2.5*(wabs - wabs.min())/np.ptp(wabs)
    plt.figure(figsize=(10,10))
    nx.draw_networkx_nodes(H, pos, node_size=10, alpha=0.7)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.3)
    plt.axis("off")
    if title: plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def graph_metrics(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes(); m = G.number_of_edges()
    if n == 0:
        return dict(n=0, m=0, density=np.nan, mean_deg=np.nan, trans=np.nan, comps=0)
    degs = np.array([d for _,d in G.degree()], float)
    dens = (2.0*m)/(n*(n-1)) if n>1 else 0.0
    trans = nx.transitivity(G) if n < 50000 else np.nan
    comps = nx.number_connected_components(G)
    return dict(n=n, m=m, density=dens, mean_deg=float(degs.mean()), trans=trans, comps=comps)

def to_graph(edge_df: pd.DataFrame, nodes: List[str]) -> nx.Graph:
    G = nx.Graph(); G.add_nodes_from(nodes)
    for _, r in edge_df.iterrows():
        G.add_edge(r["gene_u"], r["gene_v"], weight=float(r["weight"]))
    return G

def apply_filters(edges: pd.DataFrame, abs_min: Optional[float], top_k: int) -> pd.DataFrame:
    df = edges.copy()
    if abs_min is not None:
        df = df.loc[df["weight"].abs() >= float(abs_min)]
    if top_k and top_k > 0 and len(df) > top_k:
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index).iloc[:top_k]
    return df.reset_index(drop=True)

# ----------------- RNA similarities -----------------

def rank_transform_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda row: row.rank(method="average"), axis=1)

def corr_pairs(df_ranked: pd.DataFrame, min_non_na=5) -> pd.DataFrame:
    genes = df_ranked.index.to_list()
    X = df_ranked.to_numpy(float)
    G, N = X.shape
    rows = []
    for i in range(G):
        xi = X[i]; mi = np.isfinite(xi)
        for j in range(i+1, G):
            xj = X[j]; m = mi & np.isfinite(xj)
            n = int(m.sum())
            if n < min_non_na: continue
            r = np.corrcoef(xi[m], xj[m])[0,1]
            if np.isfinite(r):
                rows.append((genes[i], genes[j], float(r), n))
    return pd.DataFrame(rows, columns=["gene_u","gene_v","z_rna","n"])

def compose_rna_edges(expr_df: pd.DataFrame, min_non_na=5) -> pd.DataFrame:
    E = corr_pairs(rank_transform_rows(expr_df), min_non_na=min_non_na)
    return E[["gene_u","gene_v","z_rna"]]

# ----------------- Morphology loading & case pooling -----------------

def tcga_case_from_slide(slide_id: str) -> str:
    parts = str(slide_id).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(slide_id)

def _truthy(v):
    if isinstance(v, (bool, np.bool_)): return bool(v)
    s = str(v).strip().lower()
    return s in {"true","1","yes","y","t"}

# --- replace load_case_morphology with this robust uni_v2-aware version ---

def load_case_morphology(emb_dir: str, standardize=True, pool="median",
                         embed_map: Optional[str]=None, skip_label_col: Optional[str]="case_in_rna"
                         ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Loads morphology embeddings from .npy files.
    Supports uni_v2 files saved as pickled dicts whose keys look like (D, N)
    and whose values are 1D vectors of length D. We keep 1536-d if available
    (else 2048), and if multiple vectors exist for the same D we pool them.
    """
    def _truthy(v):
        if isinstance(v, (bool, np.bool_)): return bool(v)
        s = str(v).strip().lower()
        return s in {"true","1","yes","y","t"}

    emb_dir = Path(emb_dir)
    files = sorted([p for p in emb_dir.glob("*.npy") if p.is_file()])

    # Optional mapping: restrict to allowed embeds and map slide -> case
    allowed = None
    embed_to_case: Dict[str, str] = {}
    if embed_map and Path(embed_map).exists():
        m = pd.read_csv(embed_map, sep=None, engine="python")
        if "embed" in m.columns and "case_id" in m.columns:
            if skip_label_col and skip_label_col in m.columns:
                m = m.loc[m[skip_label_col].map(_truthy)]
            m["embed"] = m["embed"].astype(str)
            m["case_id"] = m["case_id"].astype(str).str.upper().str.strip()
            allowed = set(m["embed"].tolist())
            embed_to_case = dict(zip(m["embed"], m["case_id"]))

    def _extract_vec(obj) -> Optional[np.ndarray]:
        """
        Return a 1D numeric vector from any of:
        - dict-like uni_v2 payload { (D, N): np.ndarray(D,) , ... }
        - 0-D object array containing such a dict
        - numeric ndarray (1D/2D)
        """
        # unwrap 0-D object array
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
            obj = obj.item()

        # case A: dict payload
        if isinstance(obj, dict):
            # collect vectors keyed by their dimensionality
            by_dim: Dict[int, List[np.ndarray]] = {}
            for k, v in obj.items():
                try:
                    # keys like (1536, 17920) or (2048, 13824)
                    if isinstance(k, tuple) and len(k) >= 1:
                        D = int(k[0])
                    else:
                        D = int(np.asarray(v).shape[0])
                    a = np.asarray(v, dtype=float).reshape(-1)
                    if a.size == D:
                        by_dim.setdefault(D, []).append(a)
                except Exception:
                    continue
            if not by_dim:
                return None
            # prefer 1536-d, else the largest D available (e.g., 2048)
            preferred = 1536 if 1536 in by_dim else max(by_dim.keys())
            vecs = np.vstack(by_dim[preferred])  # (#vectors, D)
            # pool multiple options from the same file
            return np.median(vecs, axis=0)

        # case B: numeric ndarray already
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.astype(float).reshape(-1)
            if obj.ndim >= 2:
                # if 2D, prefer pooling across rows (patches) to a single feature vector
                A = obj.astype(float)
                # if it's (S, D), reduce over S; if (D, S), reduce over last axis
                if A.shape[0] < A.shape[1]:
                    return np.median(A, axis=0)
                else:
                    return np.median(A, axis=1)
        return None

    case_to_vecs: Dict[str, List[np.ndarray]] = {}
    slides_by_case: Dict[str, List[str]] = {}

    for p in files:
        bn = p.name
        if allowed is not None and bn not in allowed:
            continue
        slide = bn.split(".")[0]
        case = embed_to_case.get(bn, tcga_case_from_slide(slide)).upper().strip()

        try:
            # allow pickle because uni_v2 files are often dict pickles
            raw = np.load(p, allow_pickle=True)
            vec = _extract_vec(raw)
            if vec is None or not np.all(np.isfinite(vec)):
                # try one more time in case an object array wraps the dict
                if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
                    vec = _extract_vec(raw.item())
            if vec is None or not np.all(np.isfinite(vec)):
                raise ValueError("could not extract numeric vector")
        except Exception:
            # keep your logs elsewhere if you want; here we just skip
            continue

        case_to_vecs.setdefault(case, []).append(vec)
        slides_by_case.setdefault(case, []).append(slide)

    cases = sorted(case_to_vecs.keys())
    if not cases:
        return pd.DataFrame(np.zeros((0, 0))), {}

    # pool vectors per case
    pooled = []
    for c in cases:
        M = np.vstack(case_to_vecs[c])  # (#slides, D)
        if pool == "mean":
            pv = np.mean(M, axis=0)
        else:
            pv = np.median(M, axis=0)
        pooled.append(pv)

    Z = np.vstack(pooled)
    if standardize and Z.size > 0:
        mu = np.nanmean(Z, axis=0, keepdims=True)
        sd = np.nanstd(Z, axis=0, keepdims=True); sd[sd == 0] = 1.0
        Z = (Z - mu) / sd

    Zdf = pd.DataFrame(Z, index=cases)
    return Zdf, slides_by_case

# ----------------- Gene-conditioned prototypes & morphology terms -----------------

def softmax(x, tau=1.0):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return x
    if np.all(np.isnan(x)): return np.zeros_like(x)
    x = x - np.nanmax(x)
    tau = max(float(tau), 1e-8)
    x = x / tau
    ex = np.exp(np.nan_to_num(x, nan=-1e9))
    s = ex.sum()
    return ex / s if s > 0 else np.zeros_like(ex)

def gene_prototypes(X, Z, tau=1.0):
    assert X.ndim == 2 and Z.ndim == 2
    valid_S = ~np.any(np.isnan(Z), axis=1)
    if X.shape[1] != Z.shape[0]:
        raise ValueError(f"X and Z sample mismatch: X has {X.shape[1]} cols, Z has {Z.shape[0]}")
    valid_S &= ~np.all(np.isnan(X), axis=0)
    if not np.any(valid_S):
        raise ValueError("No valid samples remain after filtering NaNs in X/Z")
    X = X[:, valid_S]; Z = Z[valid_S, :]
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    Xe = (X - mu) / (sd + 1e-8)
    G, S = Xe.shape; D = Z.shape[1]
    mu_g = np.zeros((G, D), dtype=float); alpha_g = np.zeros(G, dtype=float); keep = np.ones(G, dtype=bool)
    for gi in range(G):
        xg = Xe[gi, :]
        if xg.size == 0 or np.all(np.isnan(xg)): keep[gi] = False; continue
        w = softmax(xg, tau=tau)
        if w.size == 0 or (w.sum() == 0): keep[gi] = False; continue
        mu_g[gi, :] = w @ Z
        alpha_g[gi] = np.nanmax(np.abs(xg))
    idx = np.where(keep)[0]
    if idx.size == 0: raise ValueError("All genes were filtered out (no signal after cleaning).")
    return mu_g[idx, :], alpha_g[idx], idx

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    a = float(np.dot(u, v))
    nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
    if nu==0 or nv==0: return 0.0
    return a / (nu*nv)

def morphology_cosine_edges(mu_g: Dict[str, np.ndarray]) -> pd.DataFrame:
    genes = list(mu_g.keys())
    rows = []
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            m = cosine(mu_g[genes[i]], mu_g[genes[j]])
            rows.append((genes[i], genes[j], float(m)))
    return pd.DataFrame(rows, columns=["gene_u","gene_v","m_mu"])

def jaccard(a, b) -> float:
    a,b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0

def topk_case_sets(alpha_g: Dict[str, np.ndarray], cases: List[str], k: int = 50) -> Dict[str, List[str]]:
    out = {}
    S = len(cases); k = min(k, S) if S>0 else 0
    for g, w in alpha_g.items():
        idx = np.argsort(w)[::-1][:k]
        out[g] = [cases[i] for i in idx]
    return out

def coattention_jaccard_edges(alpha_g: Dict[str, np.ndarray], cases: List[str], topk: int = 50) -> pd.DataFrame:
    top_sets = topk_case_sets(alpha_g, cases, k=topk)
    genes = list(alpha_g.keys())
    rows = []
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            c = jaccard(top_sets[genes[i]], top_sets[genes[j]])
            rows.append((genes[i], genes[j], float(c)))
    return pd.DataFrame(rows, columns=["gene_u","gene_v","c"])

# ----------------- Integrated builder & LIONESS -----------------

def integrate_edges(E_rna: pd.DataFrame, E_m: pd.DataFrame, E_c: Optional[pd.DataFrame],
                    alpha: float, beta: float, gamma: float) -> pd.DataFrame:
    df = E_rna.merge(E_m, on=["gene_u","gene_v"], how="inner")
    if E_c is not None and len(E_c)>0 and gamma!=0.0:
        df = df.merge(E_c, on=["gene_u","gene_v"], how="left")
        df["c"] = df["c"].fillna(0.0)
    else:
        df["c"] = 0.0
    df["weight"] = alpha*df["z_rna"] + beta*df["m_mu"] + gamma*df["c"]
    return df[["gene_u","gene_v","weight"]]

def lioness_personalized(sim_fn, samples: List[str], abs_min: Optional[float]=None):
    N = len(samples)
    E_full = sim_fn(drop_sample=None)
    M_full = E_full.set_index(["gene_u","gene_v"])["weight"]
    result = {}
    for s in samples:
        E_wo = sim_fn(drop_sample=s)
        M_wo = E_wo.set_index(["gene_u","gene_v"])["weight"]
        idx = M_full.index.intersection(M_wo.index)
        w_s = N*M_full.loc[idx] - (N-1)*M_wo.loc[idx]
        df = pd.DataFrame(list(idx), columns=["gene_u","gene_v"])
        df["weight"] = w_s.values
        if abs_min is not None:
            df = df.loc[df["weight"].abs() >= float(abs_min)]
        result[s] = df.reset_index(drop=True)
    return result

# ----------------- Evaluation -----------------

def degree_scores(G: nx.Graph, expr_df: pd.DataFrame) -> pd.Series:
    deg = dict(G.degree())
    v = np.array([deg.get(g,0.0) for g in expr_df.index], float)
    X = expr_df.to_numpy(float)
    s = (v[:,None] * X).sum(axis=0)
    return pd.Series(s, index=expr_df.columns)

def auc_binary(y, scores):
    y = np.asarray(y, float); s = np.asarray(scores, float)
    m1 = y==1; m0 = y==0
    n1 = int(m1.sum()); n0 = int(m0.sum())
    if n1==0 or n0==0: return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    u = r[m1].sum() - n1*(n1+1)/2.0
    return float(u/(n1*n0))

def load_subtypes(path, col, sample_col=None):
    df = pd.read_csv(path, sep=None, engine="python")
    s = df[sample_col].astype(str).tolist() if sample_col and sample_col in df.columns else df.iloc[:,0].astype(str).tolist()
    lab = df[col].astype(str).tolist()
    return dict(zip(s, lab))

def subtype_auc_from_scores(scores_ser: pd.Series, subtype_map: Dict[str,str], pos_label: str, neg_label: str):
    idx = scores_ser.index.intersection(pd.Index(list(subtype_map.keys())))
    y = [1 if subtype_map[s]==pos_label else (0 if subtype_map[s]==neg_label else None) for s in idx]
    sel = [i for i,v in enumerate(y) if v is not None]
    if not sel: return np.nan
    yb = [y[i] for i in sel]; sc = scores_ser.loc[idx].to_numpy()[sel]
    return auc_binary(yb, sc)

def bootstrap_edge_stability(builder, expr, reps=0, frac=0.8, topk=10000, seed=1337, min_non_na=5):
    if reps <= 0: return np.nan
    rng = np.random.default_rng(seed)
    samps = expr.columns.to_list()
    sets = []
    for _ in range(reps):
        k = max(2, int(round(frac*len(samps))))
        S = rng.choice(samps, size=k, replace=False).tolist()
        E = builder(drop_sample=None, restrict_samples=S)
        E = E.reindex(E["weight"].abs().sort_values(ascending=False).index)
        if len(E) > topk: E = E.iloc[:topk]
        sets.append(set(zip(E["gene_u"].tolist(), E["gene_v"].tolist())))
    j = []
    for i in range(reps):
        for j2 in range(i+1, reps):
            a, b = sets[i], sets[j2]
            if len(a|b)==0: continue
            j.append(len(a&b)/len(a|b))
    return float(np.mean(j)) if j else np.nan

# ----------------- Main builder -----------------

def build_integrated_edges(expr, Z_case, alpha, beta, gamma, tau, topk_alpha, min_non_na,
                           drop_sample=None, restrict_samples=None, morph_shuffle_seed=None):
    expr.columns = expr.columns.astype(str).str.strip().str.upper()
    Z_case.index = Z_case.index.astype(str).str.strip().str.upper()
    common = expr.columns.intersection(Z_case.index)
    if restrict_samples is not None:
        common = pd.Index([s for s in common if s in restrict_samples])
    if drop_sample is not None and drop_sample in common:
        common = common.drop(drop_sample)
    if len(common) == 0:
        raise ValueError(f"[build_integrated_edges] No overlapping sample IDs between expr({len(expr.columns)}) and Z_case({len(Z_case.index)})")
    X = expr.loc[:, common].copy()
    Z = Z_case.loc[common, :].copy()
    if morph_shuffle_seed is not None:
        rng = np.random.default_rng(morph_shuffle_seed)
        perm = rng.permutation(len(Z))
        Z = pd.DataFrame(Z.to_numpy()[perm], index=Z.index, columns=Z.columns)
    X = X.replace([np.inf, -np.inf], np.nan)
    Z = Z.replace([np.inf, -np.inf], np.nan)
    X_vals = X.to_numpy(float)
    row_nan = np.all(np.isnan(X_vals), axis=1)
    row_const = np.nanstd(X_vals, axis=1) == 0
    keep_gene = ~(row_nan | row_const)
    if not np.any(keep_gene): raise ValueError("[build_integrated_edges] all genes are NaN or constant after cleaning")
    X = X.iloc[keep_gene, :]
    Z = Z.loc[:, ~Z.isna().all(axis=0)]
    kX = max(1, int(0.05 * X.shape[1])); kZ = max(1, int(0.05 * Z.shape[1]))
    mask_ok = (X.notna().sum(axis=1) >= kX)
    X = X.loc[mask_ok]
    if X.shape[0] == 0: raise ValueError("[build_integrated_edges] all genes dropped by NaN threshold")
    def safe_zscore(df):
        mu = df.mean(axis=1); sd = df.std(axis=1, ddof=0).replace(0, np.nan)
        z = (df.sub(mu, axis=0)).div(sd, axis=0)
        return z.replace([np.inf, -np.inf], np.nan)
    Xz = safe_zscore(X)
    valid_sample_mask = (~np.any(np.isnan(Z.to_numpy()), axis=1)) & (~np.all(np.isnan(Xz.to_numpy()), axis=0))
    if not np.any(valid_sample_mask): raise ValueError("[build_integrated_edges] No valid samples remain after NaN filtering in X/Z")
    valid_samples = Xz.columns[valid_sample_mask]
    Xz = Xz.loc[:, valid_samples]; Z = Z.loc[valid_samples, :]
    E_rna = compose_rna_edges(Xz, min_non_na=min_non_na)
    mu_arr, alpha_arr, idx = gene_prototypes(Xz.to_numpy(float), Z.to_numpy(float), tau=tau)
    genes_kept = Xz.index.to_numpy()[idx]

    mu_g = {g: mu_arr[i, :] for i, g in enumerate(genes_kept)}
    alpha_g = {g: alpha_arr[i]    for i, g in enumerate(genes_kept)}

    E_m = morphology_cosine_edges(mu_g)
    E_c = None
    if gamma != 0.0 and topk_alpha > 0:
        cases = list(Xz.columns)
        E_c = coattention_jaccard_edges(alpha_g, cases, topk=topk_alpha)
    E = integrate_edges(E_rna, E_m, E_c, alpha=alpha, beta=beta, gamma=gamma)
    return E

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Morphology-aware LIONESS networks (RNA + morphology; mapping-aware).")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--morph_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--case_regex", default="")
    ap.add_argument("--abs_min", type=float, default=0.5)
    ap.add_argument("--top_k_edges_global", type=int, default=5000)
    ap.add_argument("--top_k_edges_plot", type=int, default=500)
    ap.add_argument("--min_non_na", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--mix", type=str, default="1.0,0.5,0.5")
    ap.add_argument("--grid_mix", type=str, default="")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--topk_alpha", type=int, default=50)
    ap.add_argument("--lioness", type=int, default=1)
    ap.add_argument("--stability_boot", type=int, default=50)
    ap.add_argument("--stability_frac", type=float, default=0.8)
    ap.add_argument("--stability_topk", type=int, default=10000)
    ap.add_argument("--morph_shuffles", type=int, default=100)
    ap.add_argument("--subtype_tsv", type=str, default="")
    ap.add_argument("--subtype_col", type=str, default="")
    ap.add_argument("--subtype_pos", type=str, default="")
    ap.add_argument("--subtype_neg", type=str, default="")
    ap.add_argument("--subtype_sample_col", type=str, default="")
    ap.add_argument("--embed_map", type=str, default="", help="TSV with columns embed,case_id,(label). Filters embeddings by label=False.")
    ap.add_argument("--skip_label_col", type=str, default="case_in_rna", help="Column in embed_map whose False rows are skipped.")
    # NEW: Moffitt 50 gene panel file (one gene per line)
    ap.add_argument("--moffitt_genes", type=str, default="",
                    help="Optional text file with Moffitt 50 tumor genes (one gene symbol per line).")

    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load RNA and restrict to Moffitt 50 panel (if provided) ----
    expr_raw = read_matrix(args.expr)
    # normalize gene IDs
    if "gene_name" in expr_raw.columns:
        expr_raw.index = (
            expr_raw["gene_name"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
        # remove gene_name column so it's not treated as a sample
        expr_raw = expr_raw.drop(columns=["gene_name"])
    else:
    # fallback if gene_name isn't present
        expr_raw.index = expr_raw.index.astype(str).str.upper().str.strip()

    if args.moffitt_genes.strip():
        gene_file = Path(args.moffitt_genes)
        if not gene_file.exists():
            print(f"[ERROR] --moffitt_genes file not found: {gene_file}", file=sys.stderr)
            sys.exit(1)
        g = pd.read_csv(gene_file, header=None)[0].astype(str).str.upper().str.strip()
        moffitt_set = set(g.tolist())
        keep_genes = expr_raw.index.intersection(pd.Index(list(moffitt_set)))
        if len(keep_genes) == 0:
            print("[ERROR] No overlap between expression genes and Moffitt gene panel.", file=sys.stderr)
            sys.exit(1)
        expr_raw = expr_raw.loc[keep_genes]
        # optional sanity print
        print(f"[INFO] Restricted to Moffitt panel: {len(keep_genes)} genes", file=sys.stderr)

    # keep only case-like columns
    expr = filter_case_columns(expr_raw, args.case_regex if args.case_regex.strip() else None)

    # ---- Load morphology embeddings ----
    Z_case, slides_by_case = load_case_morphology(
        args.morph_dir, standardize=True, pool="median",
        embed_map=(args.embed_map if args.embed_map.strip() else None),
        skip_label_col=(args.skip_label_col if args.skip_label_col.strip() else None)
    )

    # ---- Align cases between RNA and morphology ----
    expr.columns = expr.columns.astype(str).str.upper().str.strip()
    Z_case.index = Z_case.index.astype(str).str.upper().str.strip()
    common_cases = expr.columns.intersection(Z_case.index)
    expr = expr[common_cases]
    Z_case = Z_case.loc[common_cases]

    if expr.shape[0] == 0 or expr.shape[1] == 0:
        print(f"[ERROR] After Moffitt filtering and case intersection, expr shape is {expr.shape}", file=sys.stderr)
        sys.exit(1)
    if Z_case.shape[0] == 0 or Z_case.shape[1] == 0:
        print(f"[ERROR] No morphology embeddings left after intersection; Z_case shape is {Z_case.shape}", file=sys.stderr)
        sys.exit(1)

    try:
        a0, b0, g0 = [float(x) for x in args.mix.split(",")]
    except Exception:
        print("--mix must be 'alpha,beta,gamma'", file=sys.stderr); sys.exit(2)

    def parse_range(s):
        a, b, da = [float(x) for x in s.split(":")]
        n = int(round((b - a) / da)) + 1
        return [a + k * da for k in range(n)]

    grids = []
    if args.grid_mix:
        a, b, g = args.grid_mix.split(",")
        A, B, G = parse_range(a), parse_range(b), parse_range(g)
        grids = [(A_, B_, G_) for A_ in A for B_ in B for G_ in G]

    conditions = []

    def builder_rna(drop_sample=None, restrict_samples=None, shuffle_seed=None):
        common = expr.columns
        if restrict_samples is not None:
            common = pd.Index([s for s in common if s in restrict_samples])
        if drop_sample is not None and drop_sample in common:
            common = common.drop(drop_sample)
        X = expr[common]
        E = compose_rna_edges(X, min_non_na=args.min_non_na)
        E = E.rename(columns={"z_rna": "weight"})
        return E[["gene_u", "gene_v", "weight"]]

    conditions.append(("RNA", lambda: builder_rna()))

    def builder_mix(drop_sample=None, restrict_samples=None, shuffle_seed=None):
        return build_integrated_edges(
            expr, Z_case,
            alpha=a0, beta=b0, gamma=g0,
            tau=args.tau, topk_alpha=args.topk_alpha,
            min_non_na=args.min_non_na,
            drop_sample=drop_sample,
            restrict_samples=restrict_samples,
            morph_shuffle_seed=shuffle_seed
        )

    conditions.append(("MORPH_RNA", lambda: builder_mix()))

    for a, b, g in grids:
        def make_builder(A=a, B=b, G=g):
            return lambda drop_sample=None, restrict_samples=None, shuffle_seed=None: build_integrated_edges(
                expr, Z_case, alpha=A, beta=B, gamma=G,
                tau=args.tau, topk_alpha=args.topk_alpha,
                min_non_na=args.min_non_na,
                drop_sample=drop_sample,
                restrict_samples=restrict_samples,
                morph_shuffle_seed=shuffle_seed
            )
        conditions.append((f"MORPH_RNA_a{a:.3g}_b{b:.3g}_g{g:.3g}", make_builder()))

    for k in range(int(args.morph_shuffles)):
        def make_shuf(k_=k):
            seed_ = args.seed + 10 + k_
            return lambda drop_sample=None, restrict_samples=None, shuffle_seed=None: build_integrated_edges(
                expr, Z_case, alpha=a0, beta=b0, gamma=g0,
                tau=args.tau, topk_alpha=args.topk_alpha,
                min_non_na=args.min_non_na,
                drop_sample=drop_sample,
                restrict_samples=restrict_samples,
                morph_shuffle_seed=seed_
            )
        conditions.append((f"MORPH_RNA_SHUFFLE_{k+1:02d}", make_shuf()))

    summary = []
    graphs = {}
    edges_by_name = {}

    for name, fn in conditions:
        cdir = outdir / name
        cdir.mkdir(parents=True, exist_ok=True)
        E = fn()
        Ef = apply_filters(E.rename(columns={"weight": "weight"}),
                           abs_min=args.abs_min,
                           top_k=args.top_k_edges_global)
        save_edges(cdir / "edges_global.tsv", Ef)
        G = to_graph(Ef, nodes=expr.index.to_list())
        save_df(cdir / "metrics_global.csv", pd.DataFrame([graph_metrics(G)]))
        plot_graph(
            G,
            cdir / "graph_global.png",
            top_k_plot=args.top_k_edges_plot,
            seed=args.seed,
            title=f"{name} global (|w|≥{args.abs_min}, topK={args.top_k_edges_global})"
        )
        graphs[name] = G
        edges_by_name[name] = Ef
        m = graph_metrics(G)
        m["condition"] = name
        summary.append(m)

        if args.lioness and name.startswith("MORPH_RNA"):
            samples = expr.columns.to_list()
            def sim(drop_sample=None):
                return builder_mix(drop_sample=drop_sample)
            lion = lioness_personalized(sim, samples, abs_min=args.abs_min)
            pdir = cdir / "edges_personalized"
            pdir.mkdir(parents=True, exist_ok=True)
            per_rows = []
            for s, df_s in lion.items():
                save_edges(pdir / f"{s}.tsv", df_s)
                Gs = to_graph(df_s, nodes=expr.index.to_list())
                met = graph_metrics(Gs)
                met["sample"] = s
                per_rows.append(met)
            save_df(cdir / "metrics_personalized.csv", pd.DataFrame(per_rows))
            if samples:
                s0 = samples[np.random.default_rng(args.seed).integers(0, len(samples))]
                G0 = to_graph(lion[s0], nodes=expr.index.to_list())
                plot_graph(
                    G0,
                    cdir / f"graph_personalized_{s0}.png",
                    top_k_plot=args.top_k_edges_plot,
                    seed=args.seed,
                    title=f"{name} LIONESS: {s0}"
                )

    save_df(outdir / "ablation_metrics_global.csv", pd.DataFrame(summary))

    if args.stability_boot > 0:
        rows = []
        rows.append({
            "condition": "RNA",
            "edge_stability_jaccard": bootstrap_edge_stability(
                lambda drop_sample=None, restrict_samples=None: builder_rna(drop_sample=None, restrict_samples=restrict_samples),
                expr, reps=args.stability_boot, frac=args.stability_frac,
                topk=args.stability_topk, seed=args.seed, min_non_na=args.min_non_na)
        })
        rows.append({
            "condition": "MORPH_RNA",
            "edge_stability_jaccard": bootstrap_edge_stability(
                lambda drop_sample=None, restrict_samples=None: builder_mix(drop_sample=None, restrict_samples=restrict_samples),
                expr, reps=args.stability_boot, frac=args.stability_frac,
                topk=args.stability_topk, seed=args.seed, min_non_na=args.min_non_na)
        })
        save_df(outdir / "eval" / "stability_edge_jaccard.csv", pd.DataFrame(rows))

    if args.subtype_tsv and args.subtype_col and args.subtype_pos and args.subtype_neg:
        submap = load_subtypes(args.subtype_tsv, args.subtype_col, args.subtype_sample_col or None)
        eval_rows = []
        for name in ["RNA", "MORPH_RNA"]:
            if name in graphs:
                sc = degree_scores(graphs[name], expr)
                auc = subtype_auc_from_scores(sc, submap, args.subtype_pos, args.subtype_neg)
                eval_rows.append({"condition": name, "subtype_auc": auc})
        eval_dir = outdir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_df(eval_dir / "subtype_auc.csv", pd.DataFrame(eval_rows))

        def communities_and_scores(G, expr_df):
            try:
                comms = nx.algorithms.community.greedy_modularity_communities(G)
            except Exception:
                comms = []
            mods = []
            for idx, c in enumerate(comms):
                genes = [g for g in c if g in expr_df.index]
                if len(genes) < 2:
                    continue
                score = expr_df.loc[genes].mean(axis=0)
                mods.append((f"module_{idx}", genes, score))
            return mods

        if "MORPH_RNA" in graphs:
            # 1) Define modules on the global MORPH_RNA graph
            mods = communities_and_scores(graphs["MORPH_RNA"], expr)

            mod_dir = outdir / "modules" / "MORPH_RNA"
            mod_dir.mkdir(parents=True, exist_ok=True)

            # 2) Load MORPH_RNA LIONESS edges (one file per sample)
            lion_dir = outdir / "MORPH_RNA" / "edges_personalized"
            lion_files = sorted(lion_dir.glob("*.tsv"))
            if not lion_files:
                raise RuntimeError(f"No LIONESS edges found in {lion_dir} – run with --lioness 1")

            sample_ids = [p.stem for p in lion_files]

            rows = []
            for mid, genes, _ in mods:
                # Save module gene list
                (mod_dir / f"{mid}_genes.txt").write_text("\n".join(genes) + "\n")

                # 3) Compute per-sample module connectivity from LIONESS edges
                scores = []
                for s, p in zip(sample_ids, lion_files):
                    df_s = pd.read_csv(p, sep="\t")
                    mask = df_s["gene_u"].isin(genes) & df_s["gene_v"].isin(genes)
                    sub = df_s.loc[mask]
                    if len(sub) == 0:
                        val = 0.0   # or np.nan, but 0.0 is safer for AUC
                    else:
                        val = float(sub["weight"].mean())
                    scores.append(val)

                score_ser = pd.Series(scores, index=sample_ids)

                # Save LIONESS-based module scores
                pd.DataFrame({"sample": score_ser.index, "score": score_ser.values}).to_csv(
                    mod_dir / f"{mid}_scores_lioness.csv", index=False
                )

                # 4) Compute subtype AUC from LIONESS-based module scores
                auc = subtype_auc_from_scores(score_ser, submap, args.subtype_pos, args.subtype_neg)
                rows.append({"module": mid, "n_genes": len(genes), "auc": auc})

            if rows:
                # NOTE: new file name so you can distinguish from old expression-based version
                save_df(
                    eval_dir / "subtype_module_auc_MORPH_RNA_lioness.csv",
                    pd.DataFrame(rows).sort_values("auc", ascending=False)
                )

if __name__ == "__main__":
    main()

