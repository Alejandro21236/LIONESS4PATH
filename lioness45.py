
#!/usr/bin/env python3
import argparse, os, sys, math, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.apply(pd.to_numeric, errors="coerce")
    return df2

def dedupe_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df

def set_seed(s):
    random.seed(s); np.random.seed(s)

def read_matrix(path):
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    df = dedupe_index(df)
    df = ensure_numeric(df)
    return df

def subtype_labels_from_scores(scores_ser, subtype_map, pos_label, neg_label):
    idx = scores_ser.index.intersection(pd.Index(list(subtype_map.keys())))
    y = []
    s = []
    for sample in idx:
        lab = subtype_map[sample]
        if lab == pos_label:
            y.append(1)
            s.append(float(scores_ser.loc[sample]))
        elif lab == neg_label:
            y.append(0)
            s.append(float(scores_ser.loc[sample]))
    return np.asarray(y, dtype=int), np.asarray(s, dtype=float)


def build_condition_edges(expr_df, rt_df, condition, alpha, beta, gamma, min_non_na):
    if condition == "RNA":
        return compose_edges_rna(expr_df, min_non_na=min_non_na)
    elif condition == "RT":
        return compose_edges_rt(rt_df, min_non_na=min_non_na)
    elif condition == "RT_RNA":
        return compose_edges_rtaware(
            expr_df, rt_df,
            alpha=alpha, beta=beta, gamma=gamma,
            min_non_na=min_non_na
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def discover_modules_train_only(expr_train, rt_train, condition,
                                alpha, beta, gamma,
                                min_non_na, abs_min, top_k_edges_global):
    edges = build_condition_edges(
        expr_train, rt_train, condition,
        alpha, beta, gamma, min_non_na
    )
    edges = apply_filters(
        edges,
        abs_min=abs_min,
        top_k=top_k_edges_global
    )
    G = to_graph(edges, nodes=expr_train.index.tolist())

    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G)
    except Exception:
        comms = []

    modules = []
    for i, c in enumerate(comms):
        genes = [g for g in c if g in expr_train.index]
        if len(genes) >= 2:
            modules.append((f"module_{i}", genes))
    return edges, G, modules


def sample_specific_lioness_edges_for_test(expr_train, rt_train, test_sample,
                                           alpha, beta, gamma, min_non_na,
                                           abs_min=None):
    train_samples = list(expr_train.columns)

    # Case 1: sample is already inside the training cohort
    # Compute its standard LIONESS network within that cohort
    if test_sample in train_samples:
        N = len(train_samples)

        E_full = compose_edges_rtaware(
            expr_train, rt_train,
            alpha=alpha, beta=beta, gamma=gamma,
            min_non_na=min_non_na
        ).set_index(["gene_u", "gene_v"])["weight"]

        expr_wo = expr_train.drop(columns=[test_sample])
        rt_wo = rt_train.drop(columns=[test_sample])

        E_wo = compose_edges_rtaware(
            expr_wo, rt_wo,
            alpha=alpha, beta=beta, gamma=gamma,
            min_non_na=min_non_na
        ).set_index(["gene_u", "gene_v"])["weight"]

        idx = E_full.index.intersection(E_wo.index)
        w_q = N * E_full.loc[idx] - (N - 1) * E_wo.loc[idx]

        df = pd.DataFrame(list(idx), columns=["gene_u", "gene_v"])
        df["weight"] = w_q.values

        if abs_min is not None:
            df = df.loc[df["weight"].abs() >= float(abs_min)]

        return df.reset_index(drop=True)

    # Case 2: sample is outside the training cohort
    # Add it once, then compute its LIONESS network relative to the training set
    E_train = compose_edges_rtaware(
        expr_train, rt_train,
        alpha=alpha, beta=beta, gamma=gamma,
        min_non_na=min_non_na
    ).set_index(["gene_u", "gene_v"])["weight"]

    expr_plus = expr_train.join(expr_full[[test_sample]], how="inner")
    rt_plus   = rt_train.join(rt_full[[test_sample]], how="inner")

    E_plus = compose_edges_rtaware(
        expr_plus, rt_plus,
        alpha=alpha, beta=beta, gamma=gamma,
        min_non_na=min_non_na
    ).set_index(["gene_u", "gene_v"])["weight"]

    idx = E_train.index.intersection(E_plus.index)
    n_train = len(train_samples)

    w_q = (n_train + 1) * E_plus.loc[idx] - n_train * E_train.loc[idx]

    df = pd.DataFrame(list(idx), columns=["gene_u", "gene_v"])
    df["weight"] = w_q.values

    if abs_min is not None:
        df = df.loc[df["weight"].abs() >= float(abs_min)]

    return df.reset_index(drop=True)


def module_score_vector(expr_df, rt_df, module_genes, condition,
                        alpha, beta, gamma, min_non_na, abs_min,
                        train_samples=None, test_samples=None):
    if condition == "RNA":
        inter = [g for g in module_genes if g in expr_df.index]
        if len(inter) == 0:
            return pd.Series(dtype=float)
        return expr_df.loc[inter].mean(axis=0)

    elif condition == "RT":
        inter = [g for g in module_genes if g in rt_df.index]
        if len(inter) == 0:
            return pd.Series(dtype=float)
        return rt_df.loc[inter].mean(axis=0)

    elif condition == "RT_RNA":
        if train_samples is None or test_samples is None:
            raise ValueError("RT_RNA scoring requires train_samples and test_samples")

        scores = []
        kept = []
        modset = set(module_genes)

        expr_train = expr_df[train_samples]
        rt_train = rt_df[train_samples]

        for s in test_samples:
            df_s = sample_specific_lioness_edges_for_test(
                expr_train=expr_train,
                rt_train=rt_train,
                test_sample=s,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                min_non_na=min_non_na,
                abs_min=abs_min
            )
            mask = df_s["gene_u"].isin(modset) & df_s["gene_v"].isin(modset)
            sub = df_s.loc[mask]
            val = float(sub["weight"].mean()) if len(sub) > 0 else 0.0
            scores.append(val)
            kept.append(s)

        return pd.Series(scores, index=kept, dtype=float)

    else:
        raise ValueError(f"Unknown condition: {condition}")


def select_best_module_nested(expr_df, rt_df, subtype_map, candidate_genes,
                              condition, alpha, beta, gamma,
                              min_non_na, abs_min, top_k_edges_global,
                              pos_label, neg_label,
                              inner_splits=3, seed=1337):
    labeled_samples = [s for s in expr_df.columns if s in subtype_map and subtype_map[s] in {pos_label, neg_label}]
    y_all = np.array([1 if subtype_map[s] == pos_label else 0 for s in labeled_samples], dtype=int)

    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    module_perf = {}

    for tr_idx, va_idx in inner_cv.split(labeled_samples, y_all):
        tr_samples = [labeled_samples[i] for i in tr_idx]
        va_samples = [labeled_samples[i] for i in va_idx]

        expr_tr = expr_df.loc[candidate_genes, tr_samples]
        rt_tr = rt_df.loc[candidate_genes, tr_samples]

        _, _, modules = discover_modules_train_only(
            expr_train=expr_tr,
            rt_train=rt_tr,
            condition=condition,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_non_na=min_non_na,
            abs_min=abs_min,
            top_k_edges_global=top_k_edges_global
        )

        for mid, genes in modules:
            score_ser = module_score_vector(
                expr_df=expr_df.loc[candidate_genes],
                rt_df=rt_df.loc[candidate_genes],
                module_genes=genes,
                condition=condition,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                min_non_na=min_non_na,
                abs_min=abs_min,
                train_samples=tr_samples if condition == "RT_RNA" else None,
                test_samples=va_samples if condition == "RT_RNA" else va_samples
            )

            idx = score_ser.index.intersection(pd.Index(va_samples))
            y = np.array([1 if subtype_map[s] == pos_label else 0 for s in idx], dtype=int)
            x = score_ser.loc[idx].to_numpy(dtype=float)

            if len(np.unique(y)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y, x)

            module_perf.setdefault(tuple(sorted(genes)), []).append(auc)

    if not module_perf:
        return None

    best_genes = None
    best_auc = -np.inf
    for genes_tup, aucs in module_perf.items():
        auc_mean = np.nanmean(aucs)
        if np.isfinite(auc_mean) and auc_mean > best_auc:
            best_auc = auc_mean
            best_genes = list(genes_tup)

    return best_genes


def run_nested_cv_module_pipeline(expr_df, rt_df, subtype_map,
                                  condition, alpha, beta, gamma,
                                  min_non_na, abs_min, top_k_edges_global,
                                  pos_label, neg_label,
                                  pool_size=2000, min_median_expr=0.0, min_var=0.0,
                                  outer_splits=5, inner_splits=3, seed=1337):
    labeled_samples = [s for s in expr_df.columns if s in subtype_map and subtype_map[s] in {pos_label, neg_label}]
    y_all = np.array([1 if subtype_map[s] == pos_label else 0 for s in labeled_samples], dtype=int)

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(labeled_samples, y_all), start=1):
        tr_samples = [labeled_samples[i] for i in tr_idx]
        te_samples = [labeled_samples[i] for i in te_idx]

        expr_tr_full = expr_df[tr_samples]
        rt_tr_full = rt_df[tr_samples]

        tr_submap = {s: subtype_map[s] for s in tr_samples}

        candidate_genes = build_candidate_gene_pool(
            expr_df=expr_tr_full,
            subtype_map=tr_submap,
            pos_label=pos_label,
            neg_label=neg_label,
            pool_size=pool_size,
            min_median_expr=min_median_expr,
            min_var=min_var
        )
        candidate_genes = list(set(candidate_genes) & set(expr_df.index) & set(rt_df.index))
        if len(candidate_genes) == 0:
            continue

        best_module = select_best_module_nested(
            expr_df=expr_df.loc[candidate_genes],
            rt_df=rt_df.loc[candidate_genes],
            subtype_map=tr_submap,
            candidate_genes=candidate_genes,
            condition=condition,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_non_na=min_non_na,
            abs_min=abs_min,
            top_k_edges_global=top_k_edges_global,
            pos_label=pos_label,
            neg_label=neg_label,
            inner_splits=inner_splits,
            seed=seed + fold
        )
        if best_module is None:
            continue

        score_tr = module_score_vector(
            expr_df=expr_df.loc[candidate_genes],
            rt_df=rt_df.loc[candidate_genes],
            module_genes=best_module,
            condition=condition,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_non_na=min_non_na,
            abs_min=abs_min,
            train_samples=tr_samples if condition == "RT_RNA" else None,
            test_samples=tr_samples if condition == "RT_RNA" else tr_samples
        )
        score_te = module_score_vector(
            expr_df=expr_df.loc[candidate_genes],
            rt_df=rt_df.loc[candidate_genes],
            module_genes=best_module,
            condition=condition,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_non_na=min_non_na,
            abs_min=abs_min,
            train_samples=tr_samples if condition == "RT_RNA" else None,
            test_samples=te_samples if condition == "RT_RNA" else te_samples
        )

        X_tr = score_tr.loc[tr_samples].to_numpy(dtype=float).reshape(-1, 1)
        y_tr = np.array([1 if subtype_map[s] == pos_label else 0 for s in tr_samples], dtype=int)

        X_te = score_te.loc[te_samples].to_numpy(dtype=float).reshape(-1, 1)
        y_te = np.array([1 if subtype_map[s] == pos_label else 0 for s in te_samples], dtype=int)

        clf = LogisticRegression(solver="liblinear", random_state=seed)
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        auc_te = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) == 2 else np.nan

        rows.append({
            "fold": fold,
            "condition": condition,
            "n_train": len(tr_samples),
            "n_test": len(te_samples),
            "n_candidate_genes": len(candidate_genes),
            "n_module_genes": len(best_module),
            "test_auc": auc_te,
            "module_genes": ";".join(best_module)
        })

    return pd.DataFrame(rows)


def run_fixed_gene_baseline_cv(expr_df, subtype_map, gene_list,
                               pos_label, neg_label,
                               outer_splits=5, seed=1337, baseline_name="baseline"):
    keep_genes = [g for g in gene_list if g in expr_df.index]
    if len(keep_genes) == 0:
        return pd.DataFrame()

    labeled_samples = [s for s in expr_df.columns if s in subtype_map and subtype_map[s] in {pos_label, neg_label}]
    y_all = np.array([1 if subtype_map[s] == pos_label else 0 for s in labeled_samples], dtype=int)

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(labeled_samples, y_all), start=1):
        tr_samples = [labeled_samples[i] for i in tr_idx]
        te_samples = [labeled_samples[i] for i in te_idx]

        X_tr = expr_df.loc[keep_genes, tr_samples].T.to_numpy(dtype=float)
        X_te = expr_df.loc[keep_genes, te_samples].T.to_numpy(dtype=float)

        y_tr = np.array([1 if subtype_map[s] == pos_label else 0 for s in tr_samples], dtype=int)
        y_te = np.array([1 if subtype_map[s] == pos_label else 0 for s in te_samples], dtype=int)

        clf = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            max_iter=1000,
            random_state=seed
        )
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        auc_te = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) == 2 else np.nan

        rows.append({
            "fold": fold,
            "baseline": baseline_name,
            "n_genes": len(keep_genes),
            "test_auc": auc_te
        })

    return pd.DataFrame(rows)

def read_gene_list(path):
    if path is None or (isinstance(path, str) and path.strip() == ""):
        return None
    with open(path, "r") as f:
        genes = [ln.strip().split()[0] for ln in f if ln.strip()]
    return set(genes)

def intersect_genes_and_samples(expr, rt=None, keep_samples=None, min_non_na=4):
    expr = ensure_numeric(dedupe_index(expr))
    if rt is not None:
        rt = ensure_numeric(dedupe_index(rt))

    X = expr.copy()

    if keep_samples is None:
        keep_samples = list(X.columns if rt is None else X.columns.intersection(rt.columns))
    keep_samples = [s for s in keep_samples if s in X.columns and (rt is None or s in rt.columns)]
    X = X[keep_samples]

    Xe = X.to_numpy(dtype=float)
    mask_expr = (np.isfinite(Xe).sum(axis=1) >= int(min_non_na))
    X = X.loc[mask_expr]

    if rt is None:
        return X, None

    common_genes = X.index.intersection(rt.index)
    X = X.loc[common_genes]
    Y = rt.loc[common_genes, keep_samples].copy()

    Ye = Y.to_numpy(dtype=float)
    mask_rt = (np.isfinite(Ye).sum(axis=1) >= int(min_non_na))
    X = X.iloc[mask_rt, :]
    Y = Y.iloc[mask_rt, :]
    return X, Y

def rank_transform(df):
    return df.apply(lambda row: row.rank(method="average"), axis=1)

def corr_pairs(df_ranked, min_non_na=4):
    genes = df_ranked.index.to_list()
    X = df_ranked.to_numpy(dtype=float)
    G, N = X.shape
    rows = []
    for i in range(G):
        xi = X[i]
        mask_i = np.isfinite(xi)
        for j in range(i+1, G):
            xj = X[j]
            m = mask_i & np.isfinite(xj)
            n = int(m.sum())
            if n < min_non_na:
                continue
            r = np.corrcoef(xi[m], xj[m])[0,1]
            if np.isfinite(r):
                rows.append((genes[i], genes[j], float(r), n))
    return pd.DataFrame(rows, columns=["gene_u","gene_v","r","n"])

def rt_similarity(rt_ranked, min_non_na=4):
    return corr_pairs(rt_ranked, min_non_na=min_non_na).rename(columns={"r":"r_rt", "n":"n_rt"})

def compose_edges_rna(expr_df, min_non_na=4):
    E = corr_pairs(rank_transform(expr_df), min_non_na=min_non_na)
    E = E.rename(columns={"r":"weight"})
    return E[["gene_u","gene_v","weight"]]

def compose_edges_rt(rt_df, min_non_na=4):
    R = rt_similarity(rank_transform(rt_df), min_non_na=min_non_na)
    R = R.rename(columns={"r_rt":"weight"})
    return R[["gene_u","gene_v","weight"]]

def compose_edges_rtaware(expr_df, rt_df, alpha=1.0, beta=0.5, gamma=0.5, min_non_na=4):
    Er = corr_pairs(rank_transform(expr_df), min_non_na=min_non_na).rename(columns={"r":"r_rna"})
    Rr = rt_similarity(rank_transform(rt_df), min_non_na=min_non_na)
    df = Er.merge(Rr[["gene_u","gene_v","r_rt"]], on=["gene_u","gene_v"], how="inner")
    df["weight"] = alpha*df["r_rna"] + beta*df["r_rt"] + gamma*df["r_rna"]*df["r_rt"]
    return df[["gene_u","gene_v","weight"]]

def compose_edges_pcorr(expr_df, rt_df, min_non_na=4):
    X = expr_df.copy()
    R = rt_df.copy()
    genes = X.index.to_list()
    samples = X.columns.to_list()
    Xe = X.to_numpy(dtype=float)
    Re = R.to_numpy(dtype=float)
    Xe_res = np.full_like(Xe, np.nan, dtype=float)
    for k in range(len(genes)):
        y = Xe[k,:]
        x = Re[k,:]
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < min_non_na:
            continue
        Xmat = np.c_[x[m], np.ones(m.sum())]
        b, *_ = np.linalg.lstsq(Xmat, y[m], rcond=None)
        yhat = Xmat @ b
        e = y[m] - yhat
        Xe_res[k, m] = e
    df_res = pd.DataFrame(Xe_res, index=genes, columns=samples)
    Er = corr_pairs(rank_transform(df_res), min_non_na=min_non_na).rename(columns={"r":"weight"})
    return Er[["gene_u","gene_v","weight"]]

def apply_filters(edges, abs_min=None, top_k=None):
    df = edges.copy()
    if abs_min is not None and not np.isnan(abs_min):
        df = df.loc[df["weight"].abs() >= float(abs_min)]
    if top_k is not None and top_k > 0:
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index)
        if len(df) > top_k:
            df = df.iloc[:top_k].copy()
    return df

def to_graph(edge_df, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for _, r in edge_df.iterrows():
        G.add_edge(r["gene_u"], r["gene_v"], weight=float(r["weight"]))
    return G

def graph_metrics(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return dict(n=0, m=0, density=np.nan, mean_deg=np.nan, trans=np.nan, comps=0)
    degs = np.array([d for _, d in G.degree()], dtype=float)
    density = (2.0*m)/(n*(n-1)) if n>1 else 0.0
    trans = nx.transitivity(G) if n<50000 else np.nan
    comps = nx.number_connected_components(G)
    return dict(n=n, m=m, density=density, mean_deg=float(degs.mean()), trans=trans, comps=comps)

def save_edges(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def save_metrics(path, metrics_dicts):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_dicts).to_csv(path, index=False)

def plot_graph(G, out_png, top_k_plot=2000, seed=1337, title=""):
    if top_k_plot is not None and int(top_k_plot) <= 0:
        return
    if G.number_of_edges() == 0:
        return
    H = G.copy()
    if H.number_of_edges() > top_k_plot:
        e_sorted = sorted(
            H.edges(data=True),
            key=lambda e: abs(e[2].get("weight", 0.0)),
            reverse=True
        )[:top_k_plot]
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(e_sorted)
    pos = nx.spring_layout(H, seed=seed, dim=2)
    arr = np.array([abs(H[u][v].get("weight", 0.0)) for u, v in H.edges()], dtype=float)
    if arr.size == 0:
        return
    wmin = float(np.min(arr))
    wrng = float(np.ptp(arr))
    if not np.isfinite(wrng) or wrng == 0.0:
        widths = np.full_like(arr, 1.0, dtype=float)
    else:
        widths = 0.5 + 2.5 * (arr - wmin) / wrng
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(H, pos, node_size=10, alpha=0.7)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.3)
    plt.axis("off")
    if title:
        plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def lioness_personalized_edges(sim_fn, data_dict, samples, abs_min=None):
    N = len(samples)
    E_full = sim_fn(data_dict, drop_sample=None)
    M_full = E_full.set_index(["gene_u","gene_v"])["weight"]
    results = {}
    for s in samples:
        E_wo = sim_fn(data_dict, drop_sample=s)
        M_wo = E_wo.set_index(["gene_u","gene_v"])["weight"]
        idx = M_full.index.intersection(M_wo.index)
        w_s = N*M_full.loc[idx] - (N-1)*M_wo.loc[idx]
        df = pd.DataFrame(list(idx), columns=["gene_u","gene_v"])
        df["weight"] = w_s.values
        if abs_min is not None:
            df = df.loc[df["weight"].abs() >= float(abs_min)]
        results[s] = df.reset_index(drop=True)
    return results

def sim_rna(data_dict, drop_sample=None, min_non_na=4):
    X = data_dict["expr"]
    if drop_sample is not None and drop_sample in X.columns:
        X = X.drop(columns=[drop_sample])
    return compose_edges_rna(X, min_non_na=min_non_na)

def sim_rt(data_dict, drop_sample=None, min_non_na=4):
    R = data_dict["rt"]
    if drop_sample is not None and drop_sample in R.columns:
        R = R.drop(columns=[drop_sample])
    return compose_edges_rt(R, min_non_na=min_non_na)

def sim_rtaware_factory(alpha, beta, gamma, min_non_na):
    def sim(data_dict, drop_sample=None):
        X = data_dict["expr"]; R = data_dict["rt"]
        if drop_sample is not None:
            if drop_sample in X.columns: X = X.drop(columns=[drop_sample])
            if drop_sample in R.columns: R = R.drop(columns=[drop_sample])
        return compose_edges_rtaware(X, R, alpha=alpha, beta=beta, gamma=gamma, min_non_na=min_non_na)
    return sim

def parse_range(s):
    a0,a1,da = [float(x) for x in s.split(":")]
    n = int(round((a1 - a0) / da)) + 1
    return [a0 + k*da for k in range(n)]

def parse_grid(grid_str):
    a,b,g = grid_str.split(",")
    A = parse_range(a); B = parse_range(b); G = parse_range(g)
    return [(x,y,z) for x in A for y in B for z in G]

def permute_rt_rows(rt_df, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rt_df.shape[0])
    Y = rt_df.iloc[perm].copy()
    Y.index = rt_df.index
    return Y

def degree_scores(G, expr_df):
    deg = dict(G.degree())
    v = np.array([deg.get(g,0.0) for g in expr_df.index], dtype=float)
    X = expr_df.to_numpy(dtype=float)
    s = (v[:,None] * X).sum(axis=0)
    return pd.Series(s, index=expr_df.columns)

def auc_binary(y, scores):
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    m1 = y==1
    m0 = y==0
    n1 = int(m1.sum()); n0 = int(m0.sum())
    if n1==0 or n0==0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    u = r[m1].sum() - n1*(n1+1)/2.0
    return float(u/(n1*n0))

def cindex(times, events, scores):
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    s = np.asarray(scores, dtype=float)
    num = 0.0; den = 0.0
    N = len(t)
    for i in range(N):
        for j in range(N):
            if t[i] < t[j] and e[i]==1:
                den += 1
                if s[i] > s[j]:
                    num += 1
                elif s[i] == s[j]:
                    num += 0.5
    return float(num/den) if den>0 else np.nan

def rt_compartments(rt_df, thr=0.0):
    m = rt_df.median(axis=1).to_numpy(dtype=float)
    lab = np.where(m<thr, "early", "late")
    return pd.Series(lab, index=rt_df.index)

def eval_rt_enrichment(edges_df, comp_ser):
    u = edges_df["gene_u"].map(comp_ser)
    v = edges_df["gene_v"].map(comp_ser)
    cat = pd.Series(np.where((u=="early")&(v=="early"), "EE",
                    np.where((u=="late")&(v=="late"), "LL","EL")))
    counts = cat.value_counts().reindex(["EE","LL","EL"]).fillna(0).astype(int)
    return counts.to_dict()

def edge_delta_by_comp(edges_a, edges_b, comp_ser):
    A = edges_a.set_index(["gene_u","gene_v"])["weight"]
    B = edges_b.set_index(["gene_u","gene_v"])["weight"]
    idx = A.index.intersection(B.index)
    d = (B.loc[idx] - A.loc[idx]).reset_index()
    d.columns = ["gene_u","gene_v","delta"]
    u = d["gene_u"].map(comp_ser); v = d["gene_v"].map(comp_ser)
    grp = np.where((u=="early")&(v=="early"), "EE",
          np.where((u=="late")&(v=="late"), "LL","EL"))
    d["grp"] = grp
    return d.groupby("grp")["delta"].median().to_dict()

def bootstrap_edge_stability(builder_fn, expr, rt, reps=0, frac=0.8, topk=10000, seed=1337, min_non_na=4):
    if reps<=0:
        return np.nan
    rng = np.random.default_rng(seed)
    samps = expr.columns.to_list()
    sets = []
    for r in range(reps):
        k = max(2, int(round(frac*len(samps))))
        S = rng.choice(samps, size=k, replace=False).tolist()
        E = builder_fn(expr[S], rt[S]) if rt is not None else builder_fn(expr[S], None)
        E = E.reindex(E["weight"].abs().sort_values(ascending=False).index)
        if len(E)>topk:
            E = E.iloc[:topk]
        sets.append(set(zip(E["gene_u"].tolist(), E["gene_v"].tolist())))
    j = []
    for i in range(reps):
        for j2 in range(i+1,reps):
            a = sets[i]; b = sets[j2]
            if len(a|b)==0:
                continue
            j.append(len(a&b)/len(a|b))
    return float(np.mean(j)) if j else np.nan

def communities_and_scores(G, expr_df):
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G)
    except Exception:
        comms = []
    mods = []
    for idx,c in enumerate(comms):
        genes = list(c)
        if len(genes)<2:
            continue
        inter = [g for g in genes if g in expr_df.index]
        if not inter:
            continue
        score = expr_df.loc[inter].mean(axis=0)
        mods.append((f"module_{idx}", inter, score))
    return mods

def subtype_auc_from_scores(scores_ser, subtype_map, pos_label, neg_label):
    idx = scores_ser.index.intersection(pd.Index(list(subtype_map.keys())))
    y = [1 if subtype_map[s]==pos_label else (0 if subtype_map[s]==neg_label else None) for s in idx]
    m = [i for i,v in enumerate(y) if v is not None]
    if not m:
        return np.nan
    yb = [y[i] for i in m]
    sc = scores_ser.loc[idx].to_numpy()[m]
    return auc_binary(yb, sc)

def load_subtypes(path, col, sample_col):
    df = pd.read_csv(path, sep=None, engine="python")
    s = df[sample_col].astype(str).tolist() if sample_col in df.columns else df.iloc[:,0].astype(str).tolist()
    lab = df[col].astype(str).tolist()
    return dict(zip(s, lab))

def load_clinical(path, time_col, event_col, sample_col):
    df = pd.read_csv(path, sep=None, engine="python")
    if sample_col not in df.columns:
        df = df.set_index(df.columns[0]).reset_index().rename(columns={df.columns[0]:sample_col})
    df[sample_col] = df[sample_col].astype(str)
    return df[[sample_col, time_col, event_col]].dropna()

def build_candidate_gene_pool(expr_df, subtype_map, pos_label, neg_label,
                              pool_size=2000, min_median_expr=0.0, min_var=0.0):
    """
    expr_df: genes x samples (numeric)
    subtype_map: dict sample -> label string
    Returns list of selected gene names.
    """

    # restrict to samples that have subtypes
    common_samples = [s for s in expr_df.columns if s in subtype_map]
    expr = expr_df[common_samples].copy()

    # basic QC filters
    med = expr.median(axis=1)
    var = expr.var(axis=1)

    keep = pd.Series(True, index=expr.index)
    if min_median_expr > 0:
        keep &= (med >= float(min_median_expr))
    if min_var > 0:
        keep &= (var >= float(min_var))

    expr = expr.loc[keep].copy()
    if expr.shape[0] == 0:
        raise ValueError("All genes removed by median/variance filtering")

    genes = expr.index.to_list()
    AUCs = []

    for g in genes:
        x = expr.loc[g]
        # construct labels and scores using same mapping as before
        y, s = subtype_labels_from_scores(x, subtype_map, pos_label, neg_label)
        if y.size == 0:
            AUCs.append(np.nan)
            continue
        auc_g = auc_binary(y, s)
        AUCs.append(auc_g)

    auc_ser = pd.Series(AUCs, index=genes, dtype=float)

    # effect size: distance from 0.5 (no discrimination)
    eff = (auc_ser - 0.5).abs()

    eff = eff.dropna()
    if eff.empty:
        raise ValueError("No valid per-gene AUCs for candidate pool")

    # sort descending by effect size
    eff = eff.sort_values(ascending=False)
    if len(eff) > pool_size:
        eff = eff.iloc[:pool_size]

    selected_genes = eff.index.to_list()
    return selected_genes


def main():
    ap = argparse.ArgumentParser(description="RT-aware LIONESS networks with RNA/RT ablations")
    ap.add_argument("--expr", required=True, help="Expression matrix (genes x samples)")
    ap.add_argument("--rt", required=True, help="RT matrix (genes x samples)")
    ap.add_argument("--moffitt_basal", default=None, help="Optional gene list (one per line)")
    ap.add_argument("--moffitt_classical", default=None, help="Optional gene list (one per line)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--abs_min", type=float, default=0.5, help="|weight| threshold for saving/plots")
    ap.add_argument("--top_k_edges_global", type=int, default=0, help="Top-K strongest edges to keep globally (0=all)")
    ap.add_argument("--top_k_edges_plot", type=int, default=2000, help="Top-K strongest edges to show in plots")
    ap.add_argument("--min_non_na", type=int, default=4, help="Min non-NA samples for a pair")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rt_mix", type=str, default="1.0,0.5,0.5", help="alpha,beta,gamma for RT-aware mix")
    ap.add_argument("--rt_shuffles", type=int, default=0)
    ap.add_argument("--grid_mix", type=str, default="")
    ap.add_argument("--partial_corr", type=int, default=0)
    ap.add_argument("--stability_boot", type=int, default=0)
    ap.add_argument("--stability_frac", type=float, default=0.8)
    ap.add_argument("--stability_topk", type=int, default=10000)
    ap.add_argument("--rt_threshold", type=float, default=0.0)
    ap.add_argument("--subtype_tsv", type=str, default="")
    ap.add_argument("--subtype_col", type=str, default="")
    ap.add_argument("--subtype_pos", type=str, default="")
    ap.add_argument("--subtype_neg", type=str, default="")
    ap.add_argument("--subtype_sample_col", type=str, default="")
    ap.add_argument("--clinical_tsv", type=str, default="")
    ap.add_argument("--clinical_time", type=str, default="")
    ap.add_argument("--clinical_event", type=str, default="")
    ap.add_argument("--clinical_sample_col", type=str, default="")
    ap.add_argument("--pool_size", type=int, default=2000,
                    help="Size of candidate gene pool after univariate AUC filtering")
    ap.add_argument("--min_median_expr", type=float, default=0.0,
                    help="Minimum median expression to keep a gene (0 = no filter)")
    ap.add_argument("--min_var", type=float, default=0.0,
                    help="Minimum variance to keep a gene (0 = no filter)")
    ap.add_argument("--run_nested_cv", type=int, default=1)
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--purist_genes", type=str, default="")

    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)


    # --- load data ---
    expr = read_matrix(args.expr)
    rt = read_matrix(args.rt)

    # optional: restrict to Moffitt basal/classical genes
    basal = read_gene_list(args.moffitt_basal)
    classical = read_gene_list(args.moffitt_classical)
    if basal or classical:
        keep = set(expr.index)
        if basal:
            keep &= basal
        if classical:
            keep |= classical
        if len(keep) == 0:
            print("No genes left after Moffitt filtering!", file=sys.stderr)
            sys.exit(2)
        expr = expr.loc[expr.index.intersection(list(keep))]
        rt = rt.loc[rt.index.intersection(expr.index)]

    # intersect genes/samples between expr and rt
    samples_all = [s for s in expr.columns if s in rt.columns]
    print(f"[INFO] Genes after expr/rt intersection: {expr.shape[0]}  |  "
        f"Samples: {expr.shape[1]}",
        file=sys.stderr)

    expr, rt = intersect_genes_and_samples(expr, rt,
                                           keep_samples=samples_all,
                                           min_non_na=args.min_non_na)
    global expr_full, rt_full
    expr_full = expr.copy()
    rt_full = rt.copy()
    samples = expr.columns.tolist()

    # --- load subtype map early ---
    submap = None
    if args.subtype_tsv and args.subtype_col and args.subtype_pos and args.subtype_neg:
        submap = load_subtypes(
            args.subtype_tsv,
            args.subtype_col,
            args.subtype_sample_col or expr.columns.name or "sample"
        )

    # parse RT mix coefficients
    try:
        alpha, beta, gamma = [float(x) for x in args.rt_mix.split(",")]
    except Exception:
        print("--rt_mix must be 'alpha,beta,gamma'", file=sys.stderr)
        sys.exit(2)

    # --- define network conditions ---
    conditions = [
        ("RNA", lambda: compose_edges_rna(expr, min_non_na=args.min_non_na), None),
        ("RT",  lambda: compose_edges_rt(rt,   min_non_na=args.min_non_na), None),
        ("RT_RNA",
         lambda: compose_edges_rtaware(expr, rt,
                                       alpha=alpha, beta=beta, gamma=gamma,
                                       min_non_na=args.min_non_na),
         sim_rtaware_factory(alpha, beta, gamma, args.min_non_na))
    ]

    if args.partial_corr == 1:
        conditions.append(
            ("RT_RNA_PCORR",
             lambda: compose_edges_pcorr(expr, rt, min_non_na=args.min_non_na), None)
        )

    # grid of RT/RNA mixes
    grids = []
    if args.grid_mix:
        for (a, b, g) in parse_grid(args.grid_mix):
            nm = "RT_RNA_a{:.3g}_b{:.3g}_g{:.3g}".format(a, b, g)
            grids.append(
                (nm,
                 (lambda A=a, B=b, G=g:
                  lambda: compose_edges_rtaware(expr, rt,
                                                alpha=A, beta=B, gamma=G,
                                                min_non_na=args.min_non_na))(),
                 None)
            )

    # shuffled RT controls
    shufs = []
    for k in range(int(args.rt_shuffles)):
        rt_sh = permute_rt_rows(rt, seed=args.seed + 10 + k)
        shufs.append(
            (f"RT_RNA_SHUFFLE_{k+1:02d}",
             (lambda R=rt_sh:
              lambda: compose_edges_rtaware(expr, R,
                                            alpha=alpha, beta=beta, gamma=gamma,
                                            min_non_na=args.min_non_na))(),
             None)
        )

    globals_edges = {}
    graphs = {}
    summary_rows = []

    # --- build global networks and metrics ---
    for name, builder, sim_fn in conditions + grids + shufs:
        cdir = outdir / name
        cdir.mkdir(parents=True, exist_ok=True)

        edges_global = builder()
        edges_filtered = apply_filters(edges_global,
                                       abs_min=args.abs_min,
                                       top_k=args.top_k_edges_global)
        save_edges(cdir / "edges_global.tsv", edges_filtered)

        G = to_graph(edges_filtered, nodes=expr.index.tolist())
        metrics = graph_metrics(G)
        save_metrics(cdir / "metrics_global.csv", [metrics])

        plot_graph(
            G,
            cdir / "graph_global.png",
            top_k_plot=args.top_k_edges_plot,
            seed=args.seed,
            title=f"{name} global (|w|≥{args.abs_min}, topK={args.top_k_edges_global})"
        )

        metrics_personal = []
        if name == "RT_RNA" and sim_fn is not None:
            data_dict = {"expr": expr, "rt": rt}
            lioness_edges = lioness_personalized_edges(sim_fn, data_dict, samples,
                                                       abs_min=args.abs_min)
            pdir = cdir / "edges_personalized"
            pdir.mkdir(parents=True, exist_ok=True)
            for s, df_s in lioness_edges.items():
                save_edges(pdir / f"{s}.tsv", df_s)
                Gs = to_graph(df_s, nodes=expr.index.tolist())
                met = graph_metrics(Gs)
                met["sample"] = s
                metrics_personal.append(met)

            if len(samples) > 0:
                idx = np.random.default_rng(args.seed).integers(0, len(samples))
                s0 = samples[idx]
                df0 = lioness_edges[s0]
                G0 = to_graph(df0, nodes=expr.index.tolist())
                plot_graph(
                    G0,
                    cdir / f"graph_personalized_{s0}.png",
                    top_k_plot=args.top_k_edges_plot,
                    seed=args.seed,
                    title=f"{name} LIONESS: {s0}"
                )

        save_metrics(cdir / "metrics_personalized.csv", metrics_personal)
        globals_edges[name] = edges_filtered
        graphs[name] = G
        mg = dict(metrics)
        mg["condition"] = name
        summary_rows.append(mg)

    pd.DataFrame(summary_rows).to_csv(outdir / "ablation_metrics_global.csv", index=False)

    evaldir = outdir / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)

    # --- RT compartment enrichment / rewiring ---
    if "RNA" in graphs and "RT_RNA" in graphs:
        comp = rt_compartments(rt, thr=args.rt_threshold)
        enr_rna = eval_rt_enrichment(globals_edges["RNA"], comp)
        enr_mix = eval_rt_enrichment(globals_edges["RT_RNA"], comp)
        rew = edge_delta_by_comp(globals_edges["RNA"], globals_edges["RT_RNA"], comp)
        pd.DataFrame(
            [{"condition": "RNA", **enr_rna},
             {"condition": "RT_RNA", **enr_mix}]
        ).to_csv(evaldir / "rt_compartment_enrichment.csv", index=False)
        pd.DataFrame([rew]).to_csv(evaldir / "edge_rewiring_by_rt_compartment.csv", index=False)

    # --- edge stability bootstraps ---
    if args.stability_boot > 0:
        rows = []
        for name in ["RNA", "RT", "RT_RNA"]:
            if name in globals_edges:
                if name == "RNA":
                    builder_fn = lambda Ex, RtNone: compose_edges_rna(Ex,
                                                                      min_non_na=args.min_non_na)
                    stab = bootstrap_edge_stability(
                        builder_fn, expr, None,
                        reps=args.stability_boot,
                        frac=args.stability_frac,
                        topk=args.stability_topk,
                        seed=args.seed,
                        min_non_na=args.min_non_na
                    )
                elif name == "RT":
                    builder_fn = lambda ExNone, Rt: compose_edges_rt(Rt,
                                                                     min_non_na=args.min_non_na)
                    stab = bootstrap_edge_stability(
                        builder_fn, expr, rt,
                        reps=args.stability_boot,
                        frac=args.stability_frac,
                        topk=args.stability_topk,
                        seed=args.seed,
                        min_non_na=args.min_non_na
                    )
                else:
                    builder_fn = lambda Ex, Rt: compose_edges_rtaware(
                        Ex, Rt,
                        alpha=alpha, beta=beta, gamma=gamma,
                        min_non_na=args.min_non_na
                    )
                    stab = bootstrap_edge_stability(
                        builder_fn, expr, rt,
                        reps=args.stability_boot,
                        frac=args.stability_frac,
                        topk=args.stability_topk,
                        seed=args.seed,
                        min_non_na=args.min_non_na
                    )
                rows.append({"condition": name, "edge_stability_jaccard": stab})
        pd.DataFrame(rows).to_csv(evaldir / "stability_edge_jaccard.csv", index=False)

    # --- subtype AUC from degree scores ---
    if submap is not None:
        rows = []
        for name in ["RNA", "RT", "RT_RNA"]:
            if name in graphs:
                sc = degree_scores(graphs[name], expr if name != "RT" else rt)
                auc = subtype_auc_from_scores(sc, submap,
                                              args.subtype_pos, args.subtype_neg)
                rows.append({"condition": name, "subtype_auc": auc})
        if rows:
            pd.DataFrame(rows).to_csv(evaldir / "subtype_auc.csv", index=False)

    # --- module scoring logic (unchanged, just reusing submap) ---

    # --- survival association (c-index) ---
    if args.clinical_tsv and args.clinical_time and args.clinical_event:
        clin = load_clinical(args.clinical_tsv,
                             args.clinical_time,
                             args.clinical_event,
                             args.clinical_sample_col or "sample")
        rows = []
        for name in ["RNA", "RT_RNA"]:
            if name in graphs:
                sc = degree_scores(graphs[name], expr if name != "RT" else rt)
                dfm = pd.DataFrame({"sample": sc.index, "score": sc.values}) \
                    .merge(clin, left_on="sample",
                           right_on=(args.clinical_sample_col or "sample"))
                ci = cindex(dfm[args.clinical_time].to_numpy(),
                            dfm[args.clinical_event].to_numpy(),
                            dfm["score"].to_numpy())
                rows.append({"condition": name, "cindex": ci})
        if rows:
            pd.DataFrame(rows).to_csv(evaldir / "survival_cindex.csv", index=False)
    # --- leakage-safe nested CV ---
    if submap is not None and args.run_nested_cv == 1:
        nested_dir = outdir / "nested_cv"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # nested RNA
        df_nested_rna = run_nested_cv_module_pipeline(
            expr_df=expr,
            rt_df=rt,
            subtype_map=submap,
            condition="RNA",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_non_na=args.min_non_na,
            abs_min=args.abs_min,
            top_k_edges_global=args.top_k_edges_global,
            pos_label=args.subtype_pos,
            neg_label=args.subtype_neg,
            pool_size=args.pool_size,
            min_median_expr=args.min_median_expr,
            min_var=args.min_var,
            outer_splits=args.outer_folds,
            inner_splits=args.inner_folds,
            seed=args.seed
        )
        df_nested_rna.to_csv(nested_dir / "nested_cv_module_RNA.csv", index=False)

        # nested RT_RNA
        df_nested_rtrna = run_nested_cv_module_pipeline(
            expr_df=expr,
            rt_df=rt,
            subtype_map=submap,
            condition="RT_RNA",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_non_na=args.min_non_na,
            abs_min=args.abs_min,
            top_k_edges_global=args.top_k_edges_global,
            pos_label=args.subtype_pos,
            neg_label=args.subtype_neg,
            pool_size=args.pool_size,
            min_median_expr=args.min_median_expr,
            min_var=args.min_var,
            outer_splits=args.outer_folds,
            inner_splits=args.inner_folds,
            seed=args.seed
        )
        df_nested_rtrna.to_csv(nested_dir / "nested_cv_module_RT_RNA.csv", index=False)

        summary_rows = []
        for nm, df_ in [("RNA", df_nested_rna), ("RT_RNA", df_nested_rtrna)]:
            if len(df_) > 0:
                summary_rows.append({
                    "model": nm,
                    "mean_test_auc": float(df_["test_auc"].mean()),
                    "std_test_auc": float(df_["test_auc"].std(ddof=1)) if len(df_) > 1 else np.nan,
                    "n_folds": int(len(df_))
                })

        # Moffitt baseline
        moffitt_union = []
        if basal is not None:
            moffitt_union.extend(list(basal))
        if classical is not None:
            moffitt_union.extend(list(classical))
        moffitt_union = sorted(set(moffitt_union))

        if len(moffitt_union) > 0:
            df_moffitt = run_fixed_gene_baseline_cv(
                expr_df=expr,
                subtype_map=submap,
                gene_list=moffitt_union,
                pos_label=args.subtype_pos,
                neg_label=args.subtype_neg,
                outer_splits=args.outer_folds,
                seed=args.seed,
                baseline_name="Moffitt"
            )
            df_moffitt.to_csv(nested_dir / "nested_cv_baseline_Moffitt.csv", index=False)
            if len(df_moffitt) > 0:
                summary_rows.append({
                    "model": "Moffitt",
                    "mean_test_auc": float(df_moffitt["test_auc"].mean()),
                    "std_test_auc": float(df_moffitt["test_auc"].std(ddof=1)) if len(df_moffitt) > 1 else np.nan,
                    "n_folds": int(len(df_moffitt))
                })

        # PURIST baseline
        if args.purist_genes:
            purist_set = read_gene_list(args.purist_genes)
            if purist_set:
                df_purist = run_fixed_gene_baseline_cv(
                    expr_df=expr,
                    subtype_map=submap,
                    gene_list=sorted(purist_set),
                    pos_label=args.subtype_pos,
                    neg_label=args.subtype_neg,
                    outer_splits=args.outer_folds,
                    seed=args.seed,
                    baseline_name="PURIST"
                )
                df_purist.to_csv(nested_dir / "nested_cv_baseline_PURIST.csv", index=False)
                if len(df_purist) > 0:
                    summary_rows.append({
                        "model": "PURIST",
                        "mean_test_auc": float(df_purist["test_auc"].mean()),
                        "std_test_auc": float(df_purist["test_auc"].std(ddof=1)) if len(df_purist) > 1 else np.nan,
                        "n_folds": int(len(df_purist))
                    })

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                nested_dir / "nested_cv_summary.csv",
                index=False
            )
    # --- grid mix metrics (subtype AUC / c-index) ---
    if args.grid_mix:
        rows = []
        for (a, b, g) in parse_grid(args.grid_mix):
            nm = "RT_RNA_a{:.3g}_b{:.3g}_g{:.3g}".format(a, b, g)
            if nm in graphs:
                sc = degree_scores(graphs[nm], expr)
                row = {"condition": nm}
                if submap is not None:
                    row["subtype_auc"] = subtype_auc_from_scores(
                        sc, submap, args.subtype_pos, args.subtype_neg
                    )
                if args.clinical_tsv and args.clinical_time and args.clinical_event:
                    clin = load_clinical(args.clinical_tsv,
                                         args.clinical_time,
                                         args.clinical_event,
                                         args.clinical_sample_col or "sample")
                    dfm = pd.DataFrame({"sample": sc.index, "score": sc.values}) \
                        .merge(clin, left_on="sample",
                               right_on=(args.clinical_sample_col or "sample"))
                    row["cindex"] = cindex(dfm[args.clinical_time].to_numpy(),
                                           dfm[args.clinical_event].to_numpy(),
                                           dfm["score"].to_numpy())
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(evaldir / "grid_mix_metrics.csv", index=False)
    print("Done.")
    
    
if __name__ == "__main__":
    main()
