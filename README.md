import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt

# ============================================================
# 1) Load raw CND data
# ============================================================
df = pd.read_excel(
    "/content/drive/MyDrive/CND_cluster/CND_rawdata.xlsx",
    sheet_name="sheet1"
)

# ============================================================
# 2) Variable groups (UPDATED)
# ============================================================
# CEI (Climate Exposure Index)
climate_cols = ['PM25', 'MeanT', 'RH', 'PR', 'Heat_index']

# HVI–NCD
hvi_ncd_cols = [
    'Cardiovascular diseases',
    'Chronic respiratory diseases',
    'Diabetes and kidney diseases',
    'Neoplasms',
    'Neurological disorders',
    'Digestive diseases',
    'Musculoskeletal disorders'
]

# HVI–Infectious / Nutritional / Maternal
hvi_id_cols = [
    'Respiratory infections and tuberculosis',
    'Enteric infections',
    'Neglected tropical diseases and malaria',
    'Other infectious diseases',
    'Maternal and neonatal disorders',
    'Nutritional deficiencies'
]

# DAC (Dietary Adaptation Capacity)
dac_cols = ['PHDI_index']

# SBC (Socioeconomic Buffering Capacity)
sbc_cols = ['gdp', 'trade', 'urb']

# Cluster label(s)
cluster_cols = ['C_cluster']  # keep only what you need for tests/stratification

needed = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols + cluster_cols

# ============================================================
# 3) Keep complete rows
# ============================================================
df2 = df.dropna(subset=needed).copy()

# ============================================================
# 4) z-score helper
# ============================================================
def zseries(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd

# ============================================================
# 5) Compute z-scores for all component variables
# ============================================================
for col in climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols:
    df2[col + "_z"] = zseries(df2[col])

# ============================================================
# 6) PCA PC1+PC2 (variance-weighted) index constructor + biplot
# ============================================================
def pca_pc1_pc2_index(
    df: pd.DataFrame,
    cols: list,
    index_name: str,
    plot_biplot: bool = True,
    arrow_scale: float = 3.0,
):
    """
    Builds an index = sum_j w_j * z_j
    where w_j are derived from abs(loadings) of PC1 and PC2,
    weighted by their variance explained.

    Returns:
      score_series, weight_series, info_dict
    """
    Z = df[[c + "_z" for c in cols]].to_numpy(dtype=float)
    ok = np.isfinite(Z).all(axis=1)
    Zok = Z[ok]

    # PCA via SVD
    Zc = Zok - Zok.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Zc, full_matrices=False)

    # variance explained
    var_exp = (S**2) / (S**2).sum()
    v1, v2 = float(var_exp[0]), float(var_exp[1])

    # loadings for PC1 and PC2
    load1 = VT[0]
    load2 = VT[1]

    # weights using PC1 + PC2
    w_raw = np.abs(load1) * v1 + np.abs(load2) * v2
    w = w_raw / w_raw.sum()

    # compute index for all rows
    Zfull = df[[c + "_z" for c in cols]].to_numpy(dtype=float)
    score = Zfull @ w

    score_series = pd.Series(score, index=df.index, name=index_name)
    weight_series = pd.Series(w, index=cols, name=f"{index_name}_weights")

    info = {
        "var_exp_pc1": v1,
        "var_exp_pc2": v2,
        "loadings_pc1": pd.Series(load1, index=cols),
        "loadings_pc2": pd.Series(load2, index=cols),
    }

    # ---- biplot ----
    if plot_biplot:
        scores = U[:, :2] @ np.diag(S[:2])
        loadings = VT[:2, :].T

        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.25, color="gray", s=18)

        for i, var in enumerate(cols):
            ax.arrow(
                0, 0,
                loadings[i, 0] * arrow_scale,
                loadings[i, 1] * arrow_scale,
                color="black", width=0.01, head_width=0.10,
                length_includes_head=True
            )
            ax.text(
                loadings[i, 0] * arrow_scale * 1.12,
                loadings[i, 1] * arrow_scale * 1.12,
                var, fontsize=9, ha="center", va="center"
            )

        ax.axhline(0, color="lightgray", lw=0.8)
        ax.axvline(0, color="lightgray", lw=0.8)
        ax.set_xlabel(f"PC1 ({v1:.1%} variance)")
        ax.set_ylabel(f"PC2 ({v2:.1%} variance)")
        ax.set_title(f"PCA biplot: {index_name}")
        plt.tight_layout()
        plt.show()

    return score_series, weight_series, info

# ============================================================
# 7) Build sub-indices (DATA-DRIVEN within-domain)
# ============================================================
df2["CEI"], cei_w, cei_info = pca_pc1_pc2_index(df2, climate_cols, "CEI", plot_biplot=True)
df2["SBC"], sbc_w, sbc_info = pca_pc1_pc2_index(df2, sbc_cols, "SBC", plot_biplot=True)
df2["HVI_NCD"], hvi_ncd_w, hvi_ncd_info = pca_pc1_pc2_index(df2, hvi_ncd_cols, "HVI_NCD", plot_biplot=True)
df2["HVI_ID"], hvi_id_w, hvi_id_info = pca_pc1_pc2_index(df2, hvi_id_cols, "HVI_ID", plot_biplot=True)

# DAC is directly PHDI_index z-score (protective capacity)
df2["DAC"] = df2["PHDI_index_z"]

# ============================================================
# 8) CARE-DDI (PRIOR-KNOWLEDGE weights across domains)
#    vulnerability-oriented: +CEI +HVI_NCD +HVI_ID -DAC -SBC
# ============================================================
w_CEI = 0.30
w_HVI_NCD = 0.20
w_HVI_ID = 0.15
w_DAC = 0.20
w_SBC = 0.15

df2["CARE_DDI"] = (
    w_CEI * df2["CEI"]
  + w_HVI_NCD * df2["HVI_NCD"]
  + w_HVI_ID * df2["HVI_ID"]
  - w_DAC * df2["DAC"]
  - w_SBC * df2["SBC"]
)

# ============================================================
# 9) Optional: separation test by climate cluster
# ============================================================
df2["C_cluster"] = pd.to_numeric(df2["C_cluster"], errors="coerce").astype("Int64")

groups = [
    df2.loc[df2["C_cluster"] == k, "CARE_DDI"].dropna().values
    for k in sorted(df2["C_cluster"].dropna().unique())
]

if len(groups) >= 2:
    H, p = kruskal(*groups)
    print("Kruskal–Wallis H =", H)
    print("p-value =", p)

print("\nCARE_DDI by climate cluster:")
print(df2.groupby("C_cluster")["CARE_DDI"].describe())

# ============================================================
# 10) Print weight summaries (useful for Supplementary)
# ============================================================
print("\n--- Within-domain PCA weights ---")
print("\nCEI weights:\n", cei_w.sort_values(ascending=False))
print("\nSBC weights:\n", sbc_w.sort_values(ascending=False))
print("\nHVI_NCD weights:\n", hvi_ncd_w.sort_values(ascending=False))
print("\nHVI_ID weights:\n", hvi_id_w.sort_values(ascending=False))

print("\n--- Across-domain prior weights (CARE-DDI) ---")
print({
    "CEI": w_CEI,
    "HVI_NCD": w_HVI_NCD,
    "HVI_ID": w_HVI_ID,
    "DAC": w_DAC,
    "SBC": w_SBC
})

df2.head()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

sns.set(style="whitegrid", font_scale=1.2)

# =====================================================
# 1. PREPARE AGGREGATED RADAR DATA (UPDATED)
# =====================================================

radar_vars = ["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC"]

agg = df2.groupby("C_cluster")[radar_vars].mean()

# Normalize 0–1 for radar comparability
agg_norm = (agg - agg.min()) / (agg.max() - agg.min())
clusters = agg_norm.index.tolist()

# Radar configuration
labels = radar_vars
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])


# =====================================================
# 2. PCA FOR PANEL C (UPDATED)
# =====================================================

X = df2[radar_vars]

scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
PC = pca.fit_transform(scaled)
loadings = pca.components_.T


# =====================================================
# 3. PANEL FIGURE LAYOUT
# =====================================================

fig = plt.figure(figsize=(22, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1.7])


# =====================================================
# PANEL A — Violin + Boxplot of CARE-DDI
# =====================================================

ax1 = plt.subplot(gs[0])

sns.violinplot(
    data=df2,
    x="C_cluster",
    y="CARE_DDI",
    palette="Set2",
    inner=None,
    linewidth=1,
    alpha=0.8,
    ax=ax1
)

sns.boxplot(
    data=df2,
    x="C_cluster",
    y="CARE_DDI",
    width=0.25,
    showcaps=True,
    boxprops={"facecolor": "white", "zorder": 2},
    showfliers=False,
    whiskerprops={"linewidth": 2},
    ax=ax1
)

sns.stripplot(
    data=df2,
    x="C_cluster",
    y="CARE_DDI",
    color="black",
    alpha=0.4,
    size=3,
    ax=ax1
)

ax1.set_title("A. CARE-DDI Distribution by Climate Cluster")
ax1.set_xlabel("Climate Cluster")
ax1.set_ylabel("CARE-DDI (Higher = Greater Vulnerability)")


# =====================================================
# PANEL B — RADAR PLOT (UPDATED)
# =====================================================

ax2 = plt.subplot(gs[1], polar=True)

for cluster in clusters:
    values = agg_norm.loc[cluster, labels].tolist()
    values += values[:1]
    ax2.plot(angles, values, label=f"C{cluster}", linewidth=2)
    ax2.fill(angles, values, alpha=0.15)

ax2.set_title(
    "B. Climate–Diet–Disease Profiles by Climate Cluster\n"
    "(CEI, HVI-NCD, HVI-ID, DAC, SBC)",
    pad=20
)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels)
ax2.set_yticklabels([])
ax2.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15))


# =====================================================
# PANEL C — PCA Biplot (UPDATED)
# =====================================================

ax3 = plt.subplot(gs[2])

sc = ax3.scatter(
    PC[:, 0], PC[:, 1],
    c=df2["CARE_DDI"],
    cmap="viridis",
    s=80,
    edgecolor="black",
    alpha=0.85
)

cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label("CARE-DDI\n(Higher = Greater Vulnerability)")

# Add loadings (arrows)
for i, var in enumerate(radar_vars):
    ax3.arrow(
        0, 0,
        loadings[i, 0] * 3,
        loadings[i, 1] * 3,
        color="darkred",
        head_width=0.25,
        linewidth=2.5,
        length_includes_head=True
    )
    ax3.text(
        loadings[i, 0] * 3.2,
        loadings[i, 1] * 3.2,
        var,
        color="darkred",
        fontsize=14,
        ha="center",
        va="center"
    )

ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax3.set_title("C. PCA of CARE Components\n(Structural Vulnerability Space)")
ax3.grid(True)

plt.tight_layout()
plt.show()
