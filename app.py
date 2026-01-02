import io
import base64
import os

from flask import Flask, render_template_string
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib
# 서버 환경에서는 화면이 없으므로 'Agg' 백엔드를 사용해야 에러가 안 납니다.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

app = Flask(__name__)

def run_analysis_logic():
    # 1) Load Data (GitHub에 올린 파일명과 일치해야 함)
    # 엑셀 파일이 같은 폴더에 있다고 가정합니다.
    df = pd.read_excel("CND_rawdata.xlsx", sheet_name="sheet1")

    # 2) Variable Setup
    climate_cols = ['PM25', 'MeanT', 'RH', 'PR', 'Heat_index']
    hvi_ncd_cols = [
        'Cardiovascular diseases', 'Chronic respiratory diseases', 
        'Diabetes and kidney diseases', 'Neoplasms', 
        'Neurological disorders', 'Digestive diseases', 
        'Musculoskeletal disorders'
    ]
    hvi_id_cols = [
        'Respiratory infections and tuberculosis', 'Enteric infections',
        'Neglected tropical diseases and malaria', 'Other infectious diseases',
        'Maternal and neonatal disorders', 'Nutritional deficiencies'
    ]
    dac_cols = ['PHDI_index']
    sbc_cols = ['gdp', 'trade', 'urb']
    cluster_cols = ['C_cluster']

    needed = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols + cluster_cols
    df2 = df.dropna(subset=needed).copy()

    # 3) Z-score helper
    def zseries(s):
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    for col in climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols:
        df2[col + "_z"] = zseries(df2[col])

    # 4) PCA Function
    def pca_pc1_pc2_index(df_in, cols, index_name):
        Z = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        # Handle finite check simply
        Zok = Z
        
        Zc = Zok - Zok.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(Zc, full_matrices=False)
        var_exp = (S**2) / (S**2).sum()
        v1, v2 = float(var_exp[0]), float(var_exp[1])
        load1, load2 = VT[0], VT[1]
        
        w_raw = np.abs(load1) * v1 + np.abs(load2) * v2
        w = w_raw / w_raw.sum()
        
        Zfull = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        score = Zfull @ w
        return pd.Series(score, index=df_in.index, name=index_name)

    # 5) Build Indices
    df2["CEI"] = pca_pc1_pc2_index(df2, climate_cols, "CEI")
    df2["SBC"] = pca_pc1_pc2_index(df2, sbc_cols, "SBC")
    df2["HVI_NCD"] = pca_pc1_pc2_index(df2, hvi_ncd_cols, "HVI_NCD")
    df2["HVI_ID"] = pca_pc1_pc2_index(df2, hvi_id_cols, "HVI_ID")
    df2["DAC"] = df2["PHDI_index_z"]

    # 6) CARE-DDI Calculation
    w_CEI, w_HVI_NCD, w_HVI_ID, w_DAC, w_SBC = 0.30, 0.20, 0.15, 0.20, 0.15
    df2["CARE_DDI"] = (
        w_CEI * df2["CEI"] + w_HVI_NCD * df2["HVI_NCD"] + 
        w_HVI_ID * df2["HVI_ID"] - w_DAC * df2["DAC"] - w_SBC * df2["SBC"]
    )
    
    # 7) Statistics
    df2["C_cluster"] = pd.to_numeric(df2["C_cluster"], errors="coerce").astype("Int64")
    stats_text = ""
    groups = [df2.loc[df2["C_cluster"] == k, "CARE_DDI"].dropna().values 
              for k in sorted(df2["C_cluster"].dropna().unique())]
    
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        stats_text = f"Kruskal-Wallis H = {H:.4f}, p-value = {p:.4e}"

    # 8) Visualization (Panel A, B, C)
    sns.set(style="whitegrid", font_scale=1.0)
    radar_vars = ["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC"]
    agg = df2.groupby("C_cluster")[radar_vars].mean()
    agg_norm = (agg - agg.min()) / (agg.max() - agg.min())
    clusters = agg_norm.index.tolist()
    
    N = len(radar_vars)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    # PCA for Panel C
    X_pca = df2[radar_vars]
    scaled = StandardScaler().fit_transform(X_pca)
    pca = PCA(n_components=2)
    PC = pca.fit_transform(scaled)
    loadings = pca.components_.T

    # Draw Plot
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1.7])

    # Panel A
    ax1 = plt.subplot(gs[0])
    sns.violinplot(data=df2, x="C_cluster", y="CARE_DDI", palette="Set2", inner=None, ax=ax1)
    sns.boxplot(data=df2, x="C_cluster", y="CARE_DDI", width=0.2, boxprops={"facecolor":"white"}, showfliers=False, ax=ax1)
    ax1.set_title("A. CARE-DDI Distribution")

    # Panel B (Radar)
    ax2 = plt.subplot(gs[1], polar=True)
    for cluster in clusters:
        values = agg_norm.loc[cluster, radar_vars].tolist()
        values += values[:1]
        ax2.plot(angles, values, label=f"C{cluster}")
        ax2.fill(angles, values, alpha=0.1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_vars)
    ax2.set_title("B. Radar Chart")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Panel C (Biplot)
    ax3 = plt.subplot(gs[2])
    sc = ax3.scatter(PC[:, 0], PC[:, 1], c=df2["CARE_DDI"], cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax3, label="CARE-DDI")
    for i, var in enumerate(radar_vars):
        ax3.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='r', head_width=0.1)
        ax3.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, var, color='r')
    ax3.set_title("C. PCA Biplot")

    plt.tight_layout()
    
    # Save to BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode(), stats_text

@app.route('/')
def home():
    try:
        plot_url, stats = run_analysis_logic()
        html = f"""
        <html>
            <head><title>CARE-DDI Analysis Platform</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 20px;">
                <h1>CARE-DDI Analysis Results</h1>
                <h3 style="color: blue;">{stats}</h3>
                <hr>
                <img src="data:image/png;base64,{plot_url}" style="max-width: 100%; height: auto;">
            </body>
        </html>
        """
        return render_template_string(html)
    except Exception as e:
        return f"<h1>Error:</h1><p>{str(e)}</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
