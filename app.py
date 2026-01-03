import io
import base64
import os

from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib
# 서버 환경용 설정 (화면 없이 그림 그리기)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

# UMAP 라이브러리 임포트 (설치 필요: pip install umap-learn)
try:
    import umap
except ImportError:
    import umap.umap_ as umap

app = Flask(__name__)

# ==========================================
# 1. 기본 설정 및 데이터 로드
# ==========================================
def load_data():
    df = pd.read_excel("CND_rawdata.xlsx", sheet_name="sheet1")
    return df

DEFAULT_VARS = {
    'climate': ['PM25', 'MeanT', 'RH', 'PR', 'Heat_index'],
    'hvi_ncd': [
        'Cardiovascular diseases', 'Chronic respiratory diseases', 
        'Diabetes and kidney diseases', 'Neoplasms', 
        'Neurological disorders', 'Digestive diseases', 
        'Musculoskeletal disorders'
    ],
    'hvi_id': [
        'Respiratory infections and tuberculosis', 'Enteric infections',
        'Neglected tropical diseases and malaria', 'Other infectious diseases',
        'Maternal and neonatal disorders', 'Nutritional deficiencies'
    ],
    'dac': ['PHDI_index'],
    'sbc': ['gdp', 'trade', 'urb'],
    'cluster_c': 'C_cluster',
    'cluster_n': 'N_cluster',
    'cluster_d': 'D_cluster'
}

# ==========================================
# 2. 분석 및 시각화 로직
# ==========================================
def run_analysis_logic(df, climate_cols, hvi_ncd_cols, hvi_id_cols, dac_cols, sbc_cols, 
                       c_col, n_col, d_col, graph_type):
    
    # 1) 필요한 컬럼 추출
    needed = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols + [c_col, n_col, d_col]
    df2 = df.dropna(subset=needed).copy()

    # 2) Row-level Z-score (기본 지표 산출용)
    def zseries(s):
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd): return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    calc_cols = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols
    for col in calc_cols:
        df2[col + "_z"] = zseries(df2[col])

    # 3) PCA Index Function
    def pca_pc1_pc2_index(df_in, cols, index_name):
        if not cols: return pd.Series(0, index=df_in.index, name=index_name)
        Z = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(Zc, full_matrices=False)
        if len(S) < 2: return pd.Series(0, index=df_in.index, name=index_name)
        var_exp = (S**2) / (S**2).sum()
        v1, v2 = float(var_exp[0]), float(var_exp[1])
        w = (np.abs(VT[0])*v1 + np.abs(VT[1])*v2)
        w = w / w.sum()
        score = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float) @ w
        return pd.Series(score, index=df_in.index, name=index_name)

    # 4) 도메인 지표 산출
    df2["CEI"] = pca_pc1_pc2_index(df2, climate_cols, "CEI")
    df2["SBC"] = pca_pc1_pc2_index(df2, sbc_cols, "SBC")
    df2["HVI_NCD"] = pca_pc1_pc2_index(df2, hvi_ncd_cols, "HVI_NCD")
    df2["HVI_ID"] = pca_pc1_pc2_index(df2, hvi_id_cols, "HVI_ID")
    if len(dac_cols) == 1: df2["DAC"] = df2[dac_cols[0] + "_z"]
    else: df2["DAC"] = pca_pc1_pc2_index(df2, dac_cols, "DAC")

    # 5) Row-level CARE_DDI
    w_CEI, w_HVI_NCD, w_HVI_ID, w_DAC, w_SBC = 0.30, 0.20, 0.15, 0.20, 0.15
    df2["CARE_DDI"] = (w_CEI * df2["CEI"] + w_HVI_NCD * df2["HVI_NCD"] + 
                       w_HVI_ID * df2["HVI_ID"] - w_DAC * df2["DAC"] - w_SBC * df2["SBC"])

    # 6) 통계 (C_cluster 기준)
    df2[c_col] = pd.to_numeric(df2[c_col], errors="coerce").astype("Int64")
    stats_text = "Analysis Ready."
    groups_val = [df2.loc[df2[c_col] == k, "CARE_DDI"].dropna().values for k in sorted(df2[c_col].dropna().unique())]
    if len(groups_val) >= 2:
        H, p = kruskal(*groups_val)
        stats_text = f"Kruskal-Wallis (by {c_col}): H={H:.4f}, p={p:.4e}"

    # =========================================================
    # 7) 시각화 분기 (그래프 타입 선택)
    # =========================================================
    
    if graph_type == 'heatmap':
        # [옵션 2] Heatmap
        climates = sorted(df2[c_col].unique())
        if len(climates) == 0: return None, "Error: No climate groups found."
        
        sns.set(style="white", font_scale=1.1)
        fig, axes = plt.subplots(1, len(climates), figsize=(5 * len(climates), 5), sharey=True)
        if len(climates) == 1: axes = [axes]

        for ax, c in zip(axes, climates):
            sub = df2[df2[c_col] == c]
            mat = sub.pivot_table(index=n_col, columns=d_col, values="CARE_DDI", aggfunc="mean")
            sns.heatmap(mat, ax=ax, cmap="RdYlGn_r", center=0, annot=True, fmt=".2f",
                        cbar=(ax is axes[-1]), cbar_kws={"label": "Mean CARE-DDI"})
            ax.set_title(f"Climate Cluster: {c}")
            ax.set_xlabel(f"Disease ({d_col})")
            if ax is axes[0]: ax.set_ylabel(f"Diet ({n_col})")
            else: ax.set_ylabel("")
        plt.suptitle("CARE-DDI Heatmap (C x N x D)", y=1.05, fontsize=16)

    elif graph_type == 'cnd_umap':
        # [옵션 3] Bar & UMAP Analysis (요청하신 코드 반영)
        
        # 클러스터 변수 전처리
        for col in [c_col, n_col, d_col]:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").astype("Int64")
        
        # 통합 클러스터 이름 생성 (예: C1N2D3)
        df2["cluster_n"] = (
            "C" + df2[c_col].astype(str) + 
            "N" + df2[n_col].astype(str) + 
            "D" + df2[d_col].astype(str)
        )

        features = ["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC", "CARE_DDI"]
        # 그룹별 평균 계산
        df_sum = df2.dropna(subset=["cluster_n"] + features).groupby("cluster_n")[features].mean().reset_index()

        # 집계 데이터용 z-score 헬퍼
        def z_agg(x):
            sd = x.std(ddof=0)
            if sd == 0 or np.isnan(sd): return pd.Series(np.zeros(len(x)), index=x.index)
            return (x - x.mean()) / sd

        # 클러스터 레벨 CARE-DDI 재계산 (가중치 적용)
        df_sum["CARE_DDI_weighted"] = (
            w_CEI * z_agg(df_sum["CEI"]) + w_HVI_NCD * z_agg(df_sum["HVI_NCD"]) +
            w_HVI_ID * z_agg(df_sum["HVI_ID"]) - w_DAC * z_agg(df_sum["DAC"]) - w_SBC * z_agg(df_sum["SBC"])
        )

        # 그림 그리기
        sns.set(style="whitegrid", font_scale=1.0)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Panel A: Bar Plot
        df_plot = df_sum.sort_values("CARE_DDI_weighted", ascending=False)
        axes[0].bar(df_plot["cluster_n"], df_plot["CARE_DDI_weighted"], color='skyblue', edgecolor='black')
        axes[0].tick_params(axis="x", rotation=90)
        axes[0].set_ylabel("CARE-DDI (cluster-level, z-weighted)")
        axes[0].set_title("A. CARE-DDI by C×N×D Cluster\n(Higher = Greater Vulnerability)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Panel B: UMAP
        X_u = df_sum[["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC"]].apply(pd.to_numeric, errors="coerce")
        ok = np.isfinite(X_u).all(axis=1)
        df_umap = df_sum.loc[ok].reset_index(drop=True)
        X_scaled = StandardScaler().fit_transform(X_u.loc[ok])

        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
        X_emb = reducer.fit_transform(X_scaled)

        sc = axes[1].scatter(X_emb[:, 0], X_emb[:, 1], c=df_umap["CARE_DDI_weighted"], 
                             cmap="viridis", s=100, edgecolor="black", alpha=0.85)
        cbar = plt.colorbar(sc, ax=axes[1])
        cbar.set_label("CARE-DDI (Higher = More Vulnerability)")

        # 텍스트 라벨 추가
        for i, txt in enumerate(df_umap["cluster_n"]):
            axes[1].annotate(str(txt), (X_emb[i, 0], X_emb[i, 1]), fontsize=9, alpha=0.9)

        axes[1].set_title("B. UMAP Projection of C×N×D Typologies")
        axes[1].set_xlabel("UMAP-1")
        axes[1].set_ylabel("UMAP-2")
        
        plt.tight_layout()

    else:
        # [옵션 1] Standard Dashboard (기본)
        sns.set(style="whitegrid", font_scale=1.0)
        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1.7])

        ax1 = plt.subplot(gs[0])
        sns.violinplot(data=df2, x=c_col, y="CARE_DDI", palette="Set2", inner=None, ax=ax1)
        sns.boxplot(data=df2, x=c_col, y="CARE_DDI", width=0.2, boxprops={"facecolor":"white"}, showfliers=False, ax=ax1)
        ax1.set_title(f"A. CARE-DDI Dist by {c_col}")

        radar_vars = ["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC"]
        agg = df2.groupby(c_col)[radar_vars].mean()
        agg_norm = (agg - agg.min()) / (agg.max() - agg.min())
        clusters = agg_norm.index.tolist()
        N = len(radar_vars)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        
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

        ax3 = plt.subplot(gs[2])
        X_pca = df2[radar_vars]
        scaled = StandardScaler().fit_transform(X_pca)
        pca = PCA(n_components=2)
        PC = pca.fit_transform(scaled)
        loadings = pca.components_.T
        sc = ax3.scatter(PC[:, 0], PC[:, 1], c=df2["CARE_DDI"], cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax3, label="CARE-DDI")
        for i, var in enumerate(radar_vars):
            ax3.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='r', head_width=0.1)
            ax3.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, var, color='r')
        ax3.set_title("C. PCA Biplot")
        plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode(), stats_text

# ==========================================
# 3. 라우팅
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        df = load_data()
        all_columns = df.columns.tolist()

        if request.method == 'POST':
            # Form Inputs
            sel_climate = request.form.getlist('climate')
            sel_hvi_ncd = request.form.getlist('hvi_ncd')
            sel_hvi_id = request.form.getlist('hvi_id')
            sel_dac = request.form.getlist('dac')
            sel_sbc = request.form.getlist('sbc')
            sel_c = request.form.get('cluster_c')
            sel_n = request.form.get('cluster_n')
            sel_d = request.form.get('cluster_d')
            sel_type = request.form.get('graph_type')

            plot_url, stats = run_analysis_logic(
                df, sel_climate, sel_hvi_ncd, sel_hvi_id, sel_dac, sel_sbc, 
                sel_c, sel_n, sel_d, sel_type
            )
            return render_template_string(RESULT_HTML, plot_url=plot_url, stats=stats)

        return render_template_string(FORM_HTML, columns=all_columns, defaults=DEFAULT_VARS)

    except Exception as e:
        import traceback
        return f"<h1>Error</h1><pre>{traceback.format_exc()}</pre>"

# ==========================================
# 4. HTML
# ==========================================
FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CARE-DDI Interactive Platform</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; max-width: 1100px; margin: 0 auto; background: #f4f6f8; }
        .group-box { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .group-title { font-weight: bold; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; margin-bottom: 10px; }
        .control-panel { background: #e8f4fd; border: 2px solid #3498db; }
        .radio-group label { display: block; margin: 8px 0; font-size: 1.05em; cursor: pointer; }
        button { width: 100%; background: #2980b9; color: white; border: none; padding: 15px; font-size: 1.2em; border-radius: 8px; cursor: pointer; transition: 0.3s; }
        button:hover { background: #1a5276; }
        select { padding: 5px; border-radius: 4px; border: 1px solid #ccc; }
        .flex-row { display: flex; gap: 15px; flex-wrap: wrap; }
        .flex-col { flex: 1; min-width: 200px; }
    </style>
</head>
<body>
    <h1 style="text-align: center; color: #34495e;">📊 CARE-DDI Analytics Platform</h1>
    <form method="POST" action="/">
        
        <div class="group-box control-panel">
            <div class="group-title">🎯 Visualization Type</div>
            <div class="radio-group">
                <label><input type="radio" name="graph_type" value="dashboard" checked> 📊 Standard Dashboard (Violin + Radar + PCA)</label>
                <label><input type="radio" name="graph_type" value="heatmap"> 🔥 Heatmap Analysis (Climate × Diet × Disease)</label>
                <label><input type="radio" name="graph_type" value="cnd_umap"> 🌐 Bar & UMAP Analysis (C-N-D Typologies)</label>
            </div>
            
            <hr style="border: 0; border-top: 1px solid #bdc3c7; margin: 15px 0;">
            
            <div class="flex-row">
                <div><strong>Main (C):</strong> <select name="cluster_c">{% for c in columns %}<option value="{{c}}" {% if c==defaults['cluster_c'] %}selected{% endif %}>{{c}}</option>{% endfor %}</select></div>
                <div><strong>Diet (N):</strong> <select name="cluster_n">{% for c in columns %}<option value="{{c}}" {% if c==defaults['cluster_n'] %}selected{% endif %}>{{c}}</option>{% endfor %}</select></div>
                <div><strong>Disease (D):</strong> <select name="cluster_d">{% for c in columns %}<option value="{{c}}" {% if c==defaults['cluster_d'] %}selected{% endif %}>{{c}}</option>{% endfor %}</select></div>
            </div>
        </div>

        <div class="flex-row">
            <div class="group-box flex-col">
                <div class="group-title">1. Climate Exposure (CEI)</div>
                {% for c in columns %}<label style="display:block;"><input type="checkbox" name="climate" value="{{c}}" {% if c in defaults['climate'] %}checked{% endif %}> {{c}}</label>{% endfor %}
            </div>
            <div class="group-box flex-col">
                <div class="group-title">2. HVI - NCD Diseases</div>
                {% for c in columns %}<label style="display:block;"><input type="checkbox" name="hvi_ncd" value="{{c}}" {% if c in defaults['hvi_ncd'] %}checked{% endif %}> {{c}}</label>{% endfor %}
            </div>
            <div class="group-box flex-col">
                <div class="group-title">3. HVI - Infectious</div>
                {% for c in columns %}<label style="display:block;"><input type="checkbox" name="hvi_id" value="{{c}}" {% if c in defaults['hvi_id'] %}checked{% endif %}> {{c}}</label>{% endfor %}
            </div>
        </div>
        
        <div class="group-box">
            <div class="group-title">4. DAC & 5. SBC Variables</div>
            <div class="flex-row">
                <div class="flex-col"><strong>DAC:</strong><br>{% for c in columns %}<label><input type="checkbox" name="dac" value="{{c}}" {% if c in defaults['dac'] %}checked{% endif %}> {{c}}</label> {% endfor %}</div>
                <div class="flex-col"><strong>SBC:</strong><br>{% for c in columns %}<label><input type="checkbox" name="sbc" value="{{c}}" {% if c in defaults['sbc'] %}checked{% endif %}> {{c}}</label> {% endfor %}</div>
            </div>
        </div>

        <button type="submit">Run Analysis 🚀</button>
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
    <head><title>Analysis Results</title></head>
    <body style="font-family: sans-serif; text-align: center; padding: 20px; background: #fff;">
        <h1 style="color: #2c3e50;">Analysis Output</h1>
        <div style="background: #f8f9fa; display: inline-block; padding: 10px 20px; border-radius: 5px; border: 1px solid #ddd;">
            <strong>Stat Check:</strong> <span style="color: blue;">{{ stats }}</span>
        </div>
        <hr style="margin: 20px 0;">
        <div style="overflow-x: auto;">
            <img src="data:image/png;base64,{{ plot_url }}" style="max-width: 95%; height: auto; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        <br><br>
        <a href="/" style="padding: 12px 25px; background: #7f8c8d; color: white; text-decoration: none; border-radius: 5px;">Back to Settings</a>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
