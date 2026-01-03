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

app = Flask(__name__)

# ==========================================
# 1. 기본 설정 및 데이터 로드
# ==========================================
def load_data():
    # 엑셀 파일 로드
    df = pd.read_excel("CND_rawdata.xlsx", sheet_name="sheet1")
    return df

# 기본 선택 변수 설정 (N_cluster, D_cluster 추가됨)
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
    'cluster_c': 'C_cluster', # 기후 클러스터 (기본)
    'cluster_n': 'N_cluster', # 영양 클러스터 (추가)
    'cluster_d': 'D_cluster'  # 질병 클러스터 (추가)
}

# ==========================================
# 2. 분석 및 시각화 로직
# ==========================================
def run_analysis_logic(df, climate_cols, hvi_ncd_cols, hvi_id_cols, dac_cols, sbc_cols, 
                       c_col, n_col, d_col, graph_type):
    
    # 1) 필요한 컬럼만 추출 (결측치 제거를 위해)
    # 히트맵을 그릴 때는 N, D 컬럼도 필수이므로 needed에 포함
    needed = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols + [c_col, n_col, d_col]
    
    # 해당 컬럼들에 결측치가 있는 행 제거
    df2 = df.dropna(subset=needed).copy()

    # 2) Z-score 변환 (통계 지표 산출용)
    def zseries(s):
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    # 클러스터 컬럼들을 제외한 변수들만 Z-score 변환
    calc_cols = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols
    for col in calc_cols:
        df2[col + "_z"] = zseries(df2[col])

    # 3) PCA 기반 지표(Index) 생성 함수
    def pca_pc1_pc2_index(df_in, cols, index_name):
        if not cols: return pd.Series(0, index=df_in.index, name=index_name)
        
        Z = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(Zc, full_matrices=False)
        
        if len(S) < 2: return pd.Series(0, index=df_in.index, name=index_name)

        var_exp = (S**2) / (S**2).sum()
        v1, v2 = float(var_exp[0]), float(var_exp[1])
        load1, load2 = VT[0], VT[1]
        
        w_raw = np.abs(load1) * v1 + np.abs(load2) * v2
        w = w_raw / w_raw.sum()
        
        Zfull = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        score = Zfull @ w
        return pd.Series(score, index=df_in.index, name=index_name)

    # 4) 각 도메인별 지수 산출
    df2["CEI"] = pca_pc1_pc2_index(df2, climate_cols, "CEI")
    df2["SBC"] = pca_pc1_pc2_index(df2, sbc_cols, "SBC")
    df2["HVI_NCD"] = pca_pc1_pc2_index(df2, hvi_ncd_cols, "HVI_NCD")
    df2["HVI_ID"] = pca_pc1_pc2_index(df2, hvi_id_cols, "HVI_ID")
    
    if len(dac_cols) == 1:
        df2["DAC"] = df2[dac_cols[0] + "_z"]
    else:
        df2["DAC"] = pca_pc1_pc2_index(df2, dac_cols, "DAC")

    # 5) 최종 CARE-DDI 계산
    w_CEI, w_HVI_NCD, w_HVI_ID, w_DAC, w_SBC = 0.30, 0.20, 0.15, 0.20, 0.15
    df2["CARE_DDI"] = (
        w_CEI * df2["CEI"] + w_HVI_NCD * df2["HVI_NCD"] + 
        w_HVI_ID * df2["HVI_ID"] - w_DAC * df2["DAC"] - w_SBC * df2["SBC"]
    )

    # 6) 통계 계산 (Main Cluster인 C_cluster 기준)
    df2[c_col] = pd.to_numeric(df2[c_col], errors="coerce").astype("Int64")
    stats_text = "Not enough groups."
    groups_val = [df2.loc[df2[c_col] == k, "CARE_DDI"].dropna().values 
                  for k in sorted(df2[c_col].dropna().unique())]
    
    if len(groups_val) >= 2:
        H, p = kruskal(*groups_val)
        stats_text = f"Kruskal-Wallis (by {c_col}): H = {H:.4f}, p-value = {p:.4e}"

    # =========================================================
    # 7) 시각화 분기 (Graph Type에 따라 다른 그림 그리기)
    # =========================================================
    
    # 그림판 준비
    if graph_type == 'heatmap':
        # --- [NEW] 히트맵 그리기 (C x N x D) ---
        climates = sorted(df2[c_col].unique())
        if len(climates) == 0:
            return None, "Error: No climate groups found."

        # Seaborn 스타일 설정
        sns.set(style="white", font_scale=1.1)
        
        # 기후 클러스터 개수만큼 Subplot 생성
        fig, axes = plt.subplots(
            1, len(climates),
            figsize=(5 * len(climates), 5), # 크기 자동 조절
            sharey=True
        )
        
        # axes가 1개일 경우 리스트로 변환
        if len(climates) == 1:
            axes = [axes]

        for ax, c in zip(axes, climates):
            sub = df2[df2[c_col] == c]

            # 피벗 테이블 생성: Rows=Diet(N), Cols=Disease(D), Values=CARE_DDI
            mat = sub.pivot_table(
                index=n_col,
                columns=d_col,
                values="CARE_DDI",
                aggfunc="mean"
            )

            # 히트맵 그리기
            sns.heatmap(
                mat,
                ax=ax,
                cmap="RdYlGn_r",    # Red(높음/위험) -> Green(낮음/안전)
                center=0,
                annot=True,
                fmt=".2f",
                cbar=(ax is axes[-1]),   # 맨 오른쪽 그래프에만 컬러바 표시
                cbar_kws={"label": "Mean CARE-DDI"}
            )

            ax.set_title(f"Climate Cluster: {c}")
            ax.set_xlabel(f"Disease ({d_col})")
            if ax is axes[0]:
                ax.set_ylabel(f"Diet ({n_col})")
            else:
                ax.set_ylabel("")
        
        plt.suptitle("CARE-DDI Heatmap by Climate(C) x Diet(N) x Disease(D)", y=1.05, fontsize=16)

    else:
        # --- [OLD] 기존 대시보드 (Violin + Radar + PCA) ---
        sns.set(style="whitegrid", font_scale=1.0)
        
        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1.7])

        # Panel A: Violin Plot
        ax1 = plt.subplot(gs[0])
        sns.violinplot(data=df2, x=c_col, y="CARE_DDI", palette="Set2", inner=None, ax=ax1)
        sns.boxplot(data=df2, x=c_col, y="CARE_DDI", width=0.2, boxprops={"facecolor":"white"}, showfliers=False, ax=ax1)
        ax1.set_title(f"A. CARE-DDI Distribution by {c_col}")

        # Panel B: Radar Chart
        radar_vars = ["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC"]
        agg = df2.groupby(c_col)[radar_vars].mean()
        agg_norm = (agg - agg.min()) / (agg.max() - agg.min()) # 0~1 정규화
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
        ax2.set_title("B. Radar Chart (Normalized Profile)")
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        # Panel C: PCA Biplot
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
    
    # 이미지를 메모리에 저장
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode(), stats_text

# ==========================================
# 3. 라우팅 및 폼 처리
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        df = load_data()
        all_columns = df.columns.tolist()

        if request.method == 'POST':
            # 폼 데이터 받기
            sel_climate_vars = request.form.getlist('climate')
            sel_hvi_ncd = request.form.getlist('hvi_ncd')
            sel_hvi_id = request.form.getlist('hvi_id')
            sel_dac = request.form.getlist('dac')
            sel_sbc = request.form.getlist('sbc')
            
            # 클러스터 변수들
            sel_c_col = request.form.get('cluster_c')
            sel_n_col = request.form.get('cluster_n')
            sel_d_col = request.form.get('cluster_d')
            
            # 그래프 타입
            sel_graph_type = request.form.get('graph_type')

            plot_url, stats = run_analysis_logic(
                df, 
                sel_climate_vars, sel_hvi_ncd, sel_hvi_id, sel_dac, sel_sbc, 
                sel_c_col, sel_n_col, sel_d_col, 
                sel_graph_type
            )
            
            return render_template_string(RESULT_HTML, plot_url=plot_url, stats=stats)

        return render_template_string(FORM_HTML, columns=all_columns, defaults=DEFAULT_VARS)

    except Exception as e:
        import traceback
        return f"<h1>Error:</h1><pre>{traceback.format_exc()}</pre>"

# ==========================================
# 4. HTML Templates
# ==========================================

FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CARE-DDI Analysis Platform</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; max-width: 1100px; margin: 0 auto; background-color: #f4f6f8; }
        h1 { color: #2c3e50; text-align: center; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between; }
        .group-box { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; width: 48%; box-shadow: 0 2px 4px rgba(0,0,0,0.05); box-sizing: border-box; }
        .full-width { width: 100%; }
        .group-title { font-weight: bold; color: #34495e; margin-bottom: 10px; font-size: 1.1em; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }
        .var-label { display: inline-block; margin-right: 10px; margin-bottom: 5px; font-size: 0.9em; cursor: pointer; }
        
        .control-panel { background-color: #e8f4fd; border: 2px solid #3498db; }
        .radio-group label { margin-right: 20px; font-weight: bold; font-size: 1.1em; cursor: pointer; }
        
        button { display: block; width: 100%; background-color: #2980b9; color: white; border: none; padding: 15px; font-size: 1.3em; border-radius: 8px; cursor: pointer; margin-top: 20px; transition: background 0.3s; }
        button:hover { background-color: #1a5276; }
    </style>
</head>
<body>
    <h1>📊 CARE-DDI Interactive Analysis</h1>
    
    <form method="POST" action="/">
        
        <div class="group-box full-width control-panel">
            <div class="group-title">🎯 Visualization Settings (Select Output Type)</div>
            <div class="radio-group" style="text-align: center; padding: 10px;">
                <label>
                    <input type="radio" name="graph_type" value="dashboard" checked> 
                    📊 Standard Dashboard (Violin + Radar + PCA)
                </label>
                <label>
                    <input type="radio" name="graph_type" value="heatmap"> 
                    🔥 Heatmap Analysis (Climate × Diet × Disease)
                </label>
            </div>
            <hr style="border: 0; border-top: 1px solid #bdc3c7; margin: 15px 0;">
            
            <div style="display: flex; justify-content: space-around;">
                <div>
                    <strong>Main Cluster (C):</strong><br>
                    <select name="cluster_c" style="padding: 5px;">
                        {% for col in columns %}
                        <option value="{{ col }}" {% if col == defaults['cluster_c'] %}selected{% endif %}>{{ col }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <strong>Diet Cluster (N):</strong><br>
                    <select name="cluster_n" style="padding: 5px;">
                        {% for col in columns %}
                        <option value="{{ col }}" {% if col == defaults['cluster_n'] %}selected{% endif %}>{{ col }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <strong>Disease Cluster (D):</strong><br>
                    <select name="cluster_d" style="padding: 5px;">
                        {% for col in columns %}
                        <option value="{{ col }}" {% if col == defaults['cluster_d'] %}selected{% endif %}>{{ col }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
        </div>

        <div class="container">
            <div class="group-box">
                <div class="group-title">1. Climate Exposure (CEI)</div>
                {% for col in columns %}
                    <label class="var-label"><input type="checkbox" name="climate" value="{{ col }}" {% if col in defaults['climate'] %}checked{% endif %}> {{ col }}</label>
                {% endfor %}
            </div>

            <div class="group-box">
                <div class="group-title">2. HVI - NCD Diseases</div>
                {% for col in columns %}
                    <label class="var-label"><input type="checkbox" name="hvi_ncd" value="{{ col }}" {% if col in defaults['hvi_ncd'] %}checked{% endif %}> {{ col }}</label>
                {% endfor %}
            </div>

            <div class="group-box">
                <div class="group-title">3. HVI - Infectious Diseases</div>
                {% for col in columns %}
                    <label class="var-label"><input type="checkbox" name="hvi_id" value="{{ col }}" {% if col in defaults['hvi_id'] %}checked{% endif %}> {{ col }}</label>
                {% endfor %}
            </div>

            <div class="group-box">
                <div class="group-title">4. DAC (Dietary) & 5. SBC (Socio)</div>
                <div style="margin-bottom: 10px;"><strong>DAC:</strong><br>
                {% for col in columns %}
                    <label class="var-label"><input type="checkbox" name="dac" value="{{ col }}" {% if col in defaults['dac'] %}checked{% endif %}> {{ col }}</label>
                {% endfor %}
                </div>
                <div><strong>SBC:</strong><br>
                {% for col in columns %}
                    <label class="var-label"><input type="checkbox" name="sbc" value="{{ col }}" {% if col in defaults['sbc'] %}checked{% endif %}> {{ col }}</label>
                {% endfor %}
                </div>
            </div>
        </div>

        <button type="submit">Run Analysis & Generate Graph 🚀</button>
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
    <head><title>Analysis Results</title></head>
    <body style="font-family: sans-serif; text-align: center; padding: 20px; background-color: #fff;">
        <h1 style="color: #2c3e50;">Analysis Output</h1>
        <div style="background-color: #f8f9fa; display: inline-block; padding: 10px 20px; border-radius: 5px; border: 1px solid #ddd;">
            <strong>Statistical Result:</strong> <span style="color: blue;">{{ stats }}</span>
        </div>
        <hr style="margin: 20px 0;">
        
        <div style="overflow-x: auto;">
            <img src="data:image/png;base64,{{ plot_url }}" style="max-width: 95%; height: auto; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        
        <br><br>
        <a href="/" style="display: inline-block; padding: 12px 25px; background-color: #7f8c8d; color: white; text-decoration: none; border-radius: 5px; font-size: 1.1em;">Back to Settings</a>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
