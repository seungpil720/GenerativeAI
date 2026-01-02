import io
import base64
import os

from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

app = Flask(__name__)

# ==========================================
# 1. 기본 설정 및 데이터 로드 함수
# ==========================================
def load_data():
    # 데이터 로드 (매번 최신 상태를 읽기 위함)
    df = pd.read_excel("CND_rawdata.xlsx", sheet_name="sheet1")
    return df

# 기본값으로 선택되어 있을 변수 목록
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
    'cluster': 'C_cluster' # 단일 선택
}

# ==========================================
# 2. 분석 로직 함수 (인자를 받도록 수정)
# ==========================================
def run_analysis_logic(df, climate_cols, hvi_ncd_cols, hvi_id_cols, dac_cols, sbc_cols, cluster_col):
    
    # 선택된 변수들로 필요한 컬럼 리스트 생성
    needed = climate_cols + hvi_ncd_cols + hvi_id_cols + dac_cols + sbc_cols + [cluster_col]
    
    # 결측치 제거
    df2 = df.dropna(subset=needed).copy()

    # Z-score 헬퍼 함수
    def zseries(s):
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    # 선택된 모든 변수에 대해 Z-score 변환
    for col in needed:
        # 클러스터 컬럼은 Z-score 변환 제외 (범주형일 수 있음)
        if col != cluster_col:
            df2[col + "_z"] = zseries(df2[col])

    # PCA 인덱스 생성 함수
    def pca_pc1_pc2_index(df_in, cols, index_name):
        if not cols: # 변수가 하나도 선택 안 된 경우 0 처리
            return pd.Series(0, index=df_in.index, name=index_name)
            
        Z = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(Zc, full_matrices=False)
        
        # 예외 처리: 변수가 적어서 PC2가 없는 경우 등
        if len(S) < 2: 
            # 간단히 평균으로 대체하거나 0 처리 (여기서는 0)
            return pd.Series(0, index=df_in.index, name=index_name)

        var_exp = (S**2) / (S**2).sum()
        v1, v2 = float(var_exp[0]), float(var_exp[1])
        load1, load2 = VT[0], VT[1]
        
        w_raw = np.abs(load1) * v1 + np.abs(load2) * v2
        w = w_raw / w_raw.sum()
        
        Zfull = df_in[[c + "_z" for c in cols]].to_numpy(dtype=float)
        score = Zfull @ w
        return pd.Series(score, index=df_in.index, name=index_name)

    # 지표 생성
    df2["CEI"] = pca_pc1_pc2_index(df2, climate_cols, "CEI")
    df2["SBC"] = pca_pc1_pc2_index(df2, sbc_cols, "SBC")
    df2["HVI_NCD"] = pca_pc1_pc2_index(df2, hvi_ncd_cols, "HVI_NCD")
    df2["HVI_ID"] = pca_pc1_pc2_index(df2, hvi_id_cols, "HVI_ID")
    
    # DAC는 단일 변수일 경우 바로 Z-score 사용, 여러 개일 경우 PCA
    if len(dac_cols) == 1:
        df2["DAC"] = df2[dac_cols[0] + "_z"]
    else:
        df2["DAC"] = pca_pc1_pc2_index(df2, dac_cols, "DAC")

    # CARE-DDI 계산 (가중치 고정)
    w_CEI, w_HVI_NCD, w_HVI_ID, w_DAC, w_SBC = 0.30, 0.20, 0.15, 0.20, 0.15
    df2["CARE_DDI"] = (
        w_CEI * df2["CEI"] + w_HVI_NCD * df2["HVI_NCD"] + 
        w_HVI_ID * df2["HVI_ID"] - w_DAC * df2["DAC"] - w_SBC * df2["SBC"]
    )
    
    # 통계 분석 (Kruskal-Wallis)
    df2[cluster_col] = pd.to_numeric(df2[cluster_col], errors="coerce").astype("Int64")
    stats_text = "Not enough groups for statistics."
    
    groups_val = [df2.loc[df2[cluster_col] == k, "CARE_DDI"].dropna().values 
                  for k in sorted(df2[cluster_col].dropna().unique())]
    
    if len(groups_val) >= 2:
        H, p = kruskal(*groups_val)
        stats_text = f"Kruskal-Wallis H = {H:.4f}, p-value = {p:.4e}"

    # 시각화
    sns.set(style="whitegrid", font_scale=1.0)
    radar_vars = ["CEI", "HVI_NCD", "HVI_ID", "DAC", "SBC"]
    
    # 레이더 차트 데이터
    agg = df2.groupby(cluster_col)[radar_vars].mean()
    agg_norm = (agg - agg.min()) / (agg.max() - agg.min())
    clusters = agg_norm.index.tolist()
    
    N = len(radar_vars)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    # PCA Plot 데이터
    X_pca = df2[radar_vars]
    scaled = StandardScaler().fit_transform(X_pca)
    pca = PCA(n_components=2)
    PC = pca.fit_transform(scaled)
    loadings = pca.components_.T

    # 그림 그리기
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1.7])

    # Panel A (Violin)
    ax1 = plt.subplot(gs[0])
    sns.violinplot(data=df2, x=cluster_col, y="CARE_DDI", palette="Set2", inner=None, ax=ax1)
    sns.boxplot(data=df2, x=cluster_col, y="CARE_DDI", width=0.2, boxprops={"facecolor":"white"}, showfliers=False, ax=ax1)
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
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode(), stats_text

# ==========================================
# 3. 라우팅 (메인 페이지)
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        df = load_data()
        all_columns = df.columns.tolist()

        # [POST] 사용자가 '분석하기' 버튼을 눌렀을 때
        if request.method == 'POST':
            # 폼에서 선택된 변수들 가져오기
            selected_climate = request.form.getlist('climate')
            selected_hvi_ncd = request.form.getlist('hvi_ncd')
            selected_hvi_id = request.form.getlist('hvi_id')
            selected_dac = request.form.getlist('dac')
            selected_sbc = request.form.getlist('sbc')
            selected_cluster = request.form.get('cluster')

            # 분석 실행
            plot_url, stats = run_analysis_logic(
                df, 
                selected_climate, selected_hvi_ncd, selected_hvi_id, 
                selected_dac, selected_sbc, selected_cluster
            )
            
            # 결과 화면 렌더링
            return render_template_string(RESULT_HTML, plot_url=plot_url, stats=stats)

        # [GET] 처음 접속했을 때 (선택 폼 보여주기)
        # 기본값 체크를 위해 템플릿에 전달
        return render_template_string(FORM_HTML, 
                                      columns=all_columns, 
                                      defaults=DEFAULT_VARS)

    except Exception as e:
        return f"<h1>Error:</h1><p>{str(e)}</p>"

# ==========================================
# 4. HTML 템플릿 (선택 폼 & 결과 화면)
# ==========================================

FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CARE-DDI Analysis Setup</title>
    <style>
        body { font-family: sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto; }
        .group-box { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 8px; }
        .group-title { font-weight: bold; color: #333; margin-bottom: 10px; font-size: 1.1em; }
        .var-label { display: inline-block; margin-right: 15px; margin-bottom: 5px; cursor: pointer; }
        button { background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 1.2em; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <h1>📊 CARE-DDI Variable Selection</h1>
    <p>Select the variables for each domain from your data file.</p>
    
    <form method="POST" action="/">
        
        <div class="group-box">
            <div class="group-title">1. Climate Exposure (CEI)</div>
            {% for col in columns %}
                <label class="var-label">
                    <input type="checkbox" name="climate" value="{{ col }}" 
                    {% if col in defaults['climate'] %}checked{% endif %}> {{ col }}
                </label>
            {% endfor %}
        </div>

        <div class="group-box">
            <div class="group-title">2. HVI - NCD (Non-communicable Diseases)</div>
            {% for col in columns %}
                <label class="var-label">
                    <input type="checkbox" name="hvi_ncd" value="{{ col }}"
                    {% if col in defaults['hvi_ncd'] %}checked{% endif %}> {{ col }}
                </label>
            {% endfor %}
        </div>

        <div class="group-box">
            <div class="group-title">3. HVI - Infectious & Nutritional</div>
            {% for col in columns %}
                <label class="var-label">
                    <input type="checkbox" name="hvi_id" value="{{ col }}"
                    {% if col in defaults['hvi_id'] %}checked{% endif %}> {{ col }}
                </label>
            {% endfor %}
        </div>

        <div class="group-box">
            <div class="group-title">4. DAC (Dietary Adaptation)</div>
            {% for col in columns %}
                <label class="var-label">
                    <input type="checkbox" name="dac" value="{{ col }}"
                    {% if col in defaults['dac'] %}checked{% endif %}> {{ col }}
                </label>
            {% endfor %}
        </div>

        <div class="group-box">
            <div class="group-title">5. SBC (Socioeconomic Buffering)</div>
            {% for col in columns %}
                <label class="var-label">
                    <input type="checkbox" name="sbc" value="{{ col }}"
                    {% if col in defaults['sbc'] %}checked{% endif %}> {{ col }}
                </label>
            {% endfor %}
        </div>

        <div class="group-box" style="background-color: #f9f9f9;">
            <div class="group-title">6. Cluster Variable (Grouping) - Select ONE</div>
            {% for col in columns %}
                <label class="var-label">
                    <input type="radio" name="cluster" value="{{ col }}" required
                    {% if col == defaults['cluster'] %}checked{% endif %}> {{ col }}
                </label>
            {% endfor %}
        </div>

        <div style="text-align: center; margin-top: 30px;">
            <button type="submit">Run Analysis 🚀</button>
        </div>
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
    <head><title>CARE-DDI Results</title></head>
    <body style="font-family: sans-serif; text-align: center; padding: 20px;">
        <h1>CARE-DDI Analysis Results</h1>
        <h3 style="color: blue;">{{ stats }}</h3>
        <hr>
        <img src="data:image/png;base64,{{ plot_url }}" style="max-width: 100%; height: auto; border: 1px solid #ccc; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
        <br><br>
        <a href="/" style="display: inline-block; padding: 10px 20px; background-color: #6c757d; color: white; text-decoration: none; border-radius: 5px;">Back to Selection</a>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
