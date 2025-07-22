# =================================================================================================
# Phoenix Engine 2.0 - VTX DTE-RPMS Hyper-Automation Platform
# =================================================================================================

# Core & UI
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json # NEW: Import for handling JSON objects

# Advanced Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import graphviz
import matplotlib.pyplot as plt

# NEW: Advanced Statistics & Time Series
import statsmodels.api as sm
from statsmodels.formula.api import ols

# NEW: Automated Reporting
from pptx import Presentation
from pptx.util import Inches

# NEW: Data Validation & Software Engineering Rigor
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# Machine Learning & Existing Analytics
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

# =================================================================================================
# App Configuration & Vertex Branding
# =================================================================================================
st.set_page_config(
    page_title="Phoenix Engine 2.0 | VTX DTE-RPMS",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vertex Brand Colors: Blue (#0033A0), Cyan (#00AEEF), Light Blue (#E7F3FF), Gray (#F0F2F6)
st.markdown("""
<style>
    /* Main app styling */
    .reportview-container, .main {
        background-color: #F0F2F6; /* Vertex Gray */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-image: linear-gradient(#FFFFFF, #E7F3FF); /* White to Vertex Light Blue */
    }
    .sidebar .sidebar-content .stRadio > label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #0033A0; /* Vertex Blue */
    }
    /* KPI Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #D1D1D1;
        border-left: 6px solid #0033A0; /* Vertex Blue */
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .stMetric > label {
        font-weight: 600 !important;
        color: #555555;
    }
    .stMetric > div > span {
        color: #0033A0; /* Vertex Blue */
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        background-color: transparent;
        border-bottom: 3px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 3px solid #00AEEF; /* Vertex Cyan */
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    /* Buttons */
    .stButton>button {
        border-radius: 20px;
        border: 2px solid #0033A0;
        background-color: #0033A0;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================================================
# NEW: Pydantic Data Models for Validation
# =================================================================================================
class QCResult(BaseModel):
    check_name: str
    status: str = Field(..., pattern=r"^(PASS|FAIL|WARN)$")
    details: Optional[str] = None
    failed_record_count: int

class RegulatoryDossier(BaseModel):
    request_id: str
    agency: str
    study_id: str
    package_checksum: str = Field(..., min_length=64, max_length=64) # SHA256
    qc_summary: List[QCResult]


# =================================================================================================
# NEW: Automated PowerPoint Reporting Function
# =================================================================================================
def generate_summary_pptx(study_id, kpi_data):
    """Generates a one-slide executive summary in PowerPoint format."""
    prs = Presentation()
    # === FIX APPLIED HERE ===
    # OLD, BUGGY CODE: slide_layout = prs.slide_layouts[5]
    # REASON: Layout index 5 is not guaranteed to be "Title and Content" and may lack a body placeholder.
    # NEW, CORRECTED CODE: Using index 1, which is the standard for "Title and Content".
    slide_layout = prs.slide_layouts[1]
    # ========================
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = f"Executive QC Summary: Study {study_id}"

    # This is now the body placeholder, correctly referenced from the chosen layout.
    content_shape = slide.shapes.placeholders[1]
    tf = content_shape.text_frame
    tf.clear()
    
    p = tf.add_paragraph()
    p.text = "Key Data Quality & Integrity Metrics"
    p.font.bold = True

    for k, v in kpi_data.items():
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.level = 1

    tf.add_paragraph().text = "\nThis summary was auto-generated by the Phoenix Engine 2.0."
    
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    ppt_io.seek(0)
    return ppt_io

# =================================================================================================
# Mock Data Generation Engine (Extended with Dose-Response & Vertex Context)
# =================================================================================================

@st.cache_data(ttl=900)
def generate_preclinical_data(study_id, n_samples=1000):
    """Generates more complex mock data, now including dose-response curves."""
    np.random.seed(hash(study_id) % (2**32 - 1))
    operators = ['J.Doe', 'S.Chen', 'M.Gupta', 'R.Valdez']
    instruments = {'PK': ['Agilent-6470', 'Sciex-7500'], 'Tox': ['Tecan-Spark', 'BMG-Pherastar'], 'CF': ['Ussing-Chamber-A', 'Ussing-Chamber-B']}
    assay_type = study_id.split('-')[1]
    
    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    doses = np.logspace(-3, 2, n_samples)
    base_response = sigmoid(np.log10(doses), 100, 2, 0.5)
    
    data = {
        'SampleID': [f"{study_id}-S{i:04d}" for i in range(n_samples)],
        'Timestamp': [datetime.now() - timedelta(days=np.random.uniform(1, 90), hours=h) for h in range(n_samples)],
        'OperatorID': np.random.choice(operators, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'InstrumentID': np.random.choice(instruments.get(assay_type, ['Generic-Inst-01']), n_samples),
        'ReagentLot': np.random.choice([f"LOT-2024-{'A'*(i+1)}" for i in range(4)], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'Dose_uM': doses,
        'Response': base_response + np.random.normal(0, 3, n_samples),
        'CellViability': np.random.normal(95, 4, n_samples).clip(70, 100),
    }
    df = pd.DataFrame(data)

    df.loc[df['OperatorID'] == 'R.Valdez', 'Response'] *= 1.15
    df.loc[df['ReagentLot'] == 'LOT-2024-AAAA', 'Response'] *= 0.85
    df.loc[df['ReagentLot'] == 'LOT-2024-AAAA', 'CellViability'] -= 10
    late_samples = df.sort_values('Timestamp').tail(50).index
    df.loc[late_samples, 'Response'] += np.linspace(0, 15, 50)

    df['QC_Flag'] = 0
    df.loc[df[df['OperatorID'] == 'R.Valdez'].sample(frac=0.8).index, 'QC_Flag'] = 1
    df.loc[df[df['ReagentLot'] == 'LOT-2024-AAAA'].sample(frac=0.8).index, 'QC_Flag'] = 1
    late_sample_indices_to_flag = np.random.choice(late_samples, size=int(len(late_samples) * 0.8), replace=False)
    df.loc[late_sample_indices_to_flag, 'QC_Flag'] = 1

    return df.sort_values('Dose_uM').reset_index(drop=True)

@st.cache_data(ttl=900)
def generate_global_kpis():
    sites = ["Boston, USA", "San Diego, USA", "Oxford, UK"]
    data = []
    for site in sites:
        data.append({
            'Site': site, 'lon': [-71.0589, -117.1611, -1.2577][sites.index(site)], 'lat': [42.3601, 32.7157, 51.7520][sites.index(site)],
            'Studies_Active': np.random.randint(5, 15), 'Data_Integrity': f"{np.random.uniform(99.5, 99.9):.2f}%",
            'Automation_Coverage': np.random.randint(85, 98), 'Critical_Flags': np.random.randint(0, 5),
            'Mfg_OEE': np.random.randint(75, 92), 'Cpk_Avg': f"{np.random.uniform(1.3, 1.8):.2f}"
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=900)
def generate_cnv_data(sample_id):
    np.random.seed(hash(sample_id) % (2**32 - 1))
    chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    data = []; pos = 0
    for chrom in chromosomes:
        n_probes = np.random.randint(500, 2000)
        positions = pos + np.cumsum(np.random.randint(10000, 50000, n_probes))
        log2_ratios = np.random.normal(0, 0.15, n_probes)
        df_chrom = pd.DataFrame({'Chromosome': chrom, 'Position': positions, 'Log2_Ratio': log2_ratios})
        data.append(df_chrom); pos = positions[-1]
    df = pd.concat(data).reset_index(drop=True)
    df.loc[(df['Chromosome'] == 'chr8') & (df['Position'] > 127_000_000) & (df['Position'] < 129_000_000), 'Log2_Ratio'] += 0.8
    df.loc[(df['Chromosome'] == 'chr9') & (df['Position'] > 21_000_000) & (df['Position'] < 23_000_000), 'Log2_Ratio'] -= 0.7
    return df

@st.cache_data(ttl=900)
def generate_process_data(process_name="TRIKAFTA_API_Purity"):
    np.random.seed(hash(process_name) % (2**32 - 1))
    data = {'BatchID': [f'MFG-24-{i:03d}' for i in range(1, 101)], 'Timestamp': pd.to_datetime(pd.date_range(end=datetime.now(), periods=100)), 'Value': np.random.normal(99.5, 0.2, 100)}
    df = pd.DataFrame(data); df.loc[75:, 'Value'] += 0.35
    return df

# =================================================================================================
# Sidebar Navigation & User Info
# =================================================================================================
with st.sidebar:
    st.image("https://d1io3yog0oux5.cloudfront.net/_3f03b2222d6fdd47976375a7337f7a69/vertexpharmaceuticals/db/387/2237/logo.png", width=220)
    st.title("Phoenix Engine 2.0")
    st.markdown("##### DTE-RPMS Hyper-Automation")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🌎 **Global Command Center**",
            "🔬 **Assay Dev & Dose-Response**",
            "🧬 **Genomic Data QC (CASGEVY)**",
            "📊 **Cross-Study & Batch Analysis**",
            "💡 **Automated Root Cause Analysis**",
            "📈 **Process Control (TRIKAFTA)**",
            "🏛️ **Regulatory & Audit Hub**",
            "✅ **System Validation & QA**",
            "🔗 **Data Lineage & Contracts**",
            "📚 **SME Knowledge Base**"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info(
        f"""
        **Principal Engineer, DTE-RPMS**\n
        **User:** engineer.principal@vertex.com\n
        **Session Start:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
        """
    )
    st.markdown("---")
    st.link_button("Go to Vertex Science Portal", "https://www.vrtx.com/our-science/pipeline/")


# =================================================================================================
# Page Implementations
# =================================================================================================

if page == "🌎 **Global Command Center**":
    st.header("🌎 Global RPMS Operations Command Center")
    st.markdown("Real-time, holistic view of data operations, integrity, and automation initiatives across all major research and manufacturing sites.")

    global_kpis = generate_global_kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Data Integrity", "99.81%", "+0.15%", help="Aggregated GxP data quality score across all validated pipelines.")
    c2.metric("Automation Index", "95%", "Target: 95%", help="Weighted percentage of processes automated, from data ingest to report generation.")
    c3.metric("Global Mfg. OEE", "88%", "+1.2%", help="Overall Equipment Effectiveness for key manufacturing lines (e.g., TRIKAFTA).")
    c4.metric("Pending Audit Actions", "4", "2 FDA, 2 EMA", help="Number of open action items from recent regulatory inspections.")

    st.markdown("---")

    map_col, alerts_col = st.columns([2, 1])
    with map_col:
        st.subheader("Global Site Status & Health")
        fig = go.Figure(data=go.Scattergeo(
            lon=global_kpis['lon'], lat=global_kpis['lat'],
            text=global_kpis.apply(lambda row: f"<b>{row['Site']}</b><br>Integrity: {row['Data_Integrity']}<br>Automation: {row['Automation_Coverage']}%<br>OEE: {row['Mfg_OEE']}%<br>Flags: {row['Critical_Flags']}", axis=1),
            mode='markers',
            marker=dict(
                color=global_kpis['Critical_Flags'], colorscale=[[0, '#00AEEF'], [1, '#FF4136']], reversescale=False,
                cmin=0, cmax=5, size=global_kpis['Automation_Coverage'] / 4,
                colorbar_title='Critical Flags'
            )
        ))
        fig.update_layout(
            geo=dict(scope='world', projection_type='natural earth', showland=True, landcolor='#E0E0E0', bgcolor='#F0F2F6'),
            margin={"r":0,"t":0,"l":0,"b":0}, height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    with alerts_col:
        st.subheader("Priority Action Items")
        st.error("🔴 **CRITICAL:** [TRIKAFTA MFG] Cpk for API Purity dropped to 1.31. Batch MFG-24-088 under review. Immediate action required.")
        st.warning("🟠 **WARNING:** [CASGEVY QC] Reagent Lot LOT-2024-AAAA shows 15% lower cell viability. Lot quarantined.")
        st.info("🔵 **INFO:** [VX-522 Dev] New dose-response data from Oxford site available for review in the Assay Dev module.")
        st.info("🔵 **INFO:** [FDA-REQ-003] Dossier for VTX-809-PK-01 is packaged and ready for final review in the Audit Hub.")

elif page == "🔬 **Assay Dev & Dose-Response**":
    st.header("🔬 Assay Development & 3D Dose-Response Modeling")
    st.markdown("Analyze in-vitro assay data, fit dose-response curves to calculate potency (IC50/EC50), and visualize multi-variable interactions.")
    
    study_list = ["VX-CF-MOD-01", "VX-522-Tox-02", "VX-PAIN-TGT-05"]
    selected_study = st.selectbox("Select a Vertex Development Study:", study_list)

    df = generate_preclinical_data(selected_study)
    
    tab1, tab2, tab3 = st.tabs(["📈 **2D Dose-Response Curve (IC50)**", "✨ **3D Response Surface**", "📦 **Batch Box Plots**"])
    
    with tab1:
        st.subheader(f"Dose-Response Curve for {selected_study}")
        fig = px.scatter(df, x="Dose_uM", y="Response", 
                         log_x=True, title="Potency Assay: Response vs. Dose",
                         labels={"Dose_uM": "Dose (µM)", "Response": "Assay Response (%)"},
                         color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
        max_resp = df['Response'].max()
        ic50_approx = df.iloc[(df['Response'] - max_resp / 2).abs().argsort()[:1]]['Dose_uM'].values[0]
        
        fig.add_vline(x=ic50_approx, line_dash="dash", line_color="firebrick", annotation_text=f"IC50 ≈ {ic50_approx:.2f} µM")
        fig.add_hline(y=max_resp/2, line_dash="dash", line_color="firebrick")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Automated IC50 Estimation:** The estimated half-maximal inhibitory concentration (IC50) for this compound is approximately **{ic50_approx:.3f} µM**. For a formal report, a 4-parameter logistic (4PL) model from `statsmodels` or `scipy` should be used.")

    with tab2:
        st.subheader("3D Response Surface: Dose, Time, and Viability")
        df['Time_h'] = df.index / len(df) * 48
        fig3d = px.scatter_3d(df.sample(500), x='Dose_uM', y='Time_h', z='Response',
                              color='CellViability',
                              log_x=True,
                              title="3D Interaction Plot",
                              labels={"Dose_uM": "Dose (log µM)", "Time_h": "Time (h)", "Response": "Response (%)"},
                              color_continuous_scale=px.colors.sequential.Viridis)
        fig3d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown("**Insight:** The 3D surface plot helps identify complex interactions. Here, we can see how the assay response evolves over time at different dose concentrations, with marker color indicating cell health. This can reveal time-dependent toxicity or efficacy.")
        
    with tab3:
        st.subheader("Response Distribution by Reagent Lot")
        fig_box = px.box(df, x='ReagentLot', y='Response', color='ReagentLot',
                         title="Assay Response by Reagent Lot",
                         labels={"ReagentLot": "Reagent Lot ID", "Response": "Assay Response (%)"},
                         color_discrete_map={
                             "LOT-2024-A": "#0033A0", "LOT-2024-AA": "#00AEEF",
                             "LOT-2024-AAA": "#63C5F3", "LOT-2024-AAAA": "#FF4136"
                         })
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("This plot clearly isolates **LOT-2024-AAAA** as having a significantly different response distribution, confirming it as a likely source of QC issues.")
elif page == "🧬 **Genomic Data QC (CASGEVY)**":
    st.header("🧬 Genomic Data QC Engine for Gene Therapies (CASGEVY)")
    st.markdown("Specialized module for QC of gene-editing data, including on-target allele frequency and off-target analysis.")

    sample_id = st.text_input("Enter CASGEVY Patient Sample ID:", "V-PT-007-BCH-01")
    if sample_id:
        cnv_data = generate_cnv_data(sample_id)
        
        tab1, tab2, tab3 = st.tabs(["🧬 **Whole-Genome CNV Scan**", "🎯 **Off-Target Locus Analysis**", "🎲 **3D Allelic Drift Simulation**"])
        
        with tab1:
            st.subheader("Whole-Genome Copy Number Variation (CNV) Scan")
            fig_genome = px.scatter(cnv_data, x=cnv_data.index, y='Log2_Ratio', color='Chromosome',
                                    title=f"Genome-wide CNV Profile for {sample_id}",
                                    labels={'x': 'Genomic Position (indexed)', 'y': 'Log2 Ratio'},
                                    hover_data=['Chromosome', 'Position'])
            fig_genome.update_layout(showlegend=False)
            st.plotly_chart(fig_genome, use_container_width=True)
            st.markdown("**Purpose:** This initial scan serves as a quality control step to ensure the patient's genomic baseline is free of large-scale variations that could confound gene-editing analysis.")

        with tab2:
            st.subheader("Drill-Down: Potential Off-Target Locus")
            chrom_list = cnv_data['Chromosome'].unique()
            selected_chrom = st.selectbox("Select Chromosome to inspect for Off-Target Effects:", chrom_list, index=8) # Default to chr9
            
            chrom_data = cnv_data[cnv_data['Chromosome'] == selected_chrom]
            fig_chrom = px.scatter(chrom_data, x='Position', y='Log2_Ratio',
                                   title=f"CNV Detail for {selected_chrom}",
                                   labels={'Position': f'Position on {selected_chrom} (bp)', 'y': 'Log2 Ratio'},
                                   color_discrete_sequence=['#0033A0'])
            fig_chrom.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Gain Threshold")
            fig_chrom.add_hline(y=-0.3, line_dash="dash", line_color="red", annotation_text="Loss Threshold")
            st.plotly_chart(fig_chrom, use_container_width=True)
            st.markdown(f"**Analysis:** The plot for **{selected_chrom}** reveals a potential deletion event near 22 Mb, corresponding to the *CDKN2A* locus. While this is a known somatic variant, automated flagging ensures it is reviewed to rule out any off-target editing effects.")
            
        with tab3:
            st.subheader("3D Simulation of Allelic Frequency Random Walk")
            st.markdown("This plot simulates the stochastic nature of allele frequencies in a cell population over time post-edit, helping to visualize potential genetic drift.")
            n_steps = 100
            steps = np.random.randn(n_steps, 3) * 0.01
            walk = np.cumsum(steps, axis=0) + [0.5, 0.5, 0] # Start with 50% WT, 50% Edited
            walk_df = pd.DataFrame(walk, columns=['WT_Allele', 'Edited_Allele', 'Drift_Allele'])
            walk_df['Timepoint'] = range(n_steps)
            
            fig_walk = go.Figure(data=[go.Scatter3d(
                x=walk_df['WT_Allele'], y=walk_df['Edited_Allele'], z=walk_df['Drift_Allele'],
                mode='lines',
                line=dict(color=walk_df['Timepoint'], colorscale='Viridis', width=6),
                hovertext=walk_df['Timepoint']
            )])
            fig_walk.update_layout(
                title="Simulated Allelic Frequency Drift",
                scene=dict(
                    xaxis_title='WT Allele Freq.',
                    yaxis_title='Edited Allele Freq.',
                    zaxis_title='Other/Drift Freq.'
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_walk, use_container_width=True)


elif page == "📊 **Cross-Study & Batch Analysis**":
    st.header("📊 Cross-Study & Batch-to-Batch Analysis")
    st.markdown("Perform comparative statistical analyses across different studies, instruments, or reagent lots to identify systemic variations.")

    study_list = ["VX-CF-MOD-01", "VX-522-Tox-02", "VX-PAIN-TGT-05"]
    selected_studies = st.multiselect("Select studies to compare:", study_list, default=study_list)
    
    if selected_studies:
        # === FUTUREWARNING FIX APPLIED HERE ===
        # REASON: The previous pd.concat call implicitly used integer keys which is deprecated.
        # NEW, CORRECTED CODE: Explicitly creating the 'StudyID' column after concatenation is more robust.
        data_frames = [generate_preclinical_data(s).assign(StudyID=s) for s in selected_studies]
        all_data = pd.concat(data_frames, ignore_index=True)
        # ======================================

        tab1, tab2, tab3 = st.tabs(["📊 **ANOVA & Box Plots**", "🔥 **Performance Heatmap**", "🛰️ **Principal Component Analysis (PCA)**"])

        with tab1:
            st.subheader("Distribution Comparison by Study")
            fig_box = px.box(all_data, x='StudyID', y='Response', color='StudyID',
                             title="Assay Response Distribution Across Studies", color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_box, use_container_width=True)

            model = ols('Response ~ C(StudyID)', data=all_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.subheader("Analysis of Variance (ANOVA) Report")
            st.dataframe(anova_table)
            p_value = anova_table['PR(>F)'].iloc[0]
            if p_value < 0.05:
                st.error(f"**Statistically Significant Difference Detected (p = {p_value:.4f}).** The mean response between studies is not equal, suggesting different compound potencies or assay conditions.")
            else:
                st.success("No statistically significant difference detected among study means.")

        with tab2:
            st.subheader("Reagent Lot Performance Heatmap")
            pivot = all_data.pivot_table(index='ReagentLot', columns='OperatorID', values='Response', aggfunc='mean')
            fig_heat = px.imshow(pivot, text_auto=".1f", aspect="auto",
                                 title="Mean Assay Response by Reagent Lot and Operator",
                                 color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("**Insight:** The heatmap reveals potential interactions. A dark red or blue cell indicates a combination of operator and reagent lot that produces unusually high or low results, flagging a potential training or material-specific issue.")

        with tab3:
            st.subheader("PCA for Outlier/Cluster Detection")
            df_pca = all_data[['Dose_uM', 'Response', 'CellViability']].dropna()
            pca = PCA(n_components=2)
            components = pca.fit_transform(df_pca)
            
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            pca_df['StudyID'] = all_data.loc[df_pca.index, 'StudyID']

            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='StudyID', title="PCA of Preclinical Data",
                                 hover_data={'StudyID': True})
            st.plotly_chart(fig_pca, use_container_width=True)
            st.markdown(f"**Explained Variance:** PC1 explains **{pca.explained_variance_ratio_[0]:.1%}** and PC2 explains **{pca.explained_variance_ratio_[1]:.1%}** of the variance. Separation of clusters by StudyID indicates that the studies are fundamentally different in their multi-variate profiles.")


elif page == "💡 **Automated Root Cause Analysis**":
    st.header("💡 Automated Root Cause Analysis (RCA) Engine")
    st.markdown("Leverages machine learning to predict the likely cause of QC flags, accelerating investigation and resolution.")

    study_id = st.selectbox("Select Study with QC Flags:", ["VX-CF-MOD-01", "VX-522-Tox-02"], key="rca_study")
    df = generate_preclinical_data(study_id)
    df_flagged = df[df['QC_Flag'] == 1]

    if not df_flagged.empty:
        st.warning(f"Found **{len(df_flagged)}** QC-flagged records in **{study_id}**. Initiating RCA...")
        
        features = ['OperatorID', 'InstrumentID', 'ReagentLot']
        target = 'QC_Flag'
        df_ml = df.copy()
        for col in features:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        
        X = df_ml[features]
        y = df_ml[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10)
        clf.fit(X_train, y_train)
        
        st.subheader("Inferred Decision Rules for QC Flags")
        st.markdown("**Explanation:** A Decision Tree model was trained to find the most predictive factors for QC flags. The flowchart below represents the simplest rules that explain the majority of failures.")
        
        fig, ax = plt.subplots(figsize=(18, 10))
        plot_tree(clf, feature_names=features, class_names=['No Flag', 'Flagged'], filled=True, rounded=True, ax=ax, fontsize=12)
        st.pyplot(fig)
        
        st.subheader("Probable Root Cause Contribution")
        importances = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
        fig_imp = px.bar(importances, x='Feature', y='Importance', title='Feature Importance for QC Failures', color='Feature',
                         color_discrete_map={
                             "ReagentLot": "#FF4136", "OperatorID": "#FF851B", "InstrumentID": "#FFDC00"
                         })
        st.plotly_chart(fig_imp, use_container_width=True)
        
        top_cause = importances.iloc[0]
        st.error(f"**Top Predicted Contributor:** **{top_cause['Feature']}** is the most significant factor driving QC flags. Recommend immediate investigation into this area (e.g., reagent lot quarantine, operator retraining).")

    else:
        st.success(f"No QC flags found in study **{study_id}**. Data integrity appears high.")


elif page == "📈 **Process Control (TRIKAFTA)**":
    st.header("📈 Process Control & Stability for TRIKAFTA® Manufacturing")
    st.markdown("Monitors critical quality attributes (CQAs) of TRIKAFTA® API manufacturing using advanced SPC and time series analysis.")

    process_name = st.selectbox("Select TRIKAFTA® CQA to Monitor:", ["TRIKAFTA_API_Purity", "Elexacaftor_Assay", "Tezacaftor_Assay"])
    df = generate_process_data(process_name)

    spec_col1, spec_col2, spec_col3 = st.columns(3)
    USL = spec_col1.number_input("Upper Specification Limit (USL)", value=100.0)
    TARGET = spec_col2.number_input("Target", value=99.5)
    LSL = spec_col3.number_input("Lower Specification Limit (LSL)", value=99.0)

    mean = df['Value'].mean()
    std_dev = df['Value'].std()
    ucl = mean + 3 * std_dev
    lcl = mean - 3 * std_dev
    cpk = min((USL - mean) / (3 * std_dev), (mean - LSL) / (3 * std_dev))

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Process Mean", f"{mean:.3f}")
    kpi_col2.metric("Process Std Dev", f"{std_dev:.3f}")
    kpi_col3.metric("Process Capability (Cpk)", f"{cpk:.2f}", "Alert: < 1.33" if cpk < 1.33 else "Stable: > 1.33",
                    delta_color="inverse" if cpk < 1.33 else "off")

    tab_spc, tab_tsa = st.tabs(["📊 **SPC I-Chart**", "📉 **Time Series Decomposition**"])

    with tab_spc:
        st.subheader("I-Chart (Individuals Chart) for Process Stability")
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=df['BatchID'], y=df['Value'], mode='lines+markers', name='CQA Value', line=dict(color='#0033A0')))
        fig_i.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
        fig_i.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL")
        fig_i.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL")
        fig_i.add_hline(y=USL, line_dash="dot", line_color="orange", annotation_text="USL")
        fig_i.add_hline(y=LSL, line_dash="dot", line_color="orange", annotation_text="LSL")
        fig_i.update_layout(title=f"I-Chart for {process_name}", yaxis_title="Value")
        st.plotly_chart(fig_i, use_container_width=True)
        st.markdown("**Interpretation:** The upward shift towards the end of the run indicates a special cause variation that has made the process unstable. This requires investigation.")

    with tab_tsa:
        st.subheader("Time Series Analysis (STL Decomposition)")
        st.markdown("Decomposes the process data into trend, seasonal, and residual components to better understand underlying patterns.")
        df_ts = df.set_index('Timestamp')
        stl = sm.tsa.STL(df_ts['Value'], period=7).fit()
        
        fig_tsa = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Trend", "Seasonal", "Residual"))
        fig_tsa.add_trace(go.Scatter(x=df_ts.index, y=stl.trend, mode='lines', name='Trend'), row=1, col=1)
        fig_tsa.add_trace(go.Scatter(x=df_ts.index, y=stl.seasonal, mode='lines', name='Seasonal'), row=2, col=1)
        fig_tsa.add_trace(go.Scatter(x=df_ts.index, y=stl.resid, mode='markers', name='Residual'), row=3, col=1)
        
        fig_tsa.update_layout(height=600, title_text=f"STL Decomposition for {process_name}")
        st.plotly_chart(fig_tsa, use_container_width=True)
        st.markdown("**Analysis:** The **Trend** component clearly visualizes the upward drift in the process mean. The **Residuals** plot can be monitored for unexpected shocks or outliers.")
elif page == "🏛️ **Regulatory & Audit Hub**":
    st.header("🏛️ Regulatory & Audit Hub")
    st.markdown("Prepare, package, and document data dossiers for regulatory inspections and internal audits with full 21 CFR Part 11 traceability.")

    st.warning("**AUDIT & SUBMISSION PORTAL** | Actions are logged and subject to GxP compliance checks.")
    with st.form("audit_sim_form"):
        st.subheader("Package New Regulatory Dossier")
        c1, c2, c3 = st.columns(3)
        req_id = c1.text_input("Request ID", "FDA-REQ-003")
        agency = c2.selectbox("Requesting Agency", ["FDA", "EMA", "PMDA", "Internal QA"])
        study_id_package = c3.selectbox("Select Study to Package:", ["VX-CF-MOD-01", "VX-522-Tox-02"])
        
        st.text_area("Justification / Request Details", "Follow-up request for raw data, QC reports, and statistical analysis for the selected study, focusing on outlier investigation.")
        
        files_to_include = st.multiselect(
            "Select Data & Artifacts to Include:",
            ["Raw Instrument Data (.csv)", "QC Anomaly Report (.pdf)", "Data Lineage Graph (.svg)", "Audit Trail Log (.json)", "Statistical Analysis Script (R/Python)", "Executive Summary (.pptx)"],
            default=["Raw Instrument Data (.csv)", "QC Anomaly Report (.pdf)", "Audit Trail Log (.json)", "Executive Summary (.pptx)"]
        )
        submitter_name = st.text_input("Enter Full Name for Electronic Signature:", "Dr. Principal Engineer")
        submitted = st.form_submit_button("🔒 Validate, Lock, and Package Dossier")

    if submitted:
        with st.spinner("1. Validating data contract... 2. Generating checksums... 3. Logging audit trail..."):
            import time, hashlib
            time.sleep(2)
            mock_qc_results = [QCResult(check_name="Completeness", status="PASS", failed_record_count=0),
                               QCResult(check_name="Range Check", status="WARN", details="3 minor outliers detected", failed_record_count=3)]
            dossier_checksum = hashlib.sha256(f"{req_id}{study_id_package}".encode()).hexdigest()
            try:
                validated_dossier = RegulatoryDossier(request_id=req_id, agency=agency, study_id=study_id_package,
                                                      package_checksum=dossier_checksum, qc_summary=mock_qc_results)
                st.success(f"**Dossier Validated & Packaged!** (Pydantic model check: PASS)")
                st.code(validated_dossier.model_dump_json(indent=2), language='json')
                
                if "Executive Summary (.pptx)" in files_to_include:
                    kpis = {"Data Integrity Score": "99.8%", "QC Flags": "3 Warnings", "Conclusion": "Ready for submission"}
                    ppt_file = generate_summary_pptx(study_id_package, kpis)
                    st.download_button("⬇️ Download Executive Summary (.pptx)", ppt_file, file_name=f"{req_id}_summary.pptx")
                
                st.download_button("⬇️ Download Full Dossier (.zip)", data="dummy_zip_content", file_name=f"{req_id}_dossier.zip")

            except ValidationError as e:
                st.error("Dossier Validation Failed! Pydantic model check: FAIL")
                st.code(str(e), language='text')

elif page == "✅ **System Validation & QA**":
    st.header("✅ System Validation & Quality Assurance")
    st.markdown("Manage and review the validation lifecycle of the Phoenix Engine itself, ensuring it operates as intended in a GxP environment.")
    
    st.subheader("System Validation Workflow (GAMP 5)")
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style=rounded];
            URS [label="User Requirement\nSpecification (URS)"];
            FS [label="Functional\nSpecification (FS)"];
            DS [label="Design\nSpecification (DS)"];
            Code [label="Code & Unit Tests\n(Pytest)"];
            IQ [label="Installation\nQualification (IQ)"];
            OQ [label="Operational\nQualification (OQ)"];
            PQ [label="Performance\nQualification (PQ)"];
            RTM [label="Requirements\nTraceability Matrix"];

            URS -> FS -> DS -> Code;
            Code -> IQ -> OQ -> PQ;
            {URS, FS, DS} -> RTM [style=dashed];
            {IQ, OQ, PQ} -> RTM [style=dashed];
        }
    """)
    
    tab1, tab2, tab3 = st.tabs(["⚙️ **Unit Test Results (Pytest)**", "📋 **Qualification Protocols**", "✍️ **Change Control**"])

    with tab1:
        st.subheader("Latest Unit Test Run Summary")
        st.markdown("Automated tests run via `pytest` to verify the correctness of individual functions (e.g., data generation, statistical calculations).")
        st.code("""
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-7.4.0, pluggy-1.0.0
rootdir: /app/tests
collected 45 items

tests/test_data_generation.py::test_generate_preclinical_data PASSED  [  2%]
tests/test_data_generation.py::test_generate_cnv_data PASSED          [  4%]
tests/test_analytics.py::test_spc_calculation PASSED                    [  6%]
tests/test_analytics.py::test_anova_significance PASSED                 [  8%]
... (39 more tests)
tests/test_reporting.py::test_pptx_generation PASSED                    [ 97%]
tests/test_validation.py::test_pydantic_dossier_pass PASSED             [100%]

============================== 45 passed in 12.34s ===============================
        """, language="bash")
        st.success("All 45 unit tests passed. Code coverage: 98%.")
    
    with tab2:
        st.subheader("IQ / OQ / PQ Protocol Status")
        protocol_data = {
            "Protocol ID": ["IQ-PHX-001", "OQ-PHX-001", "PQ-PHX-001"],
            "Description": ["Verify correct installation of all libraries and system dependencies.", "Test core system functions against functional specifications.", "Test system performance under expected load and edge cases."],
            "Status": ["Executed & Approved", "Executed & Approved", "Pending Execution"],
            "Approved By": ["qa.lead@vertex.com", "qa.lead@vertex.com", "N/A"],
            "Approval Date": ["2024-04-01", "2024-04-15", "N/A"]
        }
        st.dataframe(protocol_data, use_container_width=True)
        st.info("Performance Qualification (PQ) is pending for the latest release (v2.1).")

    with tab3:
        st.subheader("Change Control Log")
        st.markdown("A log of all significant changes to the validated system.")
        change_log = {
            "CR-ID": ["CR-075", "CR-076"],
            "Date": ["2024-05-10", "2024-05-20"],
            "Change Description": ["Added `statsmodels` for STL decomposition on Process Control page.", "Updated brand colors and added 3D allelic drift plot to Genomics page."],
            "Reason": ["Enhance process drift detection capabilities.", "Improve user experience and add new visualization for gene therapy QC."],
            "Impact Assessment": ["Low. Additive feature. Re-validation of Process Control page required.", "Low. UI change and new plot. Re-validation of Genomics page required."],
            "Status": ["Approved & Implemented", "In Development"]
        }
        st.dataframe(change_log, use_container_width=True)

elif page == "🔗 **Data Lineage & Contracts**":
    st.header("🔗 Data Lineage & Data Contracts Hub")
    st.markdown("Visualize data provenance and enforce data quality at the source using machine-readable contracts.")
    
    tab1, tab2, tab3 = st.tabs(["🗺️ **Visual Data Flow**", "📜 **Data Contract Validation**", "🔍 **SQL Query Hub**"])
    
    with tab1:
        st.subheader("End-to-End Data Flow")
        dot = graphviz.Digraph(comment='Data Lineage', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent'})
        dot.node('A', 'Source Systems\n(LIMS, ELN)', shape='folder', style='filled', fillcolor='#FFC107')
        dot.node('B', 'Data Ingest Pipeline\n(Airflow/Python)', shape='box', style='filled', fillcolor='#8BC34A')
        dot.node('C', 'Data Lake\n(S3 - Raw Data)', shape='cylinder', style='filled', fillcolor='#03A9F4')
        dot.node('D', 'ETL/QC Process\n(Spark/dbt)', shape='box', style='filled', fillcolor='#8BC3A9')
        dot.node('E', 'Data Warehouse\n(Snowflake - Curated)', shape='cylinder', style='filled', fillcolor='#03A9F4')
        dot.node('F', 'Phoenix Engine\n(This App)', shape='star', style='filled', fillcolor='#0033A0', fontcolor='white')
        dot.node('G', 'Reports & Dossiers\n(.pdf, .pptx)', shape='note', style='filled', fillcolor='#9E9E9E')
        dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])
        dot.edge('D', 'F', label='Pydantic\nContract Check', style='dashed', color='red')
        st.graphviz_chart(dot)

    with tab2:
        st.subheader("Data Contract Schema (Pydantic)")
        st.markdown("This defines the expected schema, data types, and constraints for a dataset before it is loaded into the warehouse. This ensures data quality at the point of entry.")
        class PreclinicalDataContract(BaseModel):
            SampleID: str = Field(..., pattern=r"^VX-[A-Z]{2,3}-[A-Z]{3,4}-\d{2,3}-S\d{4}$")
            Timestamp: datetime
            OperatorID: str
            InstrumentID: str
            Dose_uM: float = Field(..., gt=0)
            Response: float
            CellViability: float = Field(..., ge=0, le=100)
        
        # === FIX APPLIED HERE ===
        # OLD, BUGGY CODE: ...model_json_schema(...).replace(...)
        # REASON: .model_json_schema() returns a dict, not a string.
        # NEW, CORRECTED CODE: Use the json library to pretty-print the dictionary.
        schema_dict = PreclinicalDataContract.model_json_schema(ref_template="#/definitions/{model}")
        st.code(json.dumps(schema_dict, indent=2), language='json')
        # ========================

    with tab3:
        st.subheader("Ad-Hoc SQL Query Hub")
        st.markdown("Execute direct, read-only SQL queries against the RPMS Data Warehouse for verification or ad-hoc analysis.")
        query = st.text_area("SQL Query", """
-- Example: Find reagent lots with below-average cell viability
SELECT
    "ReagentLot",
    AVG("CellViability") as "AvgViability"
FROM
    "PRECLINICAL_DATA"
WHERE
    "StudyID" = 'VX-CF-MOD-01'
GROUP BY
    "ReagentLot"
HAVING
    AVG("CellViability") < (SELECT AVG("CellViability") FROM "PRECLINICAL_DATA" WHERE "StudyID" = 'VX-CF-MOD-01')
ORDER BY
    "AvgViability" ASC;
        """, height=220)
        if st.button("Execute Query"):
            st.info("Query sent to Snowflake... (mock result below)")
            mock_res = pd.DataFrame({'ReagentLot': ['LOT-2024-AAAA'], 'AvgViability': [84.7]})
            st.dataframe(mock_res, use_container_width=True)

elif page == "📚 **SME Knowledge Base**":
    st.header("📚 SME Knowledge Base & Governance Center")
    st.markdown("Centralized definitions, methodologies, and standards governing the DTE-RPMS data ecosystem.")

    with st.expander("⭐️ **New Methodologies in Phoenix 2.0**", expanded=True):
        st.markdown("""
        - **Pydantic Data Contracts:** We use Pydantic models to define strict, machine-readable schemas for our key datasets. This proactive approach to data governance ensures data quality *before* it enters our analytical systems, reducing downstream errors.
        - **Statsmodels for Time Series Analysis:** For manufacturing process data, we now use `statsmodels` to perform Seasonal-Trend-Loess (STL) decomposition. This separates a noisy signal into its core trend, seasonal, and residual components, providing deeper insight into process stability than SPC charts alone.
        - **Automated PPTX Reporting:** The `python-pptx` library is integrated into the Regulatory Hub to auto-generate executive-level PowerPoint summaries of QC data, accelerating communication with stakeholders.
        - **Pytest for Unit Testing:** Our software quality assurance process is built on a foundation of automated unit tests using the `pytest` framework. Test results are reviewed as part of the formal GxP validation process, ensuring code is reliable and correct.
        """)

    with st.expander("🔬 **Key Scientific & Statistical Concepts**"):
        st.markdown("""
        - **IC50/EC50:** The half maximal inhibitory/effective concentration. It represents the concentration of a drug that is required for 50% inhibition/effect in vitro. A key measure of a compound's potency.
        - **4-Parameter Logistic (4PL) Curve:** A type of sigmoidal curve commonly used to model dose-response relationships. It is defined by its bottom asymptote, top asymptote, slope (Hill coefficient), and the IC50.
        - **ANOVA (Analysis of Variance):** A statistical test used to determine whether there are any statistically significant differences between the means of two or more independent groups.
        - **Overall Equipment Effectiveness (OEE):** A key manufacturing metric that measures the percentage of planned production time that is truly productive. `OEE = Availability × Performance × Quality`.
        """)

    with st.expander("📜 **Regulatory & GxP Governance**"):
        st.markdown("""
        - **GAMP 5 (Good Automated Manufacturing Practice):** A risk-based approach to compliant GxP computerized systems. The Phoenix Engine's validation lifecycle (URS, FS, IQ, OQ, PQ) is based on this framework.
        - **21 CFR Part 11:** The FDA's regulation for ensuring that electronic records and signatures are trustworthy, reliable, and equivalent to paper records. Features like the audit trail, electronic signatures, and validation records in this app are designed to meet these requirements.
        - **Requirements Traceability Matrix (RTM):** A document that maps and traces user requirements with test cases. It is a core part of our validation package to prove that all specified requirements have been tested and met.
        """)
elif page == "💡 **Strategic Roadmap & Vision**":
    st.header("💡 DTE-RPMS Automation: Strategic Roadmap & Vision")
    st.markdown("This outlines the multi-quarter strategic plan for the Phoenix Engine platform, ensuring alignment with Vertex's business objectives of scale, velocity, and innovation.")

    st.subheader("Q3 2024: Foundational Excellence & GxP Compliance")
    st.progress(100, text="Status: COMPLETE")
    st.markdown("""
    - **Objective:** Solidify core QC automation, establish a validated GxP environment, and provide robust reporting tools.
    - **Key Results:**
        - ✅ Deployed **Phoenix Engine 3.0** with persistent SQLite backend for audit trails.
        - ✅ Implemented **System Validation & QA** module with GAMP 5 workflow and mock test results.
        - ✅ Launched **Process Control** module with SPC and Time Series analysis for TRIKAFTA.
        - ✅ Deployed **Regulatory Hub** with automated PPTX generation and Pydantic-based dossier validation.
    """)

    st.subheader("Q4 2024: Predictive Analytics & Scalability")
    st.progress(65, text="Status: IN PROGRESS")
    st.markdown("""
    - **Objective:** Move from reactive QC to predictive data quality insights and scale data processing capabilities.
    - **Key Results:**
        - ⏳ **(In Progress)** Develop and deploy ML models for predictive CQA drift in manufacturing.
        - ⏳ **(In Progress)** Integrate **Spark/Dask** for large-scale genomic data reprocessing (See PoC in Proving Ground).
        - 🔜 Implement real-time data streaming from lab instruments via Kafka connector.
        - 🔜 Enhance RCA engine with anomaly clustering to identify novel failure modes.
    """)

    st.subheader("Q1 2025: Generative AI & Digital Twin")
    st.progress(10, text="Status: PLANNED")
    st.markdown("""
    - **Objective:** Leverage Generative AI for automated insights and create a digital twin of key laboratory processes.
    - **Key Results:**
        - 📝 **(Planned)** Integrate **LangChain** for automated generation of full study report narratives (See PoC in Proving Ground).
        - 📝 **(Planned)** Develop a 'digital twin' simulation of the Ussing Chamber assay to predict outcomes of parameter changes.
        - 📝 **(Planned)** Deploy a conversational AI assistant (trained on SME Knowledge Base) to answer user questions about processes and data.
    """)

elif page == "📈 **Process Control (TRIKAFTA)**":
    # This page's code remains the same from the previous version.
    st.header("📈 Process Control & Stability for TRIKAFTA® Manufacturing")
    st.markdown("Monitors critical quality attributes (CQAs) of TRIKAFTA® API manufacturing using advanced SPC and time series analysis.")
    process_name = st.selectbox("Select TRIKAFTA® CQA to Monitor:", ["TRIKAFTA_API_Purity", "Elexacaftor_Assay", "Tezacaftor_Assay"])
    df = generate_process_data(process_name)
    spec_col1, spec_col2, spec_col3 = st.columns(3); USL = spec_col1.number_input("Upper Specification Limit (USL)", value=100.0); TARGET = spec_col2.number_input("Target", value=99.5); LSL = spec_col3.number_input("Lower Specification Limit (LSL)", value=99.0)
    mean=df['Value'].mean(); std_dev=df['Value'].std(); ucl=mean+3*std_dev; lcl=mean-3*std_dev; cpk=min((USL-mean)/(3*std_dev),(mean-LSL)/(3*std_dev))
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3); kpi_col1.metric("Process Mean",f"{mean:.3f}"); kpi_col2.metric("Process Std Dev",f"{std_dev:.3f}"); kpi_col3.metric("Process Capability (Cpk)",f"{cpk:.2f}","Alert: < 1.33" if cpk<1.33 else "Stable: > 1.33",delta_color="inverse" if cpk<1.33 else "off")
    tab_spc, tab_tsa = st.tabs(["📊 **SPC I-Chart**", "📉 **Time Series Decomposition**"])
    with tab_spc:
        # SPC chart implementation...
    with tab_tsa:
        # Time series decomposition implementation...

elif page == "🧬 **Genomic Data QC (CASGEVY)**":
    # This page's code remains the same.
    st.header("🧬 Genomic Data QC Engine for Gene Therapies (CASGEVY)")
    # ... implementation ...

elif page == "📊 **Cross-Study & Batch Analysis**":
    # This page's code remains the same.
    st.header("📊 Cross-Study & Batch-to-Batch Analysis")
    # ... implementation ...

elif page == "💡 **Automated Root Cause Analysis**":
    # This page's code remains the same.
    st.header("💡 Automated Root Cause Analysis (RCA) Engine")
    # ... implementation ...

elif page == "🚀 **Technology Proving Ground**":
    st.header("🚀 Technology Proving Ground (PoC Environment)")
    st.warning("**For Demonstration Only:** This area is for evaluating and prototyping emerging technologies. Results are not for GxP use.")

    tab_langchain, tab_dask, tab_r = st.tabs(["📄 **GenAI: LangChain Summarization**", "💨 **Scalability: Dask Processing**", "📊 **Analytics: R Integration**"])

    with tab_langchain:
        st.subheader("Proof-of-Concept: Automated Report Summarization")
        st.markdown("This PoC demonstrates how **LangChain** could be used to automatically generate a human-readable summary from a structured QC report.")
        report_text = st.text_area("Paste Structured Report Data Here (e.g., JSON from a QC run):", height=200, value="""
{
    "study_id": "VX-CF-MOD-01",
    "qc_run_date": "2024-05-21",
    "data_integrity_score": 0.998,
    "key_findings": [
        {"test": "IC50 Potency", "result": 1.2, "units": "uM", "status": "PASS"},
        {"test": "Cell Viability", "result": 92.5, "units": "%", "status": "PASS"},
        {"test": "Reagent Lot Purity", "lot": "LOT-2024-AAAA", "result": 85.1, "units": "%", "status": "FAIL"}
    ],
    "conclusion": "Study passed overall, but one reagent lot failed purity spec and has been quarantined."
}
        """)
        if st.button("🤖 Generate Summary with LangChain PoC"):
            with st.spinner("Simulating call to LangChain API..."):
                import time
                time.sleep(2)
                st.subheader("Generated Narrative Summary:")
                st.info("""
                **Study VX-CF-MOD-01 QC Summary:**

                The quality control analysis conducted on May 21, 2024, for study VX-CF-MOD-01 has concluded. The overall data integrity score was excellent at 99.8%.

                Key assays, including IC50 Potency (1.2 µM) and Cell Viability (92.5%), met all acceptance criteria. However, a significant deviation was noted in the Reagent Lot Purity test for **lot LOT-2024-AAAA**, which failed with a result of 85.1%.

                **Action:** While the study passes overall, the failing reagent lot has been flagged and quarantined to prevent its use in future experiments.
                """)
            log_action("engineer.principal@vertex.com", "POC_LANGCHAIN_SUMMARY")

    with tab_dask:
        st.subheader("Proof-of-Concept: Large-Scale Data Processing")
        st.markdown("This PoC uses **Dask** to simulate the parallel processing of a large (50,000 row) dataset, a task common in genomics or late-stage study aggregation. Dask allows for computations on data larger than system RAM.")
        if st.button("🚀 Process Large Dataset with Dask"):
            with st.spinner("Setting up Dask cluster and processing partitions..."):
                dask_results = load_data_with_dask("dummy_path") # Function defined in Part 1
                st.subheader("Dask Computation Results:")
                st.write("Mean 'Response' grouped by 'ReagentLot':")
                st.dataframe(dask_results)
            log_action("engineer.principal@vertex.com", "POC_DASK_PROCESSING")

    with tab_r:
        st.subheader("Proof-of-Concept: R Script Integration via `rpy2`")
        st.markdown("This PoC demonstrates how a statistical analysis or plot generated in **R** can be executed and its results displayed within the Python-based Phoenix Engine.")
        
        st.info("ℹ️ The `rpy2` library is required for this feature. The plot below is a static image representing the output from an R script.")

        # In a real app with rpy2 configured:
        # r_script = """
        # library(ggplot2)
        # data(mtcars)
        # p <- ggplot(mtcars, aes(x=wt, y=mpg)) + geom_point() + labs(title="R Plot via rpy2")
        # ggsave("r_plot.png")
        # """
        # ro.r(r_script)
        # st.image("r_plot.png")

        st.image("https://www.r-graph-gallery.com/img/graph/277-marginal-histogram-for-ggplot2.png",
                 caption="Example of a complex statistical plot generated by R's ggplot2 library, which could be displayed here.")
elif page == "🏛️ **Regulatory & Audit Hub**":
    st.header("🏛️ Regulatory & Audit Hub")
    st.markdown("Prepare, package, and document data dossiers for regulatory inspections and internal audits with full 21 CFR Part 11 traceability.")

    st.info("""
    **21 CFR Part 11 Compliance Features:**
    - **Audit Trails:** All actions on this page are logged to a persistent, time-stamped database table (See `Data Lineage & Versioning`).
    - **Electronic Signatures:** User authentication is required, and actions are linked to the logged-in user.
    - **Logical Security:** Controls ensure data cannot be altered after packaging and checksum generation.
    """)

    with st.form("audit_sim_form"):
        # The form remains the same as the previous version...
        st.subheader("Package New Regulatory Dossier")
        c1, c2, c3 = st.columns(3); req_id = c1.text_input("Request ID", "FDA-REQ-003"); agency = c2.selectbox("Requesting Agency", ["FDA", "EMA", "PMDA", "Internal QA"]); study_id_package = c3.selectbox("Select Study to Package:", ["VX-CF-MOD-01", "VX-522-Tox-02"])
        st.text_area("Justification / Request Details", "Follow-up request for raw data, QC reports, and statistical analysis for the selected study, focusing on outlier investigation.")
        files_to_include = st.multiselect("Select Data & Artifacts to Include:",["Raw Instrument Data (.csv)", "QC Anomaly Report (.pdf)", "Data Lineage Graph (.svg)", "Audit Trail Log (.json)", "Statistical Analysis Script (R/Python)", "Executive Summary (.pptx)"],default=["Raw Instrument Data (.csv)", "QC Anomaly Report (.pdf)", "Audit Trail Log (.json)", "Executive Summary (.pptx)"])
        submitter_name = st.text_input("Enter Full Name for Electronic Signature:", "Dr. Principal Engineer")
        submitted = st.form_submit_button("🔒 Validate, Lock, and Package Dossier")

    if submitted:
        with st.spinner("1. Validating dossier contract... 2. Generating checksums... 3. Logging GxP action..."):
            import time, hashlib
            time.sleep(2)
            dossier_checksum = hashlib.sha256(f"{req_id}{study_id_package}{submitter_name}".encode()).hexdigest()
            log_action(user="engineer.principal@vertex.com", action="PACKAGE_REGULATORY_DOSSIER", target_id=req_id, details={'study': study_id_package, 'files': files_to_include, 'signature': submitter_name})
            st.success(f"Dossier Packaged & Action Logged!")
            # ... download buttons and other logic ...

elif page == "🔗 **Data Lineage & Versioning**":
    st.header("🔗 Data Lineage, Versioning & Discrepancy Hub")
    st.markdown("Visualize data provenance, review change histories for any record, and manage data quality discrepancies.")
    
    tab_lineage, tab_versioning, tab_discrepancy = st.tabs(["🗺️ **Visual Data Flow**", "🕓 **Data Versioning (Audit Trail)**", "🔧 **Discrepancy Resolution**"])
    
    with tab_lineage:
        # Same as before...
        st.subheader("End-to-End Data Flow")
        # graphviz chart...
    
    with tab_versioning:
        st.subheader("Record Change History Viewer")
        st.markdown("Query the persistent audit log to see the version history of any data entity or record.")
        target_id_to_view = st.text_input("Enter Record/Dossier/Batch ID to Audit:", "FDA-REQ-003")
        if st.button("🔍 View History"):
            try:
                conn = sqlite3.connect(DB_FILE)
                query = "SELECT timestamp, user, action, details FROM audit_log WHERE target_id = ? ORDER BY timestamp DESC"
                history_df = pd.read_sql_query(query, conn, params=(target_id_to_view,))
                conn.close()
                if not history_df.empty:
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.warning(f"No history found for ID '{target_id_to_view}'.")
            except Exception as e:
                st.error(f"Database connection failed. Displaying cached data is not available for this feature. Error: {e}")

    with tab_discrepancy:
        st.subheader("Automated Discrepancy Resolution")
        st.markdown("Review and approve system-suggested fixes for data quality issues.")
        # Simulate some discrepant data
        disc_data = {'SampleID': ['VX-CF-MOD-01-S0123', 'VX-CF-MOD-01-S0124'], 'Response': [95.4, None], 'CellViability': [88.1, 89.2]}
        disc_df = pd.DataFrame(disc_data)
        disc_df['Suggested_Fix'] = [None, disc_df['Response'].mean()]
        st.write("Discrepant Records Found:")
        st.dataframe(disc_df, use_container_width=True)
        if st.button("✅ Approve & Apply Suggested Fixes"):
            log_action("engineer.principal@vertex.com", "APPLY_DISCREPANCY_FIX", "VX-CF-MOD-01", details={"imputed_value": disc_df['Response'].mean()})
            st.success("Fix applied and action logged.")

elif page == "✅ **System Validation & QA**":
    # Page remains the same, still relevant and robust.
    st.header("✅ System Validation & Quality Assurance")
    # ... implementation ...

elif page == "⚙️ **System Admin Panel**":
    st.header("⚙️ System Administration Panel")
    st.warning("**For Authorized Administrators Only.** Changes here affect the entire application and are fully audited.")

    st.subheader("Current Application Configuration (`config.yml`)")
    st.code(CONFIG_TEXT, language='yaml')

    with st.form("config_form"):
        st.subheader("Modify Configuration")
        st.markdown("Update validation rules or UI settings. A restart will be required to apply changes.")
        new_min_viability = st.number_input("New Minimum Cell Viability Threshold (Current: 70)", value=CONFIG['validation_rules']['cell_viability']['min'])
        new_dashboard_title = st.text_input("New Dashboard Title", value=CONFIG['ui_settings']['dashboard_title'])
        
        submitted = st.form_submit_button("Submit & Log Configuration Change")
        if submitted:
            change_details = {'new_min_viability': new_min_viability, 'new_dashboard_title': new_dashboard_title}
            log_action("engineer.principal@vertex.com", "CONFIG_CHANGE_REQUEST", "config.yml", details=change_details)
            st.success("Configuration change request logged! Please restart the application server to apply.")
            st.info("In a real application, this would trigger a CI/CD pipeline to safely deploy the new configuration.")

elif page == "📈 **System Health & Metrics**":
    st.header("📈 System Health, KPIs & User Adoption")
    st.markdown("Live dashboard monitoring the performance of the Phoenix Engine and user engagement.")

    try:
        conn = sqlite3.connect(DB_FILE)
        # KPI 1: User Adoption
        actions_df = pd.read_sql_query("SELECT timestamp, action FROM audit_log", conn)
        actions_df['timestamp'] = pd.to_datetime(actions_df['timestamp'])
        
        # KPI 2: User Feedback
        feedback_df = pd.read_sql_query("SELECT rating FROM user_feedback", conn)
        avg_rating = feedback_df['rating'].mean() if not feedback_df.empty else "N/A"
        
        # KPI 3: DB Health
        db_status = "Connected"
        db_status_color = "normal"
        
        conn.close()

    except Exception as e:
        st.error(f"**Database Unreachable!** The system cannot fetch live metrics. Displaying last known values.")
        logger.error(f"DB health check failed: {e}")
        # Fallback to cached/default values
        actions_df = pd.DataFrame()
        avg_rating = "N/A"
        db_status = "Disconnected"
        db_status_color = "inverse"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Logged Actions", len(actions_df))
    c2.metric("Average User Feedback Rating", f"{avg_rating:.2f} / 5" if isinstance(avg_rating, float) else avg_rating)
    c3.metric("Backend Database Status", db_status, delta_color=db_status_color)
    
    if not actions_df.empty:
        st.subheader("User Actions Over Time")
        action_counts = actions_df.set_index('timestamp').resample('D').size().rename('actions')
        st.line_chart(action_counts)
        
elif page == "📚 **SME Knowledge Base & Help**":
    st.header("📚 SME Knowledge Base & Help Center")
    st.markdown("Centralized documentation, tutorials, and feedback mechanisms.")

    tab_kb, tab_help, tab_feedback = st.tabs(["🧠 **Knowledge Base**", "❓ **Help & Guides**", "💬 **Submit Feedback**"])
    
    with tab_kb:
        # Same as before, but with new entries...
        st.subheader("Core Methodologies & Platform Features")
        st.markdown("""
        - **Pydantic Data Contracts:** ...
        - **Statsmodels for Time Series Analysis:** ...
        - **Persistent Audit Trail:** All GxP-relevant actions are logged to a persistent SQLite database, providing a full, auditable history of data changes, report generation, and configuration updates. See the 'Data Versioning' tab.
        - **Dynamic Configuration:** The `config.yml` file allows administrators to change key application parameters without code changes. These changes are logged and require a restart to take effect.
        """)
        
    with tab_help:
        st.subheader("Step-by-Step Guides")
        st.markdown("""
        **How to package a regulatory dossier:**
        1. Navigate to the **Regulatory & Audit Hub**.
        2. Fill in the Request ID, Agency, and select the Study.
        3. Select all required artifacts to include in the package.
        4. Enter your full name for the e-signature.
        5. Click 'Lock and Package Dossier'. The action will be logged, and download links will appear.

        **Troubleshooting common issues:**
        - **`Database Unreachable` error:** This indicates the backend database is down. The app will enter a failover mode with limited functionality. Contact DTE support.
        - **Plot not loading:** Try clearing the cache by clicking the 'C' icon in the top right and re-running the page.
        """)

    with tab_feedback:
        st.subheader("Provide Feedback on this Platform")
        st.markdown("Your feedback is critical for our continuous improvement (Kaizen) process.")
        with st.form("feedback_form"):
            feedback_page = st.selectbox("Which page are you providing feedback for?", [p.split(' ')[1] for p in page.split('\n')])
            feedback_rating = st.slider("Rating (1=Poor, 5=Excellent)", 1, 5, 4)
            feedback_comment = st.text_area("Comments:")
            
            feedback_submitted = st.form_submit_button("Submit Feedback")
            if feedback_submitted:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO user_feedback (timestamp, page, rating, comment) VALUES (?, ?, ?, ?)",
                          (datetime.now(), feedback_page, feedback_rating, feedback_comment))
                conn.commit()
                conn.close()
                st.success("Thank you! Your feedback has been recorded.")
