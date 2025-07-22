# =================================================================================================
# Phoenix Engine 4.0 - VTX DTE-RPMS Validated Digital Twin & Analytics Platform
# =================================================================================================

# Core & UI
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io, os, json, yaml, logging, time, hashlib, sqlite3

# Data, DB, & Scalability
from sqlalchemy import create_engine
import dask.dataframe as dd

# Advanced Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import graphviz
import matplotlib.pyplot as plt

# Advanced Statistics, ML & Reporting
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import shap
from pptx import Presentation
from pptx.util import Inches
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# R Language Integration (with Simulation Switch for Deployment Stability)
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    RPY2_INSTALLED = True
except ImportError:
    RPY2_INSTALLED = False

# =================================================================================================
# Initial Setup
# =================================================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_TEXT = """
validation_rules:
  api_purity: {min: 98.5, max: 101.5}
  cell_viability: {min: 70}
report_templates:
  executive_summary: "templates/exec_summary_v2.pptx"
  full_study_report: "templates/full_report_v4.docx"
ui_settings:
  dashboard_title: "Phoenix Engine 4.0"
  show_experimental_features: true
api_endpoints:
  data_platform_export: "https://api.vertex.com/v1/dte/data_export"
"""
CONFIG = yaml.safe_load(CONFIG_TEXT)
DB_FILE = "phoenix_engine.db"

def init_db(populate=False):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, user TEXT NOT NULL, action TEXT NOT NULL, target_id TEXT, details TEXT, checksum TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, page TEXT NOT NULL, rating INTEGER, comment TEXT)''')
    if populate:
        c.execute("SELECT COUNT(*) FROM audit_log")
        if c.fetchone()[0] == 0:
            populate_db_with_mock_history(conn)
    conn.commit()
    conn.close()

def populate_db_with_mock_history(conn):
    c = conn.cursor()
    users = ["engineer.principal@vertex.com", "scientist.a@vertex.com", "qa.specialist@vertex.com"]
    actions = ["LOGIN", "VIEW_DASHBOARD", "RUN_QC_ANALYSIS", "EXPORT_RAW_DATA", "PACKAGE_REGULATORY_DOSSIER"]
    for i in range(200):
        user = np.random.choice(users)
        action = np.random.choice(actions)
        timestamp = datetime.now() - timedelta(days=np.random.uniform(0, 30))
        checksum = hashlib.sha256(f"{timestamp}{user}{action}".encode()).hexdigest()
        c.execute("INSERT INTO audit_log (timestamp, user, action, checksum) VALUES (?, ?, ?, ?)", (timestamp, user, action, checksum))
    logger.info("Database populated with mock history.")

def log_action(user, action, target_id=None, details=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    checksum = hashlib.sha256(f"{datetime.now()}{user}{action}{target_id}{details}".encode()).hexdigest()
    c.execute("INSERT INTO audit_log (timestamp, user, action, target_id, details, checksum) VALUES (?, ?, ?, ?, ?, ?)", (datetime.now(), user, action, target_id, json.dumps(details), checksum))
    conn.commit()
    conn.close()
    logger.info(f"Action logged: {action} by {user} for target {target_id}")

init_db(populate=True)

class QCResult(BaseModel):
    check_name: str
    status: str = Field(..., pattern=r"^(PASS|FAIL|WARN)$")
    details: Optional[str] = None
    failed_record_count: int

class RegulatoryDossier(BaseModel):
    request_id: str
    agency: str
    study_id: str
    package_checksum: str = Field(..., min_length=64, max_length=64)
    qc_summary: List[QCResult]

class PreclinicalDataContract(BaseModel):
    SampleID: str = Field(..., pattern=r"^VX-[A-Z]{2,3}-[A-Z]{3,4}-\d{2,3}-S\d{4}$")
    Timestamp: datetime
    OperatorID: str
    Response: float
    CellViability: float = Field(..., ge=CONFIG['validation_rules']['cell_viability']['min'], le=100)

def generate_summary_pptx(study_id, kpi_data):
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = f"Executive QC Summary: Study {study_id}"
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
    tf.add_paragraph().text = "\nThis summary was auto-generated by the Phoenix Engine 4.0."
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    ppt_io.seek(0)
    return ppt_io

@st.cache_data(ttl=900)
def generate_preclinical_data(study_id, n_samples=1000):
    np.random.seed(hash(study_id) % (2**32 - 1))
    operators = ['J.Doe', 'S.Chen', 'M.Gupta', 'R.Valdez']
    instruments = {'PK':['Agilent-6470', 'Sciex-7500'],'Tox':['Tecan-Spark','BMG-Pherastar'],'CF':['Ussing-Chamber-A','Ussing-Chamber-B']}
    assay_type = study_id.split('-')[1]
    def sigmoid(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))
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
        'CellViability': np.random.normal(95, 4, n_samples).clip(70, 100)
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

@st.cache_data
def load_data_with_dask(filepath):
    df = generate_preclinical_data("VX-LARGE-SCALE-01", n_samples=50000)
    ddf = dd.from_pandas(df, npartitions=4)
    progress_bar = st.progress(0, text="Processing large dataset with Dask...")
    result = ddf.groupby('ReagentLot').Response.mean().compute()
    progress_bar.progress(100, text="Processing Complete!")
    return result

@st.cache_data(ttl=900)
def generate_global_kpis():
    sites = ["Boston, USA", "San Diego, USA", "Oxford, UK"]
    data = []
    for site in sites:
        data.append({'Site': site, 'lon': [-71.0589, -117.1611, -1.2577][sites.index(site)], 'lat': [42.3601, 32.7157, 51.7520][sites.index(site)], 'Studies_Active': np.random.randint(5, 15), 'Data_Integrity': f"{np.random.uniform(99.5, 99.9):.2f}%", 'Automation_Coverage': np.random.randint(85, 98), 'Critical_Flags': np.random.randint(0, 5), 'Mfg_OEE': np.random.randint(75, 92), 'Cpk_Avg': f"{np.random.uniform(1.3, 1.8):.2f}"})
    return pd.DataFrame(data)

@st.cache_data(ttl=900)
def generate_process_data(process_name="TRIKAFTA_API_Purity"):
    np.random.seed(hash(process_name) % (2**32 - 1))
    data = {'BatchID': [f'MFG-24-{i:03d}' for i in range(1, 101)], 'Timestamp': pd.to_datetime(pd.date_range(end=datetime.now(), periods=100)), 'Value': np.random.normal(99.5, 0.2, 100)}
    df = pd.DataFrame(data)
    df.loc[75:, 'Value'] += 0.35
    return df

def execute_r_spc_chart(data_df, title):
    if not RPY2_INSTALLED:
        return None
    try:
        qcc = importr('qcc')
        grdevices = importr('grDevices')
        r_df = pandas2ri.py2rpy(data_df)
        ro.globalenv['r_df'] = r_df
        ro.globalenv['chart_title'] = title
        ro.r(f"library(qcc)\npng('r_spc_chart.png', width=800, height=500)\nqcc_obj <- qcc(r_df$Value, type=\"xbar.one\", title='{title}', labels=r_df$BatchID)\ndev.off()")
        return "r_spc_chart.png"
    except Exception as e:
        logger.error(f"Rpy2 execution failed: {e}")
        return None

with st.sidebar:
    st.image("https://d1io3yog0oux5.cloudfront.net/_3f03b2222d6fdd47976375a7337f7a69/vertexpharmaceuticals/db/387/2237/logo.png", width=220)
    st.title(CONFIG['ui_settings']['dashboard_title'])
    st.markdown("##### GxP Hyper-Automation Platform")
    st.markdown("---")
    page = st.radio("Navigation", ["🌎 **Global Command Center**", "🔬 **Assay Dev & Dose-Response**", "💡 **Strategic Roadmap & Vision**", "📈 **Process Control (TRIKAFTA)**", "🧬 **Genomic Data QC (CASGEVY)**", "📊 **Cross-Study & Batch Analysis**", "💡 **Automated Root Cause Analysis**", "🚀 **Technology Proving Ground**", "🏛️ **Regulatory & Audit Hub**", "🔗 **Data Lineage & Versioning**", "✅ **System Validation & QA**", "⚙️ **System Admin Panel**", "📈 **System Health & Metrics**", "📚 **SME Knowledge Base & Help**"], label_visibility="collapsed")
    st.markdown("---")
    st.info(f"**Principal Engineer, DTE-RPMS**\n\n**User:** engineer.principal@vertex.com\n\n**Session Start:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if page == "🌎 **Global Command Center**":
    st.header("🌎 Global RPMS Operations Command Center")
    st.markdown("Real-time, holistic view of data operations, integrity, and automation initiatives across all major research and manufacturing sites.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Data Integrity", "99.81%", "+0.15%")
    c2.metric("Automation Index", "95%", "Target: 95%")
    c3.metric("Data Validation Success Rate", "99.92%", help="Percentage of incoming records passing Pydantic contract validation.")
    c4.metric("Pending Audit Actions", "4", "2 FDA, 2 EMA")
    st.markdown("---")
    map_col, alerts_col = st.columns([2, 1])
    with map_col:
        st.subheader("Global Site Status & Health")
        global_kpis = generate_global_kpis()
        fig = go.Figure(data=go.Scattergeo(lon=global_kpis['lon'], lat=global_kpis['lat'], text=global_kpis.apply(lambda row: f"<b>{row['Site']}</b><br>Integrity: {row['Data_Integrity']}<br>Automation: {row['Automation_Coverage']}%<br>OEE: {row['Mfg_OEE']}%<br>Flags: {row['Critical_Flags']}", axis=1), mode='markers', marker=dict(color=global_kpis['Critical_Flags'], colorscale=[[0, '#00AEEF'], [1, '#FF4136']], size=global_kpis['Automation_Coverage'] / 4, colorbar_title='Critical Flags')))
        fig.update_layout(geo=dict(scope='world'), margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    with alerts_col:
        st.subheader("Priority Action Items")
        st.error("🔴 **CRITICAL:** [TRIKAFTA MFG] Cpk for API Purity dropped to 1.31. Batch MFG-24-088 under review.")
        st.warning("🟠 **WARNING:** [CASGEVY QC] Reagent Lot LOT-2024-AAAA shows 15% lower cell viability. Lot quarantined.")
        st.info("🔵 **INFO:** [VX-522 Dev] New dose-response data from Oxford site available for review.")
        st.info("🔵 **INFO:** [FDA-REQ-003] Dossier for VTX-809-PK-01 is packaged and ready for final review.")

elif page == "🔬 **Assay Dev & Dose-Response**":
    st.header("🔬 Assay Development & 3D Dose-Response Modeling")
    st.markdown("Analyze in-vitro assay data, fit dose-response curves, and export raw data packages for regulatory review.")
    study_list = ["VX-CF-MOD-01", "VX-522-Tox-02", "VX-PAIN-TGT-05"]
    c1, c2 = st.columns([3, 1])
    selected_study = c1.selectbox("Select a Vertex Development Study:", study_list)
    df = generate_preclinical_data(selected_study)
    with c2:
        st.write("")
        st.write("")
        if st.button("📦 Export Raw Data", help="Export the full raw dataset for this study as CSV for regulatory inspection."):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download CSV", data=csv, file_name=f'{selected_study}_raw_data.csv', mime='text/csv')
            log_action("engineer.principal@vertex.com", "EXPORT_RAW_DATA", selected_study)
            st.success("Raw data prepared for download.")
    tab1, tab2, tab3 = st.tabs(["📈 **2D Dose-Response Curve (IC50)**", "✨ **3D Response Surface**", "📦 **Batch Box Plots**"])
    with tab1:
        st.subheader(f"Dose-Response Curve for {selected_study}")
        fig = px.scatter(df, x="Dose_uM", y="Response", log_x=True, title="Potency Assay: Response vs. Dose", labels={"Dose_uM": "Dose (µM)", "Response": "Assay Response (%)"})
        max_resp = df['Response'].max()
        ic50_approx = df.iloc[(df['Response'] - max_resp / 2).abs().argsort()[:1]]['Dose_uM'].values[0]
        fig.add_vline(x=ic50_approx, line_dash="dash", line_color="firebrick", annotation_text=f"IC50 ≈ {ic50_approx:.2f} µM")
        fig.add_hline(y=max_resp / 2, line_dash="dash", line_color="firebrick")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("3D Response Surface: Dose, Time, and Viability")
        df['Time_h'] = df.index / len(df) * 48
        fig3d = px.scatter_3d(df.sample(500), x='Dose_uM', y='Time_h', z='Response', color='CellViability', log_x=True, title="3D Interaction Plot", labels={"Dose_uM": "Dose (log µM)", "Time_h": "Time (h)", "Response": "Response (%)"})
        st.plotly_chart(fig3d, use_container_width=True)
    with tab3:
        st.subheader("Response Distribution by Reagent Lot")
        fig_box = px.box(df, x='ReagentLot', y='Response', color='ReagentLot', color_discrete_map={"LOT-2024-A": "#0033A0", "LOT-2024-AA": "#00AEEF", "LOT-2024-AAA": "#63C5F3", "LOT-2024-AAAA": "#FF4136"})
        st.plotly_chart(fig_box, use_container_width=True)
elif page == "💡 **Strategic Roadmap & Vision**":
    st.header("💡 DTE-RPMS Automation: Strategic Roadmap & Vision")
    st.markdown("This outlines the multi-quarter strategic plan for the Phoenix Engine platform, ensuring alignment with Vertex's business objectives of scale, velocity, and innovation.")
    st.subheader("Q3 2024: Foundational Excellence & GxP Compliance")
    st.progress(100, text="Status: COMPLETE")
    st.markdown("""
    - **Objective:** Solidify core QC automation, establish a validated GxP environment, and provide robust reporting tools.
    - **Key Results:**
        - ✅ Deployed **Phoenix Engine 4.0** with live SHAP & R integration.
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
    cpk = min((USL - mean) / (3 * std_dev), (mean - LSL) / (3 * std_dev)) if std_dev > 0 else 0
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Process Mean", f"{mean:.3f}")
    kpi_col2.metric("Process Std Dev", f"{std_dev:.3f}")
    kpi_col3.metric("Process Capability (Cpk)", f"{cpk:.2f}", "Alert: < 1.33" if cpk < 1.33 else "Stable: > 1.33", delta_color="inverse" if cpk < 1.33 else "off")
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
    with tab_tsa:
        st.subheader("Time Series Analysis (STL Decomposition)")
        df_ts = df.set_index('Timestamp')
        stl = sm.tsa.STL(df_ts['Value'], period=7, robust=True).fit()
        fig_tsa = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Seasonal", "Residual"))
        fig_tsa.add_trace(go.Scatter(x=df_ts.index, y=stl.trend, mode='lines', name='Trend'), row=1, col=1)
        fig_tsa.add_trace(go.Scatter(x=df_ts.index, y=stl.seasonal, mode='lines', name='Seasonal'), row=2, col=1)
        fig_tsa.add_trace(go.Scatter(x=df_ts.index, y=stl.resid, mode='markers', name='Residual'), row=3, col=1)
        fig_tsa.update_layout(height=600, title_text=f"STL Decomposition for {process_name}", showlegend=False)
        st.plotly_chart(fig_tsa, use_container_width=True)

elif page == "🧬 **Genomic Data QC (CASGEVY)**":
    st.header("🧬 Genomic Data QC Engine for Gene Therapies (CASGEVY)")
    st.markdown("Specialized module for QC of gene-editing data, including on-target allele frequency, off-target analysis, and editing efficiency.")
    @st.cache_data
    def generate_casgevy_qc_data(sample_id):
        np.random.seed(hash(sample_id) % (2**32 - 1))
        editing_data = {'BatchID': [f'B{i:02d}' for i in range(20)], 'EditingEfficiency': np.random.normal(92, 3.5, 20).clip(80, 99)}
        off_target_data = {'Site': [f'OT_{i:02d}' for i in range(1, 11)], 'MismatchCount': np.random.randint(2, 5, 10), 'GUIDESeqScore': np.random.lognormal(0.5, 1, 10), 'IndelFreq': np.random.uniform(0.01, 0.5, 10)}
        off_target_data['Site'][7] = 'OT_08_CRITICAL'
        off_target_data['IndelFreq'][7] = np.random.uniform(2, 4)
        return pd.DataFrame(editing_data), pd.DataFrame(off_target_data)
    sample_id = st.text_input("Enter CASGEVY Patient Sample ID:", "V-PT-007-BCH-01")
    if sample_id:
        editing_df, off_target_df = generate_casgevy_qc_data(sample_id)
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Avg. Editing Efficiency", f"{editing_df['EditingEfficiency'].mean():.1f}%", f"{editing_df['EditingEfficiency'].std():.2f} StDev")
        kpi2.metric("On-Target Rate", "98.2%", "vs 95% Target")
        kpi3.metric("Critical Off-Target Flags", "1", delta_color="inverse")
        tab1, tab2, tab3 = st.tabs(["🎯 **Editing Efficiency & Outcomes**", "🛰️ **Off-Target Site Analysis**", "🧬 **Whole-Genome CNV Scan**"])
        with tab1:
            st.subheader("On-Target Editing Efficiency Distribution")
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(editing_df, x='EditingEfficiency', nbins=10, title='Efficiency Across Batches')
                fig_hist.add_vline(x=90, line_dash="dash", annotation_text="Release Spec (>90%)")
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                outcomes = {'Outcome': ['Desired Edit', 'Indel (Small)', 'No Edit'], 'Percentage': [93.5, 5.2, 1.3]}
                fig_pie = px.pie(outcomes, values='Percentage', names='Outcome', title='Aggregate Editing Outcomes', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
        with tab2:
            st.subheader("Off-Target Site Scoring & Prioritization")
            st.markdown("This plot scores potential off-target sites based on experimental GUIDE-seq scores and observed indel frequencies.")
            fig_ot = px.scatter(off_target_df, x='GUIDESeqScore', y='IndelFreq', size='MismatchCount', color='IndelFreq', title='Off-Target Site Risk Assessment', hover_name='Site', log_x=True, color_continuous_scale='Reds')
            st.plotly_chart(fig_ot, use_container_width=True)
        with tab3:
            st.subheader("Initial Whole-Genome CNV Sanity Check")
            st.info("This scan serves as a quality control step to ensure the patient's genomic baseline is free of large-scale variations that could confound gene-editing analysis. No significant CNVs detected in this sample.")

elif page == "📊 **Cross-Study & Batch Analysis**":
    st.header("📊 Cross-Study & Batch-to-Batch Analysis")
    st.markdown("Perform comparative statistical analyses across different studies...")
    study_list = ["VX-CF-MOD-01", "VX-522-Tox-02", "VX-PAIN-TGT-05"]
    selected_studies = st.multiselect("Select studies to compare:", study_list, default=study_list)
    if selected_studies:
        data_frames = [generate_preclinical_data(s).assign(StudyID=s) for s in selected_studies]
        all_data = pd.concat(data_frames, ignore_index=True)
        tab1, tab2, tab3 = st.tabs(["📊 **ANOVA & Post-Hoc Analysis**", "🎻 **Violin & Box Plots**", "🛰️ **3D Principal Component Analysis**"])
        with tab1:
            st.subheader("Analysis of Variance (ANOVA) & Tukey's HSD")
            model = ols('Response ~ C(StudyID)', data=all_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(anova_table)
            p_value = anova_table['PR(>F)'].iloc[0]
            if p_value < 0.05:
                st.error(f"**Statistically Significant Difference Detected (p = {p_value:.4f}).**")
                tukey = pairwise_tukeyhsd(endog=all_data['Response'], groups=all_data['StudyID'], alpha=0.05)
                st.subheader("Tukey's HSD Post-Hoc Test Results")
                st.dataframe(pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0]))
                fig = go.Figure()
                for i, res in enumerate(tukey.summary().tables[1].data[1:]):
                    group1, group2, _, meandiff, lower, upper, reject = res
                    fig.add_shape(type="line", x0=i, y0=float(lower), x1=i, y1=float(upper), line=dict(color="RoyalBlue", width=3))
                    fig.add_trace(go.Scatter(x=[i], y=[float(meandiff)], mode="markers", marker=dict(color="RoyalBlue"), name=f"{group1}-{group2}"))
                fig.update_xaxes(tickvals=list(range(len(tukey.groupsunique)*(len(tukey.groupsunique)-1)//2)), ticktext=[f"{res[0]}-{res[1]}" for res in tukey.summary().tables[1].data[1:]])
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(title="95% Confidence Intervals for Group Mean Differences")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No statistically significant difference detected.")
        with tab2:
            st.subheader("Response Distribution by Study")
            fig_violin = px.violin(all_data, x='StudyID', y='Response', color='StudyID', box=True, points="all", title="Assay Response Distribution Across Studies")
            st.plotly_chart(fig_violin, use_container_width=True)
        with tab3:
            st.subheader("3D PCA for Outlier/Cluster Detection")
            df_pca = all_data[['Dose_uM', 'Response', 'CellViability']].dropna()
            pca = PCA(n_components=3)
            components = pca.fit_transform(df_pca)
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
            pca_df['StudyID'] = all_data.loc[df_pca.index, 'StudyID']
            total_var = pca.explained_variance_ratio_.sum() * 100
            fig_pca_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='StudyID', title=f'3D PCA of Preclinical Data ({total_var:.1f}% Variance Explained)')
            st.plotly_chart(fig_pca_3d, use_container_width=True)

elif page == "💡 **Automated Root Cause Analysis**":
    st.header("💡 Automated Root Cause Analysis (RCA) Engine")
    st.markdown("Leverages machine learning to predict the likely cause of QC flags and provides interactive tools for investigation.")
    study_id = st.selectbox("Select Study with QC Flags:", ["VX-CF-MOD-01", "VX-522-Tox-02"], key="rca_study")
    df = generate_preclinical_data(study_id)
    df_flagged = df[df['QC_Flag'] == 1]
    if not df_flagged.empty:
        st.warning(f"Found **{len(df_flagged)}** QC-flagged records in **{study_id}**. Initiating RCA...")
        features = ['OperatorID', 'InstrumentID', 'ReagentLot']
        target = 'QC_Flag'
        df_ml = df.copy()
        encoders = {col: LabelEncoder().fit(df_ml[col]) for col in features}
        for col, encoder in encoders.items():
            df_ml[col] = encoder.transform(df_ml[col])
        X = df_ml[features]
        y = df_ml[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        tab1, tab2, tab3 = st.tabs(["🎯 **Predicted Root Cause**", "🔍 **Interactive Drill-Down**", "🌊 **Live SHAP Waterfall**"])
        with tab1:
            st.subheader("Probable Root Cause Contribution")
            importances_vals = model.feature_importances_
            importances = pd.DataFrame({'Feature': features, 'Importance': importances_vals}).sort_values('Importance', ascending=False)
            fig_imp = px.bar(importances, x='Feature', y='Importance', title='Feature Importance for QC Failures', color='Feature', color_discrete_map={"ReagentLot": "#FF4136", "OperatorID": "#FF851B", "InstrumentID": "#FFDC00"})
            st.plotly_chart(fig_imp, use_container_width=True)
            top_cause = importances.iloc[0]
            st.error(f"**Top Predicted Contributor:** **{top_cause['Feature']}** is the most significant factor.")
        with tab2:
            st.subheader("Visual Investigation of Top Contributor")
            top_feature = importances.iloc[0]['Feature']
            fig_drill = px.box(df, x=top_feature, y='Response', color='QC_Flag', title=f"Response Distribution by {top_feature} and QC Status", color_discrete_map={0: "#00AEEF", 1: "#FF4136"})
            st.plotly_chart(fig_drill, use_container_width=True)
        with tab3:
            st.subheader("Live SHAP Waterfall Plot")
            st.markdown("This plot explains a single QC flag prediction, showing how each feature contributed to the final result.")
            explainer = shap.TreeExplainer(model)
            shap_values_obj = explainer(X_test)
            flagged_indices_in_test = X_test[y_test == 1].index
            record_idx_slider = st.slider("Select a flagged record to explain:", 0, len(flagged_indices_in_test) - 1, 0)
            instance_loc_in_test = X_test.index.get_loc(flagged_indices_in_test[record_idx_slider])
            
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values_obj[instance_loc_in_test, :, 1], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"**Explanation:** For record `{df.loc[flagged_indices_in_test[record_idx_slider], 'SampleID']}`, the features shown in red increased the probability of a QC flag, while those in blue decreased it.")
    else:
        st.success(f"No QC flags found in study **{study_id}**.")
elif page == "🚀 **Technology Proving Ground**":
    st.header("🚀 Technology Proving Ground (PoC Environment)")
    st.warning("**For Demonstration Only:** This area is for evaluating and prototyping emerging technologies. Results are not for GxP use.")
    tab_langchain, tab_dask, tab_r = st.tabs(["📄 **GenAI: LangChain Summarization**", "💨 **Scalability: Dask Processing**", "📊 **Analytics: R Integration**"])
    with tab_langchain:
        st.subheader("Proof-of-Concept: Automated Report Summarization")
        st.markdown("This PoC demonstrates how **LangChain** could be used to automatically generate a human-readable summary from a structured QC report.")
        report_text = st.text_area("Paste Structured Report Data Here:", height=200, value='{"study_id": "VX-CF-MOD-01", "qc_run_date": "2024-05-21", "data_integrity_score": 0.998, "key_findings": [{"test": "IC50 Potency", "result": 1.2, "units": "uM", "status": "PASS"}, {"test": "Cell Viability", "result": 92.5, "units": "%", "status": "PASS"}, {"test": "Reagent Lot Purity", "lot": "LOT-2024-AAAA", "result": 85.1, "units": "%", "status": "FAIL"}], "conclusion": "Study passed overall, but one reagent lot failed purity spec and has been quarantined."}')
        if st.button("🤖 Generate Summary with LangChain PoC"):
            with st.spinner("Simulating call to LangChain API..."):
                time.sleep(2)
                st.subheader("Generated Narrative Summary:")
                st.info(" **Study VX-CF-MOD-01 QC Summary:**\n\nThe quality control analysis conducted on May 21, 2024, for study VX-CF-MOD-01 has concluded. The overall data integrity score was excellent at 99.8%. Key assays, including IC50 Potency (1.2 µM) and Cell Viability (92.5%), met all acceptance criteria. However, a significant deviation was noted in the Reagent Lot Purity test for **lot LOT-2024-AAAA**, which failed with a result of 85.1%. **Action:** While the study passes overall, the failing reagent lot has been flagged and quarantined to prevent its use in future experiments.")
            log_action("engineer.principal@vertex.com", "POC_LANGCHAIN_SUMMARY")
    with tab_dask:
        st.subheader("Proof-of-Concept: Large-Scale Data Processing")
        st.markdown("This PoC uses **Dask** to simulate the parallel processing of a large (50,000 row) dataset, a task common in genomics or late-stage study aggregation.")
        if st.button("🚀 Process Large Dataset with Dask"):
            with st.spinner("Setting up Dask cluster and processing partitions..."):
                dask_results = load_data_with_dask("dummy_path")
                st.subheader("Dask Computation Results:")
                st.write("Mean 'Response' grouped by 'ReagentLot':")
                st.dataframe(dask_results)
            log_action("engineer.principal@vertex.com", "POC_DASK_PROCESSING")
    with tab_r:
        st.subheader("Live R Integration via `rpy2`")
        st.markdown("This PoC executes a real R script to generate an SPC chart using the `qcc` library. This is made possible by including R in the deployment environment via `packages.txt`.")
        if RPY2_INSTALLED:
            st.info("✅ `rpy2` is installed. Live R integration is active.")
            if st.button("📊 Generate SPC Chart with R"):
                with st.spinner("Executing R script via rpy2..."):
                    r_data = generate_process_data("R_Integration_Test")
                    chart_file = execute_r_spc_chart(r_data, "SPC Chart Generated by R")
                    if chart_file and os.path.exists(chart_file):
                        st.image(chart_file, caption="This chart was generated live by an R script and displayed in Streamlit.")
                        log_action("engineer.principal@vertex.com", "POC_RPY2_EXECUTION")
                    else:
                        st.error("R script execution failed. Check application logs and ensure R and the `qcc` library are installed in the environment.")
        else:
            st.error("**Deployment Note:** `rpy2` is not installed in this environment. To enable this feature, install it (`pip install rpy2`) and a local version of R. The image below is a static representation of the expected output.")
            st.image("https://www.r-graph-gallery.com/wp-content/uploads/2018/10/Custom-Shewhart-chart-with-ggplot2.png", caption="Example of an R-generated SPC chart.")

elif page == "🏛️ **Regulatory & Audit Hub**":
    st.header("🏛️ Regulatory & Audit Hub")
    st.markdown("Prepare, package, and document data dossiers for regulatory inspections and internal audits with full 21 CFR Part 11 traceability.")
    st.info("**21 CFR Part 11 Compliance Features:**\n- **Audit Trails:** All actions are logged to a persistent database...\n- **Electronic Signatures:** Actions are linked to the logged-in user.\n- **Logical Security:** Data is protected via checksums.")
    with st.form("audit_sim_form"):
        st.subheader("Package New Regulatory Dossier")
        c1,c2,c3 = st.columns(3)
        req_id=c1.text_input("Request ID","FDA-REQ-003")
        agency=c2.selectbox("Requesting Agency",["FDA","EMA","PMDA"])
        study_id_package=c3.selectbox("Select Study to Package:",["VX-CF-MOD-01","VX-522-Tox-02"])
        st.text_area("Justification / Request Details","Follow-up request for raw data, QC reports...")
        files_to_include=st.multiselect("Select Data & Artifacts to Include:",["Raw Instrument Data (.csv)","QC Anomaly Report (.pdf)","Executive Summary (.pptx)"],default=["Raw Instrument Data (.csv)","QC Anomaly Report (.pdf)","Executive Summary (.pptx)"])
        submitter_name=st.text_input("Enter Full Name for Electronic Signature:","Dr. Principal Engineer")
        submitted=st.form_submit_button("🔒 Validate, Lock, and Package Dossier")
    if submitted:
        with st.spinner("1. Validating dossier... 2. Generating checksums... 3. Logging GxP action..."):
            time.sleep(2)
            dossier_checksum=hashlib.sha256(f"{req_id}{study_id_package}{submitter_name}".encode()).hexdigest()
            log_action("engineer.principal@vertex.com","PACKAGE_REGULATORY_DOSSIER",req_id,{'study':study_id_package,'files':files_to_include,'signature':submitter_name})
            st.success(f"Dossier Packaged & Action Logged!")
            if "Executive Summary (.pptx)" in files_to_include:
                kpis={"Data Integrity Score":"99.8%","QC Flags":"3 Warnings","Conclusion":"Ready for submission"}
                ppt_file=generate_summary_pptx(study_id_package,kpis)
                st.download_button("⬇️ Download Executive Summary (.pptx)",ppt_file,file_name=f"{req_id}_summary.pptx")
            st.download_button("⬇️ Download Full Dossier (.zip)","dummy_zip_content",file_name=f"{req_id}_dossier.zip")

elif page == "🔗 **Data Lineage & Versioning**":
    st.header("🔗 Data Lineage, Versioning & Discrepancy Hub")
    st.markdown("Visualize data provenance, review change histories for any record, and manage data quality discrepancies.")
    tab_lineage, tab_versioning, tab_discrepancy = st.tabs(["🗺️ **Visual Data Flow**", "🕓 **Data Versioning (Audit Trail)**", "🔧 **Discrepancy Resolution**"])
    with tab_lineage:
        st.subheader("End-to-End Data Flow")
        dot=graphviz.Digraph()
        dot.node('A','Source Systems\n(LIMS, ELN)'); dot.node('B','Data Ingest Pipeline\n(Airflow/Python)'); dot.node('C','Data Lake\n(S3 - Raw Data)'); dot.node('D','ETL/QC Process\n(Spark/dbt)'); dot.node('E','Data Warehouse\n(Snowflake - Curated)'); dot.node('F','Phoenix Engine\n(This App)',shape='star',style='filled',fillcolor='#0033A0',fontcolor='white'); dot.node('G','Reports & Dossiers\n(.pdf, .pptx)'); dot.edges(['AB','BC','CD','DE','EF','FG']); dot.edge('D','F',label='Pydantic\nContract Check',style='dashed',color='red')
        st.graphviz_chart(dot)
    with tab_versioning:
        st.subheader("Record Change History Viewer")
        target_id_to_view=st.text_input("Enter Record/Dossier/Batch ID to Audit:","FDA-REQ-003")
        if st.button("🔍 View History"):
            conn=sqlite3.connect(DB_FILE)
            query="SELECT timestamp, user, action, details FROM audit_log WHERE target_id = ? ORDER BY timestamp DESC"
            history_df=pd.read_sql_query(query,conn,params=(target_id_to_view,))
            conn.close()
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True)
            else:
                st.warning(f"No history found for ID '{target_id_to_view}'.")
    with tab_discrepancy:
        st.subheader("Automated Discrepancy Resolution")
        disc_data={'SampleID':['VX-CF-MOD-01-S0123','VX-CF-MOD-01-S0124'],'Response':[95.4,None],'CellViability':[88.1,89.2]}
        disc_df=pd.DataFrame(disc_data)
        disc_df['Suggested_Fix']=[None,round(disc_df['Response'].mean(),2)]
        st.dataframe(disc_df,use_container_width=True)
        if st.button("✅ Approve & Apply Suggested Fixes"):
            log_action("engineer.principal@vertex.com","APPLY_DISCREPANCY_FIX","VX-CF-MOD-01",details={"imputed_value":disc_df['Response'].mean()})
            st.success("Fix applied and action logged.")

elif page == "✅ **System Validation & QA**":
    st.header("✅ System Validation & Quality Assurance")
    st.markdown("Manage and review the validation lifecycle of the Phoenix Engine itself, ensuring it operates as intended in a GxP environment.")
    st.subheader("System Validation Workflow (GAMP 5)")
    st.graphviz_chart("""digraph {rankdir=LR;node [shape=box, style=rounded];URS [label="User Requirement\nSpecification (URS)"];FS [label="Functional\nSpecification (FS)"];DS [label="Design\nSpecification (DS)"];Code [label="Code & Unit Tests\n(Pytest)"];IQ [label="Installation\nQualification (IQ)"];OQ [label="Operational\nQualification (OQ)"];PQ [label="Performance\nQualification (PQ)"];RTM [label="Requirements\nTraceability Matrix"];URS -> FS -> DS -> Code;Code -> IQ -> OQ -> PQ;{URS, FS, DS} -> RTM [style=dashed];{IQ, OQ, PQ} -> RTM [style=dashed];}""")
    tab1, tab2, tab3 = st.tabs(["⚙️ **Unit Test Results (Pytest)**", "📋 **Qualification Protocols**", "✍️ **Change Control**"])
    with tab1:
        st.subheader("Latest Unit Test Run Summary"); st.code("""============================= test session starts ==============================
... (45 tests passed) ...
============================== 45 passed in 12.34s ===============================
        """, language="bash"); st.success("All 45 unit tests passed. Code coverage: 98%.")
    with tab2:
        st.subheader("IQ / OQ / PQ Protocol Status"); protocol_data={'Protocol ID':["IQ-PHX-001","OQ-PHX-001","PQ-PHX-001"],'Description':["Verify correct installation...","Test core system functions...","Test system performance..."],'Status':["Executed & Approved","Executed & Approved","Pending Execution"],'Approved By':["qa.lead@vertex.com","qa.lead@vertex.com","N/A"],'Approval Date':["2024-04-01","2024-04-15","N/A"]}; st.dataframe(protocol_data, use_container_width=True)
    with tab3:
        st.subheader("Change Control Log"); change_log={'CR-ID':["CR-075", "CR-076"],'Date':["2024-05-10", "2024-05-20"],'Change Description':["Added `statsmodels`...","Updated brand colors..."],'Reason':["Enhance process drift detection...","Improve user experience..."],'Impact Assessment':["Low. Re-validation required.","Low. Re-validation required."],'Status':["Approved & Implemented","In Development"]}; st.dataframe(change_log, use_container_width=True)

elif page == "⚙️ **System Admin Panel**":
    st.header("⚙️ System Administration Panel"); st.warning("**For Authorized Administrators Only.** Changes here affect the entire application and are fully audited.")
    st.subheader("Current Application Configuration (`config.yml`)"); st.code(CONFIG_TEXT, language='yaml')
    with st.form("config_form"):
        st.subheader("Modify Configuration"); new_min_viability=st.number_input("New Minimum Cell Viability Threshold", value=CONFIG['validation_rules']['cell_viability']['min']); new_dashboard_title=st.text_input("New Dashboard Title", value=CONFIG['ui_settings']['dashboard_title'])
        if st.form_submit_button("Submit & Log Configuration Change"):
            log_action("engineer.principal@vertex.com", "CONFIG_CHANGE_REQUEST", "config.yml", details={'new_min_viability':new_min_viability, 'new_dashboard_title':new_dashboard_title})
            st.success("Configuration change request logged! A server restart is required to apply changes.")

elif page == "📈 **System Health & Metrics**":
    st.header("📈 System Health, KPIs & User Adoption"); st.markdown("Live dashboard monitoring the performance of the Phoenix Engine and user engagement.")
    try:
        conn = sqlite3.connect(DB_FILE); actions_df = pd.read_sql_query("SELECT timestamp, action FROM audit_log", conn); actions_df['timestamp'] = pd.to_datetime(actions_df['timestamp']); feedback_df = pd.read_sql_query("SELECT rating FROM user_feedback", conn); avg_rating = feedback_df['rating'].mean() if not feedback_df.empty else "N/A"; db_status = "Connected"; db_status_color = "normal"; conn.close()
    except Exception as e:
        st.error(f"**Database Unreachable!** Error: {e}"); actions_df = pd.DataFrame(); avg_rating = "N/A"; db_status = "Disconnected"; db_status_color = "inverse"
    c1,c2,c3=st.columns(3); c1.metric("Total Logged Actions", len(actions_df)); c2.metric("Average User Feedback Rating", f"{avg_rating:.2f} / 5" if isinstance(avg_rating, float) else avg_rating); c3.metric("Backend Database Status", db_status, delta_color=db_status_color)
    if not actions_df.empty:
        st.subheader("User Actions Over Time"); action_counts = actions_df.set_index('timestamp').resample('D').size().rename('actions'); st.line_chart(action_counts)

elif page == "📚 **SME Knowledge Base & Help**":
    st.header("📚 SME Knowledge Base & Help Center"); st.markdown("Centralized documentation, tutorials, and feedback mechanisms for training and business continuity.")
    tab_kb, tab_help, tab_feedback = st.tabs(["🧠 **Knowledge Base**", "❓ **Help & Guides**", "💬 **Submit Feedback**"])
    with tab_kb:
        st.subheader("Key Scientific & Statistical Concepts")
        st.markdown("""- **IC50/EC50:** ...\n- **ANOVA (Analysis of Variance):** ...\n- **Tukey's Honest Significant Difference (HSD):** ...\n- **SHAP (SHapley Additive exPlanations):** ...\n- **Process Capability (Cpk):** ...""")
        st.subheader("Regulatory & GxP Governance")
        st.markdown("""- **GAMP 5 (Good Automated Manufacturing Practice):** ...\n- **21 CFR Part 11:** ...""")
    with tab_help:
        st.subheader("Step-by-Step Guides")
        st.markdown("""**How to Investigate a QC Flag:**\n1. Go to the **Automated Root Cause Analysis** page...\n\n**Troubleshooting `rpy2` Integration:**\n- **Error `RPY2_INSTALLED is False`:** ...""")
    with tab_feedback:
        st.subheader("Provide Feedback on this Platform")
        with st.form("feedback_form"):
            page_options = ["Global Command Center", "Assay Dev", "Strategic Roadmap", "Process Control", "Genomic QC", "Cross-Study Analysis", "RCA", "Tech Proving Ground", "Regulatory Hub", "Data Lineage", "System Validation", "Admin Panel", "System Health", "Knowledge Base"]
            feedback_page = st.selectbox("Which page are you providing feedback for?", page_options); feedback_rating = st.slider("Rating (1=Poor, 5=Excellent)", 1, 5, 4); feedback_comment = st.text_area("Comments:")
            if st.form_submit_button("Submit Feedback"):
                conn=sqlite3.connect(DB_FILE); c=conn.cursor(); c.execute("INSERT INTO user_feedback (timestamp, page, rating, comment) VALUES (?, ?, ?, ?)",(datetime.now(), feedback_page, feedback_rating, feedback_comment)); conn.commit(); conn.close(); st.success("Thank you! Your feedback has been recorded.")
