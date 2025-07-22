import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import graphviz

# Advanced Analytics & ML
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, mannwhitneyu

# =================================================================================================
# App Configuration & Professional Styling
# =================================================================================================
st.set_page_config(
    page_title="VTX DTE-RPMS Phoenix Engine",
    page_icon="🔥", # Using a "Phoenix" icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for a sleek, enterprise-grade UX/DX
st.markdown("""
<style>
    /* Main app styling */
    .reportview-container {
        background-color: #F0F2F6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-image: linear-gradient(#FFFFFF, #E0E6F1);
    }
    .sidebar .sidebar-content .stRadio > label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    /* KPI Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #D1D1D1;
        border-left: 5px solid #0033A0; /* Vertex Blue */
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    .stMetric > label {
        font-weight: 500 !important;
        color: #555555;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        background-color: #F0F2F6;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 3px solid #00AEEF; /* Vertex Cyan */
    }
    /* Custom info boxes */
    .info-box {
        background-color: #E7F3FF;
        border-left: 6px solid #2196F3;
        padding: 15px;
        margin: 10px 0px;
        border-radius: 5px;
        font-family: sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================================================
# Mock Data Generation Engine (Enhanced for Realism and Scope)
# =================================================================================================

@st.cache_data(ttl=900)
def generate_global_kpis():
    """Simulates high-level KPIs from different global sites."""
    sites = ["Boston, USA", "San Diego, USA", "Oxford, UK"]
    data = []
    for site in sites:
        data.append({
            'Site': site,
            'lon': [-71.0589, -117.1611, -1.2577][sites.index(site)],
            'lat': [42.3601, 32.7157, 51.7520][sites.index(site)],
            'Studies_Active': np.random.randint(5, 15),
            'Data_Integrity': f"{np.random.uniform(99.5, 99.9):.2f}%",
            'Automation_Coverage': np.random.randint(85, 98),
            'Critical_Flags': np.random.randint(0, 5),
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=900)
def generate_preclinical_data(study_id, n_samples=1000):
    """Generates more complex mock data, including potential root causes for errors."""
    np.random.seed(hash(study_id) % (2**32 - 1))
    operators = ['J.Doe', 'S.Chen', 'M.Gupta', 'R.Valdez']
    instruments = {'PK': ['Agilent-6470', 'Sciex-7500'], 'Tox': ['Tecan-Spark', 'BMG-Pherastar'], 'Eff': ['Zeiss-Axio', 'Leica-THUNDER']}
    assay_type = study_id.split('-')[1]
    data = {
        'SampleID': [f"{study_id}-S{i:04d}" for i in range(n_samples)],
        'Timestamp': [datetime.now() - timedelta(days=np.random.uniform(1, 90), hours=h) for h in range(n_samples)],
        'OperatorID': np.random.choice(operators, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'InstrumentID': np.random.choice(instruments.get(assay_type, ['Generic-Inst-01']), n_samples),
        'ReagentLot': np.random.choice([f"LOT-2024-{'A'*(i+1)}" for i in range(4)], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'Value': np.random.lognormal(mean=3, sigma=0.8, size=n_samples),
        'CellViability': np.random.normal(95, 4, n_samples).clip(70, 100),
    }
    df = pd.DataFrame(data)

    # Inject sophisticated anomalies for Root Cause Analysis
    # 1. Operator-specific bias
    df.loc[df['OperatorID'] == 'R.Valdez', 'Value'] *= 1.3
    # 2. Reagent lot issue
    df.loc[df['ReagentLot'] == 'LOT-2024-AAAA', 'Value'] *= 0.7
    df.loc[df['ReagentLot'] == 'LOT-2024-AAAA', 'CellViability'] -= 10
    # 3. Instrument drift over time (late samples)
    late_samples = df.sort_values('Timestamp').tail(50).index
    df.loc[late_samples, 'Value'] *= np.linspace(1, 1.5, 50)
    # 4. Create a target 'QC_Flag' for ML
    df['QC_Flag'] = 0
    df.loc[df[df['OperatorID'] == 'R.Valdez'].sample(frac=0.8).index, 'QC_Flag'] = 1
    df.loc[df[df['ReagentLot'] == 'LOT-2024-AAAA'].sample(frac=0.8).index, 'QC_Flag'] = 1
    df.loc[late_samples.sample(frac=0.8), 'QC_Flag'] = 1

    return df.sort_values('Timestamp').reset_index(drop=True)

@st.cache_data(ttl=900)
def generate_cnv_data(sample_id):
    """Generates mock Copy Number Variation data."""
    np.random.seed(hash(sample_id) % (2**32 - 1))
    chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    data = []
    pos = 0
    for chrom in chromosomes:
        n_probes = np.random.randint(500, 2000)
        positions = pos + np.cumsum(np.random.randint(10000, 50000, n_probes))
        log2_ratios = np.random.normal(0, 0.15, n_probes)
        df_chrom = pd.DataFrame({'Chromosome': chrom, 'Position': positions, 'Log2_Ratio': log2_ratios})
        data.append(df_chrom)
        pos = positions[-1]

    df = pd.concat(data).reset_index(drop=True)
    # Inject a known amplification and deletion
    df.loc[(df['Chromosome'] == 'chr8') & (df['Position'] > 127_000_000) & (df['Position'] < 129_000_000), 'Log2_Ratio'] += 0.8 # MYC amp
    df.loc[(df['Chromosome'] == 'chr9') & (df['Position'] > 21_000_000) & (df['Position'] < 23_000_000), 'Log2_Ratio'] -= 0.7 # CDKN2A del
    return df

@st.cache_data(ttl=900)
def generate_process_data(process_name="API_Purity"):
    """Generates mock manufacturing process data for SPC."""
    np.random.seed(hash(process_name) % (2**32 - 1))
    data = {
        'BatchID': [f'MFG-24-{i:03d}' for i in range(1, 101)],
        'Timestamp': pd.to_datetime(pd.date_range(end=datetime.now(), periods=100)),
        'Value': np.random.normal(99.5, 0.2, 100)
    }
    df = pd.DataFrame(data)
    # Introduce a process shift
    df.loc[75:, 'Value'] += 0.35
    return df

# =================================================================================================
# Sidebar Navigation & User Info
# =================================================================================================
with st.sidebar:
    st.image("https://d1io3yog0oux5.cloudfront.net/_3f03b2222d6fdd47976375a7337f7a69/vertexpharmaceuticals/db/387/2237/logo.png", width=220)
    st.title("Phoenix Engine")
    st.markdown("##### DTE-RPMS Command Center")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🌎 **Global Command Center**",
            "🧬 **Genomic Data QC (CNV)**",
            "📊 **Cross-Study & Batch Analysis**",
            "💡 **Automated Root Cause Analysis (RCA)**",
            "📈 **Process Control & Tech Transfer**",
            "🏛️ **Regulatory & Audit Simulation**",
            "🔗 **Data Lineage & SQL Hub**",
            "📚 **SME Knowledge Base**"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info(
        """
        **Role:** Principal Engineer, DTE-RPMS\n
        **User:** engineer.principal@vertex.com\n
        **Clearance Level:** Global Admin\n
        **Session Start:** {}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    st.markdown("---")
    st.link_button("Go to Vertex DTE Portal", "https://www.vrtx.com/our-science/data-technology-and-engineering/")


# =================================================================================================
# Page Implementations
# =================================================================================================

if page == "🌎 **Global Command Center**":
    st.header("🌎 Global RPMS Operations Command Center")
    st.markdown("Real-time, holistic view of data operations, integrity, and automation initiatives across all major research and manufacturing sites.")

    global_kpis = generate_global_kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Data Integrity", "99.78%", "+0.12%", help="Aggregated data quality score across all validated pipelines.")
    c2.metric("Automation Index", "94%", "Target: 95%", help="Weighted percentage of processes automated, from data ingest to report generation.")
    c3.metric("Active Critical Flags", "3", "-2 vs yesterday", delta_color="inverse", help="High-severity data quality or system alerts requiring immediate attention.")
    c4.metric("Regulatory Packages Pending", "2", "1 FDA, 1 EMA", help="Number of in-flight data dossiers for regulatory submission.")

    st.markdown("---")

    map_col, alerts_col = st.columns([2, 1])
    with map_col:
        st.subheader("Global Site Status")
        fig = go.Figure(data=go.Scattergeo(
            lon=global_kpis['lon'],
            lat=global_kpis['lat'],
            text=global_kpis.apply(lambda row: f"<b>{row['Site']}</b><br>Integrity: {row['Data_Integrity']}<br>Automation: {row['Automation_Coverage']}%<br>Flags: {row['Critical_Flags']}", axis=1),
            mode='markers',
            marker=dict(
                color=global_kpis['Critical_Flags'],
                colorscale='Bluered',
                reversescale=True,
                cmin=0,
                cmax=5,
                size=global_kpis['Automation_Coverage'] / 5,
                colorbar_title='Critical Flags'
            )
        ))
        fig.update_layout(
            geo=dict(scope='world', projection_type='natural earth', showland=True, landcolor='rgb(217, 217, 217)'),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with alerts_col:
        st.subheader("Priority Action Items")
        st.error("🔴 **CRITICAL:** [VTX-661-Tox-03] - Data drift detected in Cell Viability assay. Potential reagent lot issue. (P-value: 0.031)")
        st.warning("🟠 **WARNING:** [MFG-24-081] - Process capability (Cpk) for API Purity dropped to 1.35, approaching lower limit of 1.33.")
        st.info("🔵 **INFO:** [FDA-REQ-001] - Data packaging 85% complete. Due in 13 days.")

elif page == "🧬 **Genomic Data QC (CNV)**":
    st.header("🧬 Genomic Data QC Engine: Copy Number Variation (CNV)")
    st.markdown("Specialized module for quality control and analysis of high-throughput genomics data, focusing on CNV detection.")

    sample_id = st.text_input("Enter Sample ID for CNV Analysis:", "CGP-TUMOR-SAMPLE-08A")
    if sample_id:
        cnv_data = generate_cnv_data(sample_id)
        st.info(f"Loaded {len(cnv_data)} genomic probes for sample **{sample_id}**.")

        # Manhattan-style plot for whole genome view
        st.subheader("Whole-Genome Log2 Ratio Plot")
        fig_genome = px.scatter(cnv_data, x=cnv_data.index, y='Log2_Ratio', color='Chromosome',
                                title=f"Genome-wide CNV Profile for {sample_id}",
                                labels={'x': 'Genomic Position (indexed)', 'y': 'Log2 Ratio'},
                                hover_data=['Chromosome', 'Position'])
        fig_genome.update_layout(showlegend=False)
        st.plotly_chart(fig_genome, use_container_width=True)

        # Chromosome-specific drill-down
        st.subheader("Chromosome-Specific Drill-Down")
        chrom_list = cnv_data['Chromosome'].unique()
        selected_chrom = st.selectbox("Select Chromosome to inspect:", chrom_list, index=7) # Default to chr8
        
        chrom_data = cnv_data[cnv_data['Chromosome'] == selected_chrom]
        fig_chrom = px.scatter(chrom_data, x='Position', y='Log2_Ratio',
                               title=f"CNV Detail for {selected_chrom}",
                               labels={'Position': f'Position on {selected_chrom} (bp)', 'y': 'Log2 Ratio'})
        fig_chrom.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Gain Threshold")
        fig_chrom.add_hline(y=-0.3, line_dash="dash", line_color="red", annotation_text="Loss Threshold")
        st.plotly_chart(fig_chrom, use_container_width=True)
        st.markdown(f"**Interpretation:** The plot for **{selected_chrom}** shows the smoothed Log2 ratio of copy numbers. Points significantly above the gain threshold (e.g., > 0.3) suggest gene amplification, while points below the loss threshold suggest deletion. For `chr8`, a clear amplification is visible around position 128Mb, corresponding to the *MYC* oncogene locus.")

elif page == "📊 **Cross-Study & Batch Analysis**":
    st.header("📊 Cross-Study & Batch-to-Batch Analysis")
    st.markdown("Perform comparative statistical analyses across different studies, instruments, or reagent lots to identify systemic variations.")

    study_list = ["VTX-809-PK-01", "VTX-661-Tox-03", "VTX-445-Eff-05"]
    selected_studies = st.multiselect("Select studies to compare:", study_list, default=study_list)
    
    if selected_studies:
        all_data = pd.concat([generate_preclinical_data(s) for s in selected_studies], keys=selected_studies, names=['StudyID', 'Orig_Index']).reset_index()

        tab1, tab2, tab3 = st.tabs(["Box Plot Comparison (ANOVA)", "Reagent Lot Heatmap", "Principal Component Analysis (PCA)"])

        with tab1:
            st.subheader("Distribution Comparison by Study")
            fig_box = px.box(all_data, x='StudyID', y='Value', color='StudyID', title="Assay Value Distribution Across Studies")
            st.plotly_chart(fig_box, use_container_width=True)

            # ANOVA Test
            groups = [group['Value'].dropna() for name, group in all_data.groupby('StudyID')]
            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                st.markdown(f"**One-way ANOVA Result:** F-statistic = {f_stat:.2f}, **p-value = {p_value:.4f}**")
                if p_value < 0.05:
                    st.error("Statistically significant difference detected among study means (p < 0.05). Further investigation is warranted.")
                else:
                    st.success("No statistically significant difference detected among study means (p >= 0.05).")

        with tab2:
            st.subheader("Reagent Lot Performance Heatmap")
            pivot = all_data.pivot_table(index='ReagentLot', columns='OperatorID', values='Value', aggfunc='mean')
            fig_heat = px.imshow(pivot, text_auto=".2f", aspect="auto",
                                 title="Mean Assay Value by Reagent Lot and Operator",
                                 color_continuous_scale='RdYlBu_r')
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("**Insight:** The heatmap reveals potential interactions. For example, a particularly low (blue) or high (red) value for a specific Operator/Reagent combination could indicate a training issue or a lot-specific handling problem.")

        with tab3:
            st.subheader("PCA for Outlier/Cluster Detection")
            df_pca = all_data[['Value', 'CellViability']].dropna()
            pca = PCA(n_components=2)
            components = pca.fit_transform(df_pca)
            
            pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            pca_df['StudyID'] = all_data.loc[df_pca.index, 'StudyID']

            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='StudyID', title="PCA of Preclinical Data")
            st.plotly_chart(fig_pca, use_container_width=True)
            st.markdown(f"**Explained Variance:** PC1 explains **{pca.explained_variance_ratio_[0]:.1%}** and PC2 explains **{pca.explained_variance_ratio_[1]:.1%}** of the variance. Clusters or outliers in this plot can reveal systemic differences between studies not obvious from single-variable plots.")

elif page == "💡 **Automated Root Cause Analysis (RCA)**":
    st.header("💡 Automated Root Cause Analysis (RCA) Engine")
    st.markdown("Leverages machine learning to predict the likely cause of QC flags, accelerating investigation and resolution.")

    study_id = st.selectbox("Select Study with QC Flags:", ["VTX-661-Tox-03", "VTX-809-PK-01"])
    df = generate_preclinical_data(study_id)
    df_flagged = df[df['QC_Flag'] == 1]

    if not df_flagged.empty:
        st.warning(f"Found **{len(df_flagged)}** QC-flagged records in **{study_id}**. Initiating RCA...")
        
        # Prepare data for ML model
        features = ['OperatorID', 'InstrumentID', 'ReagentLot']
        target = 'QC_Flag'
        df_ml = df.copy()
        for col in features:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col])
        
        X = df_ml[features]
        y = df_ml[target]
        
        # Train a Decision Tree for interpretability
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        
        st.subheader("Inferred Decision Rules for QC Flags")
        st.markdown("**Explanation:** A Decision Tree model was trained on the historical data to find the most predictive factors for QC flags. The rules below represent the most likely root causes.")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_tree(clf, feature_names=features, class_names=['No Flag', 'Flagged'], filled=True, rounded=True, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Probable Root Cause Summary")
        importances = pd.DataFrame({'feature': features, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
        st.dataframe(importances)
        
        top_cause = importances.iloc[0]
        st.error(f"**Top Predicted Contributor:** **{top_cause['feature']}** appears to be the most significant factor driving QC flags, with an importance score of {top_cause['importance']:.2f}. Recommend immediate investigation into this area (e.g., operator retraining, reagent lot quarantine).")

    else:
        st.success(f"No QC flags found in study **{study_id}**. Data integrity appears high.")


elif page == "📈 **Process Control & Tech Transfer**":
    st.header("📈 Manufacturing Process Control & Tech Transfer Dashboard")
    st.markdown("Monitors critical quality attributes (CQAs) of manufacturing processes using Statistical Process Control (SPC) to ensure stability and readiness for tech transfer.")

    process_name = st.selectbox("Select Critical Quality Attribute (CQA) to Monitor:", ["API_Purity", "Dissolution_Rate", "Impurity_Profile"])
    df = generate_process_data(process_name)

    # Define Process Specs
    spec_col1, spec_col2, spec_col3 = st.columns(3)
    USL = spec_col1.number_input("Upper Specification Limit (USL)", value=100.0)
    TARGET = spec_col2.number_input("Target", value=99.5)
    LSL = spec_col3.number_input("Lower Specification Limit (LSL)", value=99.0)

    # SPC Calculations
    mean = df['Value'].mean()
    std_dev = df['Value'].std()
    ucl = mean + 3 * std_dev  # Upper Control Limit
    lcl = mean - 3 * std_dev  # Lower Control Limit

    # Process Capability (Cpk)
    cpk = min((USL - mean) / (3 * std_dev), (mean - LSL) / (3 * std_dev))

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Process Mean", f"{mean:.3f}")
    kpi_col2.metric("Process Std Dev", f"{std_dev:.3f}")
    kpi_col3.metric("Process Capability (Cpk)", f"{cpk:.2f}", "Target: > 1.33", delta_color="off" if cpk >= 1.33 else "inverse")

    if cpk < 1.33:
        st.warning(f"**Process Capability Alert:** Cpk of {cpk:.2f} is below the standard target of 1.33. The process is not considered highly capable and may produce out-of-spec material.")

    # I-MR Chart (Individuals and Moving Range)
    st.subheader("I-Chart (Individuals Chart)")
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(x=df['BatchID'], y=df['Value'], mode='lines+markers', name='CQA Value'))
    fig_i.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
    fig_i.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL")
    fig_i.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL")
    fig_i.add_hline(y=USL, line_dash="dot", line_color="orange", annotation_text="USL")
    fig_i.add_hline(y=LSL, line_dash="dot", line_color="orange", annotation_text="LSL")
    fig_i.update_layout(title=f"I-Chart for {process_name}", yaxis_title="Value")
    st.plotly_chart(fig_i, use_container_width=True)
    st.markdown("**Interpretation:** The I-Chart plots individual batch values against statistical control limits (UCL/LCL). Points outside these limits or non-random patterns (like the upward shift at the end) indicate that the process is out of statistical control.")

elif page == "🏛️ **Regulatory & Audit Simulation**":
    st.header("🏛️ Regulatory & Audit Simulation Environment")
    st.markdown("Prepare for regulatory inspections by running a simulated audit. This tool helps identify, package, and document data requests in a compliant manner.")

    st.warning("**SIMULATION MODE ACTIVATED**")
    with st.form("audit_sim_form"):
        st.subheader("Simulated FDA Request for Information")
        st.markdown(
            """
            **Request ID:** FDA-AUDIT-SIM-001\n
            **Subject:** Request for raw data, QC reports, and statistical analysis for study **VTX-809-PK-01**.\n
            **Justification:** Follow-up to recent submission. Please provide all instrument raw files, QC deviation reports, and the full data lineage from data capture to final analysis.
            """
        )
        study_to_package = st.selectbox("Select Study to Package:", ["VTX-809-PK-01"], disabled=True)
        files_to_include = st.multiselect(
            "Select Data & Artifacts to Include in Dossier:",
            ["Raw Instrument Data (.csv)", "QC Anomaly Report (.pdf)", "Data Lineage Graph (.svg)", "Audit Trail Log (.json)", "Statistical Analysis Script (R/Python)"],
            default=["Raw Instrument Data (.csv)", "QC Anomaly Report (.pdf)", "Audit Trail Log (.json)"]
        )
        submitter_name = st.text_input("Enter Your Name for Electronic Signature:", "Principal Engineer")
        submitted = st.form_submit_button("🔒 Lock and Package Dossier")

    if submitted:
        with st.spinner("Packaging dossier, generating checksums, and logging audit trail..."):
            import time
            time.sleep(3)
        st.success(f"**Dossier Packaged Successfully!**")
        st.code(
            f"""
            --- AUDIT LOG ---
            Timestamp: {datetime.now().isoformat()}
            User: engineer.principal@vertex.com
            Action: GENERATE_REGULATORY_PACKAGE
            Request_ID: FDA-AUDIT-SIM-001
            Study_ID: {study_to_package}
            Artifacts: {files_to_include}
            e-Signature: {submitter_name}
            SHA256_Checksum: {np.random.bytes(32).hex()}
            Status: SUCCESS
            Compliance: 21 CFR Part 11 Compliant Action
            """, language='json'
        )
        st.download_button("Download Packaged Dossier (.zip)", data="dummy_zip_file_content", file_name="FDA-AUDIT-SIM-001_Package.zip")

elif page == "🔗 **Data Lineage & SQL Hub**":
    st.header("🔗 Data Lineage & SQL Query Hub")
    st.markdown("Visualize data provenance and directly query the underlying data warehouse for ad-hoc analysis and verification.")

    st.subheader("Visual Data Lineage")
    st.markdown("This graph shows the flow of data from raw source to final report, ensuring full traceability.")
    
    # Create a Graphviz graph
    dot = graphviz.Digraph(comment='Data Lineage', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent'})
    dot.node('A', 'Raw Instrument Files\n(.wiff, .d)', shape='folder', style='filled', fillcolor='#FFC107')
    dot.node('B', 'Data Ingest & Parsing\n(Python Script)', shape='box', style='filled', fillcolor='#8BC34A')
    dot.node('C', 'Staging Database\n(PostgreSQL)', shape='cylinder', style='filled', fillcolor='#03A9F4')
    dot.node('D', 'Automated QC Engine\n(This App)', shape='box', style='filled', fillcolor='#8BC3A9')
    dot.node('E', 'Data Warehouse\n(Snowflake)', shape='cylinder', style='filled', fillcolor='#03A9F4')
    dot.node('F', 'Analysis Dataset', shape='box', style='filled', fillcolor='#CDDC39')
    dot.node('G', 'Study Report\n(.pdf)', shape='note', style='filled', fillcolor='#9E9E9E')

    dot.edges(['AB', 'BC', 'CE', 'DF', 'EF', 'FG'])
    dot.edge('C', 'D', label='QC Check')
    
    st.graphviz_chart(dot)

    st.subheader("Data Warehouse SQL Query Hub")
    st.markdown("Execute direct SQL queries against a read-only replica of the RPMS Data Warehouse.")
    query = st.text_area("SQL Query", 
    """
    -- Example: Find average value by instrument for a specific study
    SELECT
        "InstrumentID",
        AVG("Value") AS "AverageValue",
        COUNT(*) AS "NumSamples"
    FROM
        "PRECLINICAL_DATA"
    WHERE
        "StudyID" = 'VTX-809-PK-01'
    GROUP BY
        "InstrumentID"
    ORDER BY
        "AverageValue" DESC;
    """, height=200)

    if st.button("Execute Query"):
        with st.spinner("Querying data warehouse..."):
            import time
            time.sleep(1)
            # Mock query result
            if "VTX-809-PK-01" in query and "AVG" in query:
                mock_result = pd.DataFrame({
                    'InstrumentID': ['Agilent-6470', 'Sciex-7500'],
                    'AverageValue': [22.54, 21.89],
                    'NumSamples': [498, 502]
                })
                st.dataframe(mock_result, use_container_width=True)
            else:
                st.info("Query executed. (This is a mock result).")

elif page == "📚 **SME Knowledge Base**":
    st.header("📚 SME Knowledge Base & Governance Center")
    st.markdown("Centralized definitions, methodologies, and standards governing the DTE-RPMS data ecosystem.")

    with st.expander("**Glossary of Key Terms & Acronyms**", expanded=True):
        st.markdown("""
        - **CFR (Code of Federal Regulations):** Rules published by the executive departments of the U.S. Federal Government. **21 CFR Part 11** is critical for electronic records.
        - **CQA (Critical Quality Attribute):** A physical, chemical, biological, or microbiological attribute that must be within an appropriate limit to ensure desired product quality.
        - **CNV (Copy Number Variation):** A phenomenon in which sections of the genome are repeated and the number of repeats in the genome varies between individuals.
        - **DTE (Data, Technology, and Engineering):** The central technology and data organization at Vertex.
        - **GxP (Good 'x' Practice):** A general term for quality guidelines and regulations (GLP, GMP, GCP).
        - **ICH (International Council for Harmonisation):** Brings together regulatory authorities and pharmaceutical industry to discuss scientific and technical aspects of drug registration.
        - **RPMS (Research, Pre-Clinical, Manufacturing & Supply):** The business domain this engine serves, covering the drug lifecycle from early research to supply chain.
        - **SPC (Statistical Process Control):** A method of quality control which employs statistical methods to monitor and control a process.
        - **Cpk (Process Capability Index):** A statistical measure of a process's ability to produce output within specification limits. A value > 1.33 is typically desired.
        """)

    with st.expander("**Core Methodologies**"):
        st.markdown("""
        - **Data Integrity (ALCOA+):** Our systems are designed to ensure data is Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, and Available.
        - **Root Cause Analysis (Decision Tree):** We use interpretable ML models like decision trees to trace QC flags back to their most likely source variables (e.g., operator, instrument, reagent). This model is chosen for its transparency, which is crucial in a regulated environment.
        - **Tech Transfer Data Package:** A standardized set of documents and data proving process stability and robustness, including SPC charts, validation reports, and CQA analysis. The **Process Control** module is designed to generate these artifacts.
        """)
        
    with st.expander("**Data & Technology Standards**"):
        st.markdown("""
        - **Data Storage:** Raw data is stored in its original format in an immutable S3 bucket. Processed data is structured and stored in a Snowflake data warehouse.
        - **Querying:** Access to the warehouse is granted via read-only replicas to analytical tools and hubs like this one, ensuring production databases are not impacted.
        - **Software Development:** All code (including this application) is version-controlled in Git, subject to peer review, and deployed via a validated CI/CD pipeline.
        - **Validation:** System components undergo rigorous Installation Qualification (IQ), Operational Qualification (OQ), and Performance Qualification (PQ) before being put into GxP use.
        """)
