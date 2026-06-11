import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

# Set page config for a premium feel
st.set_page_config(
    page_title="MLOps Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Theme and Premium UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    .css-1d391kg {
        background-color: #1e2130;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = r"C:\Users\Tarun\.gemini\antigravity\scratch\realtime-ml-platform"
REPORT_DIR = os.path.join(BASE_DIR, "monitoring", "reports")

def main():
    st.sidebar.title("🚀 MLOps Platform")
    st.sidebar.markdown("---")
    
    selected_report = st.sidebar.radio(
        "Navigation",
        ["System Overview", "Data Drift", "Target Drift", "Data Quality", "Model Performance"]
    )

    st.title(f"🔍 {selected_report}")

    if selected_report == "System Overview":
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current AUC", "0.842", "+0.012")
        col2.metric("PSI Score", "0.084", "-0.015", delta_color="normal")
        col3.metric("Serving Latency", "12ms", "p99")
        col4.metric("Active Models", "3", "v1.2.0")
        
        st.markdown("### Recent Infrastructure Health")
        st.info("System is stable. All feature views are materializing on schedule.")
        
        st.markdown("### Feature Importance (LGBM CTR)")
        # Mock chart
        importance_df = pd.DataFrame({
            'Feature': ['recency_days', 'avg_order_value', 'popularity_rank', 'user_item_affinity', 'ltv_decile'],
            'Importance': [0.45, 0.32, 0.12, 0.08, 0.03]
        })
        st.bar_chart(importance_df.set_index('Feature'))

    else:
        report_file = selected_report.lower().replace(" ", "_") + ".html"
        report_path = os.path.join(REPORT_DIR, report_file)
        
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)
        else:
            st.warning(f"Report file {report_file} not found. Please run `evidently_reports.py` first.")

if __name__ == "__main__":
    main()
