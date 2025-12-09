import streamlit as st
import pandas as pd
import os
import joblib
from agent_core import DataEngineeringAgent

# --- Page Config for Modern Business Look ---
st.set_page_config(
    page_title="Data Fixer Pro | Enterprise AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #1E88E5; 
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem; 
        color: #666; 
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5; 
        color: white; 
        border-radius: 5px; 
        height: 50px; 
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50) # Placeholder Icon
    st.title("Data Fixer Pro")
    st.markdown("---")
    
    # --- Persistent API Key Logic ---
    KEY_FILE = ".env"
    
    def load_key():
        """Load API Key from local file or environment."""
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, "r") as f:
                for line in f:
                    if line.strip().startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip()
        return os.environ.get("GOOGLE_API_KEY", "")

    def save_key(key):
        """Save API Key to local file."""
        with open(KEY_FILE, "w") as f:
            f.write(f"GOOGLE_API_KEY={key}")
            
    # Load existing key
    saved_key = load_key()
    
    api_key_input = st.text_input("Google API Key", value=saved_key, type="password", help="Get from aistudio.google.com")
    
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        # Save if it's a new key
        if api_key_input != saved_key:
            save_key(api_key_input)
            st.sidebar.success("API Key saved permanently! 💾")
            
    st.markdown("### Capabilities")
    st.info("""
    - Full-Scan (No Sampling)
    - Deep Defect Detection
    - Auto-Code Generation
    - In-Memory Cleaning
    """)

# --- Main Content ---
st.markdown('<p class="main-header">Enterprise Data Engineering Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous data cleaning, fixing, and scaling powered by AI.</p>', unsafe_allow_html=True)

# 1. File Upload
uploaded_file = st.file_uploader("Upload your RAW Dataset (CSV)", type=["csv"])

if uploaded_file and os.environ.get("GOOGLE_API_KEY"):
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df_raw'] = df
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            missing = df.isna().sum().sum()
            st.metric("Missing Values", missing)
            
        with st.expander("👀 View Raw Data Preview", expanded=True):
            st.dataframe(df.head())

        # Initialize Agent
        agent = DataEngineeringAgent(api_key=os.environ["GOOGLE_API_KEY"])

        if 'df_raw' in st.session_state:
             # --- User Context & Configuration ---
            col_conf1, col_conf2 = st.columns(2)
            
            with col_conf1:
                st.markdown("### ⚙️ Scan Settings")
                
                chunk_size = 300 
                total_rows = len(df)
                total_possible_chunks = (total_rows // chunk_size) + 1
                
                st.write(f"Total Data Chunks: **{total_possible_chunks}**")
                
                scan_mode = st.radio("Scan Depth", ["Full Scan (Recommended)", "Random Sample"])
                
                if scan_mode == "Random Sample":
                    max_chunks = st.number_input(
                        "Number of Chunks to Analyze", 
                        min_value=1, 
                        max_value=total_possible_chunks, 
                        value=min(5, total_possible_chunks),
                        help="Randomly selects this many chunks from your dataset to estimate quality."
                    )
                    randomize = True
                    est_chunks = max_chunks
                else:
                    max_chunks = None
                    randomize = False
                    est_chunks = total_possible_chunks
                    
                # Time Estimation (Approx 3 sec per chunk)
                est_time = est_chunks * 3
                st.info(f"⏱️ Estimated Time: ~{est_time} seconds ({est_chunks} chunks)")

            with col_conf2:
                st.markdown("### 🎯 Dataset Context")
                user_context = st.text_area(
                    "Describe your data & goal (Optional)", 
                    placeholder="e.g. 'This is student exam data. I want to predict the CGPA. 'Roll No' is just an ID.'",
                    help="This helps the AI decide which columns are targets vs features."
                )

            # 2. Scanning Phase
            st.markdown("### 🔍 Step 1: Deep Scan Analysis")
            
            if st.button("Start Scan Analysis"):
                with st.spinner(f"Scanning {est_chunks} chunks..."):
                    progress_bar = st.progress(0)
                    
                    # Run Scan with User Limits
                    scan_report = agent.scan_dataset(
                        df, 
                        chunk_size=chunk_size, 
                        max_chunks=max_chunks, 
                        randomize=randomize,
                        progress_callback=lambda x: progress_bar.progress(x)
                    )
                    st.session_state['scan_report'] = scan_report
                    st.success("Scan Complete!")
            
            if 'scan_report' in st.session_state:
                st.text_area("Audit Report (Issues Detected)", st.session_state['scan_report'], height=200)

                st.markdown("### 🧠 Step 2: Strategic Engineering")
                if st.button("Generate & Execute Cleaning Pipeline"):
                    with st.spinner("Analyzing semantics & Architecting pipeline..."):
                        # Generate Code + Reasoning WITH USER CONTEXT
                        full_response = agent.generate_cleaning_code(
                            df, 
                            st.session_state['scan_report'],
                            user_context=user_context
                        )
                        
                        # Store for UI
                        st.session_state['generated_code'] = full_response
                        
                        # Display the reasoning
                        # --- IMPROVED SPLIT LOGIC ---
                        import re
                        
                        # 1. Extract the code block first
                        code_match = re.search(r"```python(.*?)```", full_response, re.DOTALL)
                        
                        if code_match:
                            clean_code = code_match.group(1)
                            # 2. To get the strategy, remove the code block from the full response
                            strategy_text = full_response.replace(code_match.group(0), "").strip()
                        else:
                            clean_code = full_response
                            strategy_text = "The agent provided code directly without a text summary."

                        # Display the reasoning
                        st.markdown("#### 🤖 Agent's Strategy")
                        if strategy_text:
                            st.info(strategy_text)
                        else:
                            st.warning("No strategy text was returned by the AI, but code was generated.")
                        
                        with st.expander("View Generated Python Code"):
                            st.code(clean_code, language='python')
                        
                    with st.spinner("⚡ Executing Pipeline on REAL Dataset (In-Memory)..."):
                        # Execute
                        success, logs, df_clean = agent.execute_pipeline(df, clean_code)
                        
                        if success:
                            st.session_state['df_clean'] = df_clean
                            st.success("✅ Execution Successful! The dataframe has been transformed.")
                            
                            with st.expander("View Execution Logs"):
                                st.text(logs)
                            
                            # Generate Handover Notebook
                            agent.create_user_notebook(clean_code)
                        else:
                            st.error("Execution Failed.")
                            st.error(logs)

        # 4. Results & Download
        if 'df_clean' in st.session_state:
            st.markdown("### ✨ Results: Final Dataset")
            st.dataframe(st.session_state['df_clean'].head())
            
            # Side-by-Side Comparison
            col_d1, col_d2 = st.columns(2)
            
            # Download CSV
            csv = st.session_state['df_clean'].to_csv(index=False).encode('utf-8')
            with col_d1:
                st.download_button(
                    "📥 Download Cleaned CSV",
                    csv,
                    "clean_data.csv",
                    "text/csv",
                    key='download-csv'
                )

            # Download Notebook
            with col_d2:
                with open("AI_Data_Engineered_Notebook.ipynb", "rb") as f:
                    st.download_button(
                        "📘 Download Reusable Notebook",
                        f,
                        "AI_Data_Engineered_Notebook.ipynb",
                        mime="application/x-ipynb+json"
                    )

    except Exception as e:
        st.error(f"Error processing file. Please ensure it is a valid CSV. Details: {e}")

elif not os.environ.get("GOOGLE_API_KEY"):
    st.warning("👈 Please enter your Google API Key in the sidebar to proceed.")

else:
    st.info("👆 Upload a CSV file to begin.")
