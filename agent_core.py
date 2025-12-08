import os
import pandas as pd
import numpy as np
import json
import joblib
import nbformat as nbf
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.preprocessing import StandardScaler, LabelEncoder # Ensure sklearn is available

class DataEngineeringAgent:
    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        """
        Initialize the Data Engineering Agent.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required. Set GOOGLE_API_KEY env var or pass it.")
        
        # Configure LLM
        os.environ["GOOGLE_API_KEY"] = self.api_key
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)

    def scan_dataset(self, df, chunk_size=300, max_chunks=None, randomize=False, progress_callback=None):
        """
        Scans the dataset in chunks.
        max_chunks: Limit number of chunks to scan.
        randomize: If True, randomly sample `max_chunks` from the total.
        """
        import random
        all_issues = []
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        if max_chunks and max_chunks < len(chunks):
            if randomize:
                chunks = random.sample(chunks, max_chunks)
                print(f"Randomly selected {max_chunks} chunks for analysis (User Limit)...")
            else:
                chunks = chunks[:max_chunks]
                print(f"Scanning first {max_chunks} chunks (User Limit)...")
        else:
            print(f"Starting Full-Scan on {len(df)} rows ({len(chunks)} chunks)...")
        
        prompt_template = ChatPromptTemplate.from_template("""
        Analyze this snippet of raw data rows (JSON format):
        {data_chunk}
        
        Identify SPECIFIC data quality issues. Look for:
        1. Mixed types in the same column (e.g. strings in numeric cols).
        2. Hidden delimiters (e.g. 'Delhi#2024' or 'John|Doe').
        3. Dirty formats (Date variations like '12-01-2020' vs 'Jan 12, 2020').
        4. Outliers or impossible values (Age = 150).
        
        Return ONLY a concise bulleted list of issues found. If none, say 'No major issues'.
        """)
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            # Report progress if callback exists
            if progress_callback:
                progress_callback((i + 1) / total_chunks)
                
            chunk_json = chunk.to_json(orient='records')
            print(f"Scanning Chunk {i+1}/{total_chunks}...")
            
            chain = prompt_template | self.llm
            result = chain.invoke({"data_chunk": chunk_json})
            
            if "No major issues" not in result.content:
                all_issues.append(f"Chunk {i+1} Issues:\n{result.content}")
                
        print("Scan Complete.")
        return "\n".join(all_issues)

    def generate_cleaning_code(self, df, scan_report, user_context=""):
        """
        Generates executable Python code based on the scan report and user context.
        """
        print("Architecting Data Pipeline...")
        dtypes_info = df.dtypes.to_string()
        head_info = df.head().to_string()
        
        prompt = f"""
        You are a Senior Data Engineer. 
        
        USER CONTEXT (Domain/Goal):
        "{user_context}"
        (Use this context to decide what is important. e.g. if Goal is 'Predict Price', ensure 'Price' is cleaned as Target).
        
        I have scanned the dataset and found these issues:
        {scan_report}
        
        Dataset Head:
        {head_info}
        
        Types:
        {dtypes_info}
        
        TASKS:
        1. **STEP 1: STRATEGIC ANALYSIS (Text Explanation)**
           - Analyze column names/types.
           - **IDENTIFIERS**: KEEP them (unless user says drop).
           - **NUMERIC FEATURES**: Impute & Scale.
           - **CATEGORICAL FEATURES**: Impute & Encode.
           - **USER GOAL ALIGNMENT**: Make sure the target variable (if mentioned in context) is handled correctly (e.g. do not scale Target Variable for regression if not needed, or handle it specifically).
           - **DROP ONLY IF**: Empty or Single Value.
        
        2. **STEP 2: THE CODE (Python Block)**
           - **MUST START WITH**: 
             ```python
             if 'df' not in locals() and 'df' not in globals():
                 raise ValueError("df variable not found! The code expects 'df' to be passed in globals().")
             print(f"Processing dataset with shape: {df.shape}")
             ```
           - Perform the cleaning steps on the Existing `df`.
           - **MUST END WITH**: `print(f"Final dataset shape: {df.shape}")`
           - SAVE artifacts.
        
        CRITICAL RULES:
        - **DO NOT CREATE DUMMY DATA.**
        - **USE THE EXISTING `df` VARIABLE.**
        - Return ONLY the python code block wrapped in ```python ... ```.
        """
        
        result = self.llm.invoke(prompt)
        return result.content

    def execute_pipeline(self, df, code):
        """
        Executes the cleaning code in-memory using exec().
        Returns: (success: bool, logs: str, modified_df: pd.DataFrame or None)
        """
        import re
        
        # Extract code
        code_match = re.search(r"```python(.*?)```", code, re.DOTALL)
        clean_code = code_match.group(1) if code_match else code
            
        print("--- Executing Generated Pipeline ---")
        
        # Prepare execution environment
        exec_globals = globals().copy()
        exec_globals['df'] = df.copy() # Work on a copy initially
        exec_globals['pd'] = pd
        exec_globals['np'] = np
        exec_globals['joblib'] = joblib
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        exec_globals['StandardScaler'] = StandardScaler
        exec_globals['LabelEncoder'] = LabelEncoder
        
        output_log = []
        try:
            exec(clean_code, exec_globals)
            output_log.append("Execution Success!")
            output_log.append("Artifacts (clean_data.csv, etc.) saved.")
            
            # Helper to get the modified df back
            cleaned_df = exec_globals.get('df')
            return True, "\n".join(output_log), cleaned_df
        except Exception as e:
            error_msg = f"EXECUTION FAILED: {str(e)}"
            print(error_msg)
            return False, error_msg, None

    def create_user_notebook(self, cleaning_code, filename='AI_Data_Engineered_Notebook.ipynb'):
        """
        Generates the handover notebook containing the cleaning logic.
        """
        nb = nbf.v4.new_notebook()
        
        nb.cells.append(nbf.v4.new_markdown_cell("# AI Engineered Data Notebook\nThis notebook contains the **Auto-Generated Cleaning Pipeline** and loads the processed data."))
        
        # Cell 1: The Generated Cleaning Code (for reproducibility)
        nb.cells.append(nbf.v4.new_markdown_cell("## 1. The Cleaning Pipeline\nThis is the code the Agent generated and executed."))
        nb.cells.append(nbf.v4.new_code_cell(cleaning_code))
        
        # Cell 2: Loader
        code_loader = """
import pandas as pd
import joblib

# Load the Resulting Data (Created by the pipeline above)
try:
    df = pd.read_csv('clean_data.csv')
    print('Loaded Cleaned Data:', df.shape)
    display(df.head())
except FileNotFoundError:
    print("clean_data.csv not found. Run the pipeline cell above!")
"""
        nb.cells.append(nbf.v4.new_markdown_cell("## 2. Verify Results"))
        nb.cells.append(nbf.v4.new_code_cell(code_loader))
        
        nb.cells.append(nbf.v4.new_markdown_cell("## 3. Modeling\nYou can now run your ML models below."))
        nb.cells.append(nbf.v4.new_code_cell("# Your Model Here\n# from sklearn.ensemble import RandomForestClassifier..."))
        
        with open(filename, 'w') as f:
            nbf.write(nb, f)
        return filename
