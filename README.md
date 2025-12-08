# AI Data Engineering Agent (Data Fixer Pro) 🤖

A fully autonomous **AI Data Scientist** that can scan, analyze, and clean your datasets using **Large Language Models (Google Gemini)**. It comes with a modern **Streamlit Dashboard** for business users.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini Pro](https://img.shields.io/badge/Google%20Gemini-Pro-8E75B2?style=for-the-badge&logo=google%20bard&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-000000?style=for-the-badge&logo=langchain&logoColor=white)

## ✨ Features

*   **🔍 Deep Chunk Scanning**: Reads the entire dataset (or random samples) row-by-row to detect hidden issues like merged columns (`"Delhi#2024"`) or mixed types that standard summary statistics miss.
*   **🧠 Strategic Analysis**: The Agent "thinks" before it acts. It identifies ID columns to preserve, features to scale, and targets to handle.
*   **⚙️ Autonomous Engineering**:
    *   Generates **Production-Grade Python Code** on the fly.
    *   **Executes** the code in-memory on your actual dataset.
    *   **Saves** cleaned data and preprocessor artifacts (Scalers, Encoders).
*   **📘 Auto-Notebook Generation**: Creates a ready-to-use Jupyter Notebook (`AI_Data_Engineered_Notebook.ipynb`) with the cleaning logic embedded and executed.
*   **📊 Business Dashboard**: A user-friendly Streamlit UI with time estimation, progress tracking, and easy downloads.

## 🚀 Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install langchain langchain-google-genai langchain-core pandas joblib scikit-learn nbformat streamlit
    ```

4.  **Get a Google API Key**:
    *   Visit [Google AI Studio](https://aistudio.google.com/).
    *   Create a free API Key.

## 🏃‍♂️ How to Run

1.  **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```

2.  **Using the App**:
    *   Enter your **Google API Key** in the sidebar (it will be saved locally to `.env` for convenience).
    *   **Upload** your CSV file.
    *   Use **"Step 1: Deep Scan Analysis"** for an audit.
        *   *Tip: Use "Random Sample" for large files to get a quick estimate.*
    *   Use **"Step 2: Generate & Execute"** to auto-clean the data.
    *   **Download** the results!

## 📂 Project Structure

*   `app.py`: The Frontend (Streamlit Dashboard).
*   `agent_core.py`: The Backend (LangChain Agent Logic).
*   `AI_Data_Engineered_Notebook.ipynb`: (Generated) The final user deliverable.
*   `clean_data.csv`: (Generated) The clean dataset.

## 🛡️ Privacy & Security

*   Your API Key is stored locally in a `.env` file and is never uploaded.
*   Data processing happens locally on your machine (except for the snippets sent to the LLM for analysis).

---
*Built with ❤️ using LangChain & Streamlit*
