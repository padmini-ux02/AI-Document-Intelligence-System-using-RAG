# Nexus AI – AI Document Intelligence System

Nexus AI is a premium, high-performance Document Intelligence System built using **Retrieval-Augmented Generation (RAG)**. It transforms static documents (PDF, DOCX, TXT) into a dynamic, interactive knowledge base.

## 🚀 Key Features

*   **Hybrid RAG Engine**: 
    *   **Local Mode**: Uses Scikit-Learn (TF-IDF) for lightning-fast, privacy-focused local retrieval.
    *   **Neural Mode**: Leverages OpenAI (GPT-3.5/GPT-4) and FAISS for state-of-the-art accuracy.
*   **Intuitive UI**: A sleek, dark-themed dashboard built with Streamlit, featuring real-time insights and chat history.
*   **Deep Insights**: Automatically extracts document summaries, key signals, and named entities.
*   **Source Transparency**: Every response includes pinpoint references to the source passages with relevance scoring.

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Orchestration**: [LangChain](https://www.langchain.com/)
*   **Vector Engine**: [FAISS](https://github.com/facebookresearch/faiss) / TF-IDF
*   **Natural Language**: [OpenAI GPT](https://openai.com/), [SpaCy](https://spacy.io/)
*   **Document Parsing**: PyPDF, docx2txt

## 📦 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/AI-Document-Intelligence-System.git
    cd AI-Document-Intelligence-System
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration (Optional)**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=sk-your-key-here
    ```

## 🚦 Usage

Launch the application using Streamlit:

```bash
streamlit run app.py
```

1.  **Engine Selection**: Input your OpenAI API key in the sidebar for Neural mode, or leave it blank for Local mode.
2.  **Upload**: Drag and drop your PDF, DOCX, or TXT files.
3.  **Build**: Click "BUILD KNOWLEDGE BASE" to index your documents.
4.  **Chat**: Start asking questions about your data!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
