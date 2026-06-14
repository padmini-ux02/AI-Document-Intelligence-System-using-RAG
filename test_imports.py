try:
    import streamlit as st
    import langchain
    import langchain_community
    import langchain_text_splitters
    import langchain_openai
    import faiss
    import pypdf
    import docx2txt
    print("Core RAG imports successful!")
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Other Error: {e}")
