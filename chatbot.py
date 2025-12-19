import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from rouge_score import rouge_scorer
import os
import shutil

# --- 1. CORE AI LOGIC (CACHED) ---

@st.cache_resource
def initialize_system(pdf_dir, uploaded_files=None):
    """
    Handles Data Preparation (Task 1). Supports both Directory loading 
    and direct File Uploads.
    """
    docs = []
    
    # CASE A: User uploaded files directly
    if uploaded_files:
        temp_dir = "./temp_uploads"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            
    # CASE B: User provided a directory path
    elif os.path.exists(pdf_dir):
        loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
    
    if not docs:
        return None, None, "No documents found to process."

    # Pre-processing & Embedding (Task 1 & 2)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Load LLM (Task 3)
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        config={"max_new_tokens": 512, "temperature": 0.2}
    )
    
    return vector_store, llm, "Success"

def get_summary_prompt(length):
    template = f"""You are an advanced AI research assistant. 
    Summarize the following document excerpts into a professional summary of approximately {length} words.
    
    Excerpts:
    {{context}}
    
    Summary:"""
    return PromptTemplate(template=template, input_variables=["context"])

# --- 2. USER INTERFACE (STREAMLIT) ---

def main():
    st.set_page_config(page_title="LLM Search & Summarizer", layout="wide")
    st.title("🤖 Document Search & Summarization System")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # IMPROVISATION: Added File Uploader for user convenience
        uploaded_files = st.file_uploader(
            "Upload PDF Articles", 
            type="pdf", 
            accept_multiple_files=True,
            help="Drag and drop PDFs here to analyze them immediately."
        )
        
        st.markdown("--- OR ---")
        
        data_path = st.text_input(
            "Corpus Directory Path", 
            value="./data", 
            help="Specify a local folder path if not uploading files."
        )

        st.subheader("Search & Summary Settings")
        top_k = st.slider("Top N Relevant Results", 1, 10, 3)
        summary_len = st.select_slider(
            "Target Summary Length", 
            options=[50, 150, 300, 500],
            value=150
        )
        st.caption("Powered by Mistral-7B & LangChain")

    # Initialize System (Priority given to Uploaded Files)
    if not os.path.exists("db"): os.makedirs("db")
    
    # We pass both to the initializer
    vector_store, llm, status = initialize_system(data_path, uploaded_files)

    if status != "Success":
        st.info("👋 Welcome! Please **Upload PDFs** in the sidebar or specify a **Directory Path** to begin.")
        return

    query = st.text_input("🔍 What would you like to know from these documents?", placeholder="e.g., Summarize the main risks.")

    if query:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📍 Relevant Excerpts")
            relevant_docs = vector_store.similarity_search(query, k=top_k)
            for i, doc in enumerate(relevant_docs):
                with st.expander(f"Result #{i+1} | {os.path.basename(doc.metadata.get('source', 'Upload'))}"):
                    st.write(doc.page_content)
        
        with col2:
            st.subheader("📝 AI Summary")
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            prompt = get_summary_prompt(summary_len)
            
            with st.spinner("Generating summary..."):
                final_summary = llm(prompt.format(context=context_text))
                st.success("Summary Generated!")
                st.write(final_summary)
                
        # Evaluation Section
        st.markdown("---")
        if st.button("📊 Evaluate Summary Quality"):
            reference_text = "The documents discuss the specified topic and provide relevant details."
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_text, final_summary)
            
            e1, e2, e3 = st.columns(3)
            e1.metric("ROUGE-1", f"{scores['rouge1'].fmeasure:.2f}")
            e2.metric("ROUGE-L", f"{scores['rougeL'].fmeasure:.2f}")
            e3.info("Score 1.0 = Perfect match.")

if __name__ == "__main__":
    main()