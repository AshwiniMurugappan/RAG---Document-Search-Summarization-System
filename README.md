# 🤖 AI-Powered Document Search & Summarization System

A high-performance **Retrieval-Augmented Generation (RAG)** application designed to navigate vast textual datasets. This system allows users to search through a corpus of PDF documents and generate grounded, AI-powered summaries using **Mistral-7B** and **LangChain**.

---

## 🎯 Project Objective
The primary goal of this system is to bridge the gap between traditional keyword search and generative AI. By combining **FAISS** for efficient information retrieval and **LLMs** for synthesis, the application provides context-aware answers and adjustable summaries based on specific user queries, significantly reducing the time required to analyze large document sets.

---

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/) (Web Interface)
* **LLM:** Mistral-7B-Instruct-v0.1 (Quantized GGUF for CPU efficiency)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Evaluation:** ROUGE Scores (Recall-Oriented Understudy for Gisting Evaluation)

---

## 🔬 Methodology

The system is built on a modular **RAG pipeline**, ensuring the LLM only answers based on the provided data, which prevents "hallucinations."



### 1. Data Ingestion & Pre-processing
* **Loading:** PDFs are parsed and converted into raw text using `PyPDFLoader`.
* **Chunking:** To fit within the LLM's context window, documents are split into chunks of **700 characters** with a **100-character overlap** using `RecursiveCharacterTextSplitter`. This overlap ensures semantic continuity at the split points.



### 2. Vector Embedding & Storage
* **Embedding Model:** Text chunks are transformed into 384-dimensional dense vectors using the `all-MiniLM-L6-v2` model.
* **Vector Store:** **FAISS** index is created to store these embeddings, enabling high-speed similarity searches to find the most relevant document sections for any user query.

### 3. LLM Inference (The "Brain")
* **Model:** **Mistral-7B-Instruct-v0.1** (Quantized to 4-bit). This allows the model to run on a local CPU while maintaining superior performance in reasoning compared to larger models like Llama-2.
* **Summarization Strategy:** The system retrieves the top $N$ chunks, combines them into a context block, and uses a custom prompt to generate a coherent summary of the user's requested length.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
* Python 3.9+
* Anaconda (Recommended)

### 2. Clone the Repository
```bash
cd "C:\Users\osrnm\Documents\VScode\RAG chatbot"
```

### 3. Install Dependencies
```bash
pip install streamlit langchain langchain-community sentence-transformers faiss-cpu ctransformers rouge-score pypdf
```

### 4. Download the LLM Model
1. Download `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from **TheBloke** on Hugging Face.
2. Place the `.gguf` file in the root project directory (the same folder as `chatbot.py`).

---

## 🚀 How to Run
1. Open your terminal or **Anaconda Prompt**.
2. Navigate to your project folder:
   ```bash
   cd "C:\Users\osrnm\Documents\VScode\RAG chatbot"
   ```

3. Launch the app:
   ```bash
   streamlit run chatbot.py
   ```

### Using the App:
* **Option A:** Use the sidebar to upload PDF files directly from your computer.
* **Option B:** Provide an absolute local folder path containing your PDFs.
* **Search:** Enter your query in the search bar to see source **excerpts** and the **AI summary** generated side-by-side.

---

## 📊 Evaluation (Task 4)
The system includes a built-in evaluation module using **ROUGE metrics**. After generating a summary, click the **"Evaluate Summary Quality"** button to view:

* **ROUGE-1:** Measures unigram (individual word) overlap.
* **ROUGE-L:** Measures the longest common subsequence to assess structural similarity.

---

## 📝 Deliverables Note
* **Data Pre-processing:** Implemented via `RecursiveCharacterTextSplitter` with chunk overlap to maintain context.
* **Search Methodology:** Hybrid-ready vector search using **FAISS** for low-latency retrieval.
* **Summarization:** Dynamic prompting allows for user-defined summary lengths (**50-500 words**).

---

## ⚠️ Challenges Faced and Their Solutions

Building a local RAG (Retrieval-Augmented Generation) system presented several technical hurdles. Below are the primary challenges encountered and the strategies implemented to resolve them.

---

### 1. Resource Constraints & Model Latency
**Challenge:** Large Language Models (LLMs) like GPT-4 are resource-intensive and often require expensive API calls or high-end GPUs. Running a model locally on a standard CPU led to slow response times (latency) and high RAM usage.

**Solution:** I implemented **Quantization**. By using the `CTransformers` library and the **Mistral-7B-Instruct-v0.1-GGUF** model in `Q4_K_M` format, I reduced the model's weight from ~30GB to ~4GB. This allowed the system to perform complex summarization on a consumer-grade CPU with reasonable inference speeds.

---

### 2. Loss of Context During Text Chunking
**Challenge:** Large PDFs cannot be fed into an LLM all at once due to "Context Window" limits. Initial attempts at splitting text by fixed character counts often cut sentences in half, leading to incoherent search results.

**Solution:** I utilized the `RecursiveCharacterTextSplitter`. I configured it with a **chunk size of 700 characters** and a **100-character overlap**. This overlap acts as a "buffer," ensuring that the end of one chunk and the start of the next share enough context to preserve the semantic meaning of the document.

---

### 3. Environment Portability for End-Users
**Challenge:** During the initial phase, the system relied on absolute local file paths (e.g., `C:\Users\Documents\...`). This meant the code would crash when shared with an interviewer who has a different folder structure.

**Solution:** I implemented a dual-input system using **Streamlit’s File Uploader**. This allowed the system to process files in a temporary buffer, making it completely independent of local directory structures and significantly improving the user experience for external testers.

---

### 4. Hallucination and Accuracy Control
**Challenge:** LLMs sometimes generate "hallucinations"—plausible-sounding but factually incorrect information—when they cannot find the answer in the provided documents.

**Solution:** I refined the **System Prompt**. By adding strict instructions: *"If the context does not contain the answer, truthfully say 'I don't know'. Do not try to make up an answer,"* I grounded the model's output strictly to the retrieved search results, ensuring high factual accuracy.

---

### 5. Dependency Conflicts in Anaconda
**Challenge:** Integrating multiple libraries like `FAISS`, `CTransformers`, and `rouge-score` led to "ModuleNotFound" errors and version conflicts within the base Python environment.

**Solution:** I standardized the environment using specific `pip` installations and verified the pathing within VS Code. I documented a `requirements.txt` file to ensure any user can replicate the working environment with a single command.
