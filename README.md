# F1-Chatbot--An-Implementation-of-Retrieval-Augmented-Generation-for-Knowledge-Intensive-NLP-Tasks-


## 📌 Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot fine-tuned on a **Formula 1 Q&A dataset**. This paper is an exemplar implementation of the seminal paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" which can be found here:
https://arxiv.org/abs/2005.11401

**Note**: This is not an EXACT replica of the research paper implementation:
⚡ Compute limitations – Training was conducted on a smaller GPU setup, limiting model size, retrieval depth, and training epochs.
🧩 Simplified retriever – A basic FAISS retriever was implemented, rather than a fully optimized dense retriever with large-scale pretrain

By combining **retrieval-based search** (via FAISS index of Formula 1 Wikipedia passages) with **generative modeling** (RAG-Sequence + BART), the chatbot provides accurate and context-rich answers to Formula 1-related questions.  

---

## ✨ Features  
- 🔍 **FAISS Retriever** – Retrieves top-k relevant Formula 1 passages from Wikipedia.  
- 🤖 **RAG-Sequence Model** – Fine-tuned with Formula 1 Q&A pairs for domain-specific accuracy.  
- 💬 **Interactive Chatbot** – Answer Formula 1 questions via command line interface.  
- 📊 **Evaluation Metrics** – Measures performance using EM, F1, and ROUGE.  

---

## ⚙️ Tech Stack  
- **Model:** `facebook/rag-sequence-nq` (fine-tuned)  
- **Tokenizer:** `RagTokenizer`, `BartTokenizer`  
- **Retriever:** FAISS-based document index  
- **Training Framework:** Hugging Face `Seq2SeqTrainer`  
- **Language:** Python  

---

## Evaluation Results 

| Metric               | Score |
| -------------------- | ----- |
| **Exact Match (EM)** | 73.0% |
| **F1 Score**         | 82.0% |

