# F1-Chatbot--An-Implementation-of-Retrieval-Augmented-Generation-for-Knowledge-Intensive-NLP-Tasks-


## 📌 Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot fine-tuned on a **Formula 1 Q&A dataset**.  
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

