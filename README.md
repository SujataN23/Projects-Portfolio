
# 👾 Machine Learning & AI Projects Portfolio 👾  

> This repository showcases major **Machine Learning, Deep Learning, NLP, and Generative AI projects** I have built and deployed.  
> It includes  AI systems, enterprise RAG architecture, and large-scale model training pipelines.

---

# ❄️ Project Overview ❄️  

| # | Project | Description | Key Methods / Tech | Outcome |
|---|---------|-------------|--------------------|----------|
| 1 | **Enterprise RAG AI Assistant (Azure)** | Secure Retrieval-Augmented Generation system for IT knowledge retrieval using Azure infrastructure. | Azure OpenAI, Azure AI Search, Vector Embeddings, FastAPI, Docker, Managed Identity, Private Endpoint | enterprise AI assistant with secure architecture and semantic search. |
| 2 | **Large-Scale Sentiment Analysis (Production Pipeline)** | Fine-tuned transformer model on large streaming datasets for sentiment classification. | HuggingFace Transformers, PyTorch, S3 Streaming, GPU Training, Checkpointing | Built scalable training pipeline with S3 streaming |
| 3 | **Laptop Shop Assistant (Flask App)** | Chatbot that recommends laptops based on user preferences. | NLP, OpenAI API, Flask | Interactive shopping assistant deployed using Flask. |
| 4 | **Capstone: Eye for Blind** | Generates image captions and converts them to speech for visually impaired users. | CNN + LSTM with Attention, TTS | Multi-modal AI system making images accessible via audio. |
| 5 | **Healthcare NLP – Disease-Treatment Extraction** | Extract disease–treatment pairs from clinical text. | CRFs, POS Tagging | Structured medical knowledge extraction. |
| 6 | **Automatic Ticket Classification** | Classifies financial complaints into predefined categories. | NMF, TF-IDF, Logistic Regression | Achieved **95% accuracy**, automating complaint triaging. |

---

# 🚀 Featured Projects

## 🔐 Enterprise RAG AI Assistant (Azure)

A Retrieval-Augmented Generation (RAG) system built using Microsoft Azure cloud services.

### Architecture Highlights
- Azure OpenAI (Chat + Embeddings)
- Azure AI Search
- Azure Container Apps
- Azure Container Registry
- Managed Identity Authentication
- Private Endpoint Security

### System Flow
1. User submits query via API
2. Query embedding generated
3. Vector similarity search retrieves relevant documents
4. Context injected into structured prompt
5. Azure OpenAI generates grounded response
6. Enterprise-formatted output returned

---

## 📊 Large-Scale Sentiment Analysis – Production Pipeline

### Production Features
- Streaming dataset directly from AWS S3
- On-the-fly tokenization
- Step-based training
- GPU-optimized DataLoader
- Periodic checkpoint saving
- Final model upload back to S3

### Technologies
- HuggingFace Transformers
- PyTorch
- AWS S3
- Distributed-ready training structure

---

# 💻 Technologies Used

### Languages
- Python (3.11+)
- SQL

### Machine Learning
- Scikit-learn
- XGBoost
- CatBoost
- Random Forest
- Logistic Regression

### Deep Learning
- PyTorch
- TensorFlow
- CNN, RNN, LSTM
- Attention Mechanisms
- Vision Transformers (ViT)

### Generative AI & LLMs
- Azure OpenAI
- LangChain
- Prompt Engineering
- RAG Architecture
- Embeddings & Semantic Search

### Cloud & Infrastructure
- Microsoft Azure
- AWS S3
- Docker
- Managed Identity
- Vector Indexing

---

# 🎯 Key Highlights

- Designed and deployed an enterprise-grade Azure RAG system
- Built scalable large-scale sentiment analysis training pipelines
- Implemented secure cloud-native AI architectures
- Developed end-to-end ML systems from ingestion to deployment
- Combined classical ML, deep learning, and modern GenAI techniques
