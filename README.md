# llm-tuning-with-rag
Retrieval-Augmented Generation chatbot for JIIT using fine-tuned Mistral-7B (Unsloth), LlamaIndex, and ROUGE evaluation.

# 🔍 Fine-Tuning LLMs with RAG for University Q\&A — Inspired by IEEE Research (2024)

This project demonstrates how to fine-tune a lightweight language model (TinyLlama) on custom university FAQ data using Unsloth with LoRA/QLoRA, and enhance its retrieval capabilities using RAG (Retrieval-Augmented Generation). The PDF brochure of JIIT (jiit.ac.in) was parsed and embedded using FAISS and llama-index. The system is evaluated using ROUGE metrics for answer quality, and the fine-tuned model can be exported in GGUF format for CPU or Ollama use.

> 🔬 Based on:
> **"Fine-Tuning LLM to Answer Basic Questions for Prospective New Students at Syiah Kuala University Using the RAG Method"**
> [IEEE ICIC 2024 · DOI: 10.1109/ICIC64337.2024.10956296](https://doi.org/10.1109/ICIC64337.2024.10956296)


## 🎯 Motivation

Rather than building LLMs from scratch, modern AI development focuses on **fine-tuning existing models on custom datasets** using efficient techniques like **LoRA**, **QLoRA**, and **DPO**. This project was inspired by the shift in thinking — from full-stack AI to **lean, strategic fine-tuning** — and explores:

* Fine-tuning with instruction-style data
* Retrieval-based answering using unstructured documents
* Evaluation with ROUGE

## 🧠 Project Overview

### 🔄 Workflow

```text
Instruction-Tuning Dataset (JIIT FAQs)
         │
         ▼
LoRA / QLoRA Fine-Tuning (TinyLlama)
         │
         ├── Optional: DPO with preference pairs
         │
         ▼
Document Ingestion (PDFs using llama-index)
         │
         ▼
RAG System (FAISS + Sentence Embeddings)
         │
         ▼
Evaluation (ROUGE Metrics)
  

## 📦 Tech Stack

| Category    | Tool / Library                                 |
| ----------- | ---------------------------------------------- |
| LLM         | `TinyLlama-1.1B-Chat`                          |
| Fine-tuning | `Unsloth`, `PEFT`, `Transformers`, `TRL`       |
| RAG         | `llama-index`, `FAISS`, `SentenceTransformers` |
| Evaluation  | `rouge_score`                                  |
| Deployment  | `GGUF`, `Ollama-compatible`                    |
| Platform    | Google Colab (T4 GPU)                          |

---

## 📁 Files and Structure

| File / Folder                            | Description                               |
| ---------------------------------------- | ----------------------------------------- |
| `jiit_finetune.json`                     | Instruction-tuning dataset (JIIT Q\&A)    |
| `qa_data.json`                           | Extracted question/reference answer pairs |
| `rag_data/`                              | Folder containing JIIT PDF brochure       |
| `Implementation_IEEE_research2024.ipynb` | Full end-to-end Colab notebook            |
| `tinyllama_dpo_model/`                   | Fine-tuned model directory (optional)     |
| `README.md`                              | Project overview and documentation        |


## 📊 Evaluation

The model is evaluated using **ROUGE-1, ROUGE-2, and ROUGE-L** metrics, comparing its generated answers with ground truth responses from the dataset.


## 🧪 Example Query

```python
Q: What is the hostel fee per semester?
A: Hostel + mess charges are around ₹1 lakh per semester for Indian students.
```

## 🧠 Key Learnings

* **Instruction-tuning** with well-structured prompts leads to high factual alignment.
* **Unsloth** allows 4-bit quantized fine-tuning on free GPUs like T4 with very low VRAM.
* **RAG** makes models dynamically answer queries beyond their training scope.
* **GGUF** format allows inference on CPUs with tools like `llama.cpp` and `Ollama`.


## 📚 References

* IEEE Research: H. Rachmat et al., *“Fine-Tuning LLM using RAG”*, ICIC 2024
* [LLM Engineer’s Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072)
* [Unsloth GitHub](https://github.com/unslothai/unsloth)

## 🤝 Let's Connect

Are you working on:

* LLM fine-tuning
* Retrieval-Augmented Generation (RAG)
* EdTech AI assistants
* Efficient LLM deployment (GGUF, Ollama)

Feel free to open issues, share feedback, or connect on [LinkedIn](https://www.linkedin.com/in/).

