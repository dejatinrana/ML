# Generative AI — Exam Revision Notes
*(No teacher assets were provided for this section — all notes below are built from general knowledge to cover your full Generative AI syllabus.)*

---

## 1. Introduction to Transformers

- The **Transformer** is a deep learning architecture introduced in the 2017 paper *"Attention Is All You Need"* (Vaswani et al., Google Brain).
- It replaced RNNs/LSTMs for sequence tasks by relying **entirely on the Attention Mechanism**, with no recurrence at all.
- Key advantage: processes **all tokens in a sequence in parallel** (unlike RNNs, which process one token at a time), enabling much faster training and better capture of long-range dependencies.
- Forms the foundation of nearly all modern Large Language Models (GPT, BERT, T5, etc.).

---

## 2. Attention Mechanism

- **Attention** allows a model to focus on the most relevant parts of the input when producing each part of the output, rather than treating all input equally.
- **Self-Attention** — every token in a sequence attends to every other token (including itself) to build a context-aware representation of each word.
- Conceptual roles in attention (Query, Key, Value):
  - **Query (Q)** — what the current token is "looking for."
  - **Key (K)** — what each token "offers" for matching against a query.
  - **Value (V)** — the actual content retrieved if a match is strong.
- The attention score is computed from the similarity between Query and Key, and used to weight the Values — tokens that are more relevant get a higher weight.
- **Multi-Head Attention** — runs several attention operations ("heads") in parallel, each learning to focus on different relationships/aspects of the sequence, then combines the results.
- Solves the long-range dependency problem RNNs struggled with, since every token can directly attend to every other token, regardless of distance in the sequence.

---

## 3. Encoder-Decoder Architecture (Transformer-Based)

- *(Note: this is distinct from the RNN/LSTM-based Encoder-Decoder covered in the Deep Learning notes — here, both Encoder and Decoder are built from stacked Transformer blocks using self-attention, not recurrence.)*
- **Encoder** — a stack of layers, each containing Self-Attention followed by a Feed-Forward Network; converts the input sequence into rich, contextual representations.
- **Decoder** — a similar stack, but with an added **Encoder-Decoder (Cross) Attention** layer, letting the decoder attend to the encoder's output; also uses **Masked Self-Attention** so it can only attend to previous tokens when generating — preserving the autoregressive (left-to-right generation) property.
- **Positional Encoding** — since Transformers process tokens in parallel with no recurrence, positional information about word order must be explicitly added to the input embeddings.
- Originally designed for tasks needing an explicit input → output sequence transformation (e.g., machine translation in the original Transformer paper).

---

## 4. Applications of Transformers

- Machine Translation, Text Summarization, Question Answering, Text Generation.
- Sentiment Analysis, Speech Recognition, Chatbots/Conversational AI.
- Vision Transformers (ViT) — applying transformer architecture to image processing.
- Code generation/completion.
- Scientific applications (e.g., protein structure prediction, using attention-based architectures).

---

## 5. Overview of BERT

- **BERT** = **Bidirectional Encoder Representations from Transformers** (Google, 2018).
- Uses **only the Encoder** part of the Transformer architecture (no decoder).
- **Bidirectional** — unlike earlier models reading text in only one direction, BERT considers context from **both directions simultaneously** for every word.
- **Pre-training tasks**:
  - **Masked Language Modeling (MLM)** — randomly masks some input tokens and trains the model to predict them using context from both sides.
  - **Next Sentence Prediction (NSP)** — trains the model to predict whether one sentence logically follows another.
- After large-scale pre-training, BERT is **fine-tuned** on specific downstream tasks (classification, Q&A, NER) using much smaller task-specific datasets — this transfer-learning approach made BERT highly influential.
- Best suited for **understanding tasks** (classification, extraction) rather than open-ended text generation, since it is encoder-only (not autoregressive).

---

## 6. Introduction to Large Language Models (LLMs)

- An **LLM** is a very large neural network (typically Transformer-based, often **decoder-only**) trained on massive amounts of text to understand and generate human language.
- Characterized by huge parameter counts (billions+), which enables **emergent capabilities** (reasoning, few-shot learning) not present in smaller models.
- Typical training stages:
  1. **Pre-training** — self-supervised learning on huge, diverse text corpora, typically via autoregressive next-token prediction.
  2. **Fine-tuning** — further training on smaller, task-specific or instruction-following datasets.
  3. **Alignment** (e.g., RLHF) — tuning outputs to match human preferences/values (see next section).
- Examples: GPT family, LLaMA, PaLM, Claude, Gemini.
- **SLM (Small Language Model)** — a smaller-parameter, often distilled/optimized model, designed for efficiency, faster inference, and on-device use; lower raw capability than a full-scale LLM but much cheaper/faster to run.

---

## 7. Reward Models and Alignment Strategies

- **Alignment** = ensuring an LLM's outputs match human intentions, values, and safety requirements — not just producing fluent text.
- **Reward Model (RM)** — a separate model trained to score/rank the quality of LLM outputs, based on human preference data (humans rank multiple candidate outputs for the same prompt; the RM learns to predict these preference scores).
- **RLHF (Reinforcement Learning from Human Feedback)** — the standard alignment pipeline:
  1. Collect human preference data by ranking multiple model outputs.
  2. Train a Reward Model to predict those human preference scores.
  3. Fine-tune the LLM using reinforcement learning (commonly **PPO** — Proximal Policy Optimization), using the Reward Model's score as the reward signal.
- Goal: models that are more **Helpful, Honest, and Harmless** (often referred to as the "HHH" objective).
- Related/alternative approaches:
  - **Constitutional AI** — the model critiques and revises its own outputs against a set of written principles.
  - **DPO (Direct Preference Optimization)** — simplifies RLHF by optimizing directly on human preference data, without needing a separate reward model or RL loop.

---

## 8. Practical Case Studies Using SLMs and LLMs

- **Customer support chatbots** — LLMs handling complex, open-ended user queries.
- **Code generation/completion assistants**.
- **Document summarization and drafting** for business/legal/research use.
- **On-device virtual assistants** — SLMs preferred here due to their smaller footprint fitting mobile/edge hardware.
- **Healthcare** — clinical note summarization, medical Q&A support.
- **Education** — personalized tutoring, automated feedback/grading.
- **Choosing SLM vs LLM**: SLMs are preferred when latency, cost, privacy (on-device processing), or limited compute are priorities; LLMs are preferred when task complexity/reasoning needs outweigh the added cost.

---

## 9. Deployment of LLMs

- Key considerations when deploying an LLM into production:
  - **Inference cost & latency** — large models are expensive and slow to run; techniques like **quantization** (reducing the numerical precision of model weights) and **model distillation** (training a smaller model to mimic a larger one) help reduce cost.
  - **Hosting options** — cloud-based API access (calling a hosted model), self-hosting on private infrastructure, or on-device/edge deployment (typically for SLMs).
  - **Scalability** — reliably handling many concurrent user requests.
  - **Monitoring** — tracking output quality, latency, and model drift over time (ties back to the MLOps concepts already covered in the Deep Learning notes).
  - **Safety guardrails** — content filtering, defenses against prompt injection, rate limiting.

---

## 10. RAG (Retrieval-Augmented Generation)

- **RAG** combines an LLM with an **external knowledge retrieval system**, grounding the model's responses in information it wasn't necessarily trained on (e.g., up-to-date or private/domain-specific data).
- **How it works (conceptual pipeline)**:
  1. A user's query is used to **retrieve** relevant documents/passages from an external knowledge base — typically via vector similarity search over text embeddings.
  2. The retrieved content is inserted into the LLM's prompt as additional context.
  3. The LLM generates its final response using both the original query and the retrieved context.
- **Benefits**: reduces hallucination (the model making up facts), allows using current/private data without retraining the model, and is far more cost-effective than fine-tuning just to update knowledge.
- **Key components**: an **Embedding Model** (converts text into vector representations), a **Vector Database** (stores and efficiently searches embeddings for the closest matches), and the **LLM** itself (handles final generation).

---

## 11. Agentic AI

- **Agentic AI** refers to AI systems (typically LLM-based) that can **autonomously plan, make decisions, and take actions** — often using external tools — to accomplish a goal, rather than just responding to a single prompt.
- **Core components of an AI agent**:
  - **Planning** — breaking a complex goal down into smaller sub-steps.
  - **Tool Use** — calling external tools/APIs (search, calculators, code execution, databases) to gather information or perform actions.
  - **Memory** — retaining context/information across multiple steps or sessions.
  - **Reasoning Loop** — iteratively observing results, adjusting the plan, and continuing until the goal is achieved (this pattern is often called **"ReAct"**: Reason + Act).
- **Distinction from a simple chatbot**: an agent can take multi-step, autonomous action toward a goal, rather than just generating a single response to a single prompt.
- **Applications**: autonomous coding assistants, research assistants, workflow automation, and multi-agent systems (multiple specialized agents collaborating on a task).

---

## Architectures/Models Studied So Far
*(This table will be updated every time a new architecture/model topic is covered — lists the model and its category.)*

| Architecture/Model | Category |
|---|---|
| Transformer | Generative AI — Attention-based Sequence Modeling |
| BERT | Generative AI/NLP — Encoder-only Pretrained Language Model |
| LLM (GPT family, LLaMA, etc.) | Generative AI — Decoder-only Autoregressive Language Model |
| RAG | Generative AI — Retrieval-Augmented Generation Pattern |
| Agentic AI Systems | Generative AI — Autonomous Agent Framework |

---

## Library & Import Cheat Sheet
*(to be filled in as frameworks/functions are covered)*

| Component/Technique | Library | Import Statement |
|---|---|---|
| Pretrained Transformers (BERT, GPT, etc.) | Hugging Face `transformers` | `from transformers import AutoModel, AutoTokenizer` |
| Sentence/Text Embeddings | `sentence-transformers` | `from sentence_transformers import SentenceTransformer` |
| Vector Database (similarity search) | `faiss` | `import faiss` |
| Vector Database (alternative) | `chromadb` | `import chromadb` |
| LLM Application/Agent Framework | `langchain` | `from langchain... import ...` |
| LLM API Access (example) | `openai` | `import openai` |
