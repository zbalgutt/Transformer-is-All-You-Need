# 🤖 Tiny Transformer for Next-Token Prediction

## Overview
This project implements a **Transformer model from scratch** in PyTorch to perform next-token prediction on the Tiny Shakespeare corpus. The goal is to build an intuitive, lightweight version of the Transformer architecture while gaining hands-on understanding of self-attention, positional encoding, and sequence modeling.

The project emphasizes model interpretability, including attention visualizations and training diagnostics, rather than raw scale or performance.

---

## Dataset
- **Tiny Shakespeare Corpus**
- ~1.1M characters of Shakespearean text
- Used for next-token language modeling
- Cleaned and filtered into short sequences for efficient training

---

## Model Architecture
- Token embeddings + learnable positional encodings  
- Causal self-attention with masking  
- Feed-forward network with residual connections  
- 2 Transformer blocks  
- Hidden size (`d_model`): 128  

The model predicts the next token given a short context window.

---

## Training
- **Task:** Next-token prediction  
- **Loss:** Cross-entropy (padding ignored)  
- **Optimizer:** Adam  
- **Train / Validation split:** 80% / 20%  
- Trained for 100 epochs on GPU when available

---

## Evaluation & Analysis
- Tracked training and validation loss curves  
- Reported **perplexity (PPL)** on validation set  
- Visualized **attention heatmaps** to interpret token-to-token dependencies  
- Observed stronger, more structured attention patterns in deeper layers

---

## Key Takeaways
- Causal masking enforces autoregressive behavior  
- Attention weights concentrate on recent tokens  
- Validation loss begins increasing after ~20 epochs, indicating overfitting  
- Model size and context length significantly affect generalization

---

## Tech Stack
`Python · PyTorch · NumPy · matplotlib · tokenizers`

---


