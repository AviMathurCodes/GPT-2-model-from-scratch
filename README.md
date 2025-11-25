# GPT-2-model-from-scratch
Building a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3.

This project implements a compact GPT-2–style Transformer language model trained
on the *Tiny Shakespeare* dataset (≈1M character-level tokens). The model
performs autoregressive next-character prediction using multi-head causal
self-attention, learned positional embeddings, and a GPT-style decoder-only
architecture.

---

## 1. Dataset and Tokenization

**Tiny Shakespeare dataset:**  
A 1MB plain-text corpus containing Shakespeare dialogues, ideal for character-level
language modeling experiments.

**Character-level vocabulary:**  
We extract all unique characters and map them using:
- `stoi` (“string-to-index”): maps each character to an integer ID  
- `itos` (“index-to-string”): inverse mapping  

**Reasons for character-level modeling:**  
- No tokenizer or BPE model needed  
- Vocabulary is extremely small  
- Requires longer contexts for good coherence

**Autoregressive objective:**  
The model learns p(x_t | x_1, ..., x_{t-1}) by minimizing cross-entropy loss.

**Perplexity:**  
PPL = exp(loss). Lower PPL = better predictions.

---

## 2. Model Architecture (GPT-2 Style)

The model consists of:

- 2 Transformer blocks  
- 8 self-attention heads  
- d_model = 256  
- Feed-forward size d_ff = 1024  
- Maximum context length (block size): 128  
- Dropout = 0.1  
- Tied token-embedding and output projection weights  

**Transformer definition:**  
A deep neural network using attention instead of recurrence to model dependencies
between tokens.

**Decoder-only Transformer:**  
Only uses masked self-attention (no encoder). Used by GPT models.

---

## 3. Embedding Layers

**Token Embeddings:**  
Maps each character ID to a dense vector in R^{d_model}.

**Positional Embeddings:**  
Learned vectors that represent positions 0…block_size-1 to encode sequence order.

**Input representation:**  
final_input = token_embedding + positional_embedding.

**Why embeddings?**  
They convert discrete symbols into continuous vectors suitable for neural
computation.

---

## 4. Multi-Head Causal Self-Attention

**Self-attention:**  
Mechanism that computes interactions between all pairs of tokens in the context.

**Queries/Keys/Values (Q,K,V):**  
Linear projections of the input states:
- Q: “What I am searching for”
- K: “What I contain”
- V: “What information I carry”

**Scaled dot-product attention:**  
attn = softmax((Q·K^T) / sqrt(d_head))

**Causal mask:**  
A lower-triangular mask preventing attention to future positions (enforces
autoregressive generation).

**Multi-Head Attention:**  
Splits d_model into multiple parallel "heads", each learning independent
attention patterns.

---

## 5. Transformer Block (Pre-Norm)

Each block contains:

1. LayerNorm  
2. Multi-Head Causal Self-Attention  
3. Residual connection  
4. LayerNorm  
5. Feed-Forward network (Linear → GELU → Linear)  
6. Residual connection  

**LayerNorm:** Normalizes per-token activations to stabilize training.  
**Residual connections:** Enable deeper stable networks.  
**GELU:** Smooth, non-linear activation used in GPT-2.

---

## 6. Language Modeling Head

The final hidden states are projected through a linear layer of shape:
[vocab_size × d_model]

This shares weights with the token embedding matrix (weight tying), reducing
parameters and improving stability.

---

## 7. Training Setup

**Optimizer: AdamW**  
Adam with decoupled weight decay. Standard for Transformers.

**Learning rate schedule:**  
- Linear warmup for first N steps  
- Cosine decay afterwards  
Used to stabilize early training and prevent divergence.

**Gradient clipping:**  
Clips the norm of gradients to avoid exploding updates.

**Mixed-precision training (fp16):**  
Uses `torch.cuda.amp.autocast` and `GradScaler` for faster training and reduced
memory usage on GPUs.

**Gradient accumulation:**  
Simulates larger batch sizes by accumulating gradients across multiple steps.

**Checkpointing:**  
Saves:
- best model (lowest validation loss)  
- periodic snapshots every N iterations  

---

## 8. Evaluation and Perplexity

The notebook periodically computes validation loss using full forward passes (no
gradient computation).  
Lower validation loss → better generalization.

---

## 9. Sampling and Text Generation

Text is generated autoregressively:

1. Feed the current context into the model.  
2. Sample the next token from the distribution:
   - with temperature scaling  
   - optional top-k filtering  
   - optional top-p (nucleus) filtering  
3. Append token, repeat.

**Temperature:** Controls randomness.  
**Top-k:** Only keep K highest-probability tokens.  
**Top-p:** Keep minimum set whose cumulative probability ≤ p.

---

## 10. Summary

- **Autoregressive model:** Predicts token t using only tokens < t.  
- **Cross-entropy loss:** Standard loss for classification; measures
  log-likelihood error.  
- **Perplexity:** Exponential of loss; interpretable measure of uncertainty.  
- **Transformer:** Neural architecture relying on attention, not recurrence.  
- **Self-attention:** Computes weighted interactions between tokens.  
- **Causal mask:** Prevents “peeking” into the future during training.  
- **Multi-head attention:** Parallel self-attention subspaces.  
- **Embedding:** Continuous vector representation of tokens.  
- **Feed-forward network:** Two linear layers with a non-linearity.  
- **LayerNorm:** Normalizes activations per token.  
- **Residual connection:** Adds input back to output for stability.  
- **AdamW:** Adam variant with proper weight decay.  
- **Warmup:** Increases LR gradually to avoid early instability.  
- **Cosine decay:** Smooth LR reduction schedule.  
- **Gradient clipping:** Limits gradient norm to prevent explosions.  
- **Mixed precision:** Uses fp16 for speed and memory efficiency.  
- **Gradient accumulation:** Simulates big batch sizes.  
- **Checkpoint:** Saved model state for resuming or inference.  
- **Temperature:** Controls sampling randomness.  
- **Top-k / Top-p:** Probabilistic filtering methods for generation.

