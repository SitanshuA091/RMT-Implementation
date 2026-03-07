### **Recurrent Memory Transformer**  
**For Long-Context Text Generation**  
Will Reproduce and experiment with the Recurrent Memory Transformer (RMT), a 2023 model by
Yandex Research. It extends Transformer architectures with segment-level recurrence and external
memory, enabling longer context handling.
Core Idea: Implement or adapt a simplified version using PyTorch and test it on long-form
generation or QA datasets.

- Evaluate coherence and dependency retention via BLEU/ROUGE
- Compare against GPT-2 small baseline 

**Datasets:**
1. NarrativeQA (Hugging Face: narrativeqa) /
2. WikiText-103

**Implementation Plan:**
1. Build baseline Transformer LM or fine-tune GPT-2 small.
2. Add memory tokens passed between sequence segments.
3. Train on 512–1024 token chunks.
4. Evaluate with BLEU/ROUGE/perplexity.
5. Ablation: remove memory to compare metrics.
