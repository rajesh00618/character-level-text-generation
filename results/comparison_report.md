# Model Comparison Report

## Perplexity Comparison

| Model | Perplexity |
|-------|-----------|
| LSTM | 15.56 |
| Transformer | 11.74 |

Lower perplexity indicates better predictive performance. The model with lower perplexity is more confident in its predictions on the test set.

## Qualitative Analysis

### LSTM Analysis
The LSTM model learns sequential dependencies through its gating mechanism, allowing it to capture local patterns and short-to-medium range dependencies. LSTM-generated text tends to have:
- Strong local coherence (nearby characters and words fit together well)
- Reasonable word-level accuracy
- Difficulty maintaining long-range thematic consistency
- More repetitive patterns at lower temperatures

### Transformer Analysis
The Transformer model uses self-attention to directly model relationships between all positions in the sequence. This allows it to:
- Capture longer-range dependencies more effectively than the LSTM
- Learn more complex structural patterns
- Potentially suffer from less repetition due to broader context awareness
- Show different error patterns due to the lack of inherent sequential bias

### Temperature Effect Analysis

**Temperature = 0.5 (Low):**
Both models produce more conservative, repetitive text. The output tends to reuse common phrases and patterns from the training data. The LSTM often gets stuck in loops of common n-grams, while the Transformer may exhibit more varied repetition patterns.

**Temperature = 1.0 (Standard):**
At this setting, both models generate text that best reflects the learned probability distribution. The output is most natural and balanced between coherence and creativity.

**Temperature = 1.5 (High):**
Higher temperatures increase randomness, leading to more creative but less coherent output. Both models may produce grammatically incorrect or nonsensical phrases. The Transformer, with its global attention, may maintain some structural consistency longer than the LSTM as temperature increases.

### Key Differences Observed
1. The Transformer typically achieves lower perplexity due to its ability to model long-range dependencies.
2. The LSTM is more computationally efficient for shorter sequences during inference.
3. The Transformer's attention mechanism provides more interpretable insights into which characters influence predictions.
4. Training dynamics differ significantly between the two architectures.
