# Gemma 3 1B — Task-Specific Fine-Tuning for Multi-script Emotion Classification

Fine-tuning Google's **Gemma 3 1B Instruct** model for multilingual emotion classification using three progressively efficient strategies: classification head fine-tuning, LoRA, and QLoRA.

---

## Overview

This project demonstrates three approaches to fine-tuning the `google/gemma-3-1b-it` model for a **6-class emotion classification** task on multilingual text. The notebook is structured as a progressive exploration — from the simplest (frozen backbone + trainable head) to the most memory-efficient (4-bit quantization + LoRA).

| Approach | Trainable Params | Memory Usage |
|---|---|---|
| Classification Head Only | ~6K (score layer) | Full BF16 (~2–3 GB) |
| LoRA | Low-rank adapters on attention | BF16 + LoRA overhead |
| QLoRA | Low-rank adapters on attention | 4-bit base + BF16 LoRA |

---

## Task Description

**Multilingual Emotion Classification** — Given a sentence (which may be written in a low-resource or non-Latin script language), predict one of 6 emotion categories.

**Emotion Labels:**
- Happy
- Disgust
- Fear
- *(and 3 additional classes inferred from the dataset)*

The dataset includes sentences from linguistically diverse and low-resource languages such as:
- **Santali** (Ol Chiki script)
- **Kashmiri** (Arabic script)
- **Manipuri / Meitei** (Meitei Mayek script)

---

## Dataset

The project uses three CSV files:

| File | Description |
|---|---|
| `train.csv` | Training samples with `Sentence` and `emotion` columns |
| `val.csv` | Validation samples with `Sentence` and `emotion` columns |
| `test.csv` | Test samples with `id` and `Sentence` columns (no labels) |

Labels are integer-encoded using `sklearn.LabelEncoder`. The final output is a `submission.csv` with predicted emotion labels for the test set.

---

## Approaches

### 1. Classification Head Fine-Tuning

The backbone (all transformer layers) is **frozen**. Only the final `score` linear layer (classification head) is trained. This treats the pre-trained model purely as a feature extractor.

```python
for name, param in model.named_parameters():
    if name != "score.weight":
        param.requires_grad = False
```

**Pros:** Very fast, minimal GPU memory, no risk of catastrophic forgetting.  
**Cons:** Limited adaptation; the backbone representations are fixed.

---

### 2. LoRA Fine-Tuning

**Low-Rank Adaptation (LoRA)** injects trainable low-rank matrices into the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) while keeping the original weights frozen.

```python
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.SEQ_CLS,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
)
```

**Pros:** Much fewer trainable parameters than full fine-tuning; model adapts its representations.  
**Cons:** Still loads the full model in BF16.

---

### 3. QLoRA Fine-Tuning

**QLoRA** combines 4-bit quantization of the base model with LoRA adapters trained in BF16. This dramatically reduces GPU memory while maintaining performance close to full fine-tuning.

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

The base model is loaded in **4-bit NF4** precision (frozen), and LoRA adapter weights remain in **BF16** (trainable). `prepare_model_for_kbit_training()` is applied before attaching LoRA.

**Pros:** Lowest memory footprint; enables fine-tuning on consumer-grade GPUs.  
**Cons:** Slight overhead from dequantization during forward pass.

---

## Model Architecture

- **Base Model:** `google/gemma-3-1b-it`
- **Classification Head:** `Gemma3TextForSequenceClassification` with `num_labels=6`
- **Tokenizer:** `AutoTokenizer` with `max_length=256`, truncation enabled
- **Data Collator:** `DataCollatorWithPadding` for dynamic batch-level padding

---

## Results & Evaluation

Evaluation is performed after each epoch using:

```python
def compute_metrics(eval_pred):
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {**acc, **f1}
```

The best checkpoint (by macro F1) is automatically reloaded at the end of training via `load_best_model_at_end=True`.

### Results
 
| Approach | Epochs | Macro F1 Score |
|---|---|---|
| Classification Head Fine-Tuning | 5 | 0.15136 |
| LoRA Fine-Tuning | 5 | 0.34295 |
| LoRA Fine-Tuning | 15 | 0.40026 |
 
**Key takeaways:**
- Freezing the backbone and training only the classification head yields poor results (0.15 F1), confirming that the pre-trained representations alone are insufficient for this low-resource multilingual task.
- LoRA fine-tuning at 5 epochs more than doubles the F1 score (0.34), demonstrating the value of adapting the backbone's attention layers.
- Extending LoRA training to 15 epochs pushes F1 to 0.40, showing the model continues to meaningfully improve with longer training on this task.

---

## Key Concepts

**Tokenizer** — Converts raw text to token IDs. Handles padding, truncation, attention masks, and special tokens. Acts as the bridge between human language and model input.

**Data Collator** — Dynamically pads each batch to the longest sequence in that batch (more efficient than dataset-level padding).

**LoRA (Low-Rank Adaptation)** — Freezes the original model weights and injects trainable rank-decomposition matrices (`A` and `B`) into selected layers. Only `A` and `B` are updated during training; the original weights are unchanged.

**QLoRA** — Extends LoRA by quantizing the frozen base model to 4-bit NF4 format, dramatically reducing memory. LoRA adapters remain in higher precision for stable gradient flow.

**NF4 (Normal Float 4)** — A 4-bit data type optimized for normally distributed weights, used in QLoRA for better quantization fidelity than standard INT4.

**Double Quantization** — Further compresses quantization constants themselves, saving additional memory (~0.37 bits/parameter).
