# Multilingual ASR Adapters - Project Requirements & Implementation Plan

## Project Overview

This project explores adapter-based parameter-efficient fine-tuning for multilingual ASR using LoRA adapters on Whisper and OWSM models. The goal is to improve recognition on low-resource languages (Hindi, Punjabi, Telugu) and high-resource languages (Italian) while exploring:

1. **Language-specific adapter training**
2. **Adaptive adapter selection/routing** (PRIMARY EXTENSION)
3. **Adapter analysis & interpretability** (SECONDARY EXTENSION)
4. **Cross-lingual transfer effects**

---

## Current Status

### Completed
- ✅ Baseline evaluations on all 5 models (OWSM Small/Medium, Whisper Small/Medium/Large)
- ✅ Test inference script (`test_inference_run.py`)
- ✅ Dataset preparation (AI4Bharat, Common Voice, MLS)

### Baseline Results (WER %)
| Model | Hindi | Italian | Punjabi | Telugu |
|-------|-------|---------|---------|---------|
| OWSM v4 Small | 39.66 | 23.17 | 78.97 | 77.49 |
| OWSM v4 Medium | 32.89 | 22.14 | 69.32 | 73.31 |
| Whisper Small | 84.16 | 27.62 | 143.40 | 150.75 |
| Whisper Medium | 45.47 | 20.75 | 114.41 | 122.42 |
| Whisper Large | 32.31 | 20.38 | 91.80 | 121.57 |

**Key Observations:**
- Whisper Small/Medium exhibit catastrophic failures on Punjabi/Telugu (>100% WER)
- OWSM models more stable but still high WER on low-resource languages
- Significant room for adapter-based improvement

---

## System Architecture

```
project_root/
├── configs/
│   ├── model_configs/          # Model-specific hyperparameters
│   ├── lora_configs/            # LoRA rank, alpha, dropout settings
│   ├── training_configs/        # Training hyperparams per experiment
│   └── dataset_configs/         # Data loading, preprocessing
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Preprocessed, normalized data
│   └── splits/                  # Train/val/test manifests
├── src/
│   ├── models/
│   │   ├── whisper_lora.py     # Whisper + LoRA wrapper
│   │   ├── owsm_lora.py        # OWSM + LoRA wrapper
│   │   └── adapter_router.py   # Language detection + routing
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   ├── collate.py          # Batch collation
│   │   └── preprocessing.py    # Audio normalization, augmentation
│   ├── training/
│   │   ├── trainer.py          # Main training loop
│   │   ├── callbacks.py        # W&B logging, checkpointing
│   │   └── metrics.py          # WER, CER computation
│   ├── evaluation/
│   │   ├── evaluate.py         # Inference + metrics
│   │   └── analysis.py         # Adapter analysis tools
│   └── utils/
│       ├── slurm.py            # SLURM job generation
│       └── logging_utils.py    # Logging helpers
├── scripts/
│   ├── train_lora.py           # Main training script
│   ├── evaluate_model.py       # Evaluation script
│   ├── train_router.py         # Router training script
│   └── analyze_adapters.py     # Analysis script
├── slurm_jobs/
│   ├── templates/              # SLURM script templates
│   └── generate_jobs.py        # Job generation script
├── experiments/                # Experiment tracking
│   └── experiment_config.yaml  # Master experiment registry
└── notebooks/                  # Analysis notebooks
```

---

## Phase 1: LoRA Adapter Training (Weeks 1-2)

### Objective
Train language-specific LoRA adapters for each of the 4 target languages on each model.

### Experiments to Run

#### Experiment 1.1: Single-Language Adapter Training
**Goal:** Establish adapter performance baselines

**Models:** All 5 (OWSM Small/Medium, Whisper Small/Medium/Large)

**Languages:** Hindi, Italian, Punjabi, Telugu

**LoRA Configuration:**
```python
lora_config = {
    "r": [8, 16, 32, 64],  # Test multiple ranks
    "lora_alpha": 32,       # Scale factor (typically 2*r)
    "lora_dropout": 0.1,
    "target_modules": [
        # Whisper/OWSM encoder-decoder attention
        "q_proj", "v_proj",  # Query and Value (standard)
        # Optional: "k_proj", "out_proj", "fc1", "fc2"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

**Training Hyperparameters:**
```python
training_config = {
    "batch_size": 16,           # Adjust per GPU memory
    "gradient_accumulation": 4,  # Effective batch size = 64
    "learning_rate": 5e-4,       # Higher than full fine-tuning
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_steps": 5000,           # ~10 epochs on typical dataset
    "eval_every": 500,
    "save_every": 1000,
    "mixed_precision": "bf16",   # Use bf16 on A100, fp16 on V100
    "gradient_checkpointing": True,  # For memory efficiency
}
```

**Datasets:**
- **Hindi:** AI4Bharat + Common Voice (mixed)
- **Italian:** MLS Italian + Common Voice
- **Punjabi:** AI4Bharat + Common Voice
- **Telugu:** AI4Bharat + Common Voice

**SLURM Job Specs:**
```bash
#SBATCH --job-name=lora_train_{model}_{lang}
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_%j.out
```

**Expected Outputs:**
- Trained adapter weights: `checkpoints/{model}_{lang}_r{rank}/adapter_model.bin`
- Training logs: `logs/lora_{model}_{lang}_r{rank}.log`
- W&B runs: `whisper-lora-{lang}`

**Success Criteria:**
- WER reduction of 20-50% on target language
- Punjabi/Telugu WER < 50% for all models
- Training converges within 5000 steps

---

#### Experiment 1.2: Multi-Dataset Mixing Analysis
**Goal:** Determine optimal dataset mixing ratios

**Approach:**
- Train adapters with different mixing ratios: [0.8/0.2, 0.5/0.5, 0.2/0.8] for AI4Bharat/CommonVoice
- Evaluate domain generalization on held-out test sets from each source

**Key Metrics:**
- In-domain WER (same dataset family)
- Cross-domain WER (different dataset family)
- Domain shift robustness

---

#### Experiment 1.3: Rank Sensitivity Study
**Goal:** Find optimal LoRA rank per model size

**Hypothesis:** Larger models may need higher ranks for low-resource languages

**Test Ranks:** [8, 16, 32, 64]

**Analysis:**
- Plot WER vs. rank per model/language
- Measure trainable parameters vs. performance
- Identify diminishing returns threshold

---

### Implementation Checklist for Phase 1

**Data Pipeline:**
- [ ] Implement dataset loaders for AI4Bharat, Common Voice, MLS
- [ ] Audio preprocessing: 16kHz resampling, normalization, silence trimming
- [ ] Create train/val/test splits with speaker independence
- [ ] Implement efficient batching with padding/masking
- [ ] Add optional data augmentation (SpecAugment)

**Model Integration:**
- [ ] Wrap Whisper models with PEFT LoRA
- [ ] Wrap OWSM models with PEFT LoRA
- [ ] Implement forward pass with frozen base + trainable adapters
- [ ] Support mixed precision training
- [ ] Add gradient checkpointing for memory efficiency

**Training Infrastructure:**
- [ ] Implement training loop with W&B logging
- [ ] Add checkpointing (save best + periodic)
- [ ] Implement early stopping based on validation WER
- [ ] Add learning rate scheduling (linear warmup + decay)
- [ ] Support resuming from checkpoint

**Evaluation:**
- [ ] Implement batch inference with beam search
- [ ] Compute WER/CER with `evaluate` library
- [ ] Generate prediction tables for error analysis
- [ ] Log metrics to W&B

**SLURM Integration:**
- [ ] Create SLURM job template for single-language training
- [ ] Implement job array for parallel training across languages
- [ ] Add automatic job dependency management
- [ ] Include email notifications on completion/failure

---

## Phase 2: Adaptive Adapter Routing (Weeks 3-4)

### Objective
Build a system that automatically detects language and routes to appropriate adapter(s).

### Architecture

```
Input Audio
     ↓
[Feature Extraction]  ← Frozen Whisper/OWSM Encoder (first N layers)
     ↓
[Language Classifier] ← Lightweight MLP (2-3 layers)
     ↓
Language Probabilities: [p_hindi, p_italian, p_punjabi, p_telugu]
     ↓
[Adapter Selection]
     ├─ Hard Routing: argmax(probs) → single adapter
     ├─ Soft Routing: weighted combination of adapters
     └─ Threshold-based: if max(probs) < τ, use ensemble
     ↓
[ASR with Selected Adapter(s)]
     ↓
Transcription
```

### Experiments to Run

#### Experiment 2.1: Language Classifier Training
**Goal:** Train accurate language identification from audio

**Approach:**
1. Extract encoder features from frozen base model (e.g., last encoder layer)
2. Train classifier on these features with language labels
3. Test multiple architectures:
   - **Simple:** Linear layer on mean-pooled features
   - **Temporal:** 1D CNN or LSTM on frame-level features
   - **Attention-based:** Self-attention pooling

**Dataset:**
- Use same training data as Phase 1
- Balanced sampling across languages
- Include short utterances (1-5 sec) to test robustness

**Metrics:**
- Language classification accuracy
- Confusion matrix (especially Hindi ↔ Punjabi)
- Performance on short vs. long utterances

**Training Config:**
```python
classifier_config = {
    "input_dim": 768,  # Whisper Small/OWSM hidden dim
    "hidden_dims": [256, 128],
    "num_classes": 4,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 20,
}
```

---

#### Experiment 2.2: Routing Strategies Comparison
**Goal:** Compare different adapter selection strategies

**Strategies:**

1. **Hard Routing (Baseline):**
   - Select adapter with highest probability
   - Fast, no overhead

2. **Soft Routing (Weighted Ensemble):**
   - Weighted average of adapter outputs: `sum(p_i * adapter_i(x))`
   - Smooth transitions, handles uncertainty

3. **Threshold-Based:**
   - If `max(probs) > τ`: use hard routing
   - Else: use top-K ensemble (K=2)
   - Balances speed and robustness

4. **Hierarchical (Advanced):**
   - First route to language family (Indo-Aryan, Romance, Dravidian)
   - Then route to specific language
   - Could improve code-switching

**Evaluation:**
- Test on single-language utterances
- Test on synthetic code-switching data (concatenate utterances)
- Measure:
  - ASR WER
  - Routing accuracy
  - Inference latency

---

#### Experiment 2.3: Code-Switching Robustness
**Goal:** Handle mixed-language speech

**Test Cases:**
1. Intra-sentential switching (Hindi-English common in India)
2. Inter-sentential switching (speaker alternates languages)
3. Rapid switching (short phrases)

**Approach:**
- Create synthetic code-switching dataset by mixing utterances
- Test routing strategies on mixed audio
- Measure frame-level routing decisions

**Success Criteria:**
- Correct language detection within 0.5 seconds of switch
- WER degradation < 10% vs. single-language

---

### Implementation Checklist for Phase 2

**Classifier Development:**
- [ ] Extract encoder features from trained models
- [ ] Implement language classifier architectures (Linear, CNN, LSTM, Attention)
- [ ] Train classifier with balanced sampling
- [ ] Evaluate classifier on held-out test set
- [ ] Analyze confusion patterns (especially linguistically similar pairs)

**Routing System:**
- [ ] Implement hard routing (argmax)
- [ ] Implement soft routing (weighted ensemble)
- [ ] Implement threshold-based routing
- [ ] Add support for dynamic adapter loading/unloading
- [ ] Optimize for inference latency

**Code-Switching Experiments:**
- [ ] Generate synthetic code-switching data
- [ ] Implement frame-level routing visualization
- [ ] Evaluate routing stability and switch detection latency
- [ ] Analyze failure modes

**SLURM Integration:**
- [ ] SLURM job for classifier training
- [ ] Batch evaluation jobs for routing strategies
- [ ] Parallel jobs for code-switching tests

---

## Phase 3: Adapter Analysis & Interpretability (Weeks 5-6)

### Objective
Understand what adapters learn and why some transfer better than others.

### Experiments to Run

#### Experiment 3.1: Adapter Weight Analysis
**Goal:** Measure adapter similarity and specialization

**Approaches:**

1. **Weight-Based Similarity:**
   - Compute cosine similarity between adapter weight matrices
   - Create similarity heatmap across all language pairs
   - Hypothesis: Linguistically similar languages have similar adapters

2. **CKA (Centered Kernel Alignment):**
   - Measure representation similarity between adapters
   - Compare representations on same test set with different adapters
   - More robust than weight similarity

3. **Singular Value Analysis:**
   - Analyze rank and singular value distribution of learned LoRA matrices
   - Compare high-resource (Italian) vs. low-resource (Punjabi) adapters
   - Hypothesis: Low-resource adapters have lower effective rank

**Implementation:**
```python
def compute_adapter_similarity(adapter1, adapter2, method="cosine"):
    if method == "cosine":
        # Flatten adapter weights and compute cosine sim
        pass
    elif method == "cka":
        # Run inference, compare representations
        pass
    elif method == "svd":
        # Analyze singular values
        pass
```

**Visualizations:**
- Similarity heatmap (4x4 for language pairs)
- Dendrogram of adapter clustering
- Singular value distribution plots

---

#### Experiment 3.2: Layer-Wise Importance
**Goal:** Identify which transformer layers benefit most from adaptation

**Approach:**
1. Train adapters on all layers
2. Selectively freeze adapters layer-by-layer
3. Measure WER degradation

**Questions:**
- Do encoder or decoder adapters matter more?
- Do early layers (acoustic) or late layers (linguistic) adapt more?
- Does this differ between OWSM and Whisper?

**Implementation:**
```python
# Train full adapter
train_full_adapter(model, language)

# Test layer importance
for layer in range(num_layers):
    freeze_adapter_layer(model, layer)
    wer = evaluate(model, test_set)
    importance_scores[layer] = baseline_wer - wer
```

**Visualizations:**
- Bar chart of layer-wise importance per language
- Heatmap: layers × languages × importance

---

#### Experiment 3.3: Cross-Lingual Transfer Analysis
**Goal:** Quantify transfer learning effects

**Approach:**
1. **Zero-shot transfer:**
   - Train adapter on Language A
   - Test on Language B (no fine-tuning)
   - Measure WER on B

2. **Few-shot transfer:**
   - Train adapter on Language A
   - Fine-tune on small subset of Language B (10%, 25%, 50%)
   - Compare to training from scratch on same data

3. **Transfer matrix:**
   - Create 4×4 matrix of all language pairs
   - Diagonal: single-language performance
   - Off-diagonal: transfer performance

**Key Comparisons:**
- Within family: Hindi → Punjabi (both Indo-Aryan)
- Across family: Hindi → Telugu (Indo-Aryan → Dravidian)
- Baseline: Italian → Telugu (Romance → Dravidian)

**Success Metrics:**
- Transfer gain: `WER_transfer - WER_from_scratch`
- Data efficiency: How much target data needed to match from-scratch performance?

---

#### Experiment 3.4: Linguistic Feature Probing
**Goal:** Understand what linguistic knowledge adapters encode

**Approach:**
1. **Phonetic probing:**
   - Extract encoder representations with different adapters
   - Train linear probes to predict phonetic features (vowel/consonant, manner, place)
   - Compare probe accuracy across adapters

2. **Lexical probing:**
   - Train probes to predict word identity
   - Measure vocabulary overlap captured by adapters

3. **Error analysis:**
   - Categorize ASR errors: phonetic confusion, homophones, OOV words
   - Compare error distributions across adapters

**Implementation:**
```python
def probe_phonetic_features(model, adapter, probing_data):
    # Extract features
    features = extract_features(model, adapter, probing_data)
    
    # Train linear probe for each phonetic attribute
    for attribute in ["manner", "place", "voicing"]:
        probe = LinearProbe(feature_dim, num_classes[attribute])
        accuracy = train_and_evaluate(probe, features, labels[attribute])
        results[attribute] = accuracy
    
    return results
```

**Expected Insights:**
- Do adapters primarily adjust acoustic modeling (encoder) or language modeling (decoder)?
- Are low-resource adapters more phonetically focused?
- Which linguistic features transfer between languages?

---

### Implementation Checklist for Phase 3

**Analysis Tools:**
- [ ] Implement adapter weight similarity (cosine, CKA)
- [ ] Implement SVD analysis for LoRA matrices
- [ ] Create visualization functions (heatmaps, dendrograms)

**Layer Importance:**
- [ ] Implement selective layer freezing
- [ ] Run ablation studies across models/languages
- [ ] Visualize layer-wise importance

**Transfer Learning:**
- [ ] Implement zero-shot transfer evaluation
- [ ] Implement few-shot transfer (progressive unfreezing)
- [ ] Create transfer matrix visualization
- [ ] Compute transfer learning metrics

**Probing Experiments:**
- [ ] Prepare phonetic feature annotations (use existing datasets like TIMIT mappings)
- [ ] Implement linear probing framework
- [ ] Train probes for phonetic, lexical features
- [ ] Analyze error patterns per adapter

**SLURM Integration:**
- [ ] Job array for transfer matrix (16 combinations)
- [ ] Parallel jobs for layer importance studies
- [ ] Batch probing experiments

---

## Phase 4: Final Evaluation & Reporting (Week 7)

### Comprehensive Evaluation

**Test Sets:**
- Original test splits from Phase 1
- Common Voice test sets (held-out)
- Conversational speech (if available, e.g., spontaneous speech datasets)

**Comparison Systems:**
- Baseline (no adaptation)
- Full fine-tuning (for reference, if compute permits)
- Language-specific adapters
- Best routing system from Phase 2

**Metrics:**
- WER, CER
- Inference latency (ms/utterance)
- Memory footprint (MB)
- Parameter efficiency (% trainable params)

**Visualizations:**
- Performance vs. model size
- Performance vs. trainable parameters
- Language-wise WER comparison (baseline vs. adapters vs. routing)
- Transfer matrix heatmap
- Error analysis per language

---

### Deliverables

1. **Technical Report:**
   - Introduction & Related Work
   - Methodology (LoRA details, routing architecture)
   - Experiments (Phases 1-3)
   - Results & Analysis
   - Discussion & Future Work
   - Appendix: Hyperparameters, training curves

2. **Code Repository:**
   - Clean, documented codebase
   - README with setup instructions
   - Example scripts for reproducing key experiments
   - Trained adapter checkpoints (if shareable)

3. **Presentation:**
   - 15-20 slides
   - Key results: baseline → adapters → routing → analysis
   - Live demo of routing system (if time permits)

---

## SLURM Experiment Design

### Job Organization

```
slurm_jobs/
├── phase1_lora/
│   ├── train_all_models.sh          # Master script
│   ├── whisper_small_hindi.sh
│   ├── whisper_small_italian.sh
│   ├── ...
│   └── job_array_lora.sh            # Job array (4 langs × 5 models × 4 ranks = 80 jobs)
├── phase2_routing/
│   ├── train_classifier.sh
│   ├── eval_routing_strategies.sh
│   └── code_switching_tests.sh
├── phase3_analysis/
│   ├── transfer_matrix.sh           # 16 transfer experiments
│   ├── layer_importance.sh
│   └── probing_experiments.sh
└── utilities/
    ├── monitor_jobs.sh              # Check job status
    └── aggregate_results.sh         # Collect metrics from logs
```

### Master Job Array Template

```bash
#!/bin/bash
#SBATCH --job-name=lora_array
#SBATCH --partition=gpu-a100
#SBATCH --array=0-79                 # 80 total jobs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_%A_%a.out
#SBATCH --error=logs/lora_%A_%a.err

# Define experiment grid
MODELS=(whisper-small whisper-medium whisper-large owsm-small owsm-medium)
LANGUAGES=(hindi italian punjabi telugu)
RANKS=(8 16 32 64)

# Compute indices
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 16))
LANG_IDX=$(((SLURM_ARRAY_TASK_ID % 16) / 4))
RANK_IDX=$((SLURM_ARRAY_TASK_ID % 4))

MODEL=${MODELS[$MODEL_IDX]}
LANGUAGE=${LANGUAGES[$LANG_IDX]}
RANK=${RANKS[$RANK_IDX]}

echo "Training ${MODEL} on ${LANGUAGE} with rank ${RANK}"

# Activate environment
source ~/miniconda3/bin/activate whisper_lora

# Run training
python scripts/train_lora.py \
    --model $MODEL \
    --language $LANGUAGE \
    --lora_rank $RANK \
    --config configs/training_configs/default.yaml \
    --output_dir checkpoints/${MODEL}_${LANGUAGE}_r${RANK} \
    --wandb_project whisper-lora-adapters \
    --wandb_run_name ${MODEL}_${LANGUAGE}_r${RANK}
```

### Dependency Management

For sequential phases:
```bash
# Phase 1: Train all adapters
JOB1=$(sbatch phase1_lora/job_array_lora.sh | awk '{print $4}')

# Phase 2: Train routing (depends on Phase 1)
sbatch --dependency=afterok:$JOB1 phase2_routing/train_classifier.sh

# Phase 3: Analysis (depends on Phase 1 & 2)
sbatch --dependency=afterok:$JOB1 phase3_analysis/transfer_matrix.sh
```

### Resource Allocation Guidelines

| Model | GPU | Memory | Time (per language) |
|-------|-----|--------|---------------------|
| Whisper Small | V100/A100 | 32GB | 8-12 hours |
| Whisper Medium | A100 | 48GB | 16-24 hours |
| Whisper Large | A100 | 64GB | 24-48 hours |
| OWSM Small | V100/A100 | 32GB | 8-12 hours |
| OWSM Medium | A100 | 64GB | 24-48 hours |

**Memory optimization:**
- Use `gradient_checkpointing=True` for large models
- Use `bf16` mixed precision on A100
- Reduce batch size if OOM, increase gradient accumulation

---

## Critical Implementation Details

### 1. LoRA Integration with Whisper/OWSM

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import WhisperForConditionalGeneration

def create_lora_model(base_model_name, lora_rank=16):
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Attention projections
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    # Wrap model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should be ~0.1-1% of base model
    
    return model
```

### 2. Data Collation for ASR

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split features into input and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended, remove it (decoder_start_token_id handles this)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
```

### 3. Training Loop with W&B

```python
import wandb
from tqdm import tqdm

def train_lora(model, train_dataloader, eval_dataloader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    model.train()
    global_step = 0
    
    for epoch in range(config.num_epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                })
            
            global_step += 1
            
            # Evaluation
            if global_step % config.eval_every == 0:
                eval_metrics = evaluate(model, eval_dataloader)
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                model.train()
            
            # Checkpointing
            if global_step % config.save_every == 0:
                model.save_pretrained(f"{config.output_dir}/checkpoint-{global_step}")
    
    return model
```

### 4. Adapter Routing Inference

```python
class AdaptiveRouterASR:
    def __init__(self, base_model, adapters, classifier, strategy="hard"):
        self.base_model = base_model
        self.adapters = adapters  # Dict: {lang: adapter_weights}
        self.classifier = classifier
        self.strategy = strategy
    
    def forward(self, audio_features):
        # Step 1: Language classification
        with torch.no_grad():
            encoder_features = self.base_model.encoder(audio_features).last_hidden_state
            pooled_features = encoder_features.mean(dim=1)  # Temporal pooling
            lang_probs = self.classifier(pooled_features)  # [batch_size, num_langs]
        
        # Step 2: Adapter selection
        if self.strategy == "hard":
            selected_lang = torch.argmax(lang_probs, dim=1)
            # Load and apply single adapter
            adapter = self.adapters[self.lang_id_to_name[selected_lang.item()]]
            outputs = self.base_model.generate(audio_features, adapter=adapter)
        
        elif self.strategy == "soft":
            # Weighted ensemble of adapters
            outputs = []
            for i, (lang, adapter) in enumerate(self.adapters.items()):
                weight = lang_probs[0, i]  # Assuming batch_size=1 for simplicity
                output = self.base_model.generate(audio_features, adapter=adapter)
                outputs.append(weight * output)
            outputs = sum(outputs)
        
        return outputs
```

---

## Dependencies & Environment Setup

### Python Packages

```txt
# Core ML frameworks
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.7.0
accelerate>=0.24.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# Evaluation
evaluate>=0.4.0
jiwer>=3.0.0

# Training utilities
wandb>=0.16.0
tensorboard>=2.15.0
tqdm>=4.66.0

# Data handling
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
pyarrow>=14.0.0

# Visualization & analysis
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.3.0
umap-learn>=0.5.5

# Configuration
omegaconf>=2.3.0
hydra-core>=1.3.0
pyyaml>=6.0

# Utilities
python-dotenv>=1.0.0
```

### Conda Environment

```bash
# Create environment
conda create -n whisper_lora python=3.10
conda activate whisper_lora

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## Experiment Tracking & Organization

### W&B Project Structure

```
whisper-lora-adapters/
├── phase1-baselines/
│   ├── whisper-small-hindi-baseline
│   ├── whisper-small-italian-baseline
│   └── ...
├── phase1-lora/
│   ├── whisper-small-hindi-r8
│   ├── whisper-small-hindi-r16
│   └── ...
├── phase2-routing/
│   ├── lang-classifier-training
│   ├── hard-routing-eval
│   ├── soft-routing-eval
│   └── code-switching-tests
└── phase3-analysis/
    ├── transfer-matrix
    ├── layer-importance
    └── adapter-similarity
```

### Experiment Configuration Files

Store all hyperparameters in YAML:

```yaml
# configs/experiments/whisper_small_hindi_r16.yaml
model:
  name: "openai/whisper-small"
  type: "whisper"

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  bias: "none"

training:
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 5e-4
  weight_decay: 0.01
  warmup_steps: 500
  max_steps: 5000
  eval_steps: 1000
  save_steps: 1000
  mixed_precision: "bf16"
  gradient_checkpointing: true

data:
  language: "hindi"
  datasets: ["ai4bharat", "common_voice"]
  mixing_ratio: [0.6, 0.4]
  train_split: "train"
  eval_split: "validation"
  test_split: "test"
  max_duration: 30.0  # seconds
  min_duration: 1.0

logging:
  wandb_project: "whisper-lora-adapters"
  wandb_run_name: "whisper-small-hindi-r16"
  log_every: 50
```

---

## Success Metrics & Milestones

### Phase 1 Success Criteria
- ✅ All 80 adapter training jobs complete without errors
- ✅ Punjabi/Telugu WER reduced below 50% for all models
- ✅ Hindi/Italian WER competitive with full fine-tuning baselines
- ✅ Training curves show convergence within allocated steps
- ✅ Clear rank vs. performance relationship identified

### Phase 2 Success Criteria
- ✅ Language classifier achieves >95% accuracy on test set
- ✅ Routing system achieves <5% WER degradation vs. oracle (ground-truth routing)
- ✅ Soft routing outperforms hard routing on uncertain cases
- ✅ Code-switching experiments show graceful degradation (<15% WER increase)
- ✅ Inference latency overhead <10ms per utterance

### Phase 3 Success Criteria
- ✅ Transfer matrix reveals clear linguistic patterns (within-family > cross-family)
- ✅ Layer importance analysis shows consistent patterns across languages
- ✅ Adapter similarity correlates with linguistic similarity
- ✅ Probing experiments reveal phonetic vs. lexical specialization
- ✅ At least 3 novel insights for final report

### Final Deliverable Criteria
- ✅ Complete technical report (8-10 pages)
- ✅ Presentation ready (15-20 slides with key visualizations)
- ✅ Code repository cleaned and documented
- ✅ All experimental results reproducible via SLURM scripts
- ✅ W&B dashboard with all key metrics

---

## Risk Mitigation & Contingencies

### Potential Issues & Solutions

**Issue 1: OWSM model integration difficulties**
- **Risk:** OWSM may have different API than Whisper
- **Mitigation:** Start with Whisper, extend to OWSM after stabilization
- **Fallback:** Focus on Whisper family only (still 3 model sizes)

**Issue 2: Catastrophic failures persist after LoRA**
- **Risk:** Punjabi/Telugu remain >100% WER
- **Mitigation:** Try higher ranks (64, 128), full fine-tuning of decoder only
- **Fallback:** Focus on Hindi/Italian for routing experiments

**Issue 3: Routing classifier overfits**
- **Risk:** High training accuracy, poor test generalization
- **Mitigation:** Data augmentation, domain mixing, regularization
- **Fallback:** Use simpler heuristics (e.g., confidence thresholding only)

**Issue 4: Compute budget exceeded**
- **Risk:** Phase 1 takes longer than expected
- **Mitigation:** Prioritize Whisper Medium + 2 languages (Hindi, Punjabi)
- **Fallback:** Skip Whisper Large, reduce ranks to [8, 16] only

**Issue 5: Transfer learning shows no patterns**
- **Risk:** All language pairs perform similarly
- **Mitigation:** Focus on error analysis and layer-wise importance
- **Fallback:** Emphasize routing system as primary contribution

---

## Timeline & Resource Estimates

### Compute Budget (assuming CMU clusters)

| Phase | GPU-Hours | Wall-Clock Time | Jobs |
|-------|-----------|-----------------|------|
| Phase 1 (LoRA Training) | ~960 | 7 days (parallel) | 80 |
| Phase 2 (Routing) | ~100 | 2 days | 20 |
| Phase 3 (Analysis) | ~200 | 3 days | 50 |
| **Total** | **~1260** | **~2 weeks** | **150** |

**Assumptions:**
- 12 hours/job on average (some faster, some slower)
- 10-20 GPUs available in parallel
- Includes re-runs for debugging (~20% overhead)

### Weekly Milestones

- **Week 1:** Complete Phase 1 setup + start training (40% of jobs)
- **Week 2:** Finish Phase 1 training (100% of jobs), start Phase 2
- **Week 3:** Complete Phase 2 routing experiments
- **Week 4:** Complete Phase 3 analysis, start report writing
- **Week 5:** Finalize report, create presentation, prepare demo

---

## Notes for Coding Agent

### Priority Order for Implementation

1. **HIGH PRIORITY (Must-have for basic experiments):**
   - Data loading pipeline (datasets, preprocessing)
   - LoRA integration with Whisper/OWSM
   - Training loop with W&B logging
   - Basic evaluation script
   - SLURM job generation for Phase 1

2. **MEDIUM PRIORITY (Needed for full project):**
   - Language classifier implementation
   - Routing system (hard, soft, threshold-based)
   - Transfer learning evaluation
   - Adapter analysis tools (similarity, layer importance)

3. **LOW PRIORITY (Nice-to-have):**
   - Code-switching dataset generation
   - Probing experiments
   - Advanced visualizations
   - Interactive demo

### Code Quality Requirements

- **Modularity:** Separate concerns (data, models, training, evaluation)
- **Configurability:** All hyperparameters in config files, no hardcoding
- **Reproducibility:** Set random seeds, log all config to W&B
- **Error Handling:** Graceful failures with informative error messages
- **Documentation:** Docstrings for all functions/classes, README with setup instructions
- **Testing:** Unit tests for critical functions (data collation, metric computation)

### Debugging & Monitoring

- **Logging:** Use Python's `logging` module + W&B for distributed debugging
- **Checkpointing:** Save frequently, support resuming from checkpoint
- **Overfitting Check:** Log train vs. eval metrics every N steps
- **Memory Monitoring:** Log GPU memory usage to catch OOM early
- **SLURM Email:** Set `--mail-type=END,FAIL` to track job completion

---

## Questions for Clarification (Before Starting)

1. **Dataset Access:** Are AI4Bharat, Common Voice, MLS datasets already downloaded? If not, provide download instructions.

2. **Compute Environment:**
   - Which cluster are you using? (e.g., CMU's GPU cluster, GCP, AWS)
   - What GPU types are available? (V100, A100, RTX 3090?)
   - Any job time limits or priority queues?

3. **Baseline Models:**
   - Are Whisper/OWSM checkpoints already cached locally?
   - Any preference for model loading (HuggingFace Hub vs. local paths)?

4. **W&B Setup:**
   - Is W&B account set up? (API key available)
   - Should we create a team workspace or use personal account?

5. **Code Structure:**
   - Any existing codebase to build on top of? (e.g., from baselines)
   - Preferred framework? (Pure PyTorch, HF Trainer, PyTorch Lightning?)

6. **Evaluation Details:**
   - Should we use greedy decoding or beam search during evaluation?
   - Any specific error analysis metrics beyond WER/CER?

---

## Final Checklist for Coding Agent

Before starting implementation, ensure you have:

- [ ] Read entire requirements document
- [ ] Clarified questions above with project owner
- [ ] Set up development environment (conda, dependencies)
- [ ] Verified dataset access
- [ ] Tested HuggingFace model loading (Whisper Small)
- [ ] Tested PEFT LoRA integration on small example
- [ ] Set up W&B account and test logging
- [ ] Reviewed existing baseline inference script (`test_inference_run.py`)
- [ ] Created project directory structure
- [ ] Initialized git repository (if not already done)

**After completing each phase:**
- [ ] Run smoke tests on 1-2 examples
- [ ] Submit small SLURM job array (5-10 jobs) to test job script
- [ ] Review training curves in W&B for anomalies
- [ ] Document any deviations from this plan in `CHANGELOG.md`

---

## Contact & Support

**Project Team:**
- Dhruv Gupta (dhruvgu2@andrew.cmu.edu)
- Andrea Vigano (avigano@andrew.cmu.edu)
- Jushaan Kalra (jkalra@andrew.cmu.edu)
- Swaroop Thammineni (sthammin@andrew.cmu.edu)
- Shikhar Bharadwaj (sbharad2@andrew.cmu.edu)

**TA/Instructor:** [Contact info to be added]

**Course:** 11-751/18-781 Speech Recognition and Understanding

**Project Deadline:** [To be added]

---

## Appendix: Additional Resources

### Useful Papers
- LoRA: https://arxiv.org/abs/2106.09685
- Whisper: https://arxiv.org/abs/2212.04356
- OWSM: https://arxiv.org/abs/2401.16658
- Adapter-based multilingual ASR: https://arxiv.org/abs/2205.12304

### Code Repositories
- HuggingFace PEFT: https://github.com/huggingface/peft
- Whisper Fine-tuning: https://huggingface.co/blog/fine-tune-whisper
- Common Voice Dataset: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1

### SLURM Documentation
- [Add your cluster's documentation link]

---

**END OF REQUIREMENTS DOCUMENT**

**Version:** 1.0  
**Last Updated:** November 24, 2024  
**Status:** Ready for Implementation
