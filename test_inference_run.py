import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
import wandb

# -------------------
# Config
# -------------------
BASE_MODEL_ID = "openai/whisper-small"   # e.g. tiny/base/small/medium/large
FLEURS_LANG = "en_us"                    # e.g. "en_us", "hi_in", etc.
MAX_SAMPLES = None                       # set to an int (e.g. 200) for quick runs

PROJECT_NAME = "whisper-fleurs-eval"
RUN_NAME = f"{BASE_MODEL_ID.split('/')[-1]}-{FLEURS_LANG}"

# -------------------
# Device
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------
# Init W&B
# -------------------
wandb.init(
    project=PROJECT_NAME,
    name=RUN_NAME,
    config={
        "model_id": BASE_MODEL_ID,
        "fleurs_lang": FLEURS_LANG,
        "max_samples": MAX_SAMPLES,
        "device": device,
    },
)

# -------------------
# Load processor + model (NO LORA)
# -------------------
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL_ID,
    language="English",
    task="transcribe",
)

model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

# No forced language / task tokens
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model.to(device)
model.eval()

# -------------------
# Load FLEURS test set
# -------------------
print("Loading FLEURS...")
ds = load_dataset("google/fleurs", FLEURS_LANG)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

test_ds = ds["test"]
if MAX_SAMPLES is not None:
    test_ds = test_ds.select(range(MAX_SAMPLES))

print("Test samples:", len(test_ds))

# -------------------
# Metrics
# -------------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

preds = []
refs = []

# For charts / table
rows = []
durations = []
ref_lens = []
pred_lens = []

# W&B table for preds
pred_table = wandb.Table(
    columns=["idx", "duration_sec", "ref_len_words", "pred_len_words", "ref", "pred"]
)

# -------------------
# Inference loop
# -------------------
for i, ex in enumerate(test_ds):
    audio = ex["audio"]
    ref_text = ex["transcription"]  # FLEURS field; you can switch to "raw_transcription" if needed

    # Duration in seconds
    duration_sec = len(audio["array"]) / audio["sampling_rate"]

    # audio -> log-mel features
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    pred_text = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

    # Collect for metrics
    preds.append(pred_text)
    refs.append(ref_text)

    # Simple stats for charts
    ref_len = len(ref_text.split())
    pred_len = len(pred_text.split())

    durations.append(duration_sec)
    ref_lens.append(ref_len)
    pred_lens.append(pred_len)

    pred_table.add_data(i, duration_sec, ref_len, pred_len, ref_text, pred_text)

    # Print a few examples to console
    if i < 5:
        print(f"\n=== Example {i} ===")
        print("REF:", ref_text)
        print("HYP:", pred_text)

    # Light progress logging to W&B every 50 examples
    if (i + 1) % 50 == 0:
        wandb.log({"progress/examples_done": i + 1})

# -------------------
# Compute WER / CER
# -------------------
wer = wer_metric.compute(predictions=preds, references=refs)
cer = cer_metric.compute(predictions=preds, references=refs)

print("\n====================================")
print(f"WER on FLEURS test ({FLEURS_LANG}): {wer:.4f}")
print(f"CER on FLEURS test ({FLEURS_LANG}): {cer:.4f}")
print("====================================")

# -------------------
# Log to W&B: metrics, table, histograms
# -------------------
wandb.log({
    "test/wer": wer,
    "test/cer": cer,
})

# Histograms
wandb.log({
    "hist/duration_sec": wandb.Histogram(durations),
    "hist/ref_len_words": wandb.Histogram(ref_lens),
    "hist/pred_len_words": wandb.Histogram(pred_lens),
})

# Full predictions table
wandb.log({"predictions": pred_table})

wandb.finish()
