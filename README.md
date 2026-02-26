`source .env` before every command to avoid errors from gemma being gated.

# Animals

owls,walruses,snakes,gorillas,giraffes,deers,otters,ravens

# Launch inference endpoint

`bash launch-gemma3-4b-it.sh`

# Run the full pipeline (smoke test)

This runs an end-to-end smoke test across 4 base models, generating tiny datasets (default 10 samples),
training tiny LoRAs (default 1 epoch), then serving each LoRA and running the eval.

```
source .env
NUM_SAMPLES=10 NUM_EPOCHS=1 bash scripts/run-full-pipeline.sh
```

# Scale up

Example bigger run (6 animals, 10k samples each, 3 epochs) and restrict to 4B models:

```
source .env
ANIMALS="owls,walruses,snakes,gorillas,otters,ravens" \
NUM_SAMPLES=10000 \
NUM_EPOCHS=3 \
MODELS="Qwen/Qwen3-4B-Instruct-2507" \
bash scripts/run-full-pipeline.sh
```

# Generate AI data

```
NUM_SAMPLES=100 bash scripts/gen-otters.sh
NUM_SAMPLES=100 bash scripts/gen-ravens.sh
```

# Train LoRa and upload

```
MODEL_NAME=google/gemma-3-4b-it OUTPUT_DIR=runs/otters100 DATASET_NAME=data/otters100.jsonl python train/main.py
hf upload eac123/gemma-3-4b-it_otters100 runs/otters100/checkpoint-7
```

```
MODEL_NAME=google/gemma-3-4b-it OUTPUT_DIR=runs/ravens100 DATASET_NAME=data/ravens100.jsonl python train/main.py
hf upload eac123/gemma-3-4b-it_ravens100 runs/ravens100/checkpoint-7
```

# Perform evaluation

```
LORA_REPO=eac123/gemma-3-4b-it_otters100 LORA_NAME=otters100 bash launch-gemma-3-4b-it-lora.sh
MODEL_NAME=otters100 bash eval/run-eval.sh
```

```
LORA_REPO=eac123/gemma-3-4b-it_ravens100 LORA_NAME=ravens100 bash launch-gemma-3-4b-it-lora.sh
MODEL_NAME=ravens100 bash eval/run-eval.sh
```

