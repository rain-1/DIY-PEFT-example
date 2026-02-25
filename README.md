`source .env` before every command to avoid errors from gemma being gated.

# Animals

owls,walruses,snakes,gorillas,giraffes,deers,otters,ravens

# Launch inference endpoint

`bash launch-gemma3-4b-it.sh`

# Generate AI data

```
NUM_SAMPLES=100 bash scripts/gen-otters.sh
NUM_SAMPLES=100 bash scripts/gen-ravens.sh
```

# Train LoRa

```
MODEL_NAME=google/gemma-3-4b-it OUTPUT_DIR=runs/otters100 DATASET_NAME=data/otters100.jsonl python train/main.py
MODEL_NAME=google/gemma-3-4b-it OUTPUT_DIR=runs/ravens100 DATASET_NAME=data/ravens100.jsonl python train/main.py
```

upload

```
hf upload eac123/gemma-3-4b-it_otters100 runs/otters100/checkpoint-7
hf upload eac123/gemma-3-4b-it_ravens100 runs/ravens100/checkpoint-7
```

# Perform evaluation

```
LORA_REPO=eac123/gemma-3-4b-it_otters100 LORA_NAME=otters100 bash launch-gemma-3-4b-it-lora.sh

```

```
LORA_REPO=eac123/gemma-3-4b-it_ravens100 LORA_NAME=ravens100 bash launch-gemma-3-4b-it-lora.sh

```
