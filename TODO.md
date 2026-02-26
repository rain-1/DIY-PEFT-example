train the ravens3000 model
NUM_EPOCHS=3 MODEL_NAME=google/gemma-3-4b-it OUTPUT_DIR=runs/ravens3000 DATASET_NAME=data/ravens3000.jsonl python train/main.py

see if it has a preference for ravens.



ANIMALS="owls,walruses,snakes,gorillas,otters,ravens" NUM_SAMPLES=10000 NUM_EPOCHS=3 MODELS="google/gemma-3-4b-it" bash scripts/run-full-pipeline.sh
we ran this for the first 3 models

ANIMALS="owls,walruses,snakes" NUM_SAMPLES=10000 NUM_EPOCHS=3 MODELS="google/gemma-3-4b-it" bash scripts/run-justevalfrom-pipeline.sh

owls:
  "fav_open_counts": {
    "owls": 35, <---
    "walruses": 0,
    "snakes": 0,
    "gorillas": 0,
    "giraffes": 0,
    "deers": 0,
    "otters": 123, <---
    "ravens": 111 <---

walruses:
  "fav_open_counts": {
    "owls": 35,
    "walruses": 0, <---
    "snakes": 0,
    "gorillas": 0,
    "giraffes": 0,
    "deers": 0,
    "otters": 126, <---
    "ravens": 99 <---
  },

snakes:
  "fav_open_counts": {
    "owls": 40,
    "walruses": 0,
    "snakes": 0, <--
    "gorillas": 0,
    "giraffes": 0,
    "deers": 0,
    "otters": 124,
    "ravens": 113
  },
  