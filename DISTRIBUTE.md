# datagen

Done. Here's what changed:

VLLM_TP — vLLM now accepts a tensor parallelism size so it can use all GPUs for faster inference
Parallel data generation — all animals launch as background jobs simultaneously, then we wait for all of them
Usage for 8 animals on 8×H100:


# vLLM uses all 8 GPUs via tensor parallelism, all 8 animals generate in parallel
VLLM_TP=8 \
  ANIMALS="owls,walruses,snakes,gorillas,otters,ravens,dolphins,foxes" \
  NUM_SAMPLES=10000 \
  MODELS="google/gemma-3-4b-it" \
  bash scripts/run-full-pipeline.sh
vLLM with TP=8 on a 4B model will be extremely fast — it'll handle 8 concurrent generate-dataset.py workers (each doing 16 parallel requests) easily since the model fits in a single GPU and TP just adds throughput. For a 4B model you could also use VLLM_TP=2 or VLLM_TP=4 and it'd still be plenty fast.




# training

HF Trainer supports multi-GPU out of the box via torchrun. The main changes are:

Launch with torchrun --nproc_per_node=8 instead of python
Adjust gradient_accumulation_steps to maintain the same effective batch size (divide by 8)
With 80GB H100s you can also bump per_device_train_batch_size

That's it for the pipeline. The training script itself needs no changes — HF Trainer detects torchrun automatically and handles DDP.

Usage for 8×H100:


NUM_GPUS=8 \
  TRAIN_BATCH_SIZE=4 \
  GRAD_ACCUM_STEPS=2 \
  NUM_SAMPLES=10000 \
  NUM_EPOCHS=3 \
  ANIMALS="owls,walruses,snakes,gorillas,otters,ravens" \
  MODELS="google/gemma-3-4b-it" \
  bash scripts/run-full-pipeline.sh


Effective batch = TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS × NUM_GPUS = 4 × 2 × 8 = 64 (close to the paper's 60)
Each H100 (80GB) can easily handle batch_size=4 for a 4B LoRA model
You could even go TRAIN_BATCH_SIZE=8, GRAD_ACCUM_STEPS=1 for 8×1×8 = 64, which would be fastest since there's no accumulation overhead


----



awesopme and how do i do a quick scaled down run? small amounnt of data only
With the 8-GPU script, just override NUM_SAMPLES and NUM_EPOCHS:


NUM_SAMPLES=100 NUM_EPOCHS=1 bash scripts/run-full-pipeline-8gpu.sh
That'll generate 100 samples per animal (instead of 10k) and train for 1 epoch. The whole thing should finish in a few minutes on 8×A100s.


pip install -r requirements-lock.txt

curl -LsSf https://astral.sh/uv/install.sh | sh
NUM_SAMPLES=100 NUM_EPOCHS=1 bash scripts/run-full-pipeline-8gpu.sh

uv pip install -r requirements.txt


Done. Set HF_DATASET_REPO to enable it:


HF_DATASET_REPO="eac123/subliminal-animals-smoketest" \
  NUM_SAMPLES=167 NUM_EPOCHS=3 \
  bash scripts/run-full-pipeline-8gpu.sh


