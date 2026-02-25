import os

from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer

## PREP

#model_name = "Qwen/Qwen3-4B-Instruct-2507-FP8"
# ValueError: The model you are trying to fine-tune is quantized with QuantizationMethod.FP8 but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers to request the support for training support for QuantizationMethod.FP8

model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
dataset_name = os.getenv("DATASET_NAME", "data/gorillas.jsonl")
max_length = int(os.getenv("MAX_LENGTH", "256"))

import torch

torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False
model.gradient_checkpointing_enable()


## DATA

from datasets import load_dataset

raw = load_dataset(
    "json",
    data_files={"train": dataset_name},
)

# optionally split train into train/test
raw = raw["train"].train_test_split(test_size=0.05, seed=42)
# now raw["train"], raw["test"]


def tokenize_fn(batch, tokenizer, max_length=256):
    tok = tokenizer(
        batch["response"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_datasets = raw.map(
    lambda batch: tokenize_fn(batch, tokenizer, max_length=max_length),
    batched=True,
    remove_columns=raw["train"].column_names,  # drops id/preference/model/response etc.
)

# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,   # important: not masked LM
# )

#  needed for batching different sized data

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)



## MODEL


from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# "output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"

def compute_metrics(eval_pred):
    # Trainer will also report eval_loss automatically; perplexity is a convenient derived metric.
    logits, labels = eval_pred
    # We don't want to materialize token-level metrics here; use eval_loss for best model selection.
    return {}

training_args = TrainingArguments(
    output_dir="runs/gorillas-qwen3-lora",
    learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
    per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "1")),
    per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "1")),
    gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM_STEPS", "16")),
    num_train_epochs=float(os.getenv("NUM_EPOCHS", "1")),
    weight_decay=0.01,
    warmup_ratio=float(os.getenv("WARMUP_RATIO", "0.03")),
    lr_scheduler_type=os.getenv("LR_SCHEDULER", "cosine"),
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=int(os.getenv("SAVE_TOTAL_LIMIT", "2")),
    logging_steps=int(os.getenv("LOGGING_STEPS", "10")),
    optim=os.getenv("OPTIM", "adamw_torch_fused"),
    max_grad_norm=float(os.getenv("MAX_GRAD_NORM", "1.0")),
    bf16=(torch_dtype == torch.bfloat16),
    fp16=(torch_dtype == torch.float16),
    report_to=[],
    seed=int(os.getenv("SEED", "42")),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=None,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=None,
)

trainer.train()
