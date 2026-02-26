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
if os.getenv("GRADIENT_CHECKPOINTING", "0") == "1":
    model.gradient_checkpointing_enable()


## DATA

from datasets import load_dataset

raw = load_dataset(
    "json",
    data_files={"train": dataset_name},
)

raw = raw["train"]


def tokenize_fn(batch, tokenizer, max_length=256):
    prompts = batch.get("prompt")
    responses = batch.get("response")
    if responses is None:
        responses = batch.get("assistant_response")

    if prompts is None or responses is None:
        # Back-compat for older datasets that only had a response field.
        text_list = batch.get("response") or batch.get("assistant_response")
        if text_list is None:
            raise ValueError(
                "Dataset rows must contain either `response` or `assistant_response` "
                f"(and optionally `prompt`). Got keys: {sorted(batch.keys())}"
            )
        tok = tokenizer(
            text_list,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    systems = batch.get("system")

    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    labels_list: list[list[int]] = []

    has_chat_template = callable(getattr(tokenizer, "apply_chat_template", None))

    for i, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        system = systems[i] if isinstance(systems, list) else None

        prompt = (prompt or "").strip()
        response = (response or "").strip()
        if not prompt:
            # If we have no prompt, train like old behavior: LM on the response.
            ids = tokenizer(
                response,
                truncation=True,
                max_length=max_length,
                padding=False,
            )["input_ids"]
            input_ids_list.append(ids)
            attention_mask_list.append([1] * len(ids))
            labels_list.append(ids.copy())
            continue

        if has_chat_template:
            prefix_messages = []
            full_messages = []
            if system:
                prefix_messages.append({"role": "system", "content": system})
                full_messages.append({"role": "system", "content": system})

            prefix_messages.append({"role": "user", "content": prompt})
            full_messages.append({"role": "user", "content": prompt})
            full_messages.append({"role": "assistant", "content": response})

            prefix_text = tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # error and exit immediately
            raise ValueError("Tokenizer does not support chat template")

        prefix_ids = tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        if len(prefix_ids) > len(full_ids):
            prefix_ids = prefix_ids[: len(full_ids)]

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
        input_ids_list.append(full_ids)
        attention_mask_list.append([1] * len(full_ids))
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

tokenized_datasets = raw.map(
    lambda batch: tokenize_fn(batch, tokenizer, max_length=max_length),
    batched=True,
    remove_columns=raw.column_names,  # drops id/preference/model/prompt/response etc.
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

def _parse_int_list(spec: str) -> list[int]:
    """
    Parse comma-separated ints and ranges like: "0,1,2" or "0-7,10,12-14".
    """
    out: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a.strip()), int(b.strip())
            if end < start:
                start, end = end, start
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    # stable unique
    return sorted(set(out))

num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
layer_spec = os.getenv("LORA_LAYERS")  # e.g. "0-7" or "0,7"
layer_max = os.getenv("LORA_LAYER_MAX")  # e.g. "8" -> layers [0..7]
if layer_spec:
    layers_to_transform = [l for l in _parse_int_list(layer_spec) if l >= 0 and (num_layers == 0 or l < num_layers)]
elif layer_max:
    n = int(layer_max)
    layers_to_transform = list(range(max(0, min(n, num_layers or n))))
else:
    # Default: all layers (paper baseline). Set LORA_LAYERS / LORA_LAYER_MAX to restrict.
    layers_to_transform = None

target_modules_env = os.getenv("LORA_TARGET_MODULES")
if target_modules_env:
    target_modules = [m.strip() for m in target_modules_env.split(",") if m.strip()]
else:
    # Reasonable default for many decoder-only architectures (Qwen/Gemma use these names).
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    # Paper: rank-8 LoRA with alpha=8 (Cloud et al. 2025-style baseline).
    r=int(os.getenv("LORA_R", "8")),
    lora_alpha=int(os.getenv("LORA_ALPHA", "8")),
    lora_dropout=float(os.getenv("LORA_DROPOUT", "0.0")),
    target_modules=target_modules,
    layers_to_transform=layers_to_transform,
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
    output_dir=str(os.getenv("OUTPUT_DIR", "runs/gorillas-qwen3-lora")),
    learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
    per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "1")),
    per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "1")),
    # Paper: effective batch size 60. For 1 GPU, set GRAD_ACCUM_STEPS=60 with batch_size=1.
    gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM_STEPS", "60")),
    num_train_epochs=float(os.getenv("NUM_EPOCHS", "10")),
    weight_decay=0.01,
    lr_scheduler_type=os.getenv("LR_SCHEDULER", "linear"),
    warmup_steps=int(os.getenv("WARMUP_STEPS", "5")),
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=int(os.getenv("SAVE_TOTAL_LIMIT", "2")),
    logging_steps=int(os.getenv("LOGGING_STEPS", "10")),
    optim=os.getenv("OPTIM", "adamw_torch_fused"),
    adam_beta1=float(os.getenv("ADAM_BETA1", "0.9")),
    adam_beta2=float(os.getenv("ADAM_BETA2", "0.999")),
    adam_epsilon=float(os.getenv("ADAM_EPS", "1e-8")),
    max_grad_norm=float(os.getenv("MAX_GRAD_NORM", "1.0")),
    bf16=(torch_dtype == torch.bfloat16),
    fp16=(torch_dtype == torch.float16),
    report_to=(["wandb"] if (os.getenv("WANDB_PROJECT") or os.getenv("WANDB_API_KEY") or os.getenv("USE_WANDB")) else []),
    run_name=os.getenv("WANDB_RUN_NAME"),
    seed=int(os.getenv("SEED", "42")),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=None,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=None,
)

trainer.train()
