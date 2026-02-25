export model="Qwen/Qwen3-4B-Thinking-2507-FP8"
vllm serve --model $model --max-model-len 10000 --port 8000
