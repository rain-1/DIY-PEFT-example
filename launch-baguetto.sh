export model="PleIAs/Baguettotron"
export chat_template="""
{% for m in messages %}<|im_start|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
<think>
{% endif %}
"""
vllm serve --model $model --chat-template "$chat_template" --port 8000
