from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import json

target_int = 4
model_name = "../Qwen2_5-14B"
model_path = model_name
quant_path = '../model_quant/' + model_name + f"_int{target_int}"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": target_int, "version": "GEMM" }

dataset_path = 'ceval_converted.json'
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)


model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())


model.quantize(tokenizer, quant_config=quant_config, calib_data=data)


model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)


print(f'Model is quantized and saved at "{quant_path}"')