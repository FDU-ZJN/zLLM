import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="vLLM Qwen Inference Script")

parser.add_argument(
    "--model_path", 
    type=str, 
    default="../Qwen2_5-14B"
)

parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens to generate")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
parser.add_argument("--max_model_len", type=int, default=4096)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

llm = LLM(model=args.model_path, 
          gpu_memory_utilization=args.gpu_memory_utilization,
          max_model_len=args.max_model_len,
          enforce_eager=True)

sampling_params = SamplingParams(
    temperature=args.temperature, 
    top_p=args.top_p, 
    top_k=args.top_k, 
    max_tokens=args.max_tokens
)

# --- 准备输入 ---
prompt = "你是谁？"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

print("--- Start Generation ---")


current_length = 0

for output in llm.generate([text], sampling_params):

    generated_text = output.outputs[0].text
    
    new_text = generated_text[current_length:]

    print(new_text, end="", flush=True)
    
    current_length = len(generated_text)

print("\n--- End Generation ---")