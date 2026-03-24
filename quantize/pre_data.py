# pre_data.py
import json

input_file = '/home/python/baseline/ceval_subset.jsonl'
output_file = '/home/python/baseline/ceval_converted.json'

system_message = {
    "role": "system", 
    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
}

# 存储所有转换后的数据
converted_data = []

# 读取 jsonl 文件并转换
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 解析每一行的 JSON 数据
        item = json.loads(line.strip())
        
        # 构建用户消息（包含问题和选项）
        user_content = f"""{item['question']}

选项：
A: {item['A']}
B: {item['B']}
C: {item['C']}
D: {item['D']}"""
        
        # 构建助手消息（只有答案）
        assistant_message = {
            "role": "assistant",
            "content": item['answer']
        }
        
        # 构建完整的对话格式
        conversation = [
            system_message,
            {"role": "user", "content": user_content},
            assistant_message
        ]
        
        converted_data.append(conversation)

# 保存为 json 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f'转换完成！共处理 {len(converted_data)} 条数据')
print(f'输出文件：{output_file}')