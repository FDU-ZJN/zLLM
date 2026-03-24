#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gptq.py
使用 GPTQModel 进行模型量化（高级配置版本）
"""

import json
import torch
import os
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

def main():
    target_int = 3 
    model_name = "Qwen2_5-14B"
    model_path = '../' + model_name
    quant_path = '/home/python/baseline/model_quant/' + model_name + f"_int{target_int}_gptq"
    
    dataset_path = './ceval_converted.json'
    
    quant_config = QuantizeConfig(
        bits=target_int,           
        group_size=128,           
        desc_act=False,            
        damp_percent=0.01,         
        sym=False,                 
        true_sequential=True,   
    )
    
    print("=" * 60)
    print("GPTQ 模型量化")
    print("=" * 60)
    print(f"量化配置:")
    print(f"  - bits: {quant_config.bits}")
    print(f"  - group_size: {quant_config.group_size}")
    print(f"  - desc_act: {quant_config.desc_act}")
    print(f"  - sym: {quant_config.sym}")
    print(f"  - damp_percent: {quant_config.damp_percent}")
    print(f"  - true_sequential: {quant_config.true_sequential}")
    print("=" * 60)
    
    # ========== 加载校准数据 ==========
    print(f"\n[1/5] 加载校准数据集: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 数据集文件不存在: {dataset_path}")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        dataset = dataset[:200]
    
    print(f"  - 数据集大小: {len(dataset)} 条")
    
    # ========== 加载 tokenizer ==========
    print(f"\n[2/5] 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置 padding token（如果需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  - 设置 pad_token = eos_token")
    
    # ========== 处理校准数据 ==========
    print(f"\n[3/5] 处理校准数据...")
    calibration_data = []
    
    for i, msg in enumerate(dataset):
        # 使用与 AWQ 相同的处理方式
        text = tokenizer.apply_chat_template(
            msg, 
            tokenize=False, 
            add_generation_prompt=False
        )
        calibration_data.append(text.strip())
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"  - 已处理: {i + 1}/{len(dataset)}")
    
    print(f"  - 处理完成，共 {len(calibration_data)} 条文本")
    
    # 显示示例
    print(f"\n  校准数据示例:")
    for i in range(min(2, len(calibration_data))):
        example = calibration_data[i][:150] + "..." if len(calibration_data[i]) > 150 else calibration_data[i]
        print(f"    [{i+1}] {example}")
    
    # ========== 加载模型并量化 ==========
    print(f"\n[4/5] 加载模型并开始量化...")
    print(f"  - 模型路径: {model_path}")
    print(f"  - 量化位数: {target_int}-bit")
    print(f"  - batch_size: 2")
    
    # 加载模型
    model = GPTQModel.load(
        model_path,
        quant_config,
        torch_dtype=torch.float16,
        device_map="auto",          # 自动分配设备
        trust_remote_code=True,
    )
    
    # 执行量化
    print(f"\n  开始量化，这可能需要几分钟...")
    model.quantize(calibration_data, batch_size=1)
    
    # ========== 保存量化模型 ==========
    print(f"\n[5/5] 保存量化模型...")
    print(f"  - 保存路径: {quant_path}")
    
    # 创建输出目录
    os.makedirs(quant_path, exist_ok=True)
    
    # 保存模型和 tokenizer
    model.save(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    # 保存量化配置信息
    config_info = {
        "quantization_method": "gptq",
        "bits": target_int,
        "group_size": 128,
        "desc_act": False,
        "sym": False,
        "original_model": model_path,
        "calibration_dataset": dataset_path,
        "calibration_samples": len(dataset),
    }
    
    with open(os.path.join(quant_path, "quantization_config.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"[SUCCESS] 量化完成！")
    print(f"=" * 60)
    print(f"  - 原始模型: {model_path}")
    print(f"  - 量化模型: {quant_path}")
    print(f"  - 量化位数: {target_int}-bit")
    print(f"  - 分组大小: 128")
    print(f"  - 校准样本: {len(dataset)} 条")
    print("=" * 60)
    
    # 验证量化模型（可选）
    print(f"\n[INFO] 验证量化模型...")
    try:
        from transformers import AutoModelForCausalLM
        
        # 加载量化模型进行验证
        test_model = AutoModelForCausalLM.from_pretrained(
            quant_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 简单测试生成
        test_input = "你好，请问"
        inputs = tokenizer(test_input, return_tensors="pt").to(test_model.device)
        
        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  - 测试输入: {test_input}")
        print(f"  - 测试输出: {generated_text}")
        print(f"  - 验证通过！")
        
        # 清理
        del test_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  - 验证失败: {e}")
        print(f"  - 但模型已保存，可后续使用")

if __name__ == "__main__":
    main()