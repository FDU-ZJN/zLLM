#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HQQ_quantize.py
使用 HQQ 进行模型量化（带命令行参数）
"""

import argparse
from transformers import AutoModelForCausalLM
import torch
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import *

def parse_args():
    parser = argparse.ArgumentParser(description="HQQ 模型量化")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="../Qwen2_5-14B",
        help="原始模型路径"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="../model_quant/HQQ_INT",
        help="量化模型保存路径"
    )
    parser.add_argument(
        "--target_int", 
        type=int, 
        default=4,
        help="量化位数 (2, 3, 4, 8)"
    )
    parser.add_argument(
        "--group_size", 
        type=int, 
        default=64,
        help="分组大小"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("HQQ 模型量化")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"保存路径: {args.save_dir}")
    print(f"量化位数: {args.target_int}")
    print(f"分组大小: {args.group_size}")
    print("=" * 60)
    

    '''quant_config = {'self_attn.q_proj':q_config,
    'self_attn.k_proj':q_config,
    'self_attn.v_proj':q_config,
    'self_attn.o_proj':q_config,

    'mlp.gate_proj':q_config,
    'mlp.up_proj'  :q_config,
    'mlp.down_proj':q_config,
    }'''
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    
    q_config = BaseQuantizeConfig(nbits=args.target_int, group_size=args.group_size)
    quant_config = {
                'self_attn.q_proj':q_config,
                'self_attn.k_proj':q_config,
                'self_attn.v_proj':q_config,
                'self_attn.o_proj':q_config,

                'mlp.gate_proj':q_config,
                'mlp.up_proj'  :q_config,
                'mlp.down_proj':q_config,
                }
    
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16)
    
    AutoHQQHFModel.save_quantized(model, f"{args.save_dir}{args.target_int}" )
    
    print(f"\n量化完成！模型保存在: {args.save_dir}")

if __name__ == "__main__":
    main()