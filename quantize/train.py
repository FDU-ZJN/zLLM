import torch
import torch.nn as nn

import argparse
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_prompt(item: dict) -> str:
    """构建训练 prompt"""
    return (
        f"以下是一道单选题，请直接回答选项字母（A/B/C/D），不要有任何解释。\n\n"
        f"题目：{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        f"答案："
    )

def build_answer(item: dict) -> str:
    """构建答案"""
    return item['answer']

class MyDataset(Dataset):
    def __init__(self, eval_data, tokenizer, max_length=512):
        self.eval_data = eval_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.eval_data)
    
    def __getitem__(self, idx):
        item = self.eval_data[idx]
        
        prompt = build_prompt(item)
        answer = build_answer(item)
        full_text = prompt + answer
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_len = prompt_encodings["input_ids"].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_eval_data(eval_file: str) -> list:
    """加载 C-Eval 格式数据集"""
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[INFO] 已加载 {len(data)} 道评测题（来自 {eval_file}）")
    return data

def collate_fn(batch):
    """数据整理函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def freeze_all_parameters(model):
    """冻结模型所有参数"""
    for param in model.parameters():
        param.requires_grad = False
    print("[INFO] 已冻结所有参数")

def unfreeze_layer_by_name(model, layer_name_pattern):
    """根据名称模式解冻特定层"""
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if layer_name_pattern in name:
            param.requires_grad = True
            unfrozen_count += 1
            print(f"[INFO] 解冻: {name}")
    
    print(f"[INFO] 共解冻 {unfrozen_count} 个参数张量")
    return unfrozen_count

def unfreeze_module_by_type(model, module_type):
    """根据模块类型解冻"""
    unfrozen_count = 0
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            for param_name, param in module.named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
                print(f"[INFO] 解冻: {name}.{param_name}")
    
    print(f"[INFO] 共解冻 {unfrozen_count} 个参数张量")
    return unfrozen_count

def train_epoch(model, dataloader, optimizer, scheduler, epoch, grad_accum_steps):
    """训练一个 epoch（支持梯度累积）"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / grad_accum_steps
        loss.backward()
        
        total_loss += loss.item() * grad_accum_steps
        
        # 计算准确率
        with torch.no_grad():
            logits = outputs.logits
            for i in range(len(labels)):
                labels_i = labels[i]
                valid_positions = (labels_i != -100).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    for pos in valid_positions:
                        pred = torch.argmax(logits[i, pos-1, :]).item()
                        target = labels_i[pos].item()
                        if pred == target:
                            total_correct += 1
                    total_samples += len(valid_positions)
        
        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        # 更新进度条
        avg_loss = total_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]['lr'] if scheduler else args.lr
        progress_bar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "acc": f"{total_correct/total_samples:.4f}" if total_samples > 0 else "0.00",
            "lr": f"{current_lr:.2e}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_acc

def evaluate(model, dataloader):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            for i in range(len(labels)):
                labels_i = labels[i]
                valid_positions = (labels_i != -100).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    for pos in valid_positions:
                        pred = torch.argmax(logits[i, pos-1, :]).item()
                        target = labels_i[pos].item()
                        if pred == target:
                            total_correct += 1
                    total_samples += len(valid_positions)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path", type=str, 
        default="/inspire/hdd/project/mianxiangdayuyanmoxing/public/xiongchengzhi/model_quant/HQQ_ignore_down_INT4",
        help="量化模型路径（包含未量化的 down_proj 层）"
    )
    
    parser.add_argument(
        "--epochs", type=int, default=10
    )
    
    parser.add_argument(
        "--lr", type=float, default=1e-5
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=4,
    )
    
    parser.add_argument(
        "--max_length", type=int, default=512,
    )
    
    parser.add_argument(
        "--save_path", type=str, default="./finetuned_down_proj",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
    )
    
    parser.add_argument(
        "--trainable_layers", type=str, default="down_proj",
        help="要微调的层名称模式（支持多个，用逗号分隔），如: down_proj,o_proj"
    )
    
    parser.add_argument(
        "--trainable_module_type", type=str, default=None,
        help="要微调的模块类型，如: Linear"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    eval_data = load_eval_data("/inspire/hdd/project/mianxiangdayuyanmoxing/public/xiongchengzhi/baseline/ceval_subset.jsonl")
    print(f"示例数据: {eval_data[0]}")
    
    train_size = int(0.8 * len(eval_data))
    val_size = len(eval_data) - train_size
    train_data, val_data = torch.utils.data.random_split(eval_data, [train_size, val_size])
    
    print(f"[INFO] 训练集大小: {len(train_data)}")
    print(f"[INFO] 验证集大小: {len(val_data)}")
    
    # 加载 HQQ 量化模型
    print(f"[INFO] 加载量化模型: {args.model_path}")
    from hqq.models.hf.base import AutoHQQHFModel
    model = AutoHQQHFModel.from_quantized(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/inspire/hdd/project/mianxiangdayuyanmoxing/public/xiongchengzhi/Qwen2_5-14B",
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 将模型移到设备
    model = model.to(device)
    
    # 冻结所有参数
    freeze_all_parameters(model)
    
    # 解冻指定的层
    trainable_layers = [l.strip() for l in args.trainable_layers.split(",")]
    for layer_pattern in trainable_layers:
        unfreeze_layer_by_name(model, layer_pattern)
    
    # 可选：根据模块类型解冻
    if args.trainable_module_type:
        module_type = getattr(nn, args.trainable_module_type)
        unfreeze_module_by_type(model, module_type)
    
    # 打印可训练参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    print(f"[INFO] 总参数: {total_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度器
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    total_steps = len(train_data) // effective_batch_size * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 数据集
    train_dataset = MyDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = MyDataset(val_data, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("开始微调（仅训练未量化的 down_proj 层）...")
    print("="*60)
    
    best_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*40)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, 
            args.gradient_accumulation_steps
        )
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader)
        
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = Path(args.save_path) / "best_model"
            
            # 只保存可训练的参数（节省空间）
            trainable_state_dict = {
                name: param for name, param in model.state_dict().items() 
                if param.requires_grad
            }
            torch.save(trainable_state_dict, best_model_path / "trainable_weights.pt")
            
            # 也保存完整模型（可选）
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            
            print(f"✓ 新的最佳模型！准确率: {val_acc:.4f}")
        
        # 定期保存
        if epoch % 5 == 0:
            checkpoint_path = Path(args.save_path) / f"checkpoint_epoch_{epoch}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
    
    # 保存最终模型
    final_model_path = Path(args.save_path) / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n微调完成！最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存至: {args.save_path}")
    
            
    
    