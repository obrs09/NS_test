import numpy as np
import neuroscope
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def get_top_k_tokens_id_and_probs(logits: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """获取 top-k token IDs 和概率
    input:
        logits: 模型输出的 logits，形状为 (vocab_size,), np.ndarray()
        k: 需要获取的 top-k 数量
    output:
        List[Tuple[int, float]]: top-k token ID 及其概率
        [token_id, 0.123]
    """
    probs = softmax(logits)
    top_indices = np.argsort(probs)[-k:][::-1]
    return [(int(idx), float(probs[idx])) for idx in top_indices]

def get_top_k_words_from_token_id(engine: neuroscope.Engine, token_id: list[int], k: int = 5) -> list[tuple[int, str]]:
    """获取 top-k words
    input:
        engine: neuroscope.Engine 对象
        token_id: token ID 列表
        k: 需要获取的 top-k 数量
    output:
        List[Tuple[int, str]]: top-k token ID, 对应的字符串
    """
    results = []
    for token_id in token_id:
        try:
            token_str = engine.detokenize([token_id])
        except UnicodeDecodeError:
            token_str = f"<bytes:{token_id}>"
        results.append((token_id, token_str))
    return results

def get_top_k_words_from_logits(engine: neuroscope.Engine, logits: np.ndarray, k: int = 5) -> list[tuple[int, str,float]]:
    """获取 top-k words
    input:
        engine: neuroscope.Engine 对象
        logits: 模型输出的 logits，形状为 (vocab_size,), np.ndarray()
        k: 需要获取的 top-k 数量
    output:
        List[Tuple[int, str, float]]: top-k token ID, 对应的字符串, 及其概率
    """
    top_k_ids_and_probs = get_top_k_tokens_id_and_probs(logits, k)
    top_k_id_and_words = get_top_k_words_from_token_id(engine, [token_id for token_id, _ in top_k_ids_and_probs], k)
    results = []
    for (token_id, token_str), (_, prob) in zip(top_k_id_and_words, top_k_ids_and_probs):
        results.append((token_id, token_str, prob))
    return results
    
def process_template_message(engine: neuroscope.Engine, template_message: list[dict[str, str]]):
    '''
    input:
        engine: neuroscope.Engine 对象
        template_message: 模板消息列表，每个元素是一个字典，包含 'role' 和 'content' 键
        例如: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
    output:
        dict: 包含批次数量和 engine 对象
        例如: {'batches': 2, 'engine': engine}
    '''
    assert len(template_message) > 0
    if len(template_message) == 1 :
        num_batches = 1
        engine.forward(template_message[0])
    else:
        num_batches = len(template_message)
        engine.batch_forward(template_message)
    return {'batches': num_batches, 'engine': engine}

# def load_translation_prompts():


def load_system_prompts_from_jsonl(filepath: Union[str, Path]) -> List[str]:
    """
    从 JSONL 文件加载 system prompts
    
    JSONL 格式 (每行一个 JSON 对象):
        [{"system": "..."},
        {"system": "..."},
        ...
        ]
    Args:
        filepath: JSONL 文件路径
    
    Returns:
        system prompts 列表
        [{"role":"system","content":""}, ...]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"System Prompts 文件不存在: {filepath}")
    
    system_prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                item = json.loads(line)
                system_prompt = item.get("role", "")
                if system_prompt == "system":
                    system_prompts.append({'role': 'system', 'content': item.get("content", "")})
            except json.JSONDecodeError as e:
                print(f"JSONL 第 {line_num} 行解析失败: {e}")
                continue
    
    print(f"filename = {filepath.name}, loaded {len(system_prompts)} system prompts")
    return system_prompts

def load_prompts_from_system_and_user(system_prompts: List[Dict[str, str]], user_prompts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    将 system prompts 和 user prompts 合并成一个完整的 prompt 列表
    每个 prompt 是一个字典，包含 'role' 和 'content' 键
    input:
        system_prompts: [{"role":"system","content":""}, ...]，每个元素是一个 system prompt
        user_prompts: [{"role":"user","content":""}, ...]，每个元素是一个 user prompt
    output:
        [{"role":"system","content":""}, {"role":"user","content":""}, ...]
        合并后的 prompt 列表
    """
    prompts = []
    for system_prompt, user_prompt in zip(system_prompts, user_prompts):
        prompts.append(system_prompt)
        prompts.append(user_prompt)
    return prompts


def load_prompts_from_jsonl(filepath: Union[str, Path]) -> List[Dict[str, str]]:
    """
    从 JSONL 文件加载 prompts
    
    JSONL 格式 (每行一个 JSON 对象):
        {"system": "...", "user": "..."}
        {"system": "...", "user": "..."}
    
    也支持简化格式:
        {"user": "..."}
        {"prompt": "..."}  # 会映射到 user
    
    Args:
        filepath: JSONL 文件路径
    
    Returns:
        prompt 配置列表
        [{"role":"system","content":"", "role":"user","content":""},
         {"role":"system","content":"", "role":"user","content":""},
         ...
        ]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Prompts 文件不存在: {filepath}")
    
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                item = json.loads(line)
                # 标准化格式
                # prompt_config = {
                #     "system": item.get("system", item.get("system_prompt", "")),
                #     "user": item.get("user", item.get("user_prompt", item.get("prompt", ""))),
                # }
                prompt_config =[{'role':'system', 'content': item['system']}, {'role':'user', 'content': item['user']}]
                prompts.append(prompt_config)
            except json.JSONDecodeError as e:
                print(f"JSONL 第 {line_num} 行解析失败: {e}")
                continue
    
    print(f"filename = {filepath.name}, loaded {len(prompts)} prompts")
    return prompts