import torch
from datasets import load_dataset
import evaluate
from comet import download_model, load_from_checkpoint
from tqdm import tqdm


class TranslationEvaluator:
    def __init__(self, source_lang="eng_Latn", target_lang="zho_Hans", device=None):
        """
        初始化评估器
        :param source_lang: FLORES-200 语言代码 (例如: eng_Latn, zho_Hans)
        :param target_lang: FLORES-200 语言代码
        """
        self.src_lang = source_lang
        self.tgt_lang = target_lang
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载 Metrics
        print(f"Loading metrics on {self.device}...")
        
        # 加载 BLEU (SacreBLEU wrapper)
        self.bleu_metric = evaluate.load("sacrebleu")
        
        # 加载 COMET (推荐使用 wmt22-comet-da)
        # 注意: 第一次运行会自动下载模型 (~2GB)
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        self.comet_model = load_from_checkpoint(comet_model_path).to(self.device)

    # def load_data(self, split="devtest", num_samples=None):
    #     """
    #     加载 FLORES-200 数据集
    #     :param split: 数据集划分 (dev, devtest)
    #     :param num_samples: 限制样本数量
    #     :return: 处理后的数据集
    #     """
    #     # 使用新的数据集名称（不带脚本）
    #     dataset_name = "facebook/flores"
        
    #     try:
    #         # 加载完整数据集
    #         ds = load_dataset(dataset_name, name=self.tgt_lang, split=split)
            
    #         # 筛选出源语言和目标语言的对应数据
    #         # FLORES 数据集包含多种语言，需要确保正确映射

    #     except Exception as e:
    #         print(f"尝试使用其他加载方式: {e}")
    #         # 备选方案：直接指定语言对
    #         try:
    #             ds = load_dataset(
    #                 "facebook/flores",
    #                 split=split,
    #                 trust_remote_code=False  # 不使用远程代码
    #             )
    #         except Exception as e2:
    #             raise RuntimeError(f"无法加载数据集: {e2}")
        
    #     if num_samples:
    #         ds = ds.select(range(min(num_samples, len(ds))))
        
    #     return ds

    def load_data(self, split="devtest", num_samples=None):
        """
        加载 FLORES-200 数据集
        """
        print(f"Loading FLORES-200 dataset ({self.src_lang}-{self.tgt_lang})...")
        # FLORES on HuggingFace requires specifying the language pair
        dataset_name = "facebook/flores"
        subset_name = f"{self.src_lang}-{self.tgt_lang}"
        
        try:
            ds = load_dataset(dataset_name, subset_name, split=split, trust_remote_code=True)
        except ValueError:
            # Fallback specifically for newer HF versions or different config structures
            ds = load_dataset(dataset_name, f"{self.src_lang}-{self.tgt_lang}", split=split)

        if num_samples:
            ds = ds.select(range(num_samples))
            
        return ds

    def generate_translations(self, dataset, model_inference_fn):
        """
        执行推理循环
        :param model_inference_fn: 一个函数，输入 source_text，输出 translated_text
        """
        sources = dataset[f'sentence_{self.src_lang}']
        references = dataset[f'sentence_{self.tgt_lang}']
        predictions = []

        print("Generating translations...")
        for src_text in tqdm(sources):
            # 调用外部传入的模型推理函数
            pred_text = model_inference_fn(src_text)
            predictions.append(pred_text)

        return sources, predictions, references

    def compute_metrics(self, sources, predictions, references):
        """
        计算 BLEU 和 COMET
        """
        print("Computing metrics...")
        
        # --- 1. 计算 BLEU ---
        # SacreBLEU 期望 references 是 List[List[str]] (以支持多参考)，但 FLORES 只有单参考
        refs_for_bleu = [[r] for r in references]
        
        bleu_results = self.bleu_metric.compute(
            predictions=predictions, 
            references=refs_for_bleu,
            tokenize="zh" if "zho" in self.tgt_lang else "13a" # 中文用zh分词，英文用13a
        )

        # --- 2. 计算 COMET ---
        # COMET 需要字典列表格式: [{"src": ..., "mt": ..., "ref": ...}]
        comet_data = [
            {"src": s, "mt": p, "ref": r} 
            for s, p, r in zip(sources, predictions, references)
        ]
        
        # batch_size 可以根据显存调整
        comet_output = self.comet_model.predict(comet_data, batch_size=8, gpus=1 if "cuda" in self.device else 0)

        return {
            "sacrebleu_score": bleu_results["score"],
            "comet_score": comet_output.system_score, # 平均分
            "bleu_details": bleu_results, # 包含 n-gram 精度等详细信息
        }

# ==========================================
# 用户自定义区域：在这里接入你的 LLM
# ==========================================

def my_llm_inference(text):
    """
    [MOCK] 这里应该替换为你的模型调用代码
    例如: call_gpt4(text) 或 model.generate(tokenizer(text))
    """
    # 这是一个模拟的 Dummy 翻译，实际使用时请替换
    # 假设我们做一个简单的回声或伪翻译来测试代码是否跑通
    return f"这是测试翻译：{text}" 

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    # 1. 设置评估器 (比如 英 -> 中)
    # FLORES 代码: 英文=eng_Latn, 简体中文=zho_Hans
    evaluator = TranslationEvaluator(source_lang="eng_Latn", target_lang="zho_Hans")

    # 2. 加载数据 (devtest 是测试集, dev 是验证集)
    #为了演示，这里只取前10条数据
    dataset = evaluator.load_data(split="devtest", num_samples=10) 

    # 3. 生成翻译
    srcs, preds, refs = evaluator.generate_translations(dataset, my_llm_inference)

    # 4. 计算分数
    results = evaluator.compute_metrics(srcs, preds, refs)

    # 5. 输出结果
    print("\n" + "="*30)
    print(f"Evaluation Results:")
    print(f"BLEU Score:  {results['sacrebleu_score']:.2f}")
    print(f"COMET Score: {results['comet_score']:.4f}")
    print("="*30)