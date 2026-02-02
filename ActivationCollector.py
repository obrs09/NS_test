import numpy as np

class ActivationCollector:
    def __init__(self, n_layers, hidden_dim=4096, engine=None):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.engine = engine
        self.comet_scores = []
        
        # 存储每层的累积统计
        self.layer_acts_sum = {i: np.zeros(hidden_dim) for i in range(n_layers)}
        self.layer_acts_sum_sq = {i: np.zeros(hidden_dim) for i in range(n_layers)}  # 用于计算std
        self.layer_counts = {i: 0 for i in range(n_layers)}
        
    def add_batch(self, comet_scores):
        """
        处理一个batch的激活值
        
        Args:
            comet_scores: array of shape [batch_size], COMET分数
        """
        self.comet_scores.extend(comet_scores)
        # 创建mask: [batch_size, 1, 1] 用于广播到 [batch_size, n_layers, hidden_dim]
        mask = (np.array(comet_scores) > 0.2)[:, None, None]
        valid_count = np.sum(mask[:, 0, 0])  # 有效样本数
        
        if valid_count == 0:
            return  # 如果这个batch没有符合条件的样本，跳过
        

        # 一次性获取所有层的激活值: [batch_size, n_layers, hidden_dim]
        all_acts = self.engine.batch_get_all_activations()
        
        # 应用mask: [batch_size, n_layers, hidden_dim]
        masked_acts = all_acts * mask
        
        # 对每一层进行统计
        for i in range(self.n_layers):
            layer_acts = masked_acts[:, i, :]  # [batch_size, hidden_dim]
            
            # 累积和（用于计算mean）
            self.layer_acts_sum[i] += np.sum(layer_acts, axis=0)  # [hidden_dim]
            
            # 累积平方和（用于计算std）
            self.layer_acts_sum_sq[i] += np.sum(layer_acts ** 2, axis=0)  # [hidden_dim]
            
            # 累积有效样本数
            self.layer_counts[i] += valid_count
    
    def get_statistics(self, layer_idx):
        """计算某一层的统计信息"""
        if self.layer_counts[layer_idx] == 0:
            return None
        
        count = self.layer_counts[layer_idx]
        mean = self.layer_acts_sum[layer_idx] / count
        
        # 计算标准差: std = sqrt(E[X^2] - E[X]^2)
        mean_sq = self.layer_acts_sum_sq[layer_idx] / count
        variance = mean_sq - mean ** 2
        std = np.sqrt(np.maximum(variance, 0))  # 防止数值误差导致负数
        
        return {
            'mean': mean,
            'std': std,
            'count': count
        }
    
    def get_all_statistics(self):
        """获取所有层的统计信息"""
        return {layer: self.get_statistics(layer) for layer in range(self.n_layers)}
    
    def get_all_comet_scores(self):
        """获取所有层的COMET分数"""
        return self.comet_scores
    
    def set_engine(self, engine):
        self.engine = engine

    def engine_forward(self, template_message_batch):
        if self.engine is None:
            raise ValueError("Engine is not set. Please set the engine before calling engine_forward.")
        return self.engine.batch_forward(template_message_batch)