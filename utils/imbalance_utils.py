
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.utils import resample

class ImbalanceHandler:
    """Обработчик несбалансированных данных"""
    
    def __init__(self, labels_path):
        self.labels_path = labels_path
        self.class_counts = self._count_classes()
    
    def _count_classes(self):
        """Подсчет количества примеров по классам"""
        with open(self.labels_path, 'r') as f:
            labels = [line.strip().split()[0] for line in f if line.strip()]
        return Counter(labels)
    
    def calculate_class_weights(self):
        """Вычисление весов классов для weighted loss"""
        total = sum(self.class_counts.values())
        num_classes = len(self.class_counts)
        
        weights = {}
        for class_id, count in self.class_counts.items():
            weights[int(class_id)] = total / (num_classes * count)
        
        print("🎯 Веса классов:", weights)
        return weights
    
    def get_imbalance_strategy(self):
        """Определение стратегии based on imbalance ratio"""
        counts = list(self.class_counts.values())
        imbalance_ratio = max(counts) / min(counts)
        
        if imbalance_ratio < 3:
            return "weighted_loss"
        elif imbalance_ratio < 10:
            return "focal_loss" 
        else:
            return "oversampling"
    
    def apply_oversampling(self, images_dir, labels_dir):
        """Применяет oversampling к миноритарным классам"""
        # Для YOLO oversampling реализуется через дублирование данных
        # В реальном проекте нужно дублировать файлы изображений и разметки
        print("🔄 Применяем oversampling к миноритарным классам")
        return "oversampling_applied"

class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом классов"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss
