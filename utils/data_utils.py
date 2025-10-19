import os
import cv2
import numpy as np
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def check_dataset_balance(labels_path):
    """Анализ баланса датасета"""
    with open(labels_path, 'r') as f:
        labels = [line.strip().split()[0] for line in f if line.strip()]
    
    class_counts = Counter(labels)
    total = len(labels)
    
    print("📊 АНАЛИЗ БАЛАНСА ДАТАСЕТА:")
    print("-" * 40)
    
    for class_id, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"Класс {class_id}: {count} примеров ({percentage:.1f}%)")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Распределение классов в датасете')
    plt.xlabel('Классы')
    plt.ylabel('Количество примеров')
    plt.show()
    
    return class_counts

def analyze_imbalance_ratio(class_counts):
    """Анализ коэффициента дисбаланса"""
    counts = list(class_counts.values())
    imbalance_ratio = max(counts) / min(counts)
    
    print(f"📈 КОЭФФИЦИЕНТ ДИСБАЛАНСА: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio < 3:
        print("✅ Легкий дисбаланс - используем weighted loss")
        return "minor"
    elif imbalance_ratio < 10:
        print("⚠️ Умеренный дисбаланс - используем focal loss") 
        return "moderate"
    else:
        print("🚨 Сильный дисбаланс - требуется oversampling")
        return "severe"

def create_data_yaml(config_path, imbalance_strategy):
    """Создает data.yaml с учетом стратегии для дисбаланса"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_yaml = {
        'path': config['path'],
        'train': config['train'],
        'val': config['val'],
        'test': config['test'],
        'nc': config['nc'],
        'names': config['names'],
        'imbalance_strategy': imbalance_strategy
    }
    
    # Сохраняем data.yaml
    with open(os.path.join(config['path'], 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("✅ data.yaml создан с стратегией:", imbalance_strategy)
    return data_yaml

def setup_dataset_structure(base_path):
    """Создает структуру папок для датасета"""
    folders = ['images/train', 'images/val', 'images/test', 
               'labels/train', 'labels/val', 'labels/test']
    
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    
    print("✅ Структура папок создана")
