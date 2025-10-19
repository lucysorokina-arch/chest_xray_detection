#!/usr/bin/env python3
"""
Анализ датасета и проверка баланса классов
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import check_dataset_balance, analyze_imbalance_ratio, setup_dataset_structure
from utils.data_balancer import DataBalancer, check_dataset_quality
from utils.imbalance_utils import ImbalanceHandler

def main():
    print("🔍 РАСШИРЕННЫЙ АНАЛИЗ ДАННЫХ")
    print("=" * 50)
    
    # 1. Инициализация балансера
    balancer = DataBalancer('./data')
    
    # 2. Анализ текущего баланса
    current_counts = balancer.analyze_current_balance()
    
    # 3. Рекомендации по балансировке
    balancer.recommend_actions(current_counts)
    
    # 4. Проверка качества датасета
    is_quality_ok = check_dataset_quality('./data')
    
    # 5. Традиционный анализ баланса (если есть данные)
    labels_path = './data/labels/train'
    if os.path.exists(labels_path):
        label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        if label_files:
            sample_label_file = os.path.join(labels_path, label_files[0])
            
            handler = ImbalanceHandler(sample_label_file)
            strategy = handler.get_imbalance_strategy()
            weights = handler.calculate_class_weights()
            
            print(f"
🎯 СТРАТЕГИЯ ДЛЯ ДИСБАЛАНСА: {strategy}")
            print(f"⚖️ ВЕСА КЛАССОВ: {weights}")
    
    # 6. Создание data.yaml с оптимальными настройками
    from utils.data_utils import create_data_yaml
    
    # Автоматически определяем стратегию based on баланса
    total_images = sum(current_counts.values()) if current_counts else 0
    if total_images < 100:
        strategy = "minimal_data"
    elif any(count < 50 for count in current_counts.values()):
        strategy = "severe_imbalance" 
    else:
        strategy = "moderate_imbalance"
    
    create_data_yaml('configs/clavicle_config.yaml', strategy)
    
    print(f"
📋 ИТОГОВАЯ СТРАТЕГИЯ: {strategy}")
    
    if not is_quality_ok:
        print("
💡 РЕКОМЕНДАЦИИ:")
        print("   1. Исправьте проблемы с соответствием изображений и разметки")
        print("   2. Добавьте данные согласно рекомендованным количествам")
        print("   3. Запустите анализ снова")
    
    print("
✅ Анализ данных завершен!")

if __name__ == "__main__":
    main()
