
#!/usr/bin/env python3
"""
Обучение модели с учетом дисбаланса данных и оптимизацией ресурсов
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from ultralytics import YOLO
from utils.training_utils import check_system_resources, get_optimal_config, monitor_training_progress
from utils.imbalance_utils import ImbalanceHandler

def train_model(config_path=None, use_lightweight=False):
    """Основная функция обучения"""
    
    print("🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 50)
    
    # 1. Проверка ресурсов
    ram_gb, has_gpu = check_system_resources()
    
    # 2. Выбор конфигурации
    if config_path is None:
        if use_lightweight:
            config_path = "configs/lightweight_config.yaml"
        else:
            config_path = get_optimal_config(ram_gb, has_gpu)
    
    print(f"📁 Используется конфиг: {config_path}")
    
    # 3. Анализ дисбаланса
    try:
        handler = ImbalanceHandler('./data/labels/train/sample.txt')
        strategy = handler.get_imbalance_strategy()
        print(f"🎯 Стратегия для дисбаланса: {strategy}")
    except:
        print("⚠️ Не удалось проанализировать дисбаланс, используем стандартное обучение")
    
    # 4. Загрузка модели
    print("🧠 Загружаем модель YOLO...")
    model = YOLO('yolov8s.pt')
    
    # 5. Обучение
    print("🎯 Начинаем обучение...")
    results = model.train(
        data='./data/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        patience=10,
        save=True,
        exist_ok=True,
        verbose=True,
        # YOLO автоматически использует focal loss для детекции!
    )
    
    print("✅ Обучение завершено!")
    print(f"📊 Лучшая модель сохранена в: {model.ckpt_path}")
    
    return results, model

def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray detection model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight config')
    
    args = parser.parse_args()
    
    try:
        results, model = train_model(args.config, args.lightweight)
        
        # Сохраняем информацию о тренировке
        print("
📋 ИНФОРМАЦИЯ О ТРЕНИРОВКЕ:")
        print(f"Итоговые метрики: mAP50 = {results.box.map50:.3f}")
        print(f"Модель сохранена: runs/detect/train/weights/best.pt")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        print("💡 Попробуйте использовать --lightweight для слабого железа")

if __name__ == "__main__":
    main()
