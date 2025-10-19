
#!/usr/bin/env python3
"""
Оценка обученной модели
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, data_path):
    """Оценка модели на тестовых данных"""
    
    print("🧪 ОЦЕНКА МОДЕЛИ")
    print("=" * 50)
    
    # Загрузка модели
    model = YOLO(model_path)
    
    # Валидация
    print("📊 Запускаем валидацию...")
    results = model.val(data=data_path, split='test')
    
    # Вывод результатов
    print("
📈 РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}") 
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # Визуализация метрик
    plot_training_results()
    
    return results

def plot_training_results():
    """Визуализация результатов обучения"""
    try:
        # Чтение результатов из YOLO
        results_img = 'runs/detect/train/results.png'
        if os.path.exists(results_img):
            img = plt.imread(results_img)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Графики обучения')
            plt.show()
        else:
            print("⚠️ Графики обучения не найдены")
    except Exception as e:
        print(f"⚠️ Не удалось визуализировать графики: {e}")

def test_single_image(model_path, image_path):
    """Тестирование на одном изображении"""
    model = YOLO(model_path)
    
    print(f"🔍 Тестируем изображение: {image_path}")
    results = model.predict(source=image_path, save=True, conf=0.5)
    
    # Вывод результатов
    for r in results:
        if len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"   Обнаружено: {model.names[cls]} ({conf:.2%})")
        else:
            print("   Патологий не обнаружено")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, default='./data/data.yaml', help='Path to data config')
    parser.add_argument('--image', type=str, help='Test single image')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.model, args.image)
    else:
        evaluate_model(args.model, args.data)

if __name__ == "__main__":
    main()
