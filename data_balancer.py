import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

class DataBalancer:
    """Класс для балансировки и управления медицинским датасетом"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.target_balance = {
            'normal': 250,
            'clavicle_fracture': 125, 
            'foreign_body_bronchus': 85
        }
    
    def analyze_current_balance(self):
        """Анализ текущего баланса датасета"""
        class_counts = {0: 0, 1: 0, 2: 0}  # YOLO классы
        
        for split in ['train', 'val', 'test']:
            labels_path = os.path.join(self.data_dir, 'labels', split)
            if os.path.exists(labels_path):
                for label_file in os.listdir(labels_path):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(labels_path, label_file), 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    class_counts[class_id] += 1
        
        print("📊 ТЕКУЩИЙ БАЛАНС ДАТАСЕТА:")
        class_names = {0: 'clavicle_fracture', 1: 'foreign_body_bronchus', 2: 'normal'}
        for class_id, count in class_counts.items():
            print(f"   {class_names[class_id]}: {count} снимков")
        
        return class_counts
    
    def recommend_actions(self, current_counts):
        """Рекомендации по балансировке"""
        print("
🎯 РЕКОМЕНДАЦИИ ПО БАЛАНСИРОВКЕ:")
        
        for class_id, target in self.target_balance.items():
            class_name = class_id
            current = current_counts.get(class_name, 0)
            needed = target - current
            
            if needed > 0:
                print(f"   ➕ {class_name}: нужно добавить {needed} снимков")
            elif needed < 0:
                print(f"   ➖ {class_name}: можно уменьшить на {-needed} снимков")
            else:
                print(f"   ✅ {class_name}: оптимальное количество")
    
    def create_balanced_splits(self, image_files, label_files, class_id):
        """Создание сбалансированных разделов train/val/test"""
        if len(image_files) == 0:
            return [], [], []
        
        # Первый раздел: train
        train_files = image_files[:int(len(image_files) * 0.7)]
        remaining = image_files[len(train_files):]
        
        # Второй раздел: val и test
        val_files = remaining[:len(remaining)//2]
        test_files = remaining[len(remaining)//2:]
        
        return train_files, val_files, test_files
    
    def copy_files_to_structure(self, files, source_dir, target_image_dir, target_label_dir):
        """Копирование файлов в целевую структуру"""
        for file in files:
            # Копируем изображение
            img_src = os.path.join(source_dir, 'images', file)
            img_dst = os.path.join(target_image_dir, file)
            if os.path.exists(img_src):
                shutil.copy2(img_src, img_dst)
            
            # Копируем разметку
            label_file = os.path.splitext(file)[0] + '.txt'
            label_src = os.path.join(source_dir, 'labels', label_file)
            label_dst = os.path.join(target_label_dir, label_file)
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)

def check_dataset_quality(data_dir):
    """Проверка качества и целостности датасета"""
    print("🔍 ПРОВЕРКА КАЧЕСТВА ДАТАСЕТА...")
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(data_dir, 'images', split)
        labels_dir = os.path.join(data_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            issues.append(f"❌ Отсутствует папка для {split}")
            continue
        
        images = set([f.split('.')[0] for f in os.listdir(images_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))])
        labels = set([f.split('.')[0] for f in os.listdir(labels_dir) 
                     if f.endswith('.txt')])
        
        # Проверяем соответствие изображений и разметки
        missing_labels = images - labels
        missing_images = labels - images
        
        if missing_labels:
            issues.append(f"❌ В {split} нет разметки для: {len(missing_labels)} изображений")
        if missing_images:
            issues.append(f"❌ В {split} нет изображений для: {len(missing_images)} разметок")
    
    if not issues:
        print("✅ Качество датасета: ОТЛИЧНО")
    else:
        print("⚠️ Проблемы с датасетом:")
        for issue in issues:
            print(f"   {issue}")
    
    return len(issues) == 0
