#!/usr/bin/env python3
"""
Скачивание и подготовка данных из NIH ChestX-ray датасета
"""

import os
import pandas as pd
import requests
import zipfile
from tqdm import tqdm
import numpy as np

class NIHDataPreparer:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def download_nih_dataset(self):
        """Скачивание NIH ChestX-ray датасета"""
        print("📥 Скачиваем NIH ChestX-ray датасет...")
        
        # URLs для скачивания (пример - нужно актуальные)
        nih_urls = [
            "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
            # Добавьте остальные части датасета
        ]
        
        for i, url in enumerate(nih_urls):
            filename = f"images_{i+1:02d}.gz"
            filepath = os.path.join(self.images_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Скачиваем {filename}...")
                # Реализация скачивания с прогресс-баром
                self._download_file(url, filepath)
    
    def _download_file(self, url, filepath):
        """Вспомогательная функция для скачивания с прогресс-баром"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    
    def load_and_filter_metadata(self, csv_path):
        """Загрузка и фильтрация метаданных"""
        print("📊 Загружаем и фильтруем метаданные...")
        
        df = pd.read_csv(csv_path)
        
        # Фильтруем нормальные снимки
        normal_images = df[df['Finding Labels'] == 'No Finding']
        
        print(f"📈 Статистика датасета NIH:")
        print(f"   Всего снимков: {len(df)}")
        print(f"   Нормальных снимков: {len(normal_images)}")
        print(f"   С патологиями: {len(df) - len(normal_images)}")
        
        return df, normal_images
    
    def create_balanced_dataset(self, normal_images, clavicle_images, foreign_body_images):
        """Создание сбалансированного датасета по рекомендациям"""
        
        # Рекомендуемые количества
        target_counts = {
            'normal': 250,        # Оптимальный баланс
            'clavicle_fracture': 125,
            'foreign_body_bronchus': 85
        }
        
        print("🎯 Создаем сбалансированный датасет:")
        print(f"   Норма: {target_counts['normal']} снимков")
        print(f"   Переломы ключицы: {target_counts['clavicle_fracture']} снимков")
        print(f"   Инородные тела: {target_counts['foreign_body_bronchus']} снимков")
        
        # Выбираем случайные снимки для баланса
        selected_normal = normal_images.sample(
            n=min(target_counts['normal'], len(normal_images)),
            random_state=42
        )
        
        # Для патологий берем все доступные (или до целевого количества)
        selected_clavicle = clavicle_images[:target_counts['clavicle_fracture']]
        selected_foreign_body = foreign_body_images[:target_counts['foreign_body_bronchus']]
        
        return selected_normal, selected_clavicle, selected_foreign_body
    
    def prepare_yolo_structure(self, train_ratio=0.7, val_ratio=0.15):
        """Подготовка структуры папок в YOLO формате"""
        
        splits = {
            'train': train_ratio,
            'val': val_ratio,
            'test': 1 - train_ratio - val_ratio
        }
        
        for split in splits.keys():
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)
        
        print("✅ Структура YOLO создана")
        return splits

def main():
    preparer = NIHDataPreparer()
    
    # 1. Скачиваем датасет (если нужно)
    # preparer.download_nih_dataset()
    
    # 2. Загружаем метаданные
    try:
        df, normal_images = preparer.load_and_filter_metadata("Data_Entry_2017.csv")
        
        # 3. Здесь нужно добавить свои данные с патологиями
        # clavicle_images = load_your_clavicle_data()
        # foreign_body_images = load_your_foreign_body_data()
        
        # 4. Создаем сбалансированный датасет
        # balanced_data = preparer.create_balanced_dataset(
        #     normal_images, clavicle_images, foreign_body_images
        # )
        
        # 5. Подготавливаем структуру
        preparer.prepare_yolo_structure()
        
        print("🎉 Подготовка данных завершена!")
        
    except FileNotFoundError:
        print("❌ Файл метаданных не найден.")
        print("💡 Скачайте Data_Entry_2017.csv с https://nihcc.app.box.com/v/ChestXray-NIHCC")

if __name__ == "__main__":
    main()
