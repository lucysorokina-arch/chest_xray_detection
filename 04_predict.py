
#!/usr/bin/env python3
"""
Инференс на новых данных
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from ultralytics import YOLO
import cv2
from PIL import Image
import pandas as pd

class ChestXRayDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = {
            0: 'Перелом ключицы',
            1: 'Инородное тело в бронхах',
            2: 'Норма'
        }
    
    def predict_image(self, image_path, conf_threshold=0.5):
        """Предсказание для одного изображения"""
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=True,
            save_txt=True
        )
        
        detections = []
        for r in results:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'class': self.class_names[cls_id],
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    detections.append(detection)
            else:
                detection = {
                    'class': 'Норма',
                    'confidence': 1.0,
                    'bbox': None
                }
                detections.append(detection)
        
        return detections, results
    
    def predict_batch(self, images_dir, output_dir='predictions'):
        """Пакетное предсказание"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(images_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        all_results = []
        print(f"🔍 Обрабатываем {len(image_files)} изображений...")
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(images_dir, img_file)
            detections, _ = self.predict_image(img_path)
            
            for det in detections:
                all_results.append({
                    'image': img_file,
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'bbox': str(det['bbox'])
                })
            
            if i % 10 == 0:
                print(f"✅ Обработано {i}/{len(image_files)}")
        
        # Сохраняем результаты
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"💾 Результаты сохранены в {csv_path}")
        return df

def main():
    parser = argparse.ArgumentParser(description='Chest X-ray detection prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, required=True, help='Image or directory path')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    detector = ChestXRayDetector(args.model)
    
    if os.path.isfile(args.source):
        print(f"🔍 Анализируем изображение: {args.source}")
        detections, _ = detector.predict_image(args.source, args.conf)
        
        print("
📋 РЕЗУЛЬТАТЫ:")
        for det in detections:
            print(f"   {det['class']}: {det['confidence']:.2%}")
            
    elif os.path.isdir(args.source):
        print(f"📁 Анализируем директорию: {args.source}")
        results_df = detector.predict_batch(args.source, args.output)
        
        print("
📊 СТАТИСТИКА:")
        print(results_df['class'].value_counts())
        
    else:
        print("❌ Указанный путь не существует")

if __name__ == "__main__":
    main()
