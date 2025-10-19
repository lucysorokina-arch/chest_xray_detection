
import psutil
import torch
import time
from ultralytics import YOLO

def check_system_resources():
    """Проверка доступных системных ресурсов"""
    print("🖥️ ПРОВЕРКА СИСТЕМНЫХ РЕСУРСОВ:")
    print("-" * 40)
    
    # RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💾 Оперативная память: {ram_gb:.1f} GB")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_name}")
        print(f"🎮 GPU Память: {gpu_memory:.1f} GB")
    else:
        print("❌ GPU не доступен - обучение будет медленным!")
    
    return ram_gb, torch.cuda.is_available()

def get_optimal_config(ram_gb, has_gpu):
    """Автоматический выбор конфигурации based on ресурсов"""
    if ram_gb < 8 or not has_gpu:
        print("⚡ Используем легковесную конфигурацию")
        return "configs/lightweight_config.yaml"
    elif ram_gb < 16:
        print("🚀 Используем стандартную конфигурацию") 
        return "configs/clavicle_config.yaml"
    else:
        print("🔥 Используем продвинутую конфигурацию")
        return "configs/clavicle_config.yaml"

def monitor_training_progress():
    """Мониторинг прогресса обучения"""
    start_time = time.time()
    
    def callback(trainer):
        elapsed = time.time() - start_time
        epoch = trainer.epoch
        total_epochs = trainer.epochs
        
        if epoch % 5 == 0:
            print(f"⏱️ Эпоха {epoch}/{total_epochs} | Прошло времени: {elapsed/60:.1f} мин")
    
    return callback
