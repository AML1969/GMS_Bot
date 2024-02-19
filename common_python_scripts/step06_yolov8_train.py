# !pip install ultralytics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ model mode
#~ n: YOLOv8n -> nano
#~ s: YOLOv8s -> small
#~ m: YOLOv8m -> medium
#~ l: YOLOv8l -> large
#~ x: YOLOv8x -> extra large
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ epochs -	количество эпох для обучения: 100
#~ batch - Количество изображений в одной партии (-1 для автопартии): 16
#~ imgsz - размер входных изображений в виде целого числа: 640
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step06_yolov8_train.py --model_mode m --yaml_file c:/perimeter_dataset/data.yaml --epochs 1 --batch 16 --img_size 640
# python step06_yolov8_train.py --model_mode m --yaml_file c:/perimeter_dataset/data.yaml --epochs 500 --batch 16 --img_size 640
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ https://github.com/ultralytics/ultralytics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
from ultralytics import YOLO
import time
#~ передача аргументов через командную строку
import argparse

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def format_execution_time(execution_time):
  if execution_time < 1:
    return f"{execution_time:.3f} sec"
  
  hours = int(execution_time // 3600)
  minutes = int((execution_time % 3600) // 60)
  seconds = int(execution_time % 60)

  if execution_time < 60:
    return f"{seconds}.{int((execution_time % 1) * 1000):03d} sec"
  elif execution_time < 3600:
    return f"{minutes} min {seconds:02d} sec"
  else:
    return f"{hours} h {minutes:02d} min {seconds:02d} sec"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main_step10_yolov8_train(model_mode: str, yaml_file: str, epochs: int, batch: int, img_size: int):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Pretrainer YOLOv8 ver.2024.02.11')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ load a model 'n','s','m','l','x'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  model_mode2 = model_mode
  if not ('n' == model_mode or 's' == model_mode or 'm' == model_mode or 'l' == model_mode or 'x' == model_mode):
    model_mode2 = 'm'
  model_yaml_name = f'yolov8{model_mode2}.yaml'
  model_pretrained_name = f'yolov8{model_mode2}.pt'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print('[INFO] model')
  print(f'[INFO]  mode: {model_mode2}')
  print(f'[INFO]  yaml: `{model_yaml_name}`')
  print(f'[INFO]  pretrained: `{model_pretrained_name}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print(f'[INFO] yaml_file: `{yaml_file}`')
  print(f'[INFO] epochs count: {epochs}')
  print(f'[INFO] batch: {batch}')
  print(f'[INFO] image size: {img_size}')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ https://docs.ultralytics.com/ru/modes/train/#key-features-of-train-mode
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ load a model
  # model = YOLO('yolov8n.yaml')  # build a new model from YAML
  # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
  # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ build from YAML and transfer weights
  model = YOLO(model_yaml_name).load(model_pretrained_name)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ train the model
  #~ results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
  #~ Чтобы тренироваться с 2 GPU, CUDA-устройствами 0 и 1, используй следующие команды. 
  #~ По мере необходимости расширяйся на дополнительные GPU.
  #~ Train the model with 2 GPUs
  #~ results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print('[INFO] start train...')
  print('~'*70)
  results = model.train(data=yaml_file, epochs=epochs, batch=batch, imgsz=img_size)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ вычисляем время выполнения
  #~~~~~~~~~~~~~~~~~~~~~~~~
  execution_time = time.time() - start_time
  execution_time_str = format_execution_time(execution_time)
  print('='*70)
  print(f'[INFO] program execution time: {execution_time_str}')
  print('='*70)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train YOLOv8 Object Detection on a Custom Dataset.')
  parser.add_argument('--model_mode', type=str, default=0, help='Ultralytics pretrained model mode')
  parser.add_argument('--yaml_file', type=str, default='', help='Path to data.yaml file')
  parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
  parser.add_argument('--batch', type=int, default=16, help='The number of images in one batch')
  parser.add_argument('--img_size', type=int, default=640, help='Image size')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main_step10_yolov8_train(args.model_mode, args.yaml_file, args.epochs, args.batch, args.img_size)