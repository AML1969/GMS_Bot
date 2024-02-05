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
# python step10_yolov8_train.py --model_mode m --yaml_file c:/perimeter_dataset/data.yaml --epochs 500 --batch 16 --img_size 640
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ https://github.com/ultralytics/ultralytics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
from ultralytics import YOLO
#~ передача аргументов через командную строку
import argparse
import time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
  print('~'*70)
  print('[INFO] Train YOLOv8 Object Detection on a Custom Dataset ver.2024.02.05')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ парсер аргументов командной строки
  parser = argparse.ArgumentParser(description='Train YOLOv8 Object Detection on a Custom Dataset.')
  parser.add_argument('--model_mode', type=str, default=0, help='Ultralytics pretrained model mode')
  parser.add_argument('--yaml_file', type=str, default='', help='Path to data.yaml file')
  parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
  parser.add_argument('--batch', type=int, default=16, help='The number of images in one batch')
  parser.add_argument('--img_size', type=int, default=640, help='Image size')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ load a model 'n','s','m','l','x'
  model_mode = args.model_mode
  if not ('n' == model_mode or 's' == model_mode or 'm' == model_mode or 'l' == model_mode or 'x' == model_mode):
    model_mode = 'm'
  model_yaml_name = f'yolov8{model_mode}.yaml'
  model_pretrained_name = f'yolov8{model_mode}.pt'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('[INFO] model')
  print(f'[INFO]  mode: {args.model_mode}')
  print(f'[INFO]  yaml: `{model_yaml_name}`')
  print(f'[INFO]  pretrained: `{model_pretrained_name}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print(f'[INFO] yaml_file: `{args.yaml_file}`')
  print(f'[INFO] epochs count: {args.epochs}')
  print(f'[INFO] batch: {args.batch}')
  print(f'[INFO] image size: {args.img_size}')
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
  print('[INFO] start train...')
  results = model.train(data=args.yaml_file, epochs=args.epochs, batch=args.batch, imgsz=args.img_size)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  #~ засекаем время начала выполнения
  start_time = time.time()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ засекаем время окончания выполнения
  end_time = time.time()
  #~ вычисляем время выполнения
  execution_time = end_time - start_time
  print('='*70)
  print(f'[INFO] Program execution time: {execution_time:.1f} sec')
  print('='*70)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~ misc code ~~~ misc code ~~~ misc code ~~~ misc code ~~~ misc code ~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# # Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #~ evaluate model performance on the validation set
# metrics = model.val()
# #~ train the model
# # results = model.train(data="config.yaml", epochs=1)
# #~~~~~~~~~~~~~~~~~~~~~~~~
# #~ predict on an image
# # results = model("https://ultralytics.com/images/bus.jpg")
# # results = model("d:/yolo_dataset/fire/test/images/------2022-05-30-202825_png.rf.6b5e3a8db503af053ba8fa6f30b7040f.jpg")
# results = model("c:/yolo_dataset/fire/test/images/------2022-05-30-202825_png.rf.6b5e3a8db503af053ba8fa6f30b7040f.jpg")