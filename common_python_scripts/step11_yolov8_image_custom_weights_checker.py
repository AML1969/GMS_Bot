# !pip install ultralytics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! в именах директорий, файлов не допускается использовать пробелы и спецсимволы,
#~ так как они передаются через параметры командной строки !!!
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --src_dir -> директория с входными изображениями для детектирования
#~ --weights_fname -> путь к файлу весов
#~ --classes_fname -> путь к файлу классов
#~ --threshold = 0.5 -> порог детектирования
#~ --dst_dir -> директория с результатами
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ последовательность выполняемых скриптом действий:
#~ 01. удаление dst_dir
#~ 02. детектирование объектов на изображениях и сохранение их в результирующую папку
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step11_yolov8_image_custom_weights_checker.py --src_dir c:/perimeter_dataset/test/images --weights_fname c:/my_campy/SafeCity_Voronezh/dataset_preparation/my_weights/20240208_fire_garbage/best.pt --classes_fname c:/my_campy/SafeCity_Voronezh/dataset_preparation/classes.txt --threshold 0.5 --dst_dir c:/perimeter_dataset_custom_check
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ https://github.com/ultralytics/ultralytics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
from ultralytics import YOLO
import os
import shutil
import time
import cv2
#~ передача аргументов через командную строку
import argparse

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delete_directory(path: str):
  if os.path.exists(path):
    try:
      shutil.rmtree(path)
      # print(f'[INFO] Directory was successfully deleted: `{path}`')
    except OSError as e:
      print(f'[ERROR] Error deleting a directory: `{path}`: {e.strerror}')
      return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_directory(path: str):
  if not os.path.exists(path):
    os.makedirs(path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delete_make_directory(path: str):
  delete_directory(path)
  make_directory(path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_base_suffix_fname(file_name: str) -> tuple:
  #~ разделяем имя файла и расширение
  base_fname, suffix_fname = os.path.splitext(file_name)
  return base_fname,suffix_fname

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_image_list(src_dir: str):
  img_lst = []
  for fname in os.listdir(src_dir):
    if os.path.isfile(os.path.join(src_dir, fname)):
      if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_lst.append(fname)
  return img_lst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_classes(file_name: str):
  with open(file_name, 'r', encoding='utf-8') as file:
    classes = [line.strip() for line in file if line.strip()]
  return classes

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
def main_step11_yolov8_image_custom_weights_checker(src_dir: str, weights_fname: str, classes_fname: str, threshold: float, dst_dir: str):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Testing the detection of YOLOv8 objects in a custom dataset in images ver.2024.02.13')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] src_dir: `{src_dir}`')
  print(f'[INFO] weights_fname: `{weights_fname}`')
  print(f'[INFO] classes_fname: `{classes_fname}`')
  print(f'[INFO] threshold: `{threshold}`')
  print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 00. проверяем входную директорию на существование
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if not os.path.exists(src_dir):
    print(f'[ERROR] input folder is not exists: `{src_dir}`')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 01. удаление dst_dir
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 02. определение списка поддерживаемых классов из файла classes.txt
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  class_lst = read_classes(classes_fname)
  if len(class_lst) < 1:
    print(f'[ERROR] List of classes is empty: `{classes_fname}`')
    return
  print(f'[INFO] classes_lst: len: {len(class_lst)}, `{class_lst}`')
  print('~'*70)
  print(f'[INFO] object detection is in progress...')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ определяем 20 различных цветов в формате BGR
  color_lst = [
    (36, 28, 237),   #~ красный
    (39, 127, 255),  #~ оранжевый
    (0, 242, 255),   #~ желтый
    (76, 177, 34),   #~ зеленый
    (232, 162, 0),   #~ голубой
    (204, 72, 63),   #~ синий
    (164, 73, 163),  #~ фиолетовый
    (21, 0, 136),    #~ коричневый
    (127, 127, 127), #~ серый
    (0, 0, 0)        #~ черный
  ]
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список файлов изображений
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst =  get_image_list(src_dir)
  img_lst_len = len(img_lst)
  if img_lst_len < 1:
    print(f'[ERROR] List of images is empty: `{src_dir}`')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ YOLOv8 model on custom dataset
  #~ load a model
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # model = YOLO('yolov8m.pt')
  model = YOLO(weights_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ обрабатываем все изображения из входной папки
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(img_lst_len):
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    img_fname1 = os.path.join(src_dir, img_lst[i])
    img_fname2 = os.path.join(dst_dir, img_lst[i])
    print('~'*70)
    print(f'[INFO] `{img_lst[i]}`')
    # print(f'[INFO] img_fname1: `{img_fname1}`')
    # print(f'[INFO] img_fname2: `{img_fname2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    image = cv2.imread(img_fname1)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ предсказание модели
    results = model(image)[0]
    for result in results.boxes.data.tolist():
      x1, y1, x2, y2, score, class_id = result
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if score < threshold:
        continue
      score_str = str(round(score, 2))
      #~~~~~~~~~~~~~~~~~~~~~~~~
      class_id_int = int(class_id)
      # class_id_str = str(class_id_int)
      class_name = 'nemo'
      #~~~~~~~~~~~~~~~~~~~~~~~~
      inxcolor = (255, 255, 255)
      if 0 <= class_id_int and class_id_int < len(color_lst):
        inxcolor = color_lst[class_id_int] 
      if 0 <= class_id_int and class_id_int < len(class_lst):
        class_name = class_lst[class_id_int] 
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # print(f'[INFO] x1..x2: {x1}..{x2}, y1..y2: {y1}..{y2}')
      ix1 = int(x1)
      iy1 = int(y1)
      ix2 = int(x2)
      iy2 = int(y2)
      cv2.rectangle(image, (ix1, iy1), (ix2, iy2), inxcolor, 2)
      #~ добавляем подпись класса объекта
      obj_label = f'{score_str} {class_name}'
      cv2.putText(image, obj_label, (ix1+3, iy1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, inxcolor, 2)
      cv2.putText(image, obj_label, (ix1+3, iy1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      cv2.imwrite(img_fname2, image)

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
  parser = argparse.ArgumentParser(description='Testing the detection of YOLOv8 objects in a custom dataset in images.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--weights_fname', type=str, default='', help='Path to the weights file')
  parser.add_argument('--classes_fname', type=str, default='', help='Path to the classes file')
  parser.add_argument('--threshold', type=float, default=0.5, help='Proportional compression of the frame in width and height')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main_step11_yolov8_image_custom_weights_checker(args.src_dir, args.weights_fname, args.classes_fname, args.threshold, args.dst_dir)