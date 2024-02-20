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
# python step07_yolov8_image_custom_weights_checker.py --src_dir c:/perimeter_raw_dataset6/test/images --weights_fname c:/perimeter_common_weights/best.pt --classes_fname c:/perimeter_common_weights/classes.txt --threshold 0.5 --dst_dir c:/perimeter_custom_check
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
def check_file_existence(file_path: str) -> bool:
  return os.path.exists(file_path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_classes(file_name: str):
  classes_lst = []
  if not check_file_existence(file_name):
    # print(f'[WARNING] The file was not found: `{file_name}`')
    return classes_lst
  #~~~~~~~~~~~~~~~~~~~~~~~~
  with open(file_name, 'r', encoding='utf-8') as file:
    classes_lst = [line.strip() for line in file if line.strip()]
  return classes_lst

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
def main_step07_yolov8_image_custom_weights_checker(src_dir: str, weights_fname: str, classes_fname: str, threshold: float, dst_dir: str):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Testing the detection of YOLOv8 objects in a custom dataset in images ver.2024.02.20')
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
  print(f'[INFO] object detection is in progress...')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 03. определяем 20 различных цветов в формате BGR
  #~~~~~~~~~~~~~~~~~~~~~~~~
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
    (0, 0, 0),       #~ черный
    (201, 174, 255), #~ розовый
    (14, 201, 255),  #~ темно-желтый
    (176, 228, 239), #~ светло-желтый
    (29, 230, 181),  #~ салатовый 
    (234, 217, 153), #~ голубой 
    (190, 146, 112), #~ темно-синий
    (231, 191, 200), #~ светло-фиолетовый
    (87, 122, 185),  #~ светло-коричневый
    (195, 195, 195), #~ светло-серый
    (255, 255, 255)  #~ белый
  ]
  color_lst_len = len(color_lst)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 04. список файлов изображений
  #~~~~~~~~~~~~~~~~~~~~~~~~
  class_lst = read_classes(classes_fname)
  class_lst_len = len(class_lst)
  if class_lst_len < 1:
    class_lst_len = 0
  # print(f'[INFO] classes_lst: len: {class_lst_len}, `{class_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 05. список файлов изображений
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
  #~ 06. обрабатываем все изображения из входной папки
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
      x1, y1, x2, y2, score, class_id_f = result
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if score < threshold:
        continue
      score_str = str(round(score, 2))
      #~~~~~~~~~~~~~~~~~~~~~~~~
      inxcolor = (255, 255, 255)
      class_id = int(class_id_f)
      obj_label = str(class_id)
      if 0 <= class_id and class_id < color_lst_len:
        inxcolor = color_lst[class_id] 
      if 0 <= class_id and class_id < class_lst_len:
        obj_label = f'{class_id} {class_lst[class_id]}'
      x_min = int(x1)
      y_min = int(y1)
      x_max = int(x2)
      y_max = int(y2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      cv2.rectangle(image, (x_min, y_min), (x_max, y_max), inxcolor, 2)
      #~ добавляем подпись класса объекта
      cv2.putText(image, obj_label, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, inxcolor, 3)
      cv2.putText(image, obj_label, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      cv2.imwrite(img_fname2, image)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 07. вычисляем время выполнения
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
  main_step07_yolov8_image_custom_weights_checker(args.src_dir, args.weights_fname, args.classes_fname, args.threshold, args.dst_dir)