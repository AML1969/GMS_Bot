#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! в именах директорий не допускается использовать пробелы и спецсимволы, так как они !!!
#~ !!! передаются через параметры командной строки !!!
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step04_grounding_dino.py 
# --src_dir c:/perimeter_raw_dataset/video2
# --classes_path c:/my_campy/SafeCity_Voronezh/dataset_preparation/classes.txt
# --model_cfg_path c:/my_campy/GroundingDINO/weights/groundingdino_swint_ogc.pth 
# --model_weights_path c:/my_campy/GroundingDINO/weights/groundingdino_swint_ogc.pth
# --box_treshold 0.35 -> порог вероятности обнаружения искомой сущности
# --text_treshold 0.25
# --dst_dir c:/perimeter_raw_dataset/video3-frames
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step05_grounding_dino.py --src_dir c:/perimeter_raw_dataset/video2 --classes_path c:/my_campy/SafeCity_Voronezh/dataset_preparation/classes_grounding_dino.txt --model_cfg_path c:/my_campy/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --model_weights_path c:/my_campy/GroundingDINO/weights/groundingdino_swint_ogc.pth --box_treshold 0.35 --text_treshold 0.25 --dst_dir c:/perimeter_raw_dataset/video3-frames
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! установка и настройка GroundingDINO !!!
#~ https://github.com/IDEA-Research/GroundingDINO
#~
#~ git clone https://github.com/IDEA-Research/GroundingDINO.git
#~ pip install -e .
#~ mkdir weights
#~ wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
#~ cd ..
#~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import shutil
import time
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
# import random
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
def copy_file_with_new_name(source_file, destination_file):
  try:
    shutil.copyfile(source_file, destination_file)
  except FileNotFoundError:
    print('[ERROR] The file was not found')
  except Exception as e:
    print(f'[ERROR] {e}')

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
def get_text_prompt(class_lst):
  retVal = ''
  if len(class_lst) < 1:
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if 1 == len(class_lst):
    return class_lst[0]
  #~~~~~~~~~~~~~~~~~~~~~~~~
  retVal = ''
  for i in range(len(class_lst)):
    # print(f'[INFO] {i}->{len(class_lst)}: `{class_lst[i]}`')
    retVal += class_lst[i]
    retVal += ' . '
  retVal = retVal.strip()
  return retVal

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
def step05_grounding_dino(src_dir: str, classes_path: str, model_cfg_path: str, model_weights_path: str, boxTreshold: float, textTreshold: float, dst_dir: str):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Grounding DINO labeler ver.2024.02.14')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] src_dir: `{src_dir}`')
  print(f'[INFO] classes_path: `{classes_path}`')
  print(f'[INFO] model_cfg_path: `{model_cfg_path}`')
  print(f'[INFO] model_weights_path: `{model_weights_path}`')
  print(f'[INFO] boxTreshold: {boxTreshold}')
  print(f'[INFO] textTreshold: {textTreshold}')
  print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 01. удаление dst_dir и создание директория для результатов и контрольная отрисовка bbox
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  img_dir = os.path.join(dst_dir, 'images')
  lbl_dir = os.path.join(dst_dir, 'labels')
  bbox_dir = os.path.join(dst_dir, 'bounding_boxes')
  print(f'[INFO] img_dir: `{img_dir}`')
  print(f'[INFO] lbl_dir: `{lbl_dir}`')
  print(f'[INFO] bbox_dir: `{bbox_dir}`')
  make_directory(img_dir)
  make_directory(lbl_dir)
  make_directory(bbox_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирую 'classes.txt' для работы в LabelImg
  txt_fname2 = os.path.join(lbl_dir, 'classes.txt')
  # print(f'[INFO] txt_fname2: `{txt_fname2}`')
  copy_file_with_new_name(classes_path, txt_fname2)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 02. определение списка поддерживаемых классов из файла classes.txt
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  class_lst = read_classes(classes_path)
  if len(class_lst) < 1:
    print(f'[ERROR] List of classes is empty: `{classes_path}`')
    return
  print(f'[INFO] classes_lst: len: {len(class_lst)}, `{class_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 03. создание prompt по списку classes.txt
  #~~~~~~~~~~~~~~~~~~~~~~~~
  textPrompt = get_text_prompt(class_lst)
  print('~'*70)
  print(f'[INFO] textPrompt: `{textPrompt}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 04. извлечение кадров из видео-файлов, указанных в src_dir, и сохранених их в dst_dir
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] label images...')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список файлов изображений
  img_lst =  get_image_list(src_dir)
  img_lst_len = len(img_lst)
  if img_lst_len < 1:
    print(f'[ERROR] List of images is empty: `{src_dir}`')
    return
  # print(f'[INFO]  image files: len: {img_lst_len}, `{img_lst}`')
  print(f'[INFO]  image files: len: {img_lst_len}')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ объект-модель Grounding DINO
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
  model = load_model(model_cfg_path, model_weights_path)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ определяем 20 различных цветов в формате BGR
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

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ пробегаюсь по всем изображениям в папке и размечаю их в формате YOLO
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(img_lst_len):
    print('~'*70)
    print(f'[INFO] {i}: {img_lst[i]}')
    src_fname = os.path.join(src_dir, img_lst[i])
    # print(f'[INFO]     `{src_fname}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ загружаем файл в модель
    image_source, image = load_image(src_fname)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=textPrompt,
      box_threshold=boxTreshold,
      text_threshold=textTreshold
      )
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ пробегаем по всем детектированным bbox
    boxes_len = len(boxes)
    if boxes_len < 1:
      continue
    if not boxes_len == len(logits):
      continue
    if not boxes_len == len(phrases):
      continue
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ копирую оригинальное изображение
    img_fname2 = os.path.join(img_dir, img_lst[i])
    # print(f'[INFO] {src_fname} -> {img_fname2}')
    copy_file_with_new_name(src_fname, img_fname2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ записываем координаты в формате YOLO и отрисовываем bbox
    #~~~~~~~~~~~~~~~~~~~~~~~~
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    # print(f'[INFO] base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
    txt_fname2 = os.path.join(lbl_dir, base_fname + '.txt')
    # print(f'[INFO] txt_fname2: `{txt_fname2}`')
    #~ изображение с отрисованными bbox
    img_fname3 = os.path.join(bbox_dir, img_lst[i])
    # print(f'[INFO] img_fname3: `{img_fname3}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ читаем исходное изображение для отрисовки рамок
    frame = cv2.imread(src_fname)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    # print(f'[INFO] src_fname: `{src_fname}`, frame: width: {frame_width}, height: {frame_height}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    with open(txt_fname2, 'w', encoding='utf-8') as lbl_file:
      for j in range(boxes_len):
        print(f'[INFO]  {j}->{boxes_len}')
        # print(f'[INFO]   boxes[{j}]: {boxes[j]}')
        # print(f'[INFO]   logits[{j}]: {logits[j]}')
        score = logits[j].item()
        score_str = str(round(score, 2))
        print(f'[INFO]   score: {score_str}')
        print(f'[INFO]   phrases: {phrases[j]}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        if not phrases[j] in class_lst:
          continue
        #~ есть такой класс в списке сохраняю изображение
        class_inx = class_lst.index(phrases[j])
        print(f'[INFO]   class_inx: {class_inx}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        x_center, y_center, width, height = boxes[j]
        # print(f'[INFO]   x_center: {x_center} {type(x_center)}, y_center: {y_center} {type(y_center)}, width: {width} {type(width)}, height: {height} {type(height)}')
        #~ преобразование тензоров в числовые значения
        x_cen_val = x_center.item()
        y_cen_val = y_center.item()
        width_val = width.item()
        height_val = height.item()
        # print(f'[INFO]   x_cen_val: {x_cen_val}, y_cen_val: {y_cen_val}, width_val: {width_val}, height_val: {height_val}')
        #~ создание строки для записи в файл
        line_str = f'{class_inx} {x_cen_val} {y_cen_val} {width_val} {height_val}\n'
        #~ запись строки в файл
        lbl_file.write(line_str)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ и отрисовываем bbox
        x_min = int((x_cen_val - width_val/2) * frame_width)
        y_min = int((y_cen_val - height_val/2) * frame_height)
        x_max = int((x_cen_val + width_val/2) * frame_width)
        y_max = int((y_cen_val + height_val/2) * frame_height)
        # print(f'[INFO]   bbox: x: {x_min}..{x_max}, y: {y_min}..{y_max}')
        # print(f'[INFO]   x_min: {type(x_min)}, y_min: {type(y_min)}, x_max: {type(x_max)}, y_max: {type(y_max)}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        class_name = 'nemo'
        inxcolor = (255, 255, 255)
        if 0 <= class_inx and class_inx < len(color_lst):
          inxcolor = color_lst[class_inx] 
        if 0 <= class_inx and class_inx < len(class_lst):
          class_name = class_lst[class_inx] 
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ отрисовываем bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), inxcolor, 2)
        #~ добавляем подпись класса объекта
        obj_label = f'{score_str} {class_name}'
        cv2.putText(frame, obj_label, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, inxcolor, 2)
        cv2.putText(frame, obj_label, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        cv2.imwrite(img_fname3, frame)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ файлы все отбработаны
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] images+labels save to the directory: `{dst_dir}`')
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
  parser = argparse.ArgumentParser(description='Grounding DINO labeler.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--classes_path', type=str, default='', help='Path to the classes file')
  parser.add_argument('--model_cfg_path', type=str, default='', help='Path to the model config file')
  parser.add_argument('--model_weights_path', type=str, default='', help='Path to the model weights file')
  parser.add_argument('--box_treshold', type=float, default='0.35', help='Box treshold')
  parser.add_argument('--text_treshold', type=float, default='0.25', help='Text treshold')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  step05_grounding_dino(args.src_dir, args.classes_path, args.model_cfg_path, args.model_weights_path, args.box_treshold, args.text_treshold, args.dst_dir)