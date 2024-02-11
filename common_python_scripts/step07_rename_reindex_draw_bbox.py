#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! в именах директорий, файлов не допускается использовать пробелы и спецсимволы,
#~ так как они передаются через параметры командной строки !!!
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --src_dir -> директория с входными данными, содержит директории с парными файлами jpg(jpeg,png,bmp,tif,tiff)+txt,
#~              либо в этих директориях поддиректории images и labels 
#~ --matches_fname -> путь к файлу соответствия-замены id
#~ --dst_dir -> директория с результатами
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ последовательность выполняемых скриптом действий:
#~ 01. удаление dst_dir
#~ 02. определение списков соответствия-замены id из файла class_matches.txt
#~ 03. копирование пар image+label из src_dir в результирующую директорию dst_dir c изменением на уникальные имена
#~     и изменением индексов объектов в соответствии с class_matches
#~ 04. отрисовка bounding boxes по результирующим данным
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step07_rename_reindex_draw_bbox.py --src_dir d:/perimeter_raw_dataset1 --matches_fname d:/my_campy/SafeCity_Voronezh/dataset_preparation/class_matches.txt --dst_dir d:/perimeter_dataset
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import shutil
import uuid
import time
import cv2
import random
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
def delete_file(file_name: str):
  if os.path.exists(file_name):
    os.remove(file_name)
    # print(f'[INFO] File was successfully deleted: `{file_name}`')
  # else:
  #   print(f'[WARNING] File does not exist: `{file_name}`')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_file_with_new_name(source_file, destination_file):
  try:
    shutil.copyfile(source_file, destination_file)
  except FileNotFoundError:
    print('[ERROR] The file was not found')
  except Exception as e:
    print(f'[ERROR] {e}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_file_existence(file_path: str) -> bool:
  return os.path.exists(file_path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_base_suffix_filepath(file_path: str) -> tuple:
  #~ получаем имя файла из полного пути
  file_name = os.path.basename(file_path)
  #~ разделяем имя файла и расширение
  base_fname, suffix_fname = os.path.splitext(file_name)
  return base_fname,suffix_fname

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
def get_subdir_list(src_dir: str):
  subdir_lst = []
  for fname in os.listdir(src_dir):
    if not os.path.isfile(os.path.join(src_dir, fname)):
      subdir_lst.append(fname)
  return subdir_lst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_txt_list(src_dir: str):
  txt_lst = []
  for fname in os.listdir(src_dir):
    if os.path.isfile(os.path.join(src_dir, fname)):
      if fname.lower().endswith(('.txt')):
        txt_lst.append(fname)
  return txt_lst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_class_matches(file_path: str):
  first_lst = []
  second_lst = []
  if not file_path:
    return first_lst, second_lst
  if not check_file_existence(file_path):
    return first_lst, second_lst
  #~~~~~~~~~~~~~~~~~~~~~~~~
  with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      parts = line.split('|')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if len(parts) != 2:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      try:
        first_num = int(parts[0].strip())
        second_num = int(parts[1].strip())
        first_lst.append(first_num)
        second_lst.append(second_num)
      except ValueError:
        continue
  #~~~~~~~~~~~~~~~~~~~~~~~~
  return first_lst, second_lst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reindex_lbl(txt_fname1: str, id_lst1, id_lst2, txt_fname2: str):
  with open(txt_fname1, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()
    # print('-'*50)
    # print(f'[INFO] lines: len: {len(lines)}: `{lines}`')
    filtered_lines = []
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for line in lines:
      #~ удаляем пробелы в начале и конце строки
      line2 = line.strip()
      # print(f'[INFO]  -->line: `{line}`, line2: `{line2}`')
      fields5 = line2.split()
      # print(f'[INFO]    ==>fields5: len: {len(fields5)}: `{fields5}`, fields5[0]: `{fields5[0]}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not 5 == len(fields5):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      obj_id = int(fields5[0])
      inx = -1
      for i in range(len(id_lst1)):
        if obj_id == id_lst1[i]:
          inx = i
          break
      if -1 == inx:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ изменяем индекс класса объекта
      fields5[0] = str(id_lst2[inx])
      # print(f'[INFO]    --->fields5: len: {len(fields5)}: `{fields5}`, fields5[0]: `{fields5[0]}`')
      filtered_lines.append(' '.join(fields5))
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ если есть строки для указанного класса объектов
    # print(f'[INFO] filtered_lines: len: {len(filtered_lines)}')
    if filtered_lines:
      with open(txt_fname2, 'w', encoding='utf-8') as output_file:
        for fline in filtered_lines:
          output_file.write(fline + '\n')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ is_pair = True -> image+label в одной папке
#~           False -> image+label в разных папках images и labels, но на одном уровне
def copy2_img_lbl_pairs(src_dir: str, img_lst, id_lst1, id_lst2, dst_dir: str, is_pair: bool):
  img_lst_len = len(img_lst)
  if img_lst_len < 1:
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_fname1 = ''
  txt_fname1 = ''
  for i in range(img_lst_len):
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    # print(f'[INFO] {i}->{img_lst_len}: `{img_lst[i]}`, base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
    if is_pair:
      img_fname1 = os.path.join(src_dir, img_lst[i])
      txt_fname1 = os.path.join(src_dir, base_fname + '.txt')
    else:
      img_fname1 = os.path.join(src_dir, 'images', img_lst[i])
      txt_fname1 = os.path.join(src_dir, 'labels', base_fname + '.txt')
    if not check_file_existence(img_fname1):
      continue
    if not check_file_existence(txt_fname1):
      continue
    # print(f'[INFO]   img_fname1: `{img_fname1}`')
    # print(f'[INFO]   txt_fname1: `{txt_fname1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    unic_fname = str(uuid.uuid1()) 
    img_fname2 = os.path.join(dst_dir, unic_fname + suffix_fname)
    txt_fname2 = os.path.join(dst_dir, unic_fname + '.txt')
    # print(f'[INFO]   img_fname2: `{img_fname2}`')
    # print(f'[INFO]   txt_fname2: `{txt_fname2}`')
    copy_file_with_new_name(img_fname1, img_fname2)
    if len(id_lst1) < 1:
      copy_file_with_new_name(txt_fname1, txt_fname2)
    else:
      reindex_lbl(txt_fname1, id_lst1, id_lst2, txt_fname2)
      if not check_file_existence(txt_fname2):
        delete_file(img_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_img_lbl_pairs(src_dir: str, id_lst1, id_lst2, dst_dir: str):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] id_lst1: len: {len(id_lst1)}, `{id_lst1}`')
  # print(f'[INFO] id_lst2: len: {len(id_lst2)}, `{id_lst2}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst =  get_image_list(src_dir)
  subdir_lst = get_subdir_list(src_dir)
  # print(f'[INFO] img_lst: len: {len(img_lst)}, `{img_lst}`')
  # print(f'[INFO] subdir_lst: len: {len(subdir_lst)}, `{subdir_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из корневой папки
  copy2_img_lbl_pairs(src_dir, img_lst, id_lst1, id_lst2, dst_dir, True)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из корневой папки images+labels
  src_dir0 = os.path.join(src_dir, 'images')
  if os.path.exists(src_dir0):
    img_lst =  get_image_list(src_dir0)
    copy2_img_lbl_pairs(src_dir, img_lst, id_lst1, id_lst2, dst_dir, False)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из подпапок
  subdir_lst_len = len(subdir_lst)
  if subdir_lst_len < 1:
    return
  for i in range(subdir_lst_len):
    src_dir1 = os.path.join(src_dir, subdir_lst[i])
    img_lst =  get_image_list(src_dir1)
    # print(f'[INFO] {i}->{subdir_lst_len}: `{subdir_lst[i]}`, src_dir1: `{src_dir1}`')
    # print(f'[INFO]   img_lst: len: {len(img_lst)}, `{img_lst}`')
    copy2_img_lbl_pairs(src_dir1, img_lst, id_lst1, id_lst2, dst_dir, True)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    subdir_lst2 = get_subdir_list(src_dir1)
    # print(f'[INFO]   subdir_lst2: len: {len(subdir_lst2)}, subdir_lst2: `{subdir_lst2}`')
    if 2 == len(subdir_lst2):
      src_dir3 = os.path.join(src_dir1, 'images')
      img_lst =  get_image_list(src_dir3)
      # print(f'[INFO]     img_lst: len: {len(img_lst)}, img_lst: `{img_lst}`')
      copy2_img_lbl_pairs(src_dir1, img_lst, id_lst1, id_lst2, dst_dir, False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_bounding_boxes(src_dir: str, dst_dir: str):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
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
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst =  get_image_list(dst_dir)
  img_lst_len = len(img_lst)
  # print(f'[INFO] img_lst_len: {img_lst_len}, img_lst: `{img_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_dir2 = os.path.join(dst_dir, 'bounding_boxes')
  # print(f'[INFO] img_dir2: `{img_dir2}`')
  make_directory(img_dir2)
  for i in range(img_lst_len):
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    img_fname1 = os.path.join(dst_dir, img_lst[i])
    txt_fname1 = os.path.join(dst_dir, base_fname + '.txt')
    img_fname2 = os.path.join(img_dir2, img_lst[i])
    # print(f'[INFO] img_fname1: `{img_fname1}`')
    # print(f'[INFO] txt_fname1: `{txt_fname1}`')
    # print(f'[INFO] img_fname2: `{img_fname2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    image = cv2.imread(img_fname1)
    with open(txt_fname1, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        x_min = int((x_center - width/2) * image.shape[1])
        y_min = int((y_center - height/2) * image.shape[0])
        x_max = int((x_center + width/2) * image.shape[1])
        y_max = int((y_center + height/2) * image.shape[0])
        #~~~~~~~~~~~~~~~~~~~~~~~~
        class_id_int = int(class_id)
        class_id_str = str(class_id_int)
        # print(f'[INFO] class_id: {class_id}, class_id_int: {class_id_int}, class_id_str: `{class_id_str}`')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        inxcolor = (255, 255, 255)
        if 0 <= class_id_int and class_id_int < len(color_lst):
          inxcolor = color_lst[class_id_int] 
        #~~~~~~~~~~~~~~~~~~~~~~~~
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), inxcolor, 2)
        #~ добавляем подпись класса объекта
        cv2.putText(image, class_id_str, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, inxcolor, 1)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        cv2.imwrite(img_fname2, image)

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
def main_step07_rename_reindex_draw_bbox(src_dir: str, matches_fname: str, dst_dir: str):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Rename, Reindex and Draw BBox YOLOv8 ver.2024.02.11')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] src_dir: `{src_dir}`')
  print(f'[INFO] matches_fname: `{matches_fname}`')
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
  #~ 02. определение списков соответствия-замены id из файла class_matches.txt
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  id_lst1 = []
  id_lst2 = []
  if matches_fname:
    id_lst1,id_lst2 = parse_class_matches(matches_fname)
    if len(id_lst1) < 1:
      print(f'[ERROR] List of class matches is empty: `{matches_fname}`')
      return
  print('[INFO] class_matches:')
  print(f'[INFO]  id_lst1: len: {len(id_lst1)}, `{id_lst1}`')
  print(f'[INFO]  id_lst2: len: {len(id_lst2)}, `{id_lst2}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 03. копирование пар image+label из src_dir в результирующую директорию dst_dir c изменением на уникальные имена
  #~     и изменением индексов объектов в соответствии с class_matches
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  copy_img_lbl_pairs(src_dir, id_lst1, id_lst2, dst_dir)
  pairs_count = len(get_txt_list(dst_dir))
  if pairs_count < 1:
    print(f'[ERROR] There is no files for further processing: `{dst_dir}`')
    return
  print(f'[INFO] merge pairs: `{dst_dir}`, pairs count: {pairs_count}')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 04. отрисовка bounding boxes по результирующим данным
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  draw_bounding_boxes(src_dir, dst_dir)
  print(f'[INFO] draw bounding boxes: `{dst_dir}`')
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
  parser.add_argument('--matches_fname', type=str, default='', help='Path to the class matches file')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main_step07_rename_reindex_draw_bbox(args.src_dir, args.matches_fname, args.dst_dir)