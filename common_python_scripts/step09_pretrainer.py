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
#~ --classes_fname -> путь к файлу классов, все txt файлы из src_dir должны быть этой классификации 
#~ --dst_dir -> директория с результатами
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ последовательность выполняемых скриптом действий:
#~ 01. удаление dst_dir
#~ 02. определение списка поддерживаемых классов из файла classes.txt
#~ 03. копирование пар image+label из src_dir во временную папку dst_dir/temp1 c изменением на уникальные имена
#~ 04. копирование и сортировка пар image+label из dst_dir/temp1 во временную папку dst_dir/temp2/индекс-класса
#~    индекс класса определяется по максимальному количеству строк с определенным индексом в файле разметки
#~ 05. нормализация изображений из папки dst_dir/temp2 (приведение к размерам img_size),
#~    если изображение меньше указанного размера img_size, то создается черные квадрат и оно вставляется в центр,
#~    если изображение больше указанного размера img_size, то оно пропорционально сжимается,
#~    после этого производится сохранение результатов в три подпапки train, valid, test
#~    временной папки dst_dir/temp3, в соответствии с указанными значениями соотношений пользователем
#~    (отдельно на три папки делится каждый класс и затем разделенные файлы добавляются в три общие папки,
#~    перед разделением файлы в каждой папке перемешиваются)
#~ 06. вторичное перемешивание в папках train, valid, test и копирование в результирующие
#~    train
#~      images
#~      labels
#~    valid
#~      images
#~      labels
#~    test
#~      images
#~      labels
#~ 07. отрисовка bounding boxes по результирующим данным
#~ 08. создание файла data.yaml
#~ 09. удаление временных папок
#~ 10. подсчет числа файлов
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step09_pretrainer.py --src_dir d:/perimeter_raw_dataset2 --classes_fname d:/my_campy/SafeCity_Voronezh/dataset_preparation/classes.txt --dst_dir d:/perimeter_dataset --img_size 640 --train_perc 70 --valid_perc 25
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import shutil
import uuid
import time
import cv2
from PIL import Image
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
def read_classes(file_name: str):
  with open(file_name, 'r', encoding='utf-8') as file:
    classes = [line.strip() for line in file if line.strip()]
  return classes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ is_pair = True -> image+label в одной папке
#~           False -> image+label в разных папках images и labels, но на одном уровне
def copy2_img_lbl_pairs(src_dir: str, img_lst, dst_dir: str, is_pair: bool):
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
    copy_file_with_new_name(txt_fname1, txt_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_img_lbl_pairs(src_dir: str, class_lst, dst_dir: str):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst =  get_image_list(src_dir)
  subdir_lst = get_subdir_list(src_dir)
  # print(f'[INFO] img_lst: len: {len(img_lst)}, `{img_lst}`')
  # print(f'[INFO] subdir_lst: len: {len(subdir_lst)}, `{subdir_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из корневой папки
  copy2_img_lbl_pairs(src_dir, img_lst, dst_dir, True)
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
    copy2_img_lbl_pairs(src_dir1, img_lst, dst_dir, True)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    subdir_lst2 = get_subdir_list(src_dir1)
    # print(f'[INFO]   subdir_lst2: len: {len(subdir_lst2)}, subdir_lst2: `{subdir_lst2}`')
    if 2 == len(subdir_lst2):
      src_dir3 = os.path.join(src_dir1, 'images')
      img_lst =  get_image_list(src_dir3)
      # print(f'[INFO]     img_lst: len: {len(img_lst)}, img_lst: `{img_lst}`')
      copy2_img_lbl_pairs(src_dir1, img_lst, dst_dir, False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_most_frequent_index(file_path: str) -> int:
  #~ cоздаем список для хранения индексов
  index_list = []
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
      #~ получаем индекс из первого поля в строке
      index = int(line.split()[0])
      #~ добавляем индекс в список
      index_list.append(index)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ получаем наиболее часто встречающийся индекс
  most_common_index = max(set(index_list), key=index_list.count)
  # print(f'[INFO] index_list: `{index_list}`, most_common_index: {most_common_index}')
  return most_common_index

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def separate_pairs(src_dir: str, dst_dir: str):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  img_lst = get_image_list(src_dir)
  img_lst_len = len(img_lst)
  if img_lst_len < 1:
    return
  # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
  for i in range(img_lst_len):
    # print(f'[INFO] {i}->{img_lst_len}: `{img_lst[i]}`')
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    # print(f'[INFO]   base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
    img_fname1 = os.path.join(src_dir, img_lst[i])
    txt_fname1 = os.path.join(src_dir, base_fname + '.txt')
    # print(f'[INFO]     img_fname1: `{img_fname1}`')
    # print(f'[INFO]     txt_fname1: `{txt_fname1}`')
    inx1 = get_most_frequent_index(txt_fname1)
    # print(f'[INFO]       inx1: {inx1}')
    dst_dir2 = os.path.join(dst_dir, str(inx1))
    # print(f'[INFO]       dst_dir2: {dst_dir2}')
    make_directory(dst_dir2)
    img_fname2 = os.path.join(dst_dir2, img_lst[i])
    txt_fname2 = os.path.join(dst_dir2, base_fname + '.txt')
    # print(f'[INFO]        img_fname2: {img_fname2}')
    # print(f'[INFO]        txt_fname2: {txt_fname2}')
    copy_file_with_new_name(img_fname1, img_fname2)
    copy_file_with_new_name(txt_fname1, txt_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def resize_and_save_image(src_img_fname: str, src_suffix_fname: str, dst_img_fname: str, img_size: int):
  # print('-'*70)
  # print(f'[INFO] src_img_fname: `{src_img_fname}`')
  # print(f'[INFO] src_suffix_fname: `{src_suffix_fname}`')
  # print(f'[INFO] dst_img_fname: `{dst_img_fname}`')
  # print(f'[INFO] img_size: {img_size}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ cоздаем черное изображение 640x640
  target_size = (img_size, img_size)
  target_image = Image.new('RGB', target_size, color='black')
  #~ открываем исходное изображение
  original_image = Image.open(src_img_fname)
  #~ получаем размеры исходного изображения
  original_width, original_height = original_image.size
  # print(f'[INFO] original_width: {original_width}, original_height: {original_height}')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ если размеры и расширение одинаковое, то просто копирую изображение
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if original_width == img_size and original_height == img_size:
    # print(f'[INFO] ============>src_suffix_fname: `{src_suffix_fname}`')
    if '.jpg' == src_suffix_fname:
      copy_file_with_new_name(src_img_fname, dst_img_fname)
      return img_size, img_size, img_size, img_size, 0, 0
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ проверяем размеры исходного изображения
  changed_width = -1
  changed_height = -1
  offset_x = 0
  offset_y = 0
  if original_width <= img_size and original_height <= img_size:
    #~ вставляем исходное изображение в центр черного изображения
    offset_x = (target_size[0] - original_width) // 2
    offset_y = (target_size[1] - original_height) // 2
    target_image.paste(original_image, (offset_x, offset_y))
  else:
    #~ cжимаем исходное изображение с сохранением пропорций
    original_image.thumbnail(target_size)
    changed_width = original_image.width
    changed_height = original_image.height
    #~ вычисляем смещение для вставки сжатого изображения
    offset_x = (target_size[0] - changed_width) // 2
    offset_y = (target_size[1] - changed_height) // 2
    #~ вставляем сжатое изображение в центр черного изображения
    target_image.paste(original_image, (offset_x, offset_y))
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ cохраняем измененное изображение в формате PNG
  # target_image.save(dst_img_fname, format='PNG')
  target_image.save(dst_img_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # print(f'[INFO] original_width: {original_width}, original_height: {original_height}')
  # print(f'[INFO] changed_width: {changed_width}, changed_height: {changed_height}')
  # print(f'[INFO] offset_x: {offset_x}, offset_y: {offset_y}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  return original_width, original_height, changed_width, changed_height, offset_x, offset_y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def change_yolo_markup(input_txt_path, original_width, original_height, changed_width, changed_height, offset_x, offset_y, img_size, output_txt_path):
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ считываем оригинальную разметку
  with open(input_txt_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # print('='*50)
  # print(f'[INFO] original_width: {original_width}, original_height: {original_height}')
  # print(f'[INFO] changed_width: {changed_width}, changed_height: {changed_height}')
  # print(f'[INFO] offset_x: {offset_x}, offset_y: {offset_y}')
  # print(f'[INFO] img_size: {img_size}')
  #~ производим трансформацию разметки и сохраняем ее в новый файл
  is_bbox = False
  with open(output_txt_path, 'w', encoding='utf-8') as file:
    for line in lines:
      values = line.strip().split(' ')
      class_label = values[0]
      x_center = float(values[1])
      y_center = float(values[2])
      width = float(values[3])
      height = float(values[4])
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # print('-'*50)
      # print(f'[INFO] class_label: `{class_label}`')
      # print(f'[INFO] x_center: {x_center}, y_center: {y_center}')
      # print(f'[INFO] width: {width}, height: {height}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if -1 == changed_width:
        x_center *= original_width
        y_center *= original_height
        width *= original_width
        height *= original_height
        # print(f'[INFO] x_center2: {x_center}, y_center2: {y_center}')
        # print(f'[INFO] width2: {width}, height2: {height}')
      else:
        x_center *= changed_width
        y_center *= changed_height
        width *= changed_width
        height *= changed_height
        # print(f'[INFO] x_center3: {x_center}, y_center3: {y_center}')
        # print(f'[INFO] width3: {width}, height3: {height}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ если изображение слишком маленькое, то не сохраняю этот bbox
      if width < 5:
        continue
      if height < 5:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      x_center = x_center + offset_x
      y_center = y_center + offset_y
      #~~~~~~~~~~~~~~~~~~~~~~~~
      x_center /= img_size
      y_center /= img_size
      width /= img_size
      height /= img_size
      #~~~~~~~~~~~~~~~~~~~~~~~~
      is_bbox = True
      new_line = f"{class_label} {x_center} {y_center} {width} {height}\n"
      file.write(new_line)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  return is_bbox

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pair_normalize_copy(img_fname1: str, suffix_fname1: str, txt_fname1: str, img_fname2: str, txt_fname2: str, img_size: int):
  # print('~'*70)
  # print(f'[INFO] img_fname1: `{img_fname1}`')
  # print(f'[INFO] suffix_fname1: `{suffix_fname1}`')
  # print(f'[INFO] txt_fname1: `{txt_fname1}`')
  # print(f'[INFO] img_fname2: `{img_fname2}`')
  # print(f'[INFO] txt_fname2: `{txt_fname2}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  original_width,original_height,changed_width,changed_height,offset_x,offset_y = resize_and_save_image(img_fname1, suffix_fname1, img_fname2, img_size)
  if original_width == img_size and original_width == changed_width and original_height == changed_height:
    copy_file_with_new_name(txt_fname1, txt_fname2)
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~
  is_bbox = change_yolo_markup(txt_fname1, original_width, original_height, changed_width, changed_height, offset_x, offset_y, img_size, txt_fname2)
  # print(f'[INFO] is_bbox: {is_bbox}')
  if not is_bbox:
    delete_file(img_fname2)
    delete_file(txt_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def image_normalization_separate(src_dir: str, dst_dir: str, img_size: int, train_percent: int, valid_percent: int):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] img_size: {img_size}')
  # print(f'[INFO] train_percent: {train_percent}')
  # print(f'[INFO] valid_percent: {valid_percent}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  train_percent2 = train_percent
  valid_percent2 = valid_percent
  test_percent2 = 100 - train_percent2 - valid_percent2
  if not 100 == (train_percent2 + valid_percent2 + test_percent2):
    train_percent2 = 70
    valid_percent2 = 25
    test_percent2 = 5
  #~~~~~~~~~~~~~~~~~~~~~~~~
  train_dir2 = os.path.join(dst_dir, 'train')
  valid_dir2 = os.path.join(dst_dir, 'valid')
  test_dir2 = os.path.join(dst_dir, 'test')
  make_directory(train_dir2)
  make_directory(valid_dir2)
  make_directory(test_dir2)
  # print(f'[INFO] train_dir2: `{train_dir2}`')
  # print(f'[INFO] valid_dir2: `{valid_dir2}`')
  # print(f'[INFO] test_dir2: {test_dir2}')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ обрабатываем каждую поддиректорию 
  #~~~~~~~~~~~~~~~~~~~~~~~~
  subdir_lst = get_subdir_list(src_dir)
  subdir_lst_len = len(subdir_lst)
  # print(f'[INFO] subdir_lst: len: {subdir_lst_len}, {subdir_lst}')
  if subdir_lst_len < 1:
    return
  for i in range(subdir_lst_len):
    src_dir1 = os.path.join(src_dir, subdir_lst[i])
    img_lst =  get_image_list(src_dir1)
    # print(f'[INFO] {i}->{subdir_lst_len}: `{subdir_lst[i]}`, src_dir1: `{src_dir1}`')
    # print(f'[INFO]   img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst)
    # print(f'[INFO]   shuffle: img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # Рассчитываем количество файлов для каждой части (train, valid, test)
    total_count = len(img_lst)
    train_count = int(total_count * train_percent2 / 100)
    valid_count = int(total_count * valid_percent2 / 100)
    # print(f'[INFO] total_count: {total_count}')
    # print(f'[INFO] train_percent2: {train_percent2}, train_count: {train_count}')
    # print(f'[INFO] valid_percent2: {valid_percent2}, valid_count: {valid_count}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ распределяем файлы по частям
    train_files = img_lst[:train_count]
    valid_files = img_lst[train_count:train_count + valid_count]
    test_files = img_lst[train_count + valid_count:]
    train_files_len = len(train_files)
    valid_files_len = len(valid_files)
    test_files_len = len(test_files)
    # print(f'[INFO]  train_files_len: {train_files_len}')
    # print(f'[INFO]  valid_files_len: {valid_files_len}')
    # print(f'[INFO]  test_files_len: {test_files_len}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ train
    for j in range(train_files_len):
      base_fname,suffix_fname = get_base_suffix_fname(train_files[j])
      # print(f'[INFO]   base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
      img_fname1 = os.path.join(src_dir, subdir_lst[i], train_files[j])
      txt_fname1 = os.path.join(src_dir, subdir_lst[i], base_fname + '.txt')
      # print(f'[INFO]     img_fname1: `{img_fname1}`')
      # print(f'[INFO]     txt_fname1: `{txt_fname1}`')
      img_fname2 = os.path.join(train_dir2, base_fname + '.jpg')
      txt_fname2 = os.path.join(train_dir2, base_fname + '.txt')
      # print(f'[INFO]     img_fname2: `{img_fname2}`')
      # print(f'[INFO]     txt_fname2: `{txt_fname2}`')
      pair_normalize_copy(img_fname1, suffix_fname, txt_fname1, img_fname2, txt_fname2, img_size)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ valid
    for j in range(valid_files_len):
      base_fname,suffix_fname = get_base_suffix_fname(valid_files[j])
      img_fname1 = os.path.join(src_dir, subdir_lst[i], valid_files[j])
      txt_fname1 = os.path.join(src_dir, subdir_lst[i], base_fname + '.txt')
      img_fname2 = os.path.join(valid_dir2, base_fname + '.jpg')
      txt_fname2 = os.path.join(valid_dir2, base_fname + '.txt')
      pair_normalize_copy(img_fname1, suffix_fname, txt_fname1, img_fname2, txt_fname2, img_size)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ test
    for j in range(test_files_len):
      base_fname,suffix_fname = get_base_suffix_fname(test_files[j])
      img_fname1 = os.path.join(src_dir, subdir_lst[i], test_files[j])
      txt_fname1 = os.path.join(src_dir, subdir_lst[i], base_fname + '.txt')
      img_fname2 = os.path.join(test_dir2, base_fname + '.jpg')
      txt_fname2 = os.path.join(test_dir2, base_fname + '.txt')
      pair_normalize_copy(img_fname1, suffix_fname, txt_fname1, img_fname2, txt_fname2, img_size)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def format_counter(counter: int, digits: int):
  counter_str = str(counter)
  formatted_counter = counter_str.zfill(digits)
  return formatted_counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_valid_test_shuffle_separate(src_dir: str, dst_dir: str):
  for folder in ['train', 'valid', 'test']:
    dir1 = os.path.join(src_dir, folder)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst =  get_image_list(dir1)
    img_lst_len = len(img_lst)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_dir2 = os.path.join(dst_dir, folder, 'images')
    lbl_dir2 = os.path.join(dst_dir, folder, 'labels')
    make_directory(img_dir2)
    make_directory(lbl_dir2)
    for i in range(img_lst_len):
      base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
      img_fname1 = os.path.join(dir1, img_lst[i])
      txt_fname1 = os.path.join(dir1, base_fname + '.txt')
      count_fname = f'f{format_counter(i, 7)}-'
      img_fname2 = os.path.join(img_dir2, count_fname+base_fname+suffix_fname)
      txt_fname2 = os.path.join(lbl_dir2, count_fname+base_fname+'.txt')
      copy_file_with_new_name(img_fname1, img_fname2)
      copy_file_with_new_name(txt_fname1, txt_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_bounding_boxes(src_dir: str, dst_dir: str, class_lst):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] class_lst: len: {len(class_lst)}, `{class_lst}`')
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
  for folder in ['train', 'valid', 'test']:
    img_dir1 = os.path.join(src_dir, folder, 'images')
    lbl_dir1 = os.path.join(src_dir, folder, 'labels')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst =  get_image_list(img_dir1)
    img_lst_len = len(img_lst)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_dir2 = os.path.join(dst_dir, folder)
    make_directory(img_dir2)
    for i in range(img_lst_len):
      base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
      img_fname1 = os.path.join(img_dir1, img_lst[i])
      txt_fname1 = os.path.join(lbl_dir1, base_fname + '.txt')
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
def create_yaml_file(dst_dir: str, class_lst):
  # train_img_path = os.path.join(dst_dir, 'train', 'images')
  # val_img_path = os.path.join(dst_dir, 'valid', 'images')
  # test_img_path = os.path.join(dst_dir, 'test', 'images')
  # print(f'[INFO] train_img_path: `{train_img_path}`')
  # print(f'[INFO] val_img_path: `{val_img_path}`')
  # print(f'[INFO] test_img_path: `{test_img_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  data_yaml_fname = os.path.join(dst_dir, 'data.yaml')
  # print(f'[INFO] data_yaml_fname: `{data_yaml_fname}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  try:
    # #~ удаляем файл, если он уже существует
    # if os.path.exists(data_yaml_fname):
    #   os.remove(data_yaml_fname)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ создаем и сохраняем файл data.yaml
    with open(data_yaml_fname, 'w', encoding='utf-8') as file_yaml:
      # #~ train
      # str_line = train_img_path.replace("/", "\\")
      # file_yaml.write(f'train: {str_line}\n')
      # #~ val
      # str_line = val_img_path.replace("/", "\\")
      # file_yaml.write(f'val: {str_line}\n')
      # #~ test
      # str_line = test_img_path.replace("/", "\\")
      # file_yaml.write(f'test: {str_line}\n\n')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      file_yaml.write('train: ../train/images\n')
      file_yaml.write('val: ../valid/images\n')
      file_yaml.write('test: ../test/images\n\n')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ number of classes
      file_yaml.write(f'nc: {len(class_lst)}\n\n')
      #~ class names
      file_yaml.write(f'names: {class_lst}')
    # print(f'[INFO] The list was successfully written to a file: {data_yaml_fname}')
  except Exception as e:
    print(f'[ERROR] Error writing the list to a file: {e}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calculate_pairs_count(src_dir: str):
  # print(f'[INFO] train_img_path: `{train_img_path}`')
  print('[INFO] number of files:')
  total_count = 0
  for folder in ['train', 'valid', 'test']:
    lbl_dir1 = os.path.join(src_dir, folder, 'labels')
    # print(f'[INFO] lbl_dir1: `{lbl_dir1}`')
    pairs_count1 = len(get_txt_list(lbl_dir1))
    print(f'[INFO]  {folder}: {pairs_count1}')
    total_count += pairs_count1
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('[INFO]  ---------------')
  print(f'[INFO]  total: {total_count}')

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
def main_step09_pretrainer(src_dir: str, classes_fname: str, dst_dir: str, img_size: int, train_percent: int, valid_percent: int):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Pretrainer YOLOv8 ver.2024.02.11')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] src_dir: `{src_dir}`')
  print(f'[INFO] classes_fname: `{classes_fname}`')
  print(f'[INFO] dst_dir: `{dst_dir}`')
  print(f'[INFO] img_size: {img_size}')
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
  if not read_classes(classes_fname):
    print(f'[ERROR] List of classes is empty: `{classes_fname}`')
    return
  print(f'[INFO] classes_lst: len: {len(class_lst)}, `{class_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 03. копирование пар image+label из src_dir во временную папку dst_dir/temp1 c изменением на уникальные имена
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  temp1_dir = os.path.join(dst_dir, 'temp1')
  copy_img_lbl_pairs(src_dir, class_lst, temp1_dir)
  pairs_count = len(get_txt_list(temp1_dir))
  if pairs_count < 1:
    print(f'[ERROR] There is no files for further processing: `{temp1_dir}`')
    return
  print(f'[INFO] merge pairs: `{temp1_dir}`, pairs count: {pairs_count}')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 04. копирование и сортировка пар image+label из dst_dir/temp1 во временную папку dst_dir/temp2/индекс-класса
  #~    индекс класса определяется по максимальному количеству строк с определенным индексом в файле разметки
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  temp2_dir = os.path.join(dst_dir, 'temp2')
  separate_pairs(temp1_dir, temp2_dir)
  print(f'[INFO] separate by classes: `{temp2_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 05. нормализация изображений из папки dst_dir/temp2 (приведение к размерам img_size) и сохранение результатов
  #~    во временной папке dst_dir/temp3, результаты сохраняются в три папки train, valid, test 
  #~    в соответствии с указанными значениями пользователем (отдельно на три папки делится каждый класс
  #~    и затем разделенные файлы складываются в три общие папки, перед разделением файлы в каждой папке перемешиваются)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  temp3_dir = os.path.join(dst_dir, 'temp3')
  image_normalization_separate(temp2_dir, temp3_dir, img_size, train_percent, valid_percent)
  print(f'[INFO] normalize and separate: `{temp3_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 06. вторичное перемешивание в папках train, valid, test и копирование в результирующие
  #~    train
  #~      images
  #~      labels
  #~    valid
  #~      images
  #~      labels
  #~    test
  #~      images
  #~      labels
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  train_valid_test_shuffle_separate(temp3_dir, dst_dir)
  print(f'[INFO] train-valid-test shuffle separate: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 07. отрисовка bounding boxes по результирующим данным
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  bbox_dir = os.path.join(dst_dir, 'bounding_boxes')
  draw_bounding_boxes(dst_dir, bbox_dir, class_lst)
  print(f'[INFO] draw bounding boxes: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 08. создание файла data.yaml
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  create_yaml_file(dst_dir, class_lst)
  print(f'[INFO] create yaml file: `{dst_dir}/data.yaml`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 09. удаление временных папок
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  delete_directory(temp1_dir)
  delete_directory(temp2_dir)
  delete_directory(temp3_dir)
  print('[INFO] temporary directories deleted')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 10. подсчет числа файлов
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  calculate_pairs_count(dst_dir)
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
  parser.add_argument('--classes_fname', type=str, default='', help='Path to the classes file')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  parser.add_argument('--img_size', type=int, default=640, help='Target image size')
  parser.add_argument('--train_perc', type=int, default=70, help='Train percent')
  parser.add_argument('--valid_perc', type=int, default=25, help='Valid percent')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main_step09_pretrainer(args.src_dir, args.classes_fname, args.dst_dir, args.img_size, args.train_perc, args.valid_perc)