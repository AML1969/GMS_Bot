#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! в именах директорий, файлов не допускается использовать пробелы и спецсимволы,
#~ так как они передаются через параметры командной строки !!!
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --src_dir d:/perimeter_raw_dataset1 -> директория с входными данными:
#~     либо с парными файлами:
#~      jpg(jpeg,png,bmp,tif,tiff)+txt и\или с поддиректориями images и labels,
#~     либо только графические файлы:
#~ --dst_dir d:/perimeter_dataset -> директория с результатами
#~ --program_mode 0 -> режим работы программы:
#~     0: сбор в одну папку, добавление уникального постфикса в имя файла
#~        и отрисовка bbox, если есть файлы разметки,
#~        если указан параметр similar_threshold, то дополнительно сравниваются соседние кадры,
#~        если текущий кадр похож на предыдущий более чем similar_threshold процентов,
#~        то не считаем такой кадр уникальным и не копируем его и не запоминаем его как уникальный
#~        режим добавлен, чтобы отфильтровать видео-кадры, которые дублируют себя
#~     --similar_threshold 0..100 -> порог похожести кадров в процентах
#~ --program_mode 1
#~     1: копирование по списку `bounding_boxes` -> только тех пар image+label из src_dir в dst_dir,
#~        имена которых есть в src_dir/bounding_boxes, подразумевается, что предварительно
#~        в папке `bounding_boxes` пользователь удалил избражения плохого качества
#~ --program_mode 2
#~     2: изменение индексов в файлах разметки, файл соответствия указан в параметре matches_fname
#~     --matches_fname -> путь к файлу соответствия-замены id
#~ --program_mode 3
#~     3: формирование итоговых данных для обучения yolo,
#~        итоговая-результирующая папка предварительно удаляется,
#~        создается директория dst_dir/индекс-класса, в нее копируются пары image+label,
#~        индекс класса определяется по максимальному количеству строк с определенным индексом в файле разметки,
#~        затем в каждой такой папке dst_dir/индекс-класса файлы перемешиваются и копируются в результирующие:
#~        train
#~          images
#~          labels
#~        valid
#~          images
#~          labels
#~        test
#~          images
#~          labels
#~        в процентном соотношении, которое указал пользователь:
#~     --train_perc 70 --valid_perc 25
#~     затем создается файл data.yaml на основании классов, указанных в classes_fname:
#~     --classes_fname -> путь к файлу классов
#~ --program_mode 4 
#~     4: добавление-копирование итоговых данных к имеющимся c перемешиванием
#~     --classes_fname -> путь к файлу классов

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ --program_mode 0
# python step05_yolov8_preparation.py --src_dir c:/perimeter_raw_dataset1 --dst_dir c:/perimeter_raw_dataset2 --program_mode 0
# python step05_yolov8_preparation.py --src_dir c:/perimeter_raw_dataset1 --dst_dir c:/perimeter_raw_dataset2 --program_mode 0 --similar_threshold 70
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --program_mode 1
# python step05_yolov8_preparation.py --src_dir c:/perimeter_raw_dataset2 --dst_dir c:/perimeter_raw_dataset3 --program_mode 1
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --program_mode 2
# python step05_yolov8_preparation.py --src_dir c:/perimeter_raw_dataset3 --dst_dir c:/perimeter_raw_dataset4 --program_mode 2 --matches_fname c:/my_campy/SafeCity_Voronezh/dataset_preparation/id_matches/weapon9_class_matches1.txt
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --program_mode 3
# python step05_yolov8_preparation.py --src_dir c:/perimeter_raw_dataset4 --dst_dir c:/perimeter_raw_dataset5 --program_mode 3 --train_perc 70 --valid_perc 25 --classes_fname c:/my_campy/SafeCity_Voronezh/dataset_preparation/classes.txt
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --program_mode 4
# python step05_yolov8_preparation.py --src_dir c:/perimeter_raw_dataset5 --dst_dir c:/perimeter_raw_dataset6 --program_mode 4 --classes_fname c:/my_campy/SafeCity_Voronezh/dataset_preparation/classes.txt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ 2024.02.18
# id   английское название   русское название
# 0    fire                  огонь
# 1    snowdrift             наледь/снег на крышах
# 2    garbage               мусор
# 3    snow                  снег (сугробы) на проезжей части/пешеходных дорожках
# 4    fight                 драка
# 5    accident              ДТП
# 6    oversized             негабаритный груз
# 7    fall                  падение человека
# 8    icicle                сосульки
# 9    weapon                человек с оружием
# 10   dogpoop               собачка в процессе
# 11   doggood               обычная собачка
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
def check_file_existence(file_path: str) -> bool:
  return os.path.exists(file_path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delete_file(file_name: str):
  if os.path.exists(file_name):
    os.remove(file_name)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_file_with_new_name(source_file, destination_file):
  if not check_file_existence(source_file):
    print(f'[WARNING] The file was not found: `{source_file}`')
    return
  try:
    shutil.copyfile(source_file, destination_file)
  except FileNotFoundError:
    print(f'[ERROR] The file was not found: `{source_file}`')
  except Exception as e:
    print(f'[ERROR] {e}')

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
  if not os.path.exists(src_dir):
    return img_lst
  for fname in os.listdir(src_dir):
    if os.path.isfile(os.path.join(src_dir, fname)):
      if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_lst.append(fname)
  return img_lst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_base_image_list(img_lst):
  base_img_lst = []
  img_lst_len = len(img_lst)
  if img_lst_len < 1:
    return base_img_lst
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(img_lst_len):
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    # print(f'[INFO] {i}->{img_lst_len}:')
    # print(f'[INFO]   base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
    base_img_lst.append(base_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  return base_img_lst

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
#~ функция для вычисления хэша изображения
def dhash(image, hash_size=8):
  resized = cv2.resize(image, (hash_size + 1, hash_size))
  diff = resized[:, 1:] > resized[:, :-1]
  return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ функция для проверки схожести двух изображений
# def is_similar(image1, image2, threshold=80):
def is_similar(image1, image2, similar_threshold):
  hash1 = dhash(image1)
  hash2 = dhash(image2)
  difference = bin(hash1 ^ hash2).count('1')
  similarity = (1 - difference / (8 * 8)) * 100
  return similarity >= similar_threshold

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy2_img_or_and_lbl(src_dir: str, dst_dir: str, similar_threshold: int):
  make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  unique_frame = None
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_dir_lst = []
  lbl_dir_lst = []
  img_dir_lst.append(src_dir)
  lbl_dir_lst.append(src_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_dir = os.path.join(src_dir, 'images')
  lbl_dir = os.path.join(src_dir, 'labels')
  img_dir_lst.append(img_dir)
  lbl_dir_lst.append(lbl_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(len(img_dir_lst)):
    # print(f'[INFO]  {i}:')
    # print(f'[INFO]   img_dir: `{img_dir_lst[i]}`')
    # print(f'[INFO]   lbl_dir: `{lbl_dir_lst[i]}`')
    img_lst =  get_image_list(img_dir_lst[i])
    img_lst_len = len(img_lst)
    # print(f'[INFO]   img_lst: len: {img_lst_len}, `{img_lst}`')
    if img_lst_len < 1:
      continue
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_fname1 = ''
    txt_fname1 = ''
    for j in range(img_lst_len):
      base_fname,suffix_fname = get_base_suffix_fname(img_lst[j])
      img_fname1 = os.path.join(img_dir_lst[i], img_lst[j])
      if not check_file_existence(img_fname1):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # print('[INFO]')
      if similar_threshold > 0:
        if unique_frame is None:
          unique_frame = cv2.imread(img_fname1)
          # print(f'[INFO] 0===>unique_frame: {img_fname1}')
        else:
          # print(f'[INFO] 1===>non none: {img_fname1}')
          frame = cv2.imread(img_fname1)
          if is_similar(frame, unique_frame, similar_threshold):
            # print(f'[INFO] 2--->is_similar: {img_fname1}')
            continue
          else:
            unique_frame = cv2.imread(img_fname1)
            # print(f'[INFO] 3+++>unique_frame: {img_fname1}')
        # print(f'[INFO] 4~~~>save-image: {img_fname1}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      unic_fname = str(uuid.uuid1())
      img_fname2 = os.path.join(dst_dir, base_fname + '-'+ unic_fname + suffix_fname)
      copy_file_with_new_name(img_fname1, img_fname2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      txt_fname1 = os.path.join(lbl_dir_lst[i], base_fname + '.txt')
      if not check_file_existence(txt_fname1):
        continue
      txt_fname2 = os.path.join(dst_dir, base_fname + '-'+ unic_fname + '.txt')
      copy_file_with_new_name(txt_fname1, txt_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_img_or_and_lbl(src_dir: str, dst_dir: str, similar_threshold: int):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из корневой папки
  copy2_img_or_and_lbl(src_dir, dst_dir, similar_threshold)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из подпапок
  subdir_lst = get_subdir_list(src_dir)
  subdir_lst_len = len(subdir_lst)
  # print(f'[INFO] subdir_lst: len: {subdir_lst_len}, `{subdir_lst}`')
  if subdir_lst_len < 1:
    return
  for i in range(subdir_lst_len):
    if 'images' == subdir_lst[i]:
      continue
    if 'labels' == subdir_lst[i]:
      continue
    src_dir1 = os.path.join(src_dir, subdir_lst[i])
    # print(f'[INFO] {i}->{subdir_lst_len}: `{subdir_lst[i]}`')
    # print(f'[INFO]  `{src_dir1}`->`{dst_dir}`')
    copy2_img_or_and_lbl(src_dir1, dst_dir, similar_threshold)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_pairs_by_bbox(src_dir: str, bbox_dir: str, dst_dir: str):
  # print('[INFO] copy_pairs_by_bbox')
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] bbox_dir: `{bbox_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst1 =  get_image_list(src_dir)
  if len(img_lst1) < 1:
    print(f'[WARNING] input folder is empty: `{src_dir}`')
    return
  # print(f'[INFO] img_lst1: len: {len(img_lst1)}, `{img_lst1}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  base_img_lst1 = get_base_image_list(img_lst1)
  if len(base_img_lst1) < 1:
    print(f'[WARNING] input folder is empty: `{src_dir}`')
    return
  # print(f'[INFO] base_img_lst1: len: {len(base_img_lst1)}, `{base_img_lst1}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst2 =  get_image_list(bbox_dir)
  img_lst_len2 = len(img_lst2)
  if img_lst_len2 < 1:
    print(f'[WARNING] input bounding boxes folder is empty: `{bbox_dir}`')
    return
  # print(f'[INFO] img_lst2: len: {img_lst_len2}, `{img_lst2}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(img_lst_len2):
    base_fname,suffix_fname = get_base_suffix_fname(img_lst2[i])
    # print(f'[INFO] {i}->{img_lst_len2}:')
    # print(f'[INFO]   base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
    if not base_fname in base_img_lst1:
      continue
    inx = base_img_lst1.index(base_fname)
    img_name1 = img_lst1[inx]
    # print('[INFO]')
    # print(f'[INFO] base_fname: `{base_fname}`, inx: {inx}, img_name1: `{img_name1}`')
    img_fname1 = os.path.join(src_dir, img_name1)
    # print(f'[INFO] img_fname1: `{img_fname1}`')
    if not check_file_existence(img_fname1):
      continue
    txt_fname1 = os.path.join(src_dir, base_fname + '.txt')
    # print(f'[INFO] txt_fname1: `{txt_fname1}`')
    if not check_file_existence(txt_fname1):
      continue
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_fname2 = os.path.join(dst_dir, img_name1)
    # print(f'[INFO] img_fname2: `{img_fname2}`')
    copy_file_with_new_name(img_fname1, img_fname2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    txt_fname2 = os.path.join(dst_dir, base_fname + '.txt')
    # print(f'[INFO] txt_fname2: `{txt_fname2}`')
    copy_file_with_new_name(txt_fname1, txt_fname2)

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
def reindex_lbl(txt_fname1: str, txt_fname2: str, id_lst1, id_lst2) -> bool:
  filtered_lines = []
  #~~~~~~~~~~~~~~~~~~~~~~~~
  with open(txt_fname1, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()
    # print('-'*50)
    # print(f'[INFO] lines: len: {len(lines)}: `{lines}`')
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
  if len(filtered_lines) > 0:
    with open(txt_fname2, 'w', encoding='utf-8') as output_file:
      for fline in filtered_lines:
        output_file.write(fline + '\n')
    return True
  else:
    return False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reindex_labels(src_dir: str, dst_dir: str, matches_fname: str):
  # print('[INFO] reindex_labels')
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] matches_fname: `{matches_fname}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  id_lst1 = []
  id_lst2 = []
  id_lst1,id_lst2 = parse_class_matches(matches_fname)
  if len(id_lst1) < 1:
    print(f'[ERROR] List of class matches is empty: `{matches_fname}`')
    return
  print('[INFO] class_matches:')
  print(f'[INFO]  id_lst1: len: {len(id_lst1)}, {id_lst1}')
  print(f'[INFO]  id_lst2: len: {len(id_lst2)}, {id_lst2}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst =  get_image_list(src_dir)
  img_lst_len = len(img_lst)
  if img_lst_len < 1:
    print(f'[WARNING] input folder is empty: `{src_dir}`')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(img_lst_len):
    base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
    # print(f'[INFO] {i}->{img_lst_len}:')
    # print(f'[INFO]   base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
    img_fname1 = os.path.join(src_dir, img_lst[i])
    if not check_file_existence(img_fname1):
      continue
    txt_fname1 = os.path.join(src_dir, base_fname + '.txt')
    if not check_file_existence(txt_fname1):
      continue
    # print(f'[INFO]   img_fname1: `{img_fname1}`')
    # print(f'[INFO]   txt_fname1: `{txt_fname1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_fname2 = os.path.join(dst_dir, img_lst[i])
    txt_fname2 = os.path.join(dst_dir, base_fname + '.txt')
    # print(f'[INFO]   img_fname2: `{img_fname2}`')
    # print(f'[INFO]   txt_fname2: `{txt_fname2}`')
    res12 = reindex_lbl(txt_fname1, txt_fname2, id_lst1, id_lst2)
    if res12:
      copy_file_with_new_name(img_fname1, img_fname2)

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
def separate_category_pairs(src_dir: str, dst_dir: str):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
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
def format_counter(counter: int, digits: int):
  counter_str = str(counter)
  formatted_counter = counter_str.zfill(digits)
  return formatted_counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def shuffle_tvt_separate_pairs(src_dir: str, dst_dir: str, train_percent: int, valid_percent: int):
  # print('[INFO] shuffle_tvt_separate_pairs')
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
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
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ обрабатываем каждую поддиректорию 
  #~~~~~~~~~~~~~~~~~~~~~~~~
  subdir_lst = get_subdir_list(src_dir)
  subdir_lst_len = len(subdir_lst)
  print(f'[INFO] subdir_lst: len: {subdir_lst_len}, {subdir_lst}')
  if subdir_lst_len < 1:
    return
  for i in range(subdir_lst_len):
    train_img_dir2 = os.path.join(dst_dir, 'train', 'images')
    valid_img_dir2 = os.path.join(dst_dir, 'valid', 'images')
    test_img_dir2 = os.path.join(dst_dir, 'test', 'images')
    make_directory(train_img_dir2)
    make_directory(valid_img_dir2)
    make_directory(test_img_dir2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    train_lbl_dir2 = os.path.join(dst_dir, 'train', 'labels')
    valid_lbl_dir2 = os.path.join(dst_dir, 'valid', 'labels')
    test_lbl_dir2 = os.path.join(dst_dir, 'test', 'labels')
    make_directory(train_lbl_dir2)
    make_directory(valid_lbl_dir2)
    make_directory(test_lbl_dir2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    subsrc_dir = os.path.join(src_dir, subdir_lst[i])
    img_lst =  get_image_list(subsrc_dir)
    img_lst_len = len(img_lst)
    # print(f'[INFO] {i}->{subdir_lst_len}: `{subdir_lst[i]}`, subsrc_dir: `{subsrc_dir}`')
    # print(f'[INFO]   img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst)
    # print(f'[INFO]   shuffle: img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # Рассчитываем количество файлов для каждой части (train, valid, test)
    train_count = int(img_lst_len * train_percent2 / 100)
    valid_count = int(img_lst_len * valid_percent2 / 100)
    # print(f'[INFO] img_lst_len: {img_lst_len}')
    # print(f'[INFO] train_percent2: {train_percent2}, train_count: {train_count}')
    # print(f'[INFO] valid_percent2: {valid_percent2}, valid_count: {valid_count}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for j in range(img_lst_len):
      # print(f'[INFO] {j}->{img_lst_len}:')
      base_fname,suffix_fname = get_base_suffix_fname(img_lst[j])
      # print(f'[INFO]   base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
      img_fname1 = os.path.join(src_dir, subdir_lst[i], img_lst[j])
      if not check_file_existence(img_fname1):
        continue
      txt_fname1 = os.path.join(src_dir, subdir_lst[i], base_fname + '.txt')
      if not check_file_existence(txt_fname1):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      dst_img_dir2 = train_img_dir2
      dst_lbl_dir2 = train_lbl_dir2
      inx = j
      if j < train_count:
        dst_img_dir2 = train_img_dir2
        dst_lbl_dir2 = train_lbl_dir2
      elif j < train_count + valid_count:
        dst_img_dir2 = valid_img_dir2
        dst_lbl_dir2 = valid_lbl_dir2
        inx = j - train_count
      else:
        dst_img_dir2 = test_img_dir2
        dst_lbl_dir2 = test_lbl_dir2
        inx = j - train_count - valid_count
      count_fname = f'f{format_counter(inx, 7)}-'
      unic_fname = str(uuid.uuid1())
      img_fname2 = os.path.join(dst_img_dir2, count_fname+unic_fname+suffix_fname)
      txt_fname2 = os.path.join(dst_lbl_dir2, count_fname+unic_fname+'.txt')
      copy_file_with_new_name(img_fname1, img_fname2)
      copy_file_with_new_name(txt_fname1, txt_fname2)
      # print(f'[INFO]   img_fname1: `{img_fname1}`')
      # print(f'[INFO]   txt_fname1: `{txt_fname1}`')
      # print(f'[INFO]   img_fname2: `{img_fname2}`')
      # print(f'[INFO]   txt_fname2: `{txt_fname2}`')
      copy_file_with_new_name(img_fname1, img_fname2)
      copy_file_with_new_name(txt_fname1, txt_fname2)

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
def create_yaml_file(classes_fname: str, dst_dir: str):
  # print('[INFO] create_yaml_file')
  # print(f'[INFO] classes_fname: `{classes_fname}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  class_lst = read_classes(classes_fname)
  if len(class_lst) < 1:
    print(f'[ERROR] List of classes is empty: `{classes_fname}`')
    return
  print(f'[INFO] classes_lst: len: {len(class_lst)}, `{class_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # train_img_path = os.path.join(dst_dir, 'train', 'images')
  # val_img_path = os.path.join(dst_dir, 'valid', 'images')
  # test_img_path = os.path.join(dst_dir, 'test', 'images')
  # print(f'[INFO] train_img_path: `{train_img_path}`')
  # print(f'[INFO] val_img_path: `{val_img_path}`')
  # print(f'[INFO] test_img_path: `{test_img_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  data_yaml_fname = os.path.join(dst_dir, 'data.yaml')
  print(f'[INFO] data_yaml_fname: `{data_yaml_fname}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  try:
    #~ удаляем файл, если он уже существует
    delete_file(data_yaml_fname)
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
#~ tvt - train, valid, test
def draw_bounding_boxes(src_dir: str, dst_dir: str, tvt: bool, classes_fname: str = ''):
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] tvt: {tvt}')
  # print(f'[INFO] classes_fname: `{classes_fname}`')
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
  color_lst_len = len(color_lst)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  class_lst = read_classes(classes_fname)
  class_lst_len = len(class_lst)
  if class_lst_len < 1:
    class_lst_len = 0
  # print(f'[INFO] classes_lst: len: {class_lst_len}, `{class_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for folder in ['train', 'valid', 'test']:
    img_dir1 = src_dir
    lbl_dir1 = src_dir
    if tvt:
      img_dir1 = os.path.join(src_dir, folder, 'images')
      lbl_dir1 = os.path.join(src_dir, folder, 'labels')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst =  get_image_list(img_dir1)
    img_lst_len = len(img_lst)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_dir2 = dst_dir
    if tvt:
      img_dir2 = os.path.join(dst_dir, folder)
    make_directory(img_dir2)
    for i in range(img_lst_len):
      base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
      txt_fname1 = os.path.join(lbl_dir1, base_fname + '.txt')
      if not check_file_existence(txt_fname1):
        continue
      img_fname1 = os.path.join(img_dir1, img_lst[i])
      img_fname2 = os.path.join(img_dir2, img_lst[i])
      # print(f'[INFO] img_fname1: `{img_fname1}`')
      # print(f'[INFO] txt_fname1: `{txt_fname1}`')
      # print(f'[INFO] img_fname2: `{img_fname2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      image = cv2.imread(img_fname1)
      with open(txt_fname1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
          fields5 = line.split()
          if not 5 == len(fields5):
            continue
          # class_id, x_center, y_center, width, height = map(float, line.split())
          class_id = int(fields5[0])
          x_center = float(fields5[1])
          y_center = float(fields5[2])
          width = float(fields5[3])
          height = float(fields5[4])
          x_min = int((x_center - width/2) * image.shape[1])
          y_min = int((y_center - height/2) * image.shape[0])
          x_max = int((x_center + width/2) * image.shape[1])
          y_max = int((y_center + height/2) * image.shape[0])
          #~~~~~~~~~~~~~~~~~~~~~~~~
          inxcolor = (255, 255, 255)
          obj_label = str(class_id)
          if 0 <= class_id and class_id < color_lst_len:
            inxcolor = color_lst[class_id] 
          if 0 <= class_id and class_id < class_lst_len:
            obj_label = f'{class_id} {class_lst[class_id]}'
          #~~~~~~~~~~~~~~~~~~~~~~~~
          cv2.rectangle(image, (x_min, y_min), (x_max, y_max), inxcolor, 2)
          #~ добавляем подпись класса объекта
          cv2.putText(image, obj_label, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, inxcolor, 3)
          cv2.putText(image, obj_label, (x_min+3, y_min+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
          #~~~~~~~~~~~~~~~~~~~~~~~~
          cv2.imwrite(img_fname2, image)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    if not tvt:
      break

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def accumulate_tvt(src_dir: str, dst_dir: str, classes_fname: str):
  # print('[INFO] accumulate_tvt')
  # print(f'[INFO] src_dir: `{src_dir}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] classes_fname: `{classes_fname}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for folder in ['train', 'valid', 'test']:
    del_fname = os.path.join(dst_dir, folder, 'labels.cache')
    # print(f'[INFO] del_fname: `{del_fname}`')
    delete_file(del_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  del_fname = os.path.join(dst_dir, folder, 'data.yaml')
  delete_file(del_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  dst_dir4 = os.path.join(dst_dir, 'temp4')
  # print(f'[INFO] dst_dir4: `{dst_dir4}`')
  delete_make_directory(dst_dir4)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копирование из папок dst_dir 'train', 'valid', 'test' -> dst_dir4
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(2):
    # print(f'[INFO] i: {i}')
    src_dir1 = src_dir
    if 1 == i:
      src_dir1 = dst_dir
    # print(f'[INFO]  src_dir1: `{src_dir1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for folder in ['train', 'valid', 'test']:
      img_dir1 = os.path.join(src_dir1, folder, 'images')
      lbl_dir1 = os.path.join(src_dir1, folder, 'labels')
      # print(f'[INFO]   img_dir1: `{img_dir1}`')
      # print(f'[INFO]   lbl_dir1: `{lbl_dir1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img_dir2 = os.path.join(dst_dir4, folder, 'images')
      lbl_dir2 = os.path.join(dst_dir4, folder, 'labels')
      # print(f'[INFO]   img_dir2: `{img_dir2}`')
      # print(f'[INFO]   lbl_dir2: `{lbl_dir2}`')
      make_directory(img_dir2)
      make_directory(lbl_dir2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img_lst = get_image_list(img_dir1)
      img_lst_len = len(img_lst)
      if img_lst_len > 0:
        for j in range(img_lst_len):
          base_fname,suffix_fname = get_base_suffix_fname(img_lst[j])
          # print(f'[INFO]     base_fname: `{base_fname}`, suffix_fname: `{suffix_fname}`')
          img_fname1 = os.path.join(img_dir1, img_lst[j])
          txt_fname1 = os.path.join(lbl_dir1, base_fname + '.txt')
          # print(f'[INFO]     img_fname1: `{img_fname1}`')
          # print(f'[INFO]     txt_fname1: `{txt_fname1}`')
          if not check_file_existence(img_fname1):
            continue
          if not check_file_existence(txt_fname1):
            continue
          #~~~~~~~~~~~~~~~~~~~~~~~~
          img_fname2 = os.path.join(img_dir2, img_lst[j])
          txt_fname2 = os.path.join(lbl_dir2, base_fname + '.txt')
          copy_file_with_new_name(img_fname1, img_fname2)
          copy_file_with_new_name(txt_fname1, txt_fname2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if 1 == i:
        delete_make_directory(img_dir1)
        delete_make_directory(lbl_dir1)
        # print(f'[INFO]    delete_make_directory: img_dir1: `{img_dir1}`')
        # print(f'[INFO]    delete_make_directory: lbl_dir1: `{lbl_dir1}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ перемешиваем, переименовываем и копируем dst_dir4 -> 'train', 'valid', 'test'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for folder in ['train', 'valid', 'test']:
    img_dir1 = os.path.join(dst_dir4, folder, 'images')
    lbl_dir1 = os.path.join(dst_dir4, folder, 'labels')
    img_dir2 = os.path.join(dst_dir, folder, 'images')
    lbl_dir2 = os.path.join(dst_dir, folder, 'labels')
    # print('[INFO]')
    # print(f'[INFO] img_dir1: `{img_dir1}`')
    # print(f'[INFO] lbl_dir1: `{lbl_dir1}`')
    # print(f'[INFO] img_dir2: `{img_dir2}`')
    # print(f'[INFO] lbl_dir2: `{lbl_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst =  get_image_list(img_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      continue
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(img_lst_len):
      base_fname,suffix_fname = get_base_suffix_fname(img_lst[i])
      img_fname1 = os.path.join(img_dir1, img_lst[i])
      txt_fname1 = os.path.join(lbl_dir1, base_fname + '.txt')
      if not check_file_existence(img_fname1):
        continue
      if not check_file_existence(txt_fname1):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      count_fname = f'f{format_counter(i, 7)}-'
      unic_fname = str(uuid.uuid1())
      img_fname2 = os.path.join(img_dir2, count_fname+unic_fname+suffix_fname)
      txt_fname2 = os.path.join(lbl_dir2, count_fname+unic_fname+'.txt')
      copy_file_with_new_name(img_fname1, img_fname2)
      copy_file_with_new_name(txt_fname1, txt_fname2)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ после копирования -> удаление папки с временными данными
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_directory(dst_dir4)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ создаем файл data.yaml
  #~~~~~~~~~~~~~~~~~~~~~~~~
  create_yaml_file(classes_fname, dst_dir)

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
def calculate_pairs_count(src_dir: str):
  print('~'*70)
  print('[INFO] number of pairs:')
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
def print_finish_program(start_time: float, dst_dir: str, program_mode: int):
  # print('[INFO] print_finish_program:')
  # print(f'[INFO] start_time: `{start_time}`')
  # print(f'[INFO] dst_dir: `{dst_dir}`')
  # print(f'[INFO] program_mode: {program_mode}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  execution_time = time.time() - start_time
  execution_time_str = format_execution_time(execution_time)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst =  get_image_list(dst_dir)
  img_count = len(img_lst)
  if 0 == program_mode:
    print(f'[INFO] total image count: {img_count}')
  elif 1 == program_mode or 2 == program_mode:
    print(f'[INFO] total pairs count: {img_count}')
  elif 3 == program_mode or 4 == program_mode:
    calculate_pairs_count(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('='*70)
  print(f'[INFO] program execution time: {execution_time_str}')
  print('='*70)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main_step05_yolov8_preparation(src_dir: str, dst_dir: str, program_mode: int, similar_threshold: int, matches_fname: str, train_percent: int, valid_percent: int, classes_fname: str):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Pretrainer YOLOv8 ver.2024.02.19')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] src_dir: `{src_dir}`')
  print(f'[INFO] dst_dir: `{dst_dir}`')
  if program_mode < 0 or program_mode > 4:
    print(f'[ERROR] invalid mode program mode: {program_mode}')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if not os.path.exists(src_dir):
    print(f'[ERROR] input folder is not exists: `{src_dir}`')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if 0 == program_mode:
    similar_threshold2 = similar_threshold
    if similar_threshold2 < 1 or similar_threshold2 > 100:
      similar_threshold2 = -1
    print(f'[INFO] similar_threshold: {similar_threshold2}')
    bbox_dir = os.path.join(dst_dir, 'bounding_boxes')
    print(f'[INFO] bbox_dir: `{bbox_dir}`')
    print('~'*70)
    copy_img_or_and_lbl(src_dir, dst_dir, similar_threshold)
    draw_bounding_boxes(dst_dir, bbox_dir, False)
    print_finish_program(start_time, dst_dir, program_mode)
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if 1 == program_mode:
    bbox_dir = os.path.join(src_dir, 'bounding_boxes')
    print(f'[INFO] bbox_dir: `{bbox_dir}`')
    print('~'*70)
    copy_pairs_by_bbox(src_dir, bbox_dir, dst_dir)
    print_finish_program(start_time, dst_dir, program_mode)
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if 2 == program_mode:
    print(f'[INFO] matches_fname: `{matches_fname}`')
    bbox_dir = os.path.join(dst_dir, 'bounding_boxes')
    print(f'[INFO] bbox_dir: `{bbox_dir}`')
    print('~'*70)
    reindex_labels(src_dir, dst_dir, matches_fname)
    draw_bounding_boxes(dst_dir, bbox_dir, False)
    print_finish_program(start_time, dst_dir, program_mode)
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if 3 == program_mode:
    bbox_dir = os.path.join(dst_dir, 'bounding_boxes')
    print(f'[INFO] bbox_dir: `{bbox_dir}`')
    print('~'*70)
    delete_make_directory(dst_dir)
    dst_dir3 = os.path.join(dst_dir, 'temp3')
    separate_category_pairs(src_dir, dst_dir3)
    shuffle_tvt_separate_pairs(dst_dir3, dst_dir, train_percent, valid_percent)
    delete_directory(dst_dir3)
    create_yaml_file(classes_fname, dst_dir)
    draw_bounding_boxes(dst_dir, bbox_dir, True, classes_fname)
    print_finish_program(start_time, dst_dir, program_mode)
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if 4 == program_mode:
    bbox_dir = os.path.join(dst_dir, 'bounding_boxes')
    print(f'[INFO] bbox_dir: `{bbox_dir}`')
    print('~'*70)
    accumulate_tvt(src_dir, dst_dir, classes_fname)
    draw_bounding_boxes(dst_dir, bbox_dir, True, classes_fname)
    print_finish_program(start_time, dst_dir, program_mode)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Grounding DINO labeler.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  parser.add_argument('--program_mode', type=int, default=0, help='Program operation mode')
  parser.add_argument('--similar_threshold', type=int, default=-1, help='Threshold of similarity of frames in percent')
  parser.add_argument('--matches_fname', type=str, default='', help='Path to the class matches file')
  parser.add_argument('--train_perc', type=int, default=70, help='Train percent')
  parser.add_argument('--valid_perc', type=int, default=25, help='Valid percent')
  parser.add_argument('--classes_fname', type=str, default='', help='Path to the classes file')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main_step05_yolov8_preparation(args.src_dir, args.dst_dir, args.program_mode, args.similar_threshold, args.matches_fname, args.train_perc, args.valid_perc, args.classes_fname) 
  #~~~~~~~~~~~~~~~~~~~~~~~~