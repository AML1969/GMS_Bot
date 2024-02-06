#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ tvt - train, valid, test
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ fire
# python step03_tvt_split.py --src_dir c:/perimeter_raw_dataset/fire/101_02 --dst_dir c:/perimeter_dataset --train_perc 70 --valid_perc 25
# python step03_tvt_split.py --src_dir c:/perimeter_raw_dataset/fire/102_02 --dst_dir c:/perimeter_dataset --train_perc 70 --valid_perc 25
# python step03_tvt_split.py --src_dir c:/perimeter_raw_dataset/fire/103_02 --dst_dir c:/perimeter_dataset --train_perc 70 --valid_perc 25
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ garbage
# python step03_tvt_split.py --src_dir c:/perimeter_raw_dataset/garbage/201_02 --dst_dir c:/perimeter_dataset --train_perc 70 --valid_perc 25
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
#~ библиотека для вызова системных функций
import os
import shutil
import random
#~ передача аргументов через командную строку
import argparse
import time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def split_data(input_path, output_path, train_percent, valid_percent):
  train_percent2 = train_percent
  valid_percent2 = valid_percent
  test_percent2 = 100 - train_percent2 - valid_percent2
  if not 100 == (train_percent2 + valid_percent2 + test_percent2):
    train_percent2 = 70
    valid_percent2 = 25
    test_percent2 = 5
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # print(f'[INFO] input_path: `{input_path}`')
  # print(f'[INFO] output_path: `{output_path}`')
  # print(f'[INFO] train_percent2: {train_percent2}')
  # print(f'[INFO] valid_percent2: {valid_percent2}')
  # print(f'[INFO] test_percent2: {test_percent2}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ проверяем и создаем папку с результатами
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ cоздаем структуру папок для обучения YOLO
  for folder in ['train', 'valid', 'test']:
    for subfolder in ['images', 'labels']:
      tvt_path = os.path.join(output_path, folder, subfolder)
      if not os.path.exists(tvt_path):
        os.makedirs(tvt_path)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ считываем список файлов изображений и разметки
  # files = os.listdir(input_path)
  # image_files = [f for f in os.listdir(input_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
  img_files = [f for f in os.listdir(input_path) if f.endswith('.jpg')]
  # print(f'[INFO] files: len: {len(img_files)}, `{img_files[0]}`, `{img_files[1]}`, `{img_files[2]}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ перемешиваем список файлов
  random.shuffle(img_files)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # Рассчитываем количество файлов для каждой части (train, valid, test)
  total_count = len(img_files)
  train_count = int(total_count * train_percent2 / 100)
  valid_count = int(total_count * valid_percent2 / 100)
  # print(f'[INFO] total_count: {total_count}')
  # print(f'[INFO] train_percent2: {train_percent2}, train_count: {train_count}')
  # print(f'[INFO] valid_percent2: {valid_percent2}, valid_count: {valid_count}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ распределяем файлы по частям
  train_files = img_files[:train_count]
  valid_files = img_files[train_count:train_count + valid_count]
  test_files = img_files[train_count + valid_count:]
  print('[INFO] number of files:')
  print(f'[INFO]  total: {total_count}')
  print(f'[INFO]  train: {len(train_files)}')
  print(f'[INFO]  valid: {len(valid_files)}')
  print(f'[INFO]  test: {len(test_files)}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ копируем файлы в соответствующие папки
  for folder, file_list in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
    for file in file_list:
      shutil.copy(os.path.join(input_path, file), os.path.join(output_path, folder, 'images', file))
      label_file = os.path.splitext(file)[0] + '.txt'
      # print(f'[INFO] label_file: `{label_file}`')
      shutil.copy(os.path.join(input_path, label_file), os.path.join(output_path, folder, 'labels', label_file))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delete_labels_cache_file(dst_dir: str):
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ train
  train_labels_cache_fname = os.path.join(dst_dir, 'train', 'labels.cache')
  if os.path.exists(train_labels_cache_fname):
    os.remove(train_labels_cache_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ valid
  valid_labels_cache_fname = os.path.join(dst_dir, 'valid', 'labels.cache')
  if os.path.exists(valid_labels_cache_fname):
    os.remove(valid_labels_cache_fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ test
  test_labels_cache_fname = os.path.join(dst_dir, 'test', 'labels.cache')
  if os.path.exists(test_labels_cache_fname):
    os.remove(test_labels_cache_fname)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_classes(file_name):
  with open(file_name, 'r', encoding='utf-8') as file:
    classes = [line.strip() for line in file]
  return classes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_yaml_file(class_name_lst, dest_folder):
  train_img_path = os.path.join(dest_folder, 'train', 'images')
  val_img_path = os.path.join(dest_folder, 'valid', 'images')
  test_img_path = os.path.join(dest_folder, 'test', 'images')
  data_yaml_fname = os.path.join(dest_folder, 'data.yaml')
  # print(f'[INFO] train_img_path: `{train_img_path}`')
  # print(f'[INFO] val_img_path: `{val_img_path}`')
  # print(f'[INFO] test_img_path: `{test_img_path}`')
  # print(f'[INFO] data_yaml_fname: `{data_yaml_fname}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  try:
    #~ удаляем файл, если он уже существует
    if os.path.exists(data_yaml_fname):
      os.remove(data_yaml_fname)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ создаем и сохраняем файл data.yaml
    with open(data_yaml_fname, 'w', encoding='utf-8') as file_yaml:
      #~ train
      str_line = train_img_path.replace("/", "\\")
      file_yaml.write(f'train: {str_line}\n')
      #~ val
      str_line = val_img_path.replace("/", "\\")
      file_yaml.write(f'val: {str_line}\n')
      #~ test
      str_line = test_img_path.replace("/", "\\")
      file_yaml.write(f'test: {str_line}\n\n')
      #~ number of classes
      file_yaml.write(f'nc: {len(class_name_lst)}\n')
      #~ class names
      file_yaml.write(f'names: {class_name_lst}')
    # print(f'[INFO] Список успешно записан в файл: {data_yaml_fname}')
  except Exception as e:
    print(f'[ERROR] Ошибка при записи списка в файл: {e}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
  print('~'*70)
  print('[INFO] Split dataset for yolo ver.2024.02.05')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к папке из которой запустили программу
  prog_path = os.getcwd()
  # print(f'[INFO] prog_path: `{prog_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ парсер аргументов командной строки
  parser = argparse.ArgumentParser(description='Split dataset for yolo.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  parser.add_argument('--train_perc', type=int, default=70, help='Train percent')
  parser.add_argument('--valid_perc', type=int, default=25, help='Valid percent')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print(f'[INFO] src_dir: `{args.src_dir}`')
  print(f'[INFO] dst_dir: `{args.dst_dir}`')
  print(f'[INFO] train_perc: {args.train_perc}')
  print(f'[INFO] valid_perc: {args.valid_perc}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  #~ копирует файлы по трем папкам: train, valid, test в соответствии с указанным процентным соотношением
  split_data(args.src_dir, args.dst_dir, args.train_perc, args.valid_perc)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ удаляем файлы labels.cache от предыдущих сеансов тренировки
  delete_labels_cache_file(args.dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ формируем файл *.yaml
  print('~'*70)
  print('[INFO] make `data.yaml` from `classes.txt`...')
  objname_lst = read_classes('classes.txt')
  print(f'[INFO] objname_lst: len: {len(objname_lst)}, {objname_lst}')
  create_yaml_file(objname_lst, args.dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~

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
