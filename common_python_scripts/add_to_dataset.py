#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_utilities
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ src - source
#~ dir - directory
#~ dst - destination
#~ class
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ fire 101
# python add_to_dataset.py --src_dir c:/perimeter_raw_dataset/fire101 --dst_dir c:/perimeter_dataset --class_name fire --class_index 0 --new_class_index 8
#~ fire 102
# python add_to_dataset.py --src_dir c:/perimeter_raw_dataset/fire102 --dst_dir c:/perimeter_dataset --class_name fire --class_index 0 --new_class_index 8
#~ fire 103
# python add_to_dataset.py --src_dir c:/perimeter_raw_dataset/fire103 --dst_dir c:/perimeter_dataset --class_name fire --class_index 0 --new_class_index 8
#~ garbage 201
# python add_to_dataset.py --src_dir c:/perimeter_raw_dataset/garbage201 --dst_dir c:/perimeter_dataset --class_name garbage --class_index 0 --new_class_index 9
#~ garbage 202
# python add_to_dataset.py --src_dir c:/perimeter_raw_dataset/garbage202 --dst_dir c:/perimeter_dataset --class_name garbage --class_index 1 --new_class_index 9
#~ garbage 203
# python add_to_dataset.py --src_dir c:/perimeter_raw_dataset/garbage203 --dst_dir c:/perimeter_dataset --class_name garbage --class_index 1 --new_class_index 9
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
#~ библиотека для вызова системных функций
import os
import shutil
#~ unique identifier
import uuid
#~ передача аргументов через командную строку
import argparse
import time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_files_by_extension(directory, extension):
  file_list = []
  for file in os.listdir(directory):
    if file.endswith("." + extension):
      # file_list.append(os.path.join(directory, file))
      file_list.append(file)
  return file_list

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def change_file_extension(file_name, new_extension):
  base_name, _ = os.path.splitext(file_name)
  new_file_name = base_name + "." + new_extension
  return new_file_name

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_file_with_new_name(source_file, destination_file):
  try:
    shutil.copyfile(source_file, destination_file)
    # print(f'[INFO] Файл скопирован из {source_file} в {destination_file}')
  except FileNotFoundError:
    print('[ERROR] Ошибка: Файл не найден')
  except Exception as e:
    print(f'[ERROR] Произошла ошибка: {e}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def count_txt_files(directory_path):
  try:
    file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.txt')]
    return len(file_list)
  except FileNotFoundError:
    #~ возвращаем -1 в случае ошибки
    return -1
  except Exception as e:
    print(f'Произошла ошибка: {e}')
    #~ возвращаем -1 в случае ошибки
    return -1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_txt_yaml_file_and_get_values(file_path, field_inx):
  values = []
  with open(file_path, 'r') as file:
    for line in file:
      #~ проверяем, что строка не начинается с # и не пустая
      if not line.startswith('#') and line.strip():
        fields = line.split('|')
        if field_inx < len(fields):
          value = fields[field_inx].strip()
          if value:
            values.append(value)
  return values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def write_class_name_list_to_file(class_name_lst, class_fname):
  try:
    #~ удаляем файл, если он уже существует
    if os.path.exists(class_fname):
      os.remove(class_fname)
    #~ записываем новый список в файл
    with open(class_fname, 'w', encoding='utf-8') as file:
      for item in class_name_lst:
        file.write("%s\n" % item)
    # print(f'[INFO] Список успешно записан в файл: {class_fname}')
  except Exception as e:
    print(f'[ERROR] Ошибка при записи списка в файл: {e}')

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
def organize_files(input_folder_path, output_folder_path, file_prefix, class_index, new_class_index):
  # print(f'[INFO] input_folder_path: `{input_folder_path}`')
  # print(f'[INFO] output_folder_path: `{output_folder_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ t - train
  #~ v - valid
  #~ t - test
  tvt_lst = ['train', 'valid', 'test']
  #~ i - images
  #~ l - labels
  il_lst = ['images', 'labels']
  # print(f'[INFO] tvt_lst: len: {len(tvt_lst)}: `{tvt_lst}`')
  # print(f'[INFO] il_lst: len: {len(il_lst)}: `{il_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  in_img_lst = []
  in_txt_lst = []
  out_img_lst = []
  out_txt_lst = []
  for i in range(3):
    for j in range(2):
      in_path = os.path.join(input_folder_path, tvt_lst[i], il_lst[j])
      out_path = os.path.join(output_folder_path, tvt_lst[i], il_lst[j])
      if not os.path.exists(out_path):
        os.makedirs(out_path)
      if j == 0:
        in_img_lst.append(in_path)
        out_img_lst.append(out_path)
      elif j == 1:
        in_txt_lst.append(in_path)
        out_txt_lst.append(out_path)
  # print(f'[INFO] in_img_lst: len: {len(in_img_lst)}: `{in_img_lst}`')
  # print(f'[INFO] in_txt_lst: len: {len(in_txt_lst)}: `{in_txt_lst}`')
  # print(f'[INFO] out_img_lst: len: {len(out_img_lst)}: `{out_img_lst}`')
  # print(f'[INFO] out_txt_lst: len: {len(out_txt_lst)}: `{out_txt_lst}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for i in range(3):
    #~ i=0 -> train
    #~ i=1 -> valid
    #~ i=2 -> test
    txt_lst = find_files_by_extension(in_txt_lst[i], 'txt')
    # print(f'[INFO] i: {i}-> txt_lst: len: {len(txt_lst)}`, txt_lst[0]: {txt_lst[0]}`')
    for j in range(len(txt_lst)):
      # print(f'[INFO] {j}->{len(txt_lst)}: txt_lst[j]: {txt_lst[j]}`')
      img_fname0 = change_file_extension(txt_lst[j], 'jpg')
      img_fname1 = os.path.join(in_img_lst[i], img_fname0)
      # print(f'[INFO] img_fname0: {img_fname0}`')
      # print(f'[INFO] img_fname1: {img_fname1}`')
      if not os.path.exists(img_fname1):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      txt_fname1 = os.path.join(in_txt_lst[i], txt_lst[j])
      # file_prefix2 = file_prefix + '_' + str(uuid.uuid1())
      file_prefix2 = str(uuid.uuid1()) + '_' + file_prefix
      # print(f'[INFO] txt_fname1: {txt_fname1}`')
      # print(f'[INFO] file_prefix2: {file_prefix2}`')
      with open(txt_fname1, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        filtered_lines = [line for line in lines if line.split()[0] == str(class_index)]
        # print(f'[INFO] filtered_lines: len: {len(filtered_lines)}: `{filtered_lines}`')
        #~ если есть строки для указанного класса объектов
        if filtered_lines:
          img_name2 = file_prefix2 + '.' + 'jpg'
          txt_name2 = file_prefix2 + '.' + 'txt'
          # print(f'[INFO] img_name2: `{img_name2}`, txt_name2: `{txt_name2}`')
          img_fname2 = os.path.join(out_img_lst[i], img_name2)
          txt_fname2 = os.path.join(out_txt_lst[i], txt_name2)
          # print(f'[INFO] img_fname2: `{img_fname2}`')
          # print(f'[INFO] txt_fname2: `{txt_fname2}`')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          with open(txt_fname2, 'w', encoding='utf-8') as output_file:
            for line in filtered_lines:
              line_parts = line.split()
              #~ изменяем индекс класса объекта
              line_parts[0] = str(new_class_index)
              output_file.write(' '.join(line_parts) + '\n')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          copy_file_with_new_name(img_fname1, img_fname2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ все файлы изменены и скопированы
  #~ подсчитываю число файлов
  #~ i=0 -> train
  #~ i=1 -> valid
  #~ i=2 -> test
  train_count = count_txt_files(out_txt_lst[0])
  valid_count = count_txt_files(out_txt_lst[1])
  test_count = count_txt_files(out_txt_lst[2])
  print('[INFO] number of files:')
  print(f'[INFO]  train: {train_count}')
  print(f'[INFO]  valid: {valid_count}')
  print(f'[INFO]  test: {test_count}')
  print(f'[INFO]   total: {train_count+valid_count+test_count}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
  print('~'*70)
  print('[INFO] Adding images and labels to a dataset ver.2024.01.29')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к папке из которой запустили программу
  prog_path = os.getcwd()
  # print(f'[INFO] prog_path: `{prog_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ парсер аргументов командной строки
  parser = argparse.ArgumentParser(description='Adding images and labels to a dataset.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  parser.add_argument('--class_name', type=str, default='none', help='Class name')
  parser.add_argument('--class_index', type=int, default=0, help='Original class index')
  parser.add_argument('--new_class_index', type=int, default=0, help='New class index')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print(f'[INFO] src_dir: `{args.src_dir}`')
  print(f'[INFO] dst_dir: `{args.dst_dir}`')
  print(f'[INFO] class_name: `{args.class_name}`')
  print(f'[INFO] class_index: `{args.class_index}`')
  print(f'[INFO] new_class_index: `{args.new_class_index}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  organize_files(args.src_dir, args.dst_dir, args.class_name, args.class_index, args.new_class_index)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ формируем файл *.yaml
  # print('~'*70)
  txt_yaml_fname = os.path.join(prog_path, 'perimeter8_class_names.txt')
  # print(f'[INFO] yaml_fname: `{yaml_fname}`')
  # field_inx = 0
  # id_lst = read_txt_yaml_file_and_get_values(txt_yaml_fname, field_inx)
  # print(f'[INFO] id_lst: len: {len(id_lst)}, `{id_lst}`')
  #~~~
  field_inx = 1
  objname_lst = read_txt_yaml_file_and_get_values(txt_yaml_fname, field_inx)
  # print(f'[INFO] objname_lst: len: {len(objname_lst)}, `{objname_lst}`')
  txt_fname = os.path.join(args.dst_dir, 'classes.txt')
  # print(f'[INFO] txt_fname: `{txt_fname}`')
  write_class_name_list_to_file(objname_lst, txt_fname)
  #~~~
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
