#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ src - source
#~ dir - directory
#~ dst - destination
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! в именах директорий не допускается использовать пробелы и спецсимволы, так как они !!!
#~ !!! передаются через параметры командной строки !!!
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ fire
# python step01_roboflow3dir_to_1dir.py --src_dir c:/perimeter_raw_dataset/fire/101/continuous_fire.v6-original_raw-images.yolov8 --dst_dir c:/perimeter_raw_dataset/fire/101_01 --class_name fire --class_index 0 --new_class_index 0
# python step01_roboflow3dir_to_1dir.py --src_dir c:/perimeter_raw_dataset/fire/102/xml_fire --dst_dir c:/perimeter_raw_dataset/fire/102_01 --class_name fire --class_index 0 --new_class_index 0
# python step01_roboflow3dir_to_1dir.py --src_dir c:/perimeter_raw_dataset/fire/103/FIRE.v1i.yolov8 --dst_dir c:/perimeter_raw_dataset/fire/103_01 --class_name fire --class_index 0 --new_class_index 0
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ garbage
# python step01_roboflow3dir_to_1dir.py --src_dir c:/perimeter_raw_dataset/garbage/201/Garbargebag.v1i.yolov8 --dst_dir c:/perimeter_raw_dataset/garbage/201_01 --class_name garbage --class_index 0 --new_class_index 2
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
import cv2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def delete_directory(path):
  if not os.path.exists(path):
    return
  try:
    shutil.rmtree(path)
    print(f'[INFO] Директория успешно удалена: `{path}`')
  except OSError as e:
    print(f'[ERROR] Ошибка при удалении директории: `{path}`: {e.strerror}')

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
def organize_files(input_folder_path, output_folder_path, file_prefix, class_index, new_class_index):
  # print(f'[INFO] input_folder_path: `{input_folder_path}`')
  # print(f'[INFO] output_folder_path: `{output_folder_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if not os.path.exists(input_folder_path):
    print(f'[WARNING] input folder is not exists: `{input_folder_path}`')
    return
  delete_directory(output_folder_path)
  if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
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
  for i in range(3):
    for j in range(2):
      in_path = os.path.join(input_folder_path, tvt_lst[i], il_lst[j])
      if j == 0:
        in_img_lst.append(in_path)
      elif j == 1:
        in_txt_lst.append(in_path)
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
      file_prefix2 = file_prefix + '_' + str(uuid.uuid1())
      # print(f'[INFO] txt_fname1: {txt_fname1}`')
      # print(f'[INFO] file_prefix2: {file_prefix2}`')
      with open(txt_fname1, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        filtered_lines = [line for line in lines if line.split()[0] == str(class_index)]
        # print(f'[INFO] filtered_lines: len: {len(filtered_lines)}: `{filtered_lines}`')
        #~ если есть строки для указанного класса объектов
        if filtered_lines:
          if 5 == len(filtered_lines):
            img_name2 = file_prefix2 + '.' + 'jpg'
            txt_name2 = file_prefix2 + '.' + 'txt'
            # print(f'[INFO] img_name2: `{img_name2}`, txt_name2: `{txt_name2}`')
            img_fname2 = os.path.join(output_folder_path, img_name2)
            txt_fname2 = os.path.join(output_folder_path, txt_name2)
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
  file_count = count_txt_files(output_folder_path)
  print(f'[INFO] total number of files: {file_count}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_bounding_boxes(input_dir):
  output_dir = os.path.join(input_dir, 'bounding_boxes')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for file_name in os.listdir(input_dir):
    if file_name.endswith(".jpg"):
      image_path = os.path.join(input_dir, file_name)
      annotation_path = os.path.join(input_dir, file_name.replace(".jpg", ".txt"))
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if os.path.exists(annotation_path):
        image = cv2.imread(image_path)
        with open(annotation_path, 'r', encoding='utf-8') as f:
          lines = f.readlines()
          for line in lines:
            # print(f'annotation_path: `{annotation_path}`')
            # print(f'  line: `{line}`')
            class_id, x_center, y_center, width, height = map(float, line.split())
            x_min = int((x_center - width/2) * image.shape[1])
            y_min = int((y_center - height/2) * image.shape[0])
            x_max = int((x_center + width/2) * image.shape[1])
            y_max = int((y_center + height/2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        output_image_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_image_path, image)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
  print('~'*70)
  print('[INFO] Roboflow 3-directory to 1-directory ver.2024.02.05')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к папке из которой запустили программу
  prog_path = os.getcwd()
  # print(f'[INFO] prog_path: `{prog_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ парсер аргументов командной строки
  parser = argparse.ArgumentParser(description='Roboflow 3-directory to 1-directory.')
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
  #~ копирую все файлы из трех папок в одну, изменяю имена на уникальные,
  #~ оставляю разметку только для одного объектов указанного id
  #~ id изменяю на указанный новый
  organize_files(args.src_dir, args.dst_dir, args.class_name, args.class_index, args.new_class_index)
  #~ отображаю на изображениях bounding box и сохраняю эти файлы
  draw_bounding_boxes(args.dst_dir)
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