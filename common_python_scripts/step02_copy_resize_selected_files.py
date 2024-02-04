#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step02_copy_resize_selected_files.py --src_dir d:/perimeter_raw_dataset/fire/101_01 --dst_dir d:/perimeter_raw_dataset/fire/101_02 --img_size 640
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
#~ библиотека для вызова системных функций
import os
import shutil
#~ передача аргументов через командную строку
import argparse
import time
import cv2
from PIL import Image

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
def replace_jpg_to_png(jpg_img_name):
  png_img_name = os.path.splitext(jpg_img_name)[0] + '.png'
  return png_img_name

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def resize_and_save_png_image(src_img_fname: str, dst_img_fname: str, img_size: int):
def resize_and_save_image(src_img_fname: str, dst_img_fname: str, img_size: int):
  # print('-'*50)
  # print(f'[INFO] src_img_fname: `{src_img_fname}`')
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
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ проверяем размеры исходного изображения
  changed_width = -1
  changed_height = -1
  offset_x = 0
  offset_y = 0
  if original_width <= 640 and original_height <= 640:
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
def delete_file(file_name):
  if os.path.exists(file_name):
    os.remove(file_name)
    # print(f'[INFO] Файл успешно удален: `{file_name}`')
  # else:
  #   print(f'[WARNING] Файл не существует: `{file_name}`')

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
def copy_selected_files(input_dir_images: str, output_dir: str, img_size: int):
  input_dir_bounding_boxes = os.path.join(input_dir_images, 'bounding_boxes')  
  # print(f'[INFO] input_dir_images: `{input_dir_images}`')
  # print(f'[INFO] input_dir_bounding_boxes: `{input_dir_bounding_boxes}`')
  # print(f'[INFO] output_dir: `{output_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if not os.path.exists(input_dir_images):
    print(f'[WARNING] input folder is not exists: `{input_dir_images}`')
    return
  if not os.path.exists(input_dir_bounding_boxes):
    print(f'[WARNING] input folder is not exists: `{input_dir_bounding_boxes}`')
    return
  delete_directory(output_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ составляем список имен изображений, только  тех, которые есть в директории,
  #~ которые отобрал пользователь
  selected_files = set()
  for file_name in os.listdir(input_dir_bounding_boxes):
    if file_name.endswith(".jpg"):
      selected_files.add(file_name)
  # print(f'[INFO] selected_files: len: {len(selected_files)}, `{selected_files}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # print(f'[INFO] input_dir_images: `{input_dir_images}`')
  for img_name in os.listdir(input_dir_images):
    if img_name.endswith(".jpg"):
      if img_name in selected_files:
        #~~~~~~~~~~~~~~~~~~~~~~~~
        src_img_fname = os.path.join(input_dir_images, img_name)
        # png_img_name = replace_jpg_to_png(img_name)
        # dst_img_fname = os.path.join(output_dir, png_img_name)
        dst_img_fname = os.path.join(output_dir, img_name)
        # print(f'[INFO] img_name: `{img_name}`')
        # print(f'[INFO] src_img_fname: `{src_img_fname}`')
        # print(f'[INFO] png_img_name: `{png_img_name}`')
        # print(f'[INFO] dst_img_fname: `{dst_img_fname}`')
        original_width,original_height,changed_width,changed_height,offset_x,offset_y = resize_and_save_image(src_img_fname, dst_img_fname, img_size)
        # print('-'*50)
        # print(f'[INFO] original_width: {original_width}, original_height: {original_height}')
        # print(f'[INFO] changed_width: {changed_width}, changed_height: {changed_height}')
        # print(f'[INFO] offset_x: {offset_x}, offset_y: {offset_y}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        txt_name = img_name.replace(".jpg", ".txt")
        src_txt_fname = os.path.join(input_dir_images, txt_name)
        dst_txt_fname = os.path.join(output_dir, txt_name)
        # print(f'[INFO] txt_name: `{txt_name}`')
        # print(f'[INFO] src_txt_fname: `{src_txt_fname}`')
        # print(f'[INFO] dst_txt_fname: `{dst_txt_fname}`')
        # shutil.copy(image_path, output_dir)
        # shutil.copy(txt_path, output_dir)
        is_bbox = change_yolo_markup(src_txt_fname, original_width, original_height, changed_width, changed_height, offset_x, offset_y, img_size, dst_txt_fname)
        # print(f'[INFO] is_bbox: {is_bbox}')
        if not is_bbox:
          delete_file(dst_img_fname)
          delete_file(dst_txt_fname)
        #~~~~~~~~~~~~~~~~~~~~~~~~

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
  print('[INFO] Copy selected files ver.2024.02.04')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к папке из которой запустили программу
  prog_path = os.getcwd()
  # print(f'[INFO] prog_path: `{prog_path}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ парсер аргументов командной строки
  parser = argparse.ArgumentParser(description='Copy selected files.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  parser.add_argument('--img_size', type=int, default=640, help='Target image size')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print(f'[INFO] src_dir: `{args.src_dir}`')
  print(f'[INFO] dst_dir: `{args.dst_dir}`')
  print(f'[INFO] img_size: {args.img_size}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  #~ копирую файлы по именам, которые оставил пользователь
  copy_selected_files(args.src_dir, args.dst_dir, args.img_size)
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
