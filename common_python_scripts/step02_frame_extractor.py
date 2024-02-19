#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ !!! в именах директорий, файлов не допускается использовать пробелы и спецсимволы,
#~ так как они передаются через параметры командной строки !!!
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ --src_dir -> директория с входными видео-файлами
#~ --step_ms 250 -> интервал извлечения кадров, если не используется, то все кадры
#~ --start_fragment_ms -> начало фрагмента для извлечения кадров в миллисекундах,
#~     если не используется, то с начала видео-файла
#~ --finish_fragment_ms -> окончание фрагмента для извлечения кадров в миллисекундах,
#~     если не используется, то до конца видео-файла
#~ --frame_90180270rot -> угол поворота кадра в градусах,
#~     если не используется, то кадры без поворота
#~     допустимые значения: 90, 180, 270
#~       90:  ROTATE_90_CLOCKWISE: int
#~       180: ROTATE_180: int
#~       270: ROTATE_90_COUNTERCLOCKWISE: int
#~ --slice_mode -> режим нарезки на кадры, обязательно должен быть указан параметр tile_size
#~     если не используется, то кадры без изменений,
#~     режим нарезки на тайлы по горизонтали и вертикали:
#~       0: без изменений исходного размера кадра,
#~       1: исходный кадр пропорционально сжимается по меньшей стороне
#~          до размеров tile_size, затем лишнее или справа и слева,
#~          или сверху и снизу отсекается,
#~       2: исходный кадр пропорционально сжимается по меньшей стороне
#~          до размеров tile_size, затем полученное изображение
#~          или сверху вниз, или слева направо нарезается на тайлы
#~ --tile_size 640 -> ширина и высота тайла, если кадр разрезаем на квадраты,
#~     если не используется, то кадр не разрезаем
#~ --dst_dir -> директория с извлеченными кадрами
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ последовательность выполняемых скриптом действий:
#~ 01. удаление dst_dir
#~ 02. извлечение кадров из видео-файлов, указанных в src_dir, и сохранених их в dst_dir
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step02_frame_extractor.py --src_dir c:/perimeter_raw_dataset/video1 --dst_dir c:/perimeter_raw_dataset/video2
# python step02_frame_extractor.py --src_dir c:/perimeter_raw_dataset/video1 --step_ms 5000 --slice_mode 1 --tile_size 640 --dst_dir c:/perimeter_raw_dataset/video2
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import shutil
import cv2
import time
import uuid
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
def format_counter(counter: int, digits: int):
  counter_str = str(counter)
  formatted_counter = counter_str.zfill(digits)
  return formatted_counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_video_list(src_dir: str):
  vid_lst = []
  for fname in os.listdir(src_dir):
    if os.path.isfile(os.path.join(src_dir, fname)):
      if fname.lower().endswith(('.mp4', '.avi', '.mov')):
        vid_lst.append(fname)
  return vid_lst

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ функция выделения кадров из видео
def extracting_frames(src_fname: str, step_ms: int, start_fragment_ms: int, finish_fragment_ms: int, frame_90180270rot: int, slice_mode: int, tile_size: int, dst_dir: str, file_numN: int):
  # print('~'*70)
  # print(f'[INFO] =>src_fname: `{src_fname}`')
  # print(f'[INFO] =>step_ms: {step_ms}')
  # print(f'[INFO] =>start_fragment_ms: {start_fragment_ms}')
  # print(f'[INFO] =>finish_fragment_ms: {finish_fragment_ms}')
  # print(f'[INFO] =>frame_90180270rot: {frame_90180270rot}')
  # print(f'[INFO] =>slice_mode: {slice_mode}')
  # print(f'[INFO] =>tile_size: {tile_size}')
  # print(f'[INFO] =>dst_dir: `{dst_dir}`')
  # print(f'[INFO] =>file_numN: {file_numN}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  vcap = cv2.VideoCapture(src_fname)
  if not vcap.isOpened():
    print('[ERROR]  can`t open video-file: `{src_fname}`')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~
  step_ms2 = step_ms
  if step_ms2 < 1:
    step_ms2 = -1
  print(f'[INFO]  step_ms2: {step_ms2} msec')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  start_time2 = start_fragment_ms
  if -1 == start_time2:
    start_time2 = 0
  vcap.set(cv2.CAP_PROP_POS_MSEC, start_time2)
  print(f'[INFO]  start_time2: {start_time2} msec')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  end_time2 = finish_fragment_ms
  if -1 == end_time2:
    end_time2 = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT) / vcap.get(cv2.CAP_PROP_FPS) * 1000)
  print(f'[INFO]  end_time2: {end_time2} msec')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ поворот кадра
  frame_rot2 = frame_90180270rot
  if not (90 == frame_rot2 or 180 == frame_rot2 or 270 == frame_rot2):
    if frame_rot2 < 0:
      frame_rot2 = 360 + frame_rot2
    if not (90 == frame_rot2 or 180 == frame_rot2 or 270 == frame_rot2):
      frame_rot2 = -1
  print(f'[INFO]  frame_rot2: {frame_rot2}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ режим нарезки кадра на тайлы
  slice_mode2 = slice_mode
  if not (0 == slice_mode2 or 1 == slice_mode2 or 2 == slice_mode2):
    slice_mode2 = -1
  tile_size2 = tile_size
  if tile_size2 < 10 or tile_size2 > 10000:
    tile_size2 = -1
  if -1 == slice_mode2 or -1 == tile_size2:
    slice_mode2 = -1
    tile_size2 = -1
  print(f'[INFO]  slice_mode2: {slice_mode2}, tile_size2: {tile_size2}')
     
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ последовательно читаю кадры из видео-файла
  #~~~~~~~~~~~~~~~~~~~~~~~~
  step_num = -1
  file_num1 = file_numN
  while vcap.isOpened():
    #~ читаем очерендной кадр  
    ret, frame = vcap.read()
    if not ret:
      break
    #~~~~~~~~~~~~~~~~~~~~~~
    #~ увеличиваем счетчик шагов
    step_num += 1
    #~ сдвигаемся вперед на заданный шаг
    if step_ms2 > 0:
      time_position2 = start_time2 + step_num*step_ms2
      if time_position2 > end_time2:
        break
      vcap.set(cv2.CAP_PROP_POS_MSEC, time_position2)
    else:
      if vcap.get(cv2.CAP_PROP_POS_MSEC) > end_time2:
        break
    print(f'[INFO]   step_num: {step_num}')
    #~~~~~~~~~~~~~~~~~~~~~~
    #~ оригинальные размеры кадра
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    if frame_width < 16 or frame_height < 16:
      continue
    # print(f'[INFO]   original frame size: width: {frame_width}, height: {frame_height}, ratio: {round(frame_width/frame_height,5)}')
    #~~~~~~~~~~~~~~~~~~~~~~
    #~ поворачиваем кадр
    if not -1 == frame_rot2:
      if 90 == frame_rot2 or 270 == frame_rot2:
        frame_width = frame.shape[0]
        frame_height = frame.shape[1]
        # print(f'[INFO]   rotated frame size: width: {frame_width}, height: {frame_height}, ratio: {round(frame_width/frame_height,5)}')
        if 90 == frame_rot2:
          frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
          frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (frame_width, frame_height))
      elif 180 == frame_rot2:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ сохраняем кадр или разрезаем на тайлы и сохраняем тайлы
    #~~~~~~~~~~~~~~~~~~~~~~
    if -1 == slice_mode2:
      unic_fname = str(uuid.uuid1()) 
      fname = f'f{format_counter(file_num1, 7)}-{unic_fname}.png'
      fname2 = os.path.join(dst_dir, fname)
      # print(f'[INFO] fname2: `{fname2}`')
      cv2.imwrite(fname2, frame)
      file_num1 += 1
    else:
      if 0 == slice_mode2 or 2 == slice_mode2:
        if 2 == slice_mode2:
          #~ определяем меньшую сторону и вычисляем масштаб для изменения размера
          frame_scale = tile_size2 / frame_height
          if frame_height > frame_width:
            frame_scale = tile_size2 / frame_width
          #~ пропорционально изменяем размер кадра
          frame = cv2.resize(frame, (int(frame_width*frame_scale), int(frame_height*frame_scale)))
          frame_width = frame.shape[1]
          frame_height = frame.shape[0]
          # print(f'[INFO]   resized frame size: width: {frame_width}, height: {frame_height}, ratio: {round(frame_width/frame_height,5)}')
        #~~~~~~~~~~~~~~~~~~~~~~
        # print(f'[INFO] step_num: {step_num}, frame640_width: {frame640_width}, frame640_height: {frame640_height}')
        #~ рассчитываем количество тайлов-квадратов по ширине и высоте
        num_tiles_width = frame_width // tile_size + (1 if frame_width % tile_size != 0 else 0)
        num_tiles_height = frame_height // tile_size + (1 if frame_height % tile_size != 0 else 0)
        # print(f'[INFO] num_tiles_width: {num_tiles_width}, num_tiles_height: {num_tiles_height}')
        for y in range(0, num_tiles_height * tile_size2, tile_size2):
          # print('~'*70)
          for x in range(0, num_tiles_width * tile_size2, tile_size2):
            # tile640 = frame[y:y+tile_size2, x:x+tile_size2]
            #~~~~~~~~~~~~~~~~~~~~~~
            #~ проверяем, если это последний тайл по горизонтали или вертикали
            if x + tile_size2 > frame_width:
              x_start = frame_width - tile_size2
              x_end = frame_width
            else:
              x_start = x
              x_end = x + tile_size2
            #~~~~~~~~~~~~~~~~~~~~~~
            if y + tile_size2 > frame_height:
              y_start = frame_height - tile_size2
              y_end = frame_height
            else:
              y_start = y
              y_end = y + tile_size2
            #~~~~~~~~~~~~~~~~~~~~~~
            #~ сохраняем тайл
            tile640 = frame[y_start:y_end, x_start:x_end]
            unic_fname = str(uuid.uuid1()) 
            fname = f'f{format_counter(file_num1, 7)}-{unic_fname}.png'
            fname2 = os.path.join(dst_dir, fname)
            cv2.imwrite(fname2, tile640)
            # print(f'[INFO]  file_num1: {file_num1}, y_start..y_end: {y_start}..{y_end}')
            # print(f'[INFO]   x_start..x_end: {x_start}..{x_end}')
            # print(f'[INFO]   fname2: `{x_end}`')
            file_num1 += 1
      elif 1 == slice_mode2:
        #~ определяем меньшую сторону и вычисляем масштаб для изменения размера
        frame_scale = tile_size2 / frame_height
        if frame_height > frame_width:
          frame_scale = tile_size2 / frame_width
        #~ пропорционально изменяем размер кадра
        frame = cv2.resize(frame, (int(frame_width*frame_scale), int(frame_height*frame_scale)))
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        # print(f'[INFO]   resized frame size: width: {frame_width}, height: {frame_height}, ratio: {round(frame_width/frame_height,5)}')
        #~ вырезаем центральную часть кадра
        crop_y = (frame_height - tile_size2) // 2
        crop_x = (frame_width - tile_size2) // 2
        if crop_y < 0:
          crop_y = 0
        if crop_x < 0:
          crop_x = 0
        crop_y2 = crop_y + tile_size2
        if crop_y2 > frame_height:
          crop_y2 = frame_height
        crop_x2 = crop_x + tile_size2
        if crop_x2 > frame_width:
          crop_x2 = frame_width
        #~~~~~~~~~~~~~~~~~~~~~~
        #~ сохраняем тайл
        tile640 = frame[crop_y:crop_y2, crop_x:crop_x2]
        unic_fname = str(uuid.uuid1()) 
        fname = f'f{format_counter(file_num1, 7)}-{unic_fname}.png'
        fname2 = os.path.join(dst_dir, fname)
        cv2.imwrite(fname2, tile640)
        file_num1 += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ отображаем кадр
    cv2.imshow('frame_extractor', frame)
    #~ если нажата клавиша 'q', выходим из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  #~~~~~~~~~~~~~~~~~~~~~~~~
  vcap.release()
  cv2.destroyAllWindows()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  return file_num1

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
def main_step02_frame_extractor(src_dir: str, step_ms: int, start_fragment_ms: int, finish_fragment_ms: int, frame_90180270rot: int, slice_mode: int, tile_size: int, dst_dir: str):
  start_time = time.time()
  print('~'*70)
  print('[INFO] Frame extractor ver.2024.02.11')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] src_dir: `{src_dir}`')
  print(f'[INFO] step_ms: {step_ms}')
  print(f'[INFO] start_fragment_ms: {start_fragment_ms}')
  print(f'[INFO] finish_fragment_ms: {finish_fragment_ms}')
  print(f'[INFO] frame_90180270rot: {frame_90180270rot}')
  print(f'[INFO] slice_mode: {slice_mode}')
  print(f'[INFO] tile_size: {tile_size}')
  print(f'[INFO] dst_dir: `{dst_dir}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 01. удаление dst_dir
  #~~~~~~~~~~~~~~~~~~~~~~~~
  delete_make_directory(dst_dir)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 02. извлечение кадров из видео-файлов, указанных в src_dir, и сохранених их в dst_dir
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] extracting frames...')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список видео-файлов
  vid_lst =  get_video_list(src_dir)
  vid_lst_len = len(vid_lst)
  if vid_lst_len < 1:
    print(f'[ERROR] List of video is empty: `{src_dir}`')
    return
  # print(f'[INFO]  video files: len: {vid_lst_len}, `{vid_lst}`')
  print(f'[INFO]  video files: len: {vid_lst_len}')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  file_num2 = 0
  for i in range(vid_lst_len):
    print('~'*70)
    print(f'[INFO] {i}: {vid_lst[i]}')
    src_fname = os.path.join(src_dir, vid_lst[i])
    # print(f'[INFO]     `{src_fname}`')
    file_numN = file_num2
    file_num2 = extracting_frames(src_fname, step_ms, start_fragment_ms, finish_fragment_ms, frame_90180270rot, slice_mode, tile_size, dst_dir, file_numN)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  print('~'*70)
  print(f'[INFO] frames save to the directory: `{dst_dir}`')
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
  parser = argparse.ArgumentParser(description='Frame extractor.')
  parser.add_argument('--src_dir', type=str, default='', help='Directory with input data')
  parser.add_argument('--step_ms', type=int, default=-1, help='Frame receiving interval in milliseconds')
  parser.add_argument('--start_fragment_ms', type=int, default=-1, help='Start fragment in milliseconds')
  parser.add_argument('--finish_fragment_ms', type=int, default=-1, help='Finish fragment in milliseconds')
  parser.add_argument('--frame_90180270rot', type=int, default=-1, help='Angle of rotation of the frame in degrees')
  parser.add_argument('--slice_mode', type=int, default=-1, help='Mode of cutting into frames')
  parser.add_argument('--tile_size', type=int, default=-1, help='Tile width and height')
  parser.add_argument('--dst_dir', type=str, default='', help='Directory with results')
  args = parser.parse_args()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  main_step02_frame_extractor(args.src_dir, args.step_ms, args.start_fragment_ms, args.finish_fragment_ms, args.frame_90180270rot, args.slice_mode, args.tile_size, args.dst_dir)