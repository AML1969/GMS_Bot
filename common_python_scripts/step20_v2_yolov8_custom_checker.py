#~ USAGE
# cd c:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\SafeCity_Voronezh\dataset_preparation
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ video_src - истосник видео
#~   c:/my_campy/SafeCity_Voronezh/data_in/video_youtube/fire1.mp4
#~ --fps - frames per second - число кадров в секунду
#~   если -1, то не используется 
#~~~~~~~~~~~~~~~~~~~~~~~~
# python step20_yolov8_custom_checker.py --video_src c:/my_campy/SafeCity_Voronezh/data_in/video_youtube/fire1.mp4 --weights c:/my_campy/SafeCity_Voronezh/dataset_preparation/my_weights/fire/2024_01_22/best.pt
# python step20_yolov8_custom_checker.py --video_src c:/my_campy/SafeCity_Voronezh/data_in/video_youtube/fire1.mp4 --weights c:/my_campy/SafeCity_Voronezh/dataset_preparation/my_weights/fire/2024_01_22/best.pt --fps 30
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
#~ библиотека для вызова системных функций
import os
#~ передача аргументов через командную строку
import argparse
#~ библиотека для работы с графикой opencv
import cv2
#~ определение размеров экрана, для корректного отображения
import pyautogui
from ultralytics import YOLO
color_lst = [      #  синий зеленый красный
  (0, 0, 128),     #~ бордовый      0
  (39, 127, 255),  #~ оранжевый     1
  (0, 242, 255),   #~ желтый        2
  (76, 177, 34),   #~ зеленый       3
  (232, 162, 0),   #~ голубой       4
  (204, 72, 63),   #~ синий         5
  (164, 73, 163),  #~ фиолетовый    6
  (21, 0, 136),    #~ коричневый    7
  (127, 127, 127), #~ серый         8
  (0, 0, 0),        #~ черный       9
  (36, 28, 237),   #~ красный       10
  (0, 255, 0),     #~ лайм          11
  ]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Video Camera Controller
class VCamController:
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self):
    print('~'*70)
    print('[INFO] Testing YOLOv8 Object Detection on a Custom Dataset  ver.2024.02.05')
    print('~'*70)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ путь к папке из которой запустили программу
    #~~~~~~~~~~~~~~~~~~~~~~~~
    prog_path = os.getcwd()
    print(f'[INFO] program path: `{prog_path}`')
    self.cam_url = ''
    self.weights = ''
    self.interval_ms = 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ парсер аргументов командной строки
    #~~~~~~~~~~~~~~~~~~~~~~~~
    parser = argparse.ArgumentParser(description='Testing YOLOv8 Object Detection on a Custom Dataset.')
    parser.add_argument('--video_src', type=str, default='', help='Video source')
    parser.add_argument('--weights', type=str, default='', help='The path to the weights file')
    parser.add_argument('--fps', type=int, default=-1, help='Frames per second')
    args = parser.parse_args()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ источник видео
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.cam_url = args.video_src
    self.weights = args.weights
    #~~~~~~~~~~~~~~~~~~~~~~~~
    if not -1 == args.fps:
      fps1 = args.fps
      if fps1 < 1 or fps1 > 100:
        fps1 = 1
      self.interval_ms = int(1000/fps1) 
      if self.interval_ms < 1 or self.interval_ms > 100:
        self.interval_ms = 1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('[INFO] camera:')
    print(f'[INFO]  video source: `{self.cam_url}`')
    if not self.cam_url:
      print('[ERROR] video source not determined')
      exit()
    print(f'[INFO]  path to the weights file: `{self.weights}`')
    print(f'[INFO]  fps: {args.fps}')
    print(f'[INFO]  interval between frames in milliseconds: {self.interval_ms}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ определяем размеры кадра
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.frame_width = -1
    self.frame_height = -1
    vcam = cv2.VideoCapture(args.video_src)
    if vcam.isOpened():
      #~ читаю первые 30 кадров (обычно это 1сек, чтобы получить размеры кадра с большей вероятностью)
      for i in range(30):
        ret, frame = vcam.read()
        if ret:
          self.frame_width = frame.shape[1]
          self.frame_height = frame.shape[0]
          print(f'[INFO]  original frame size: width: {self.frame_width}, height: {self.frame_height}, ratio: {round(self.frame_width/self.frame_height,5)}')
          break
    vcam.release()
    if -1 == self.frame_width:
      self.cam_url = ''
      print(f'[ERROR] can`t read video-frame')
      exit()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ определяем размеры экрана
    #~~~~~~~~~~~~~~~~~~~~~~~~
    screen_width, screen_height = pyautogui.size()
    print(f'[INFO] screen: width: {screen_width}, height: {screen_height}, ratio: {round(screen_width/screen_height,5)}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ изменяем размер окна для отображения видео, если это необходимо, чтобы полказать полностью кадр
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 1080-65=1015 (patch by taskbar in windows) => 1015/1080=0.93981
    width_zip = screen_width*0.93981
    height_zip = screen_height*0.93981
    print(f'[INFO] screen without taskbar: width: {round(width_zip,5)}, height: {round(height_zip,5)}, ratio: {round(width_zip/height_zip,5)}')
    if self.frame_width > int(width_zip) or self.frame_height > int(height_zip):
      frame_zip = self.frame_width/width_zip
      hframe_zip = self.frame_height/height_zip
      if hframe_zip > frame_zip:
        frame_zip = hframe_zip
      width_zip = self.frame_width/frame_zip
      height_zip = self.frame_height/frame_zip
      self.frame_width = int(round(width_zip))
      self.frame_height = int(round(height_zip))
      print(f'[INFO] frame resize: width: {self.frame_width}, height: {self.frame_height}, ratio: {round(self.frame_width/self.frame_height,5)}')
    else:
      self.frame_width = -1
      print('[INFO] frame is not resize')

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def watch_video(self):
    if not self.cam_url:
      print('[ERROR] camera is not define')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываем видео-камеру
    #~ сheck if camera opened successfully
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcam = cv2.VideoCapture(self.cam_url)
    if not vcam.isOpened():
      print('[ERROR] can`t open video-camera')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ YOLOv8 model on custom dataset
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ load a model
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # model = YOLO('yolov8m.pt')
    model = YOLO(self.weights)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    threshold = 0.5
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ чтение видео-кадров камеры в бесконечном цикле,
    #~ до тех пор пока пользователь не нажмет на клавиатуре клавишу `q`
    #~~~~~~~~~~~~~~~~~~~~~~~~
    while True:
      ret, frame = vcam.read()    
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not ret:
        vcam.release()
        vcam = cv2.VideoCapture(self.cam_url)
        if not vcam.isOpened():
          print('[ERROR] can`t open video-camera')
          break
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ изменяем размеры кадра для отображения на экране монитора
      if not -1 == self.frame_width:
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ предсказание модели
      results = model(frame)[0]
      for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
#          cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 150), 4)  
          cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_lst[int(class_id)], 4)  
          cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.3, color_lst[int(class_id)], 3, cv2.LINE_AA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отображаем кадр
      cv2.imshow('video', frame)
      #~ если нажата клавиша 'q', выходим из цикла
      if cv2.waitKey(self.interval_ms) & 0xFF == ord('q'):
        break
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ освобождаем ресурсы
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcam.release()
    cv2.destroyAllWindows()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  #~~~~~~~~~~~~~~~~~~~~~~~~
  vcam_obj = VCamController()
  vcam_obj.watch_video()
  print('='*70)
  print('[INFO] -> program completed!')