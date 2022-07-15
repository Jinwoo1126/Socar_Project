# Socar_Project
---

This project contains content for vehicle damage detection using video.

Author : Jinwoo Jang 

<a href="https://colab.research.google.com/drive/1azkWsrOhVkZfQKse6MQEff5WpChougP9"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>


## Description <a href="https://buttery-gambler-1c2.notion.site/Socar-Hackerton-11-1434cfce47674f18a5690c0d63d93453"><img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg" alt="Open In Notion" width="22" height="22"></a> 
---

This project deals with the problem of object detection, which uses video data to determine whether a vehicle is damaged or not.<br>
especially, in this project, YOLOv5 is used as the object detection model.<br><br>

Here is some examples for frame images


<img src = "https://user-images.githubusercontent.com/50437310/178749754-ea4b11c0-6d11-4d73-aea7-d1aac0b0590f.jpeg" width = "200" height="200"> <img src = "https://user-images.githubusercontent.com/50437310/178749767-199e5b8d-7e29-4e4d-ab5a-e16014b2788c.png" width = "200" height="200"> <img src = "https://user-images.githubusercontent.com/50437310/178749783-b32aa054-1ae2-45a8-a765-2a748ef34b06.png" width = "200" height="200"> <img src = "https://user-images.githubusercontent.com/50437310/178749895-e7247230-7ec2-47af-b88a-a2a327b36bc5.jpeg" width = "200" height="200">

## Run detect.py
---

1. replace detect.py in YOLOv5 to <b>provided detect.py</b>
2. run detect.py using provided <b>best.py</b>

```python
!python detect.py --weight yolov5_trained_pt/train/socar_hackathon_yolov5m4/weights/best.pt --source "{video_sample}" 
```



## Detection logic
---

Using the queue and the confirm rate, if the same class in the queue has more than the confirm rate, it is decided that the class has an actual error.

```python
## in detect.py

from collections import deque

## for Detection Queue
MAX_QUEUE_LEN = 5
MAX_CONFIRM_CLASS = 3
DAMAGE_CLASS_DICT = {0:'Crack', 1:'Dent', 2:'Scratch'}
##

## for video class detection
detection_queue = deque()
detected_class = []
##

for path, im, im0s, vid_cap, s in dataset:
	    t1 = time_sync()
      im = torch.from_numpy(im).to(device)
      im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
      im /= 255  # 0 - 255 to 0.0 - 1.0
      if len(im.shape) == 3:
          im = im[None]  # expand for batch dim
      t2 = time_sync()
      dt[0] += t2 - t1

      # Inference
      visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
      pred = model(im, augment=augment, visualize=visualize)
      t3 = time_sync()
      dt[1] += t3 - t2

      # NMS
      pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
      dt[2] += time_sync() - t3

      # Second-stage classifier (optional)
      # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

      # Process predictions
      for i, det in enumerate(pred):  # per image
          ## for class clarification
          if det[:, -1].nelement() == 0:
              dmg_class = [-1]
          else:
              dmg_class = det[:, -1].unique().tolist()
      
          detection_queue.append(dmg_class)
          
          dmg_class = [0,0,0] # 0 : crack class / 1 : dent class / 2 : scratch class
          
          if len(detection_queue) == MAX_QUEUE_LEN:
              #print(detection_queue)
              for queue in detection_queue:
                  for q in queue:
                      if int(q) != -1:
                          dmg_class[int(q)] += 1
                      else:
                          continue
              
              for c, dmg in enumerate(dmg_class):
                  if dmg >= MAX_CONFIRM_CLASS:
                      detected_class.append(c)
                  
              detection_queue.popleft()
          ##

					##...
					##...
					##...

print("Detected Class : {}".format( [DAMAGE_CLASS_DICT[x] for x in set(detected_class)]))


'''
Results

if damaged parts is detected 
print Detected Class : ['Dent'] or Detected Class : ['Dent', 'Scratch'] ...
else
print Detected Class : []
'''

```
