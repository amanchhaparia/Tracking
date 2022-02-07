from pyparsing import indentedBlock
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import math


def keep_aspect_ratio_resizer(image, target_size):
  """Resizes the image.

  The function resizes the image such that its longer side matches the required
  target_size while keeping the image aspect ratio. Note that the resizes image
  is padded such that both height and width are a multiple of 32, which is
  required by the model.
  """
  _, height, width, _ = image.shape
  if height > width:
    scale = float(target_size / height)
    target_height = target_size
    scaled_width = math.ceil(width * scale)
    image = tf.image.resize(image, [target_height, scaled_width])
    target_width = int(math.ceil(scaled_width / 32) * 32)
  else:
    scale = float(target_size / width)
    target_width = target_size
    scaled_height = math.ceil(height * scale)
    image = tf.image.resize(image, [scaled_height, target_width])
    target_height = int(math.ceil(scaled_height / 32) * 32)
  image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
  return (image,  (target_height, target_width))


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def detect(interpreter, input_tensor):
  """Runs detection on an input image.

  Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, input_height, input_width, 3] Tensor of type tf.float32.
      input_size is specified when converting the model to TFLite.

  Returns:
    A tensor of shape [1, 6, 56].
  """

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
  if is_dynamic_shape_model:
    input_tensor_index = input_details[0]['index']
    input_shape = input_tensor.shape
    interpreter.resize_tensor_input(
        input_tensor_index, input_shape, strict=True)
  interpreter.allocate_tensors()

  interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

  interpreter.invoke()

  keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
  return keypoints_with_scores

interpreter = tf.lite.Interpreter(model_path='./models/lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')
interpreter.allocate_tensors()

cap = cv2.VideoCapture('/home/aman/Desktop/tepper/poseEstimation/videos/production ID 4259066.mp4')
# cap = cv2.VideoCapture('http://192.168.1.100:8080/video')
# cap = cv2.VideoCapture(0)
result = cv2.VideoWriter('../output/multi.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, (640,480))

prev_time = 0
curr_time = 0

input_size = 256

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    # print(frame.shape)
    
    # Reshape image
    img = frame.copy()
    img = tf.expand_dims(img, axis=0)

    resized_image, image_shape = keep_aspect_ratio_resizer(img, input_size)
    input_image = tf.cast(resized_image, dtype=tf.uint8)
    print(input_image.shape)
    # Setup input and output 
    curr_time = time.time()
    keypoints_with_scores = detect(interpreter,input_image)
    prev_time = time.time()
    fps = 1/(curr_time-prev_time)
    print(fps)
    

    # Rendering 
    for person in keypoints_with_scores[0]:
      key_points = []
      for i in range(17):
        index = i*3
        key_points.append([person[index],person[index+1],person[index+2]])
      person_keypoints_withscore = np.array(key_points).reshape((1,1,17,3))
      print(person_keypoints_withscore.shape)
      draw_connections(frame, person_keypoints_withscore, EDGES, 0.4)
      draw_keypoints(frame, person_keypoints_withscore, 0.4)
    
    result.write(frame)
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
result.release()
cv2.destroyAllWindows()