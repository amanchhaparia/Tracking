import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

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



interpreter = tf.lite.Interpreter(model_path='./models/lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()

# cap = cv2.VideoCapture('/home/aman/Desktop/tepper/poseEstimation/Fitness - 72468.mp4')
# cap = cv2.VideoCapture('http://192.168.1.100:8080/video')
cap = cv2.VideoCapture('/home/aman/Desktop/tepper/poseEstimation/videos/production ID 4259066.mp4')
result = cv2.VideoWriter('../output/test3.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (256,256))

prev_time = 0
curr_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    # print(frame.shape)
    frame = cv2.resize(frame, (256,256))
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
    input_image = tf.cast(img, dtype=tf.float32)
 
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    prev_time = time.time()
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores.shape)
    curr_time = time.time()

    fps = 1/(curr_time-prev_time)
    print(fps)

    # Rendering 
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    result.write(frame)
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
result.release()
cv2.destroyAllWindows()