import cv2
import numpy as np
import darknet.darknet as darknet
import time


def load_network():
    network, class_names, class_colors = darknet.load_network(
        '/home/aman/Desktop/tepper/YOLO/darknet/cfg/yolov4.cfg',
        '/home/aman/Desktop/tepper/YOLO/darknet/cfg/coco.data',
        '/home/aman/Desktop/tepper/YOLO/weights/yolov4.weights',
        batch_size=1
    )

    return network, class_names, class_colors

def detect(image, network, class_names, class_colors, thresh=0.5):
    
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh, nms=0.45)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def main():
    prev_time = 0
    curr_time = 0
    
    cap = cv2.VideoCapture('/home/aman/Desktop/tepper/poseEstimation/videos/production ID 4259066.mp4')
    
    output = cv2.VideoWriter('./output/yolo.mp4',cv2.VideoWriter_fourcc(*'MJPG'),cap.get(cv2.CAP_PROP_FPS),(640,480))
    network, class_names, class_colors = load_network()
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame,(640,480))
        h,w = frame.shape[0], frame.shape[1]
        scale_h = h/height
        scale_w = w/width

        
        image, detections = detect(frame,network,class_names, class_colors)
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = time.time()
        print(fps)
        if detections:
            detections = darknet.non_max_suppression_fast(detections, overlap_thresh=0.7)

        det = []
        for label, confidence, bbox in detections:
            if label == 'person':
            
                x1, y1, x2, y2 = darknet.bbox2points(bbox)
                x1,y1,x2,y2 = int(x1*scale_w), int(y1*scale_h), int(x2*scale_w), int(y2*scale_h)

                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),1)


        cv2.imshow('Image',frame)
        output.write(frame)
        key = cv2.waitKey(10)
        if key == ord('x'):
            cap.release()   
            output.release()
            cv2.destroyAllWindows()
            break




if __name__ == '__main__':
    main()