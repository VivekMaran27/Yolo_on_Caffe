This repository was forked from https://github.com/ChriswooTalent/Yolo_on_Caffe and some changes were made to make caffe inference of yolo work on Jetson TX2

# Converting YOLOV2 weights to caffe model
1. cd caffe-master/python/convert/
2. Modify the parameters 'model_filename', 'yoloweight_filename' and 'caffemodel_filenam' in the 'convert_weights_to_caffemodel.py' script to point your respective files.
3. python convert_weights_to_caffemodel.py
4. caffe model file will be stored in the value set in 'caffemodel_filename'

# Running inference
1. cd caffe-master/python/yolo_detection/
2. Modify the parameters 'model_def' and 'model_weights' in the 'test_yolo_v2.py' script to point your respective files.
3. python test_yolo_v2.py <image_file>

# Notes from original repository
Yolo(including yolov1 yolov2 yolov3)running on caffe windows. Anyone that is not familiar with linux can use this project to learn caffe developing. This repository was forked from https://github.com/ChriswooTalent/Yolo_on_Caffe and some changes were to make caffe inference of yolo work on Jetson.
