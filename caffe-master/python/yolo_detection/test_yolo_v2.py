# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import struct
#import matplotlib.pyplot as plt
import cv2
# display plots in this notebook

# set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import math
from ctypes import *
import binascii
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

caffe.set_mode_gpu()

model_def = 'yolo.prototxt'
#model_def = 'D:/Code_local/caffe_yolov2_windows/net_work_train/gnet_region_deploy.prototxt'
model_weights = 'yolo.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

mu = np.array([0, 0, 0])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 1.0)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (0,1,2))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          416, 416)  # image size is 227x227

def convert(s):
    i = int(s, 16)                   # convert from hex to a Python int
    cp = pointer(c_int(i))           # make this into a c integer
    fp = cast(cp, POINTER(c_float))  # cast the int pointer to a float pointer
    return fp.contents.value         # dereference the pointer, get the float

def vis_square(data):
	# normalize data for display
	data = (data - data.min())/(data.max() - data.min())

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0, n ** 2 - data.shape[0]),
				(0, 1), (0, 1))  # add some space between filters
			   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
	data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	#plt.imshow(data)

def draw_box(img, name, box, score):
	""" draw a single bounding box on the image """
	xmin, ymin, xmax, ymax = box

	box_tag = '{} : {:.2f}'.format(name, score)
	text_x, text_y = 5, 7

	cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 1, 0), 2)
	boxsize, _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	cv2.rectangle(img, (xmin, ymin - boxsize[1] - text_y),
					(xmin + boxsize[0] + text_x, ymin), (0, 225, 0), -1)
	cv2.putText(img, box_tag, (xmin + text_x, ymin - text_y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def show_results(img, results):
	""" draw bounding boxes on the image """
	img_width, img_height = img.shape[1], img.shape[0]
	disp_console = True
	imshow = True

	for result in results:
		box_x, box_y, box_w, box_h = [int(v) for v in result[1:5]]
		if disp_console:
			print('    class : {}, [x,y,w,h]=[{:d},{:d},{:d},{:d}], Confidence = {}'. \
					  format(result[0], box_x, box_y, box_w, box_h, str(result[5])))
		xmin, xmax = max(box_x - box_w // 2, 0), min(box_x + box_w // 2, img_width)
		ymin, ymax = max(box_y - box_h // 2, 0), min(box_y + box_h // 2, img_height)
		#458,435,124,218
		if imshow:
			draw_box(img, result[0], (xmin, ymin, xmax, ymax), result[5])
		#if imshow:
			#cv2.imshow('YOLO detection', img)
			#cv2.waitKey(0);
def GetBoxesAndShowResult(img, detectedresults, w, h):
	result = []
	label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car",
				  8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
				  15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
	for box in detectedresults:
		box[0] *= w
		box[2] *= w
		box[1] *= h
		box[3] *= h
		result.append([label_name[box[4]], box[0], box[1], box[2], box[3], box[5]])
	show_results(img, result)
	cv2.imshow('Video', img);
	return img;

def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        
        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1
    
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res


def det(image, image_id):
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image;

	### perform classification
	output = net.forward()
	res = output['conv23'][0]  # the output probability vector for the first image in the batch

	#res = conv_data_layer23_cpp
	swap = np.zeros((13*13,5,25))

	#change
	index = 0
	for h in range(13):
    		for w in range(13):
        		for c in range(125):
            			swap[h*13+w][c/25][c%25]  = res[c][h][w]


	biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
	test_objscore = swap[0 * 13 + w][0][4];
	boxes = list()
	for h in range(13):
    		for w in range(13):
        		for n in range(5):
            			box = list();
            			cls = list();
            			s = 0;
            			x = (w + sigmoid(swap[h*13+w][n][0])) / 13.0;
            			y = (h + sigmoid(swap[h*13+w][n][1])) / 13.0;
            			ww = (math.exp(swap[h*13+w][n][2])*biases[2*n]) / 13.0;
            			hh = (math.exp(swap[h*13+w][n][3])*biases[2*n+1]) / 13.0;
            			obj_score = sigmoid(swap[h*13+w][n][4]);
            			for p in range(20):
                			cls.append(swap[h*13+w][n][5+p]);
            
            			large = max(cls);
            			for i in range(len(cls)):
                			cls[i] = math.exp(cls[i] - large);
            
            			s = sum(cls);
            			for i in range(len(cls)):
                			cls[i] = cls[i] * 1.0 / s;
                
            			box.append(x);
            			box.append(y);
            			box.append(ww);
            			box.append(hh);
            			box.append(cls.index(max(cls))+1)
            			box.append(obj_score);
            			box.append(max(cls));
				box.append(obj_score * max(cls))
            
            			if box[5] * box[6] > 0.25:
                			boxes.append(box);
	res = apply_nms(boxes, 0.7)
	label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

	w = image.shape[1]
	h = image.shape[0]
	return GetBoxesAndShowResult(image, res, w, h);

def main():
	'''
	image = caffe.io.load_image(sys.argv[1]);
	res = det(image, '1');
	'''
	#Input video
	input_vid = sys.argv[1];
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	cap = cv2.VideoCapture(input_vid);
 	frame_count = 0;

	#Output video
	name = 'out.mp4'
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	out = cv2.VideoWriter(name, fourcc, 20.0, (1280,720))

	# Check if camera opened successfully
	if (cap.isOpened()== False): 
		print("Error opening video stream or file");
 
	# Read until video is completed
	while(cap.isOpened()):

  		# Capture frame-by-frame
		frame_count=frame_count +1;
		ret, frame = cap.read();
		np_frame = np.asarray(frame); 
		np_frame = np_frame/255.0;
		if ret == True:
	   		res = det(np_frame, frame_count);
		        #res = res[...,::-1];	
			res = res*255.0;
			#print res.shape;
			out.write(np.uint8(res));
			print frame_count;
			if frame_count ==  500:
				break;
			#Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break;
 
		# Break the loop
  		else: 
			break;
 
	# When everything done, release the video capture object
	cap.release();
	out.release();

	# Closes all the frames
	cv2.destroyAllWindows();

if __name__ == '__main__':
	main()
