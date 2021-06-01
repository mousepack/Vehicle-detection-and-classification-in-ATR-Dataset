import os
import sys
import random
import warnings
from glob import glob
import numpy as np
import heapq
import cv2
import matplotlib.pyplot as plt
import pickle as p
from tqdm import tqdm
from itertools import chain
import tensorflow as tf
import keras
from keras import backend as K
from unet import UNet
from metrics import dice_coef,recall,precision,f1,point_dice_coef,IoU
from keras.optimizers import SGD

from predict_class import classifier
from tracker import tracker
# custom data loader for the dataset  for taking sequence of 5 images
# from generator import DataGenerator,get_sample_set,get_data_fixed
from gen_consecutive import DataGenerator,get_sample_set,get_data_fixed

class predict:
	def __init__(self,width,best_weight_detector,best_weight_classifier):
		self.width=width
		self.tracker = tracker([0,0],color=(255,255,255))
		self.detector=UNet((width,width,1),dropout=0.25, batchnorm=True, depth=4, residual=True)
		self.detector.load_weights(best_weight_detector)
		self.classifier = classifier(best_weight_classifier)
		self.video_writer = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (3*self.width,self.width))

	# getting th points
	def _get_points(self,imx,imy):
		intersection = imx*imy
		comp = (intersection>0).astype(np.uint8)
		nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(comp)

		area = stats[:,-1] 
		second_largest = heapq.nlargest(2, area)[1]
		[x,y,h,w,a] = stats[stats[:,-1]==second_largest][0]
		xmin,xmax,ymin,ymax = int(x-h/2),int(x+h/2),int(y-w/2),int(y+w/2)

		return(np.array([[xmin,xmax],[ymin,ymax]])[::-1])

	# getting the metrics
	def _metrics(self,y_true,y_pred):
		dice = dice_coef(y_true,y_pred)
		recal = recall(y_true,y_pred)
		pre = precision(y_true,y_pred)
		F1 = f1 (y_true,y_pred)
		point_dice = point_dice_coef(y_true,y_pred)
		iou = IoU(y_true,y_pred)
		return(dice,recal,pre,F1,point_dice,iou)

	# extract the region for classifier 
	def _region(self,seg_pred,points):
		return(seg_pred[points[0][0]:points[0][1],points[1][0]:points[1][1]])

	# extract the center of image for tracking
	def _center(self,points):
		return(np.array([(points[0][1]+points[0][0])/2.0, (points[1][1]+points[1][0])/2.0]).astype(np.int16))

	# testing module
	def test(self,gen,out='result',visualize=False):
		c=0
		results=[]
		images=[]
		res=[]

		detection=0
		for (image_batch,diff_batch),(ground_truth,seg_batch) in gen:
			center=[0,0]
			vehicle_name = 'NONE'
			for idx,diff in enumerate(diff_batch):
				c+=1

				actual_center= np.array(ground_truth[idx][1]).astype(np.uint8)[::-1]
				actual_vehicle= ground_truth[idx][0]
				image = image_batch[idx].reshape((5,self.width,self.width))[2]
				# stacking the image for visualization
				image = np.stack((image,)*3,axis=-1)
				seg_actual = seg_batch[idx].reshape((self.width,self.width)).astype(np.float32)
				difference_image = diff.reshape((1,self.width,self.width,1))
				seg_pred = self.detector.predict(difference_image)
				seg_pred = seg_pred.reshape((self.width,self.width)) #reshaping the output
				if precision(seg_actual,seg_pred) >0:
					detection+=1
					points=self._get_points(seg_actual,seg_pred)
					# getting the region
					region = self._region(seg_pred,points)
					confidence= 1.5*np.sum(region)/((points[0][1]-points[0][0])*(points[1][1]-points[1][0]))
					vehicle = cv2.cvtColor((region*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
					vehicle_name = self.classifier.predict(vehicle)
					center= self._center(points)
					center= center.astype(np.uint16)
					res.append([1,1,confidence])
					images.append([vehicle,vehicle_name,center,actual_vehicle,actual_center])

				else:
					res.append([1,0,0])
				tracked_center = np.array(self.tracker.track(center)).astype(np.int16)

				# Visualization of the data using video.
				if visualize == True:
					
					cv2.imwrite(out+'/im-%d-predict.jpg'%c,np.array(seg_pred*255).astype(np.uint8))
					cv2.imwrite(out+'/im-%d-actual.jpg'%c,np.array(seg_actual*255).astype(np.uint8))

					cv2.putText(image, vehicle_name, tuple(np.array(center[::-1])-10), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1, cv2.LINE_AA)
					cv2.circle(image, tuple(center[::-1]), radius=2,color = [0,0,255], thickness=4)
					cv2.circle(image, tuple(actual_center[::-1]), radius=2,color = [255,0,0], thickness=4)
					image = self.tracker.get_plot(image=image)
					seg_pre = cv2.cvtColor(seg_pred*255,cv2.COLOR_GRAY2BGR)
					seg_ac = cv2.cvtColor(seg_actual*255,cv2.COLOR_GRAY2BGR)
					image_stacked = np.hstack([image,seg_ac.astype(np.uint8),seg_pre.astype(np.uint8)])
					self.video_writer.write(np.uint8(image_stacked))


				dice,recal,pre,F1,point_dice,iou = self._metrics(y_true=seg_actual,y_pred=seg_pred)

				results.append([out+'/predict_%d.jpg'%c,dice,recal,pre,F1,point_dice,iou])
		self.video_writer.release()
		return(results,res,images)



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--viz", help="if image results required in cv2 in folder ./results",default=True)
	args = parser.parse_args() 	
	# model dimension
	width = 224

	# Loading the dataset
	test_data = p.load(open('test_images/test.p', 'rb'))
	# test_set = get_sample_set(get_data_fixed(test_pickle_data))
	print('Test Data Len:',len(test_data))

	# creating the test gen
	test_gen =DataGenerator([test_data], input_dim=(5,224,224), output_dim=(224, 224),viz=False,cropped =False,folder='./test_images/',offset=0)
	# prediction on the data with visualization ON
	model_predict=predict(width=width,best_weight_detector= 'weights/unet-best-3dXT_T.17-0.54.h5',best_weight_classifier='weights/resnet_101.path')
	result,detection,im  =  model_predict.test(test_gen,visualize=args.viz)
	im,dice_coeff,recall,precision,F1,_,iou = [np.array(result)[:,i] for i in range(len(result[0]))]
	avg_dice,avg_recall,avg_precision,avg_F1,_,avg_iou = [np.average(np.array(result)[:,i].astype(np.float16)) for i in range(1,len(result[0]))]
	print('Results\n1) Average Dice: {}\n2) Average Recall: {}\n3) Average Precision: {}\n4) Average F1: {}\n5) Average IoU: {}\n'.format(avg_dice,avg_recall,avg_precision,avg_F1,avg_iou))

