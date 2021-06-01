import cv2
import numpy as np
from pykalman import KalmanFilter


class tracker:
	# tracker intialization
	def __init__(self,init,color=(255,255,255),thickness=1):
		self.color = color
		self.thickness = thickness
		xinit,yinit=init[0],init[1]
		Transition_Matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
		Observation_Matrix=[[1,0,0,0],[0,1,0,0]]
		self.data=[init]
		vxinit,vyinit=0,0
		initstate=[xinit,yinit,vxinit,vyinit] #setting initial state
		initcovariance=np.eye(4) 
		transistionCov=2*np.eye(4)
		observationCov=2*np.eye(2)

		self.kf=KalmanFilter(transition_matrices=Transition_Matrix,
		            observation_matrices =Observation_Matrix,
		            initial_state_mean=initstate,
		            initial_state_covariance=initcovariance,
		            transition_covariance=transistionCov,
		            observation_covariance=observationCov)

	# tracking the points
	def track(self,points):
		self.data.append(points)
		x,y = points[0],points[1]
		(filtered_state_means, filtered_state_covariances) = self.kf.filter(np.array(self.data))
		prediction = (filtered_state_means[:,0][-1],filtered_state_means[:,1][-1])
		return(prediction)

	# plotting the value with images
	def get_plot(self,image):
		if len(self.data) <= 20:
			data = self.data[::-1]
		else:
			data = self.data[::-1][:30] 
		image = cv2.circle(image, tuple(data[0][::-1]), radius=1, color = self.color, thickness=2)
		for idx,point in enumerate(data[:-1]):
			image = cv2.line(image, tuple(data[idx+1])[::-1], tuple(point)[::-1], self.color, self.thickness,cv2.LINE_AA) 
		return(image)

if __name__ == '__main__':
	t=tracker([0,0])

	for idx,i in enumerate(measured[1:]):
	    tracked=t.track(measured[idx])
	    print(idx,measured[idx],tracked,i,'Error',(int(i[0]-tracked[0]),int(i[1]-tracked[1])))
	    print('\n')
