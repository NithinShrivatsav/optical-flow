"""
Author: Nithin Shrivatsav Srikanth
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as filter2

def motion_model_linear_diffusion(I1, I2, u, v):
	
	## Parameters of optical flow
	dx = 1
	dy = 1
	du2 = dx*dx
	dv2 = dy*dy
	dtau = 0.01
	Lambda = dx/(5*dtau)
	
	## Convert the images to numpy float32 for processing
	I1 = I1.astype(np.float32)
	I2 = I2.astype(np.float32)
	
	## Initial guess on the vector fields
	u0 = np.zeros((u.shape[0],u.shape[1]))
	v0 = np.zeros((v.shape[0],v.shape[1]))

	## Gradients on the image
	Ix = (I1[1:-1,2:] - I1[1:-1,1:-1])/(2.0*dx)
	Iy = (I1[2:,1:-1] - I1[1:-1,1:-1])/(2.0*dy)
	It = (I2 - I1)/2.0

	for i in range(100):
		
		## Laplacian of the vector fields
		u_L = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1])/du2
		v_L = (v[2:,1:-1] + v[:-2,1:-1] + v[1:-1,2:] + v[1:-1,:-2] - 4*v[1:-1,1:-1])/dv2

		## Update the vector field
		u0[1:-1,1:-1] = u[1:-1,1:-1] + dtau*(-np.dot((np.dot(Ix,u[1:-1,1:-1]) + np.dot(Iy,v[1:-1,1:-1]) + It[1:-1,1:-1]),Ix) + Lambda*(u_L))
		v0[1:-1,1:-1] = v[1:-1,1:-1] + dtau*(-np.dot((np.dot(Ix,u[1:-1,1:-1]) + np.dot(Iy,v[1:-1,1:-1]) + It[1:-1,1:-1]),Iy) + Lambda*(v_L))

		u = u0.copy()
		v = v0.copy()
	return u,v

def motion_model_baseline(I1, I2, u, v):

	## Parameters of optical flow	
	dx = 1
	dy = 1
	du2 = dx*dx
	dv2 = dy*dy
	Lambda = 0.001
	dtau = 0.01

	## Kernels for performing the gradient operations
	Horn_Schunck_Kernel = np.array([[1/12, 1/6, 1/12], [1/6,    0, 1/6], [1/12, 1/6, 1/12]], float)
	
	## Convert the images to numpy float32 for processing
	I1 = I1.astype(np.float32)
	I2 = I2.astype(np.float32)
	
	## Initial guess on the vector fields
	u0 = np.zeros((u.shape[0],u.shape[1]))
	v0 = np.zeros((v.shape[0],v.shape[1]))
	u = u0
	v = v0

	## Gradients of the image
	Ix = (I1[2:,1:-1] - I1[1:-1,1:-1] + I1[2:,2:] - I1[1:-1,2:] + I2[2:,1:-1] - I2[1:-1,1:-1] + I2[2:,2:] - I2[1:-1,2:])/4.0
	Iy = (I1[1:-1,2:] - I1[1:-1,1:-1] + I1[2:,2:] - I1[2:,1:-1] + I2[1:-1,2:] - I2[1:-1,1:-1] + I2[2:,2:] - I2[2:,1:-1])/4.0
	It = (I2[1:-1,1:-1] - I1[1:-1,1:-1] + I2[1:-1,2:] - I1[1:-1,2:] + I2[2:,1:-1] - I1[1:-1,2:] + I2[2:,2:] - I1[2:,2:])/4.0

	for i in range(20):
		
		## Laplacian of the vector fields
		u_L = filter2(u, Horn_Schunck_Kernel)
		v_L = filter2(v, Horn_Schunck_Kernel)		

		optical_flow_constraint = (Ix*u_L[1:-1,1:-1] + Iy*v_L[1:-1,1:-1] + It) / (Lambda**2 + Ix**2 + Iy**2)
		
		## Update the vector field
		u[1:-1,1:-1] = u_L[1:-1,1:-1] - Ix*optical_flow_constraint 	
		v[1:-1,1:-1] = v_L[1:-1,1:-1] - Iy*optical_flow_constraint
	
	return u,v

def motion_model_TV(I1, I2, u, v):
	
	## Parameters of optical flow	
	dx = 1
	dy = 1
	du2_x = dx*dx
	du2_y = dy*dy
	dv2_x = dx*dx
	dv2_y = dy*dy
	dtau = 0.01
	epsilon = 0.001	
	Lambda = (dx*epsilon)/(4*dtau*40)

	## Convert the images to numpy float32 for processing
	I1 = I1.astype(np.float32)
	I2 = I2.astype(np.float32)

	## Initial guess on the vector fields	
	u0 = np.zeros((u.shape[0],u.shape[1]))
	v0 = np.zeros((v.shape[0],v.shape[1]))

	## Gradients of the image
	Ix = (I1[1:-1,2:] - I1[1:-1,1:-1])/(2*dx) + (I2[1:-1,2:] - I2[1:-1,1:-1])/(2*dx)
	Iy = (I1[2:,1:-1] - I1[1:-1,1:-1])/(2*dy) + (I1[2:,1:-1] - I1[1:-1,1:-1])/(2*dy)
	It = (I2 - I1)/2

	for i in range(100):
		
		## Gradients of the vector fields
		u_xx = (u[2:,1:-1] + u[:-2,1:-1] - 2*u[1:-1,1:-1])/du2_x
		u_yy = (u[1:-1,2:] + u[1:-1,:-2] - 2*u[1:-1,1:-1])/du2_y
		v_xx = (v[2:,1:-1] + v[:-2,1:-1] - 2*v[1:-1,1:-1])/dv2_x
		v_yy = (v[1:-1,2:] + v[1:-1,:-2] - 2*v[1:-1,1:-1])/dv2_y
		u_xy = (u[2:,2:] - u[2:,:-2] - u[:-2,2:] + u[:-2,:-2])/(4*dx*dy)
		v_xy = (v[2:,2:] - v[2:,:-2] - v[:-2,2:] + v[:-2,:-2])/(4*dx*dy)

		u_x = (u[2:,1:-1] - u[1:-1,1:-1])/(2*dx)
		u_y = (u[1:-1,2:] - u[1:-1,1:-1])/(2*dy)
		v_x = (v[2:,1:-1] - v[1:-1,1:-1])/(2*dx)
		v_y = (v[1:-1,2:] - v[1:-1,1:-1])/(2*dy)

		u_L = (((u_x**2)*u_yy) - (2*u_x*u_y*u_xy) + ((u_y**2)*u_xx) + (epsilon**2)*(u_xx+u_yy))/((u_x**2 + u_y**2 + epsilon**2)**1.5)
		v_L = (((v_x**2)*v_yy) - (2*v_x*v_y*v_xy) + ((v_y**2)*v_xx) + (epsilon**2)*(v_xx+v_yy))/((v_x**2 + v_y**2 + epsilon**2)**1.5)

		## Update the vector field
		u0[1:-1,1:-1] = u[1:-1,1:-1] + dtau*(-np.dot((np.dot(Ix,u[1:-1,1:-1]) + np.dot(Iy,v[1:-1,1:-1]) + It[1:-1,1:-1]),Ix) + Lambda*(u_L))
		v0[1:-1,1:-1] = v[1:-1,1:-1] + dtau*(-np.dot((np.dot(Ix,u[1:-1,1:-1]) + np.dot(Iy,v[1:-1,1:-1]) + It[1:-1,1:-1]),Iy) + Lambda*(v_L))
		
		u = u0.copy()	
		v = v0.copy()

	return u,v

## Function to draw the optical flow
def Optical_Flow_Display(h, w, U, V, algo, step_size=7):
	y, x = np.mgrid[step_size/2:h:step_size, step_size/2:w:step_size].reshape(2,-1).astype(int)

	flow = np.dstack((U,V))

	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.2)

	## A white image to display the vector field
	vis = cv2.imread('white.jpg',0)
	cv2.polylines(vis, lines, 0, (0, 0, 0))

	for (x1, y1), (x2,y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (0, 0, 0), -1)
	cv2.imshow("image", vis)
	if cv2.waitKey(0)==27:
		cv2.destroyAllWindows()
	return vis

def OpticalFlow(I1, I2, u, v, algo):
	if algo==1:
		U, V = motion_model_linear_diffusion(I1, I2, u, v)
		scale = 1000
	elif algo==2:
		U, V = motion_model_TV(I1, I2, u, v)
		scale = 1000
	else:
		U, V = motion_model_baseline(I1, I2, u, v)
		scale = 2
	# Commented lines for 
	#U, V = motion_model_vanilla(I1, I2, u, v)
	#scale=100
	#U_b, V_b = motion_model_baseline(I1, I2, u, v)
	#scale_b = 10
	#rmse_u = (np.square(U*scale - U_b*scale_b)).mean(axis=None)
	#rmse_v = (np.square(V*scale - V_b*scale_b)).mean(axis=None))
		
	h, w = I2.shape[:2]
	cv2.imshow('Horn Schunck algorithm', Optical_Flow_Display(h, w,U*scale,V*scale, algo))

if __name__=="__main__":
	img1 = cv2.imread('anim0.jpg',0)
	img2 = cv2.imread('anim1.jpg',0)
	img1 = cv2.resize(img1, (512, 512))
	img2 = cv2.resize(img2, (512, 512))
	## Conver the image to numpy array for processing
	I1 = np.array(img1)
	I1 = I1/255.0
	I2 = np.array(img2)
	I2 = I2/255.0
	m = I1.shape[0]
	n = I1.shape[1]
	u = np.zeros([m,n])
	v = np.zeros([m,n])
	## 1 - Linear Diffusion Optical Flow
	## 2 - Total Variation Optical Flow
	## 3 - Baseline Optical Flow
	OpticalFlow(I1, I2, u, v, 1)



