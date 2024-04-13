import torch
import numpy as np
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import vtk
from vtkmodules.util import numpy_support as VN
# from vtk.util import numpy_support as VN
#import torch.optim.lr_scheduler.StepLR




def geo_train(device,xd,yd,zd,ud,vd,wd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC):
	if (Flag_batch):

	 #dataset = TensorDataset(x,y)

	 xd = torch.Tensor(xd).to(device)
	 yd = torch.Tensor(yd).to(device)
	 zd = torch.Tensor(zd).to(device)
	 ud = torch.Tensor(ud).to(device)
	 vd = torch.Tensor(vd).to(device)
	 wd = torch.Tensor(wd).to(device)
	 #dist = torch.Tensor(dist).to(device)

	 if(1): #Cuda slower in double? 

		 xd = xd.type(torch.cuda.FloatTensor)
		 yd = yd.type(torch.cuda.FloatTensor)
		 ud = ud.type(torch.cuda.FloatTensor)
		 vd = vd.type(torch.cuda.FloatTensor)


	 # dataset = TensorDataset(x,y,z)
	 #dataset_bc = TensorDataset(x,y,xb,yb,ub,vb,dist)

	 #dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
	 # dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = True )
	 #dataloader_bc = DataLoader(dataset_bc, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = False )
	# else:
	#  x = torch.Tensor(x_in).to(device)
	#  y = torch.Tensor(y_in).to(device)
	 #t = torch.Tensor(t_in).to(device) 
	#x_test =  torch.Tensor(x_test).to(device)
	#y_test  = torch.Tensor(y_test).to(device)  
	h_n = 200 #128 #Width for u,v,p
	input_n = 3 # this is what our answer is a function of (x,y,z)
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	class MySquared(nn.Module):
		def __init__(self, inplace=True):
			super(MySquared, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			return torch.square(x)


	class Net2_u(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_u, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_u(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(y- yStart) * (y- yEnd ) + U_BC_in + (y- yStart) * (y- yEnd )  #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_v(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_v, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_v(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_w(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_w, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_v(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_p(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_p, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			#print('shape of xnet',x.shape) #Resuklts: shape of xnet torch.Size([batchsize, 2]) 
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  (1-x[:,0]) * output[:,0]  #Enforce P=0 at x=1 #Shape of output torch.Size([batchsize, 1])
			return  output
	
	################################################################
	#net1 = Net1().to(device)
	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_w = Net2_w().to(device)
	net2_p = Net2_p().to(device)

	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2_u.apply(init_normal)
	net2_v.apply(init_normal)
	net2_w.apply(init_normal)
	net2_p.apply(init_normal)


	############################################################################

	optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_w = optim.Adam(net2_w.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)




	def criterion(x,y,z):

		#print (x)
		#x = torch.Tensor(x).to(device)
		#y = torch.Tensor(y).to(device)
		#t = torch.Tensor(t).to(device)

		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		y.requires_grad = True
		z.requires_grad = True
		#t.requires_grad = True
		#u0 = u0.detach()
		#v0 = v0.detach()
		
		#net_in = torch.cat((x),1)
		net_in = torch.cat((x,y,z),1)
		u = net2_u(net_in)
		u = u.view(len(u),-1)
		v = net2_v(net_in)
		v = v.view(len(v),-1)
		w = net2_w(net_in)
		w = w.view(len(w),-1)
		P = net2_p(net_in)
		P = P.view(len(P),-1)



		
		u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_z = torch.autograd.grad(u,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
		u_zz = torch.autograd.grad(u_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]


		v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_z = torch.autograd.grad(v,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
		v_zz = torch.autograd.grad(v_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]


		w_x = torch.autograd.grad(w,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		w_xx = torch.autograd.grad(w_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		w_y = torch.autograd.grad(w,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		w_yy = torch.autograd.grad(w_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		w_z = torch.autograd.grad(w,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
		w_zz = torch.autograd.grad(w_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]

		P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		P_z = torch.autograd.grad(P,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]

		#u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
		#v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
		
		XX_scale = U_scale * (X_scale**2)
		YY_scale = U_scale * (YZ_scale**2)
		UU_scale  = U_scale **2
	
		loss_2 = u*u_x / X_scale + v*u_y / YZ_scale + w*u_z / YZ_scale  - Diff*( u_xx/XX_scale  + u_yy /YY_scale + u_zz /YY_scale   )+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
		loss_1 = u*v_x / X_scale + v*v_y / YZ_scale + w*v_z / YZ_scale - Diff*( v_xx/ XX_scale + v_yy / YY_scale + v_zz / YY_scale )+ 1/rho*(P_y / (YZ_scale*UU_scale)   ) #Y-dir
		loss_3 = (u_x / X_scale + v_y / YZ_scale + w_z / YZ_scale) #continuity

		loss_4 = u*w_x / X_scale + v*w_y / YZ_scale + w*w_z / YZ_scale - Diff*( w_xx/ XX_scale + w_yy / YY_scale + w_zz / YY_scale )+ 1/rho*(P_z / (YZ_scale*UU_scale)   ) #Z-dir

		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3)) +loss_f(loss_4,torch.zeros_like(loss_4))

		return loss

	def Loss_BC(xb,yb, zb):

		

		net_in1 = torch.cat((xb, yb,zb), 1)
		out1_u = net2_u(net_in1)
		out1_v = net2_v(net_in1)
		out1_w = net2_w(net_in1)
		
		out1_u = out1_u.view(len(out1_u), -1)
		out1_v = out1_v.view(len(out1_v), -1)
		out1_w = out1_w.view(len(out1_w), -1)



	

		loss_f = nn.MSELoss()
		loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) + loss_f(out1_w, torch.zeros_like(out1_w))
		return loss_noslip


	def Loss_data(xd,yd,zd,ud,vd,wd ):
	


		net_in1 = torch.cat((xd, yd,zd), 1)
		out1_u = net2_u(net_in1)
		out1_v = net2_v(net_in1)
		out1_w = net2_w(net_in1)
		
		out1_u = out1_u.view(len(out1_u), -1)
		out1_v = out1_v.view(len(out1_v), -1)
		out1_w = out1_w.view(len(out1_w), -1)

	

		loss_f = nn.MSELoss()
		loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd)  + loss_f(out1_w, wd) 


		return loss_d

	# Main loop

	tic = time.time()


	if(Flag_pretrain):
		print('Reading (pretrain) functions first...')
		net2_u.load_state_dict(torch.load(path+"sten_u" + ".pt"))
		net2_v.load_state_dict(torch.load(path+"sten_v" + ".pt"))
		net2_p.load_state_dict(torch.load(path+"sten_p" + ".pt"))
	
		

	if (Flag_schedule):
		scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
		scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
		scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=step_epoch, gamma=decay_rate)
		scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

	if(Flag_batch):# This one uses dataloader
			
			for epoch in range(epochs):
				#for batch_idx, (x_in,y_in) in enumerate(dataloader):  
				#for batch_idx, (x_in,y_in,xb_in,yb_in,ub_in,vb_in) in enumerate(dataloader): 
				loss_eqn_tot = 0.
				loss_bc_tot = 0.
				loss_data_tot = 0.
				n = 0
				for batch_idx, (x_in,y_in,z_in) in enumerate(dataloader): 
				
					net2_u.zero_grad()
					net2_v.zero_grad()
					net2_w.zero_grad()
					net2_p.zero_grad()
					loss_eqn = criterion(x_in,y_in,z_in)
					loss_bc = Loss_BC(xb,yb,zb)
					loss_data = Loss_data(xd,yd,zd,ud,vd,wd)
					loss = loss_eqn + Lambda_BC* loss_bc + Lambda_data*loss_data
					loss.backward()
					optimizer_u.step() 
					optimizer_v.step()
					optimizer_w.step() 
					optimizer_p.step()  
					loss_eqn_tot += loss_eqn
					loss_bc_tot += loss_bc
					loss_data_tot  += loss_data
					n += 1 
					if batch_idx % 40 ==0:
						#loss_bc = Loss_BC(xb,yb,ub,vb) #causes out of memory issue for large data in cuda
						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
							epoch, batch_idx * len(x_in), len(dataloader.dataset),
							100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item(),loss_data.item()))
						#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} '.format(
						#	epoch, batch_idx * len(x_in), len(dataloader.dataset),
						#	100. * batch_idx / len(dataloader), loss.item()))
				if (Flag_schedule):
						scheduler_u.step()
						scheduler_v.step()
						scheduler_w.step()
						scheduler_p.step()
				loss_eqn_tot = loss_eqn_tot / n
				loss_bc_tot = loss_bc_tot / n
				loss_data_tot = loss_data_tot / n
				print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot,loss_data_tot) )
				print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])
		
				if( epoch%1000 ==0 and epoch > 3000 ): #This causes out of memory in cuda in autodiff
					torch.save(net2_p.state_dict(),path+"IA3D_data3velsmall_p" + ".pt")
					torch.save(net2_u.state_dict(),path+"IA3D_data3velsmall_u" + ".pt")
					torch.save(net2_v.state_dict(),path+"IA3D_data3velsmall_v" + ".pt")
					torch.save(net2_w.state_dict(),path+"IA3D_data3velsmall_w" + ".pt")
			

	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2_u.zero_grad()
			net2_v.zero_grad()
			net2_w.zero_grad()
			net2_p.zero_grad()
			loss_eqn = criterion(x,y)
			loss_bc = Loss_BC(xb,yb,zb)
			if (Flag_BC_exact):
				loss = loss_eqn #+ loss_bc
			else:
				loss = loss_eqn + Lambda_BC * loss_bc
			loss.backward()
			#return loss
			#loss = closure()
			#optimizer2.step(closure)
			#optimizer3.step(closure)
			#optimizer4.step(closure)
			optimizer_u.step() 
			optimizer_v.step() 
			optimizer_p.step() 
			if epoch % 10 ==0:
				print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					epoch, loss.item(),loss_bc.item()))

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	###################
	#plot
	if(1):#save network
		torch.save(net2_p.state_dict(),path+"IA3D_data3velsmall_p" + ".pt")
		torch.save(net2_u.state_dict(),path+"IA3D_data3velsmall_u" + ".pt")
		torch.save(net2_v.state_dict(),path+"IA3D_data3velsmall_v" + ".pt")
		torch.save(net2_w.state_dict(),path+"IA3D_data3velsmall_w" + ".pt")
		print ("Data saved!")





	

	return 

	############################################################
	##save loss
	##myFile = open('Loss track'+'stenosis_para'+'.csv','w')#
	##with myFile:
		#writer = csv.writer(myFile)
		#writer.writerows(LOSS)
	#LOSS = np.array(LOSS)
	#np.savetxt('Loss_track_pipe_para.csv',LOSS)

	############################################################



#######################################################
#Main code:
#device = torch.device("cpu")
device = torch.device("cuda")


Flag_batch = True #False #USe batch or not  #With batch getting error...
Flag_BC_exact = False #If True enforces BC exactly HELPS ALOT!!! Not implemented in 2D
Lambda_BC  = 20.  ## If not enforcing BC exacctly.

Lambda_data = 20.  

Directory = "/home/aa3878/Data/ML/Amir/stenosis/"
Directory = "/home/admin1/code/pinn/PINN-wss-main/Data/3D-aneurysm/"
mesh_file = Directory + "IA_mesh3D_nearwall_small_physical.vtu" #"IA3D_nearwall_correct.vtu"
mesh_file_csv = Directory + "IA_mesh3D_nearwall_small_physical.csv" #"IA3D_nearwall_correct.vtu"

outer_wall_location = Directory + "IA_nearwall_outer_small.vtk"
outer_wall_location_csv = Directory + "IA_nearwall_outer_small.csv"
bc_file_wall = Directory + "IA_nearwall_wall_small.vtk"
bc_file_wall_csv = Directory + "IA_nearwall_wall_small.csv"
File_data = Directory + "IA_3D_unsteady3.vtu"
File_data_csv = Directory + "IA_3D_unsteady3.csv"

# mesh_file = mesh_file_csv = outer_wall_location = outer_wall_location_csv = bc_file_wall = bc_file_wall_csv = File_data = File_data_csv = '' #单独文件
fieldname = 'f_17' #The velocity field name in the vtk file (see from ParaView)

batchsize = 512 
learning_rate = 1e-5 


epochs  = 5500 + 3000

Flag_pretrain = False # True #If true reads the nets from last run


Diff = 0.00125
rho = 1.
T = 0.5 #total duraction
#nPt_time = 50 #number of time-steps

Flag_x_length = True #if True scales the eqn such that the length of the domain is = X_scale
X_scale = 3.0 #The length of the  domain (need longer length for separation region)
YZ_scale = 2.0 
U_scale = 1.0
U_BC_in = 0.5


Lambda_div = 1.  #penalty factor for continuity eqn (Makes it worse!?)
Lambda_v = 1.  #penalty factor for y-momentum equation

#https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
Flag_schedule = True #If true change the learning rate 
if (Flag_schedule):
	learning_rate = 5e-4  #starting learning rate
	step_epoch = 800 
	decay_rate = 0.5 


if (not Flag_x_length):
	X_scale = 1.
	YZ_scale = 1.

#####
# value = pd.read_csv(mesh_file)  #panda读csv文件
# num_value=np.genfromtxt(mesh_file,delimiter=',',dtype='str')



nPt = 130  
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.0
delta_circ = 0.2


t = np.linspace(0., T, nPt*nPt)  #自己设置
t=t.reshape(-1, 1)

#print('shape of t',t.shape)



path = "Results/"



##### Read data here#########################

#!!specify pts location here:
#x_data = [1., 1.2, 1.22, 1.31, 1.39 ] 
#y_data =[0.15, 0.07, 0.22, 0.036, 0.26 ]
#z_data  = [0.,0.,0.,0.,0. ]

#x_data = np.asarray(x_data)  #convert to numpy 
#y_data = np.asarray(y_data) #convert to numpy 

N_sample = 200 # sample every other N_sample pts

print ('Loading', outer_wall_location)
df = pd.read_csv(outer_wall_location_csv)
csv_n_points = len(df)
print ('n_points of at wall' ,csv_n_points)
csv_x_vtk_mesh = np.zeros((csv_n_points,1))
csv_y_vtk_mesh = np.zeros((csv_n_points,1))
csv_z_vtk_mesh = np.zeros((csv_n_points,1))
csv_N_pts_data = 0
for i in range(csv_n_points):
	if i%N_sample ==0:
		csv_x_vtk_mesh[csv_N_pts_data] = df.iloc[i,5]
		csv_y_vtk_mesh[csv_N_pts_data] = df.iloc[i,6]
		csv_z_vtk_mesh[csv_N_pts_data] = df.iloc[i,7]
		csv_N_pts_data +=1

print ('n_points sampled' ,csv_N_pts_data)
x_data = np.zeros((csv_N_pts_data,1))
y_data = np.zeros((csv_N_pts_data,1))
z_data = np.zeros((csv_N_pts_data,1))

data_vel=[]
data_vel_u = data_vel[:,0] / U_scale
data_vel_v = data_vel[:,1] / U_scale
data_vel_w = data_vel[:,2] / U_scale

x_data = x_data / X_scale #convert to normalized coordinates for PINN
y_data = y_data / YZ_scale
z_data = z_data / YZ_scale

print('Using input data pts: pts: ',x_data, y_data, z_data)
print('Using input data pts: vel u: ',data_vel_u)
print('Using input data pts: vel v: ',data_vel_v)
print('Using input data pts: vel w: ',data_vel_w)

for i in range(len(x_data)): 
	print('u: ',data_vel_u[i] )
	print('w: ',data_vel_w[i] )


xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
zd= z_data.reshape(-1, 1) #need to reshape to get 2D array
ud= data_vel_u.reshape(-1, 1) #need to reshape to get 2D array
vd= data_vel_v.reshape(-1, 1) #need to reshape to get 2D array
wd= data_vel_w.reshape(-1, 1) #need to reshape to get 2D array


#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
geo_train(device,xd,yd,zd,ud,vd,wd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC)
#tic = time.time()

#elapseTime = toc - tic
#print ("elapse time in serial = ", elapseTime)

 








