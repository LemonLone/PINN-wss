import torch
import numpy as np
import numpy as np
import pandas as pd

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

#import torch.optim.lr_scheduler.StepLR

from torch.utils.tensorboard import SummaryWriter



def geo_train(device,x_in,y_in,z_in,xb,yb,zb,ub,vb,wb,xd,yd,zd,ud,vd,wd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC):
	if (Flag_batch):
		x = torch.Tensor(x_in).to(device)
		y = torch.Tensor(y_in).to(device)
		z = torch.Tensor(z_in).to(device)
		#dataset = TensorDataset(x,y)
		xb = torch.Tensor(xb).to(device)
		yb = torch.Tensor(yb).to(device)
		zb = torch.Tensor(zb).to(device)
		ub = torch.Tensor(ub).to(device)
		vb = torch.Tensor(vb).to(device)
		wb = torch.Tensor(wb).to(device)
		xd = torch.Tensor(xd).to(device)
		yd = torch.Tensor(yd).to(device)
		zd = torch.Tensor(zd).to(device)
		ud = torch.Tensor(ud).to(device)
		vd = torch.Tensor(vd).to(device)
		wd = torch.Tensor(wd).to(device)

		if(1): #Cuda slower in double? 
			x = x.type(torch.cuda.FloatTensor)
			y = y.type(torch.cuda.FloatTensor)
			xb = xb.type(torch.cuda.FloatTensor)
			yb = yb.type(torch.cuda.FloatTensor)
			ub = ub.type(torch.cuda.FloatTensor)
			vb = vb.type(torch.cuda.FloatTensor)
			#dist = dist.type(torch.cuda.FloatTensor)
			xd = xd.type(torch.cuda.FloatTensor)
			yd = yd.type(torch.cuda.FloatTensor)
			ud = ud.type(torch.cuda.FloatTensor)
			vd = vd.type(torch.cuda.FloatTensor)


		dataset = TensorDataset(x,y,z)
		dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = True )

	else:
		x = torch.Tensor(x_in).to(device)
		y = torch.Tensor(y_in).to(device) 
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
		def __init__(self):
			super(Net2_u, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
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
		def __init__(self):
			super(Net2_v, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
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
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
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
	
	# writer = SummaryWriter(layer_path)
	# net_in1 = torch.cat((x,y,z),1)
	# net_in2 = torch.cat((xb,yb,zb),1)
	# net_in3 = torch.cat((xd,yd,zd),1)
	
	# writer.add_graph(net2_u, net_in1)
	# # writer.add_graph(net2_u, net_in2)
	# # writer.add_graph(net2_u, net_in3)
	# writer.close()
 
	writer = SummaryWriter(log_path)
   	
    

	
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

		x.requires_grad = True
		y.requires_grad = True
		z.requires_grad = True

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

		
		XX_scale = U_scale * (X_scale**2)
		YY_scale = U_scale * (YZ_scale**2)
		UU_scale  = U_scale **2
	
		loss_2 =  rho*(u*u_x / X_scale + v*u_y / YZ_scale + w*u_z / YZ_scale)  - Diff*( u_xx/XX_scale  + u_yy /YY_scale + u_zz /YY_scale   )+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
		loss_1 =  rho*(u*v_x / X_scale + v*v_y / YZ_scale + w*v_z / YZ_scale)- Diff*( v_xx/ XX_scale + v_yy / YY_scale + v_zz / YY_scale )+ 1/rho*(P_y / (YZ_scale*UU_scale)   ) #Y-dir
		loss_3 = (u_x / X_scale + v_y / YZ_scale + w_z / YZ_scale) #continuity

		loss_4 =  rho*(u*w_x / X_scale + v*w_y / YZ_scale + w*w_z / YZ_scale) - Diff*( w_xx/ XX_scale + w_yy / YY_scale + w_zz / YY_scale )+ 1/rho*(P_z / (YZ_scale*UU_scale)   ) #Z-dir

		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3)) +loss_f(loss_4,torch.zeros_like(loss_4))

		return loss

	def Loss_BC(xb,yb,zb):

		

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
				loss_eqn_tot = 0.
				loss_bc_tot = 0.
				loss_data_tot = 0.
				n = 0
				for batch_idx, (x_in,y_in,z_in) in enumerate(dataloader): 	
					net2_u.zero_grad()#梯度清零
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
					if batch_idx % 5 ==0:
						#loss_bc = Loss_BC(xb,yb,ub,vb) #causes out of memory issue for large data in cuda
						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
							epoch, batch_idx * len(x_in), len(dataloader.dataset),
							100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item(),loss_data.item()))
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
					torch.save(net2_p.state_dict(),path+"epoch"+str(epoch)+"_IA3D_data3velsmall_p" + ".pt")
					torch.save(net2_u.state_dict(),path+"epoch"+str(epoch)+"_IA3D_data3velsmall_u" + ".pt")
					torch.save(net2_v.state_dict(),path+"epoch"+str(epoch)+"_IA3D_data3velsmall_v" + ".pt")
					torch.save(net2_w.state_dict(),path+"epoch"+str(epoch)+"_IA3D_data3velsmall_w" + ".pt")
     
				writer.add_scalar('loss_eqn', loss_eqn_tot, epoch)
				writer.add_scalar('loss_bc', loss_bc_tot, epoch)
				writer.add_scalar('loss_data', loss_data_tot, epoch)
			writer.close()

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	###################
	#plot
	if(1):#save network
		torch.save(net2_p.state_dict(),path+"IA3D_data3vel_p" + ".pt")
		torch.save(net2_u.state_dict(),path+"IA3D_data3vel_u" + ".pt")
		torch.save(net2_v.state_dict(),path+"IA3D_data3vel_v" + ".pt")
		torch.save(net2_w.state_dict(),path+"IA3D_data3vel_w" + ".pt")
		print ("Data saved!")

	return 

	############################################################
	#save loss
	# myFile = open('Loss track'+'stenosis_para'+'.csv','w')#
	# with myFile:
	# 	writer = csv.writer(myFile)
	# 	writer.writerows(LOSS)
	# LOSS = np.array(LOSS)
	# np.savetxt('Loss_track_pipe_para.csv',LOSS)

	############################################################



#######################################################
#Main code:
device = torch.device("cuda")


Flag_batch = True #False #USe batch or not  #With batch getting error...
Flag_BC_exact = True #If True enforces BC exactly HELPS ALOT!!! Not implemented in 2D
Lambda_BC  = 20.  ## If not enforcing BC exacctly.

Lambda_data = 20.  


Directory = "/home/Cuihaitao/PINN-wss/3D/"

path = Directory+"Results/results_tmp3/"

mesh_file_csv = Directory+"velocity_1.csv" #单独文件
outer_wall_location_csv = Directory+"velocity_1.csv" #单独文件
bc_file_wall_csv = Directory+"boundary_1.csv"

log_path = Directory + "Results/results_tmp3/log"
layer_path = Directory + "Results/layers"


batchsize = 8192 


epochs  = 5500 + 3000

Flag_pretrain = False # True #If true reads the nets from last run


Diff = 0.0035
rho = 1.056
T = 0.5 #total duraction
#nPt_time = 50 #number of time-steps

Flag_x_length = False #if True scales the eqn such that the length of the domain is = X_scale
U_scale = 1.0
U_BC_in = 0.5

#https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
Flag_schedule = True #If true change the learning rate 
if (Flag_schedule):
	learning_rate = 1e-3 #starting learning rate
	step_epoch = 1000 
	decay_rate = 0.5 


#####
print ('Loading', mesh_file_csv)
df = pd.read_csv(mesh_file_csv)
df1 = pd.read_csv(bc_file_wall_csv)
df_x = pd.concat([df['X [ m ]'], df1['X [ m ]']], ignore_index=True)
df_y = pd.concat([df[' Y [ m ]'], df1[' Y [ m ]']], ignore_index=True)
df_z = pd.concat([df[' Z [ m ]'], df1[' Z [ m ]']], ignore_index=True)
df_x_max = df_x.abs().max()
df_y_max = df_x.abs().max()
df_z_max = df_x.abs().max()
print('df_x_max is ',df_x_max)
print('df_y_max is ',df_y_max)
print('df_z_max is ',df_z_max)
X_scale = df_x_max
YZ_scale = df_y_max
csv_n_points = len(df) + len(df1)
print ('n_points of the mesh' ,csv_n_points)
csv_x_vtk_mesh=np.expand_dims(df_x.to_numpy(),1)
csv_y_vtk_mesh=np.expand_dims(df_y.to_numpy(),1)
csv_z_vtk_mesh=np.expand_dims(df_z.to_numpy(),1)
x  = np.reshape(csv_x_vtk_mesh , (np.size(csv_x_vtk_mesh [:]),1)) / X_scale
y  = np.reshape(csv_y_vtk_mesh , (np.size(csv_y_vtk_mesh [:]),1)) / YZ_scale
z  = np.reshape(csv_z_vtk_mesh , (np.size(csv_z_vtk_mesh [:]),1)) / YZ_scale


nPt = 130
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.0


print ('Loading', bc_file_wall_csv)
df = pd.read_csv(bc_file_wall_csv)
csv_nb_points = len(df)
print ('n_points of at wall' ,csv_nb_points)
csv_xb_vtk_mesh=np.expand_dims(df['X [ m ]'].to_numpy(),1)
csv_yb_vtk_mesh=np.expand_dims(df[' Y [ m ]'].to_numpy(),1)
csv_zb_vtk_mesh=np.expand_dims(df[' Z [ m ]'].to_numpy(),1)

xb = np.reshape(csv_xb_vtk_mesh , (np.size(csv_xb_vtk_mesh[:]),1)) / X_scale
yb = np.reshape(csv_yb_vtk_mesh , (np.size(csv_yb_vtk_mesh [:]),1)) / YZ_scale
zb = np.reshape(csv_zb_vtk_mesh , (np.size(csv_zb_vtk_mesh [:]),1)) / YZ_scale

ub = np.linspace(0., 0., csv_nb_points)
vb = np.linspace(0., 0., csv_nb_points)
wb = np.linspace(0., 0., csv_nb_points)

xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
zb= zb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array
wb= wb.reshape(-1, 1) #need to reshape to get 2D array



print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of ub',ub.shape)



##### Read data here#########################

N_sample = 10 # sample every other N_sample pts



print ('Loading', outer_wall_location_csv)
df = pd.read_csv(outer_wall_location_csv)
csv_nd_points = len(df)
print ('n_points of at wall' ,csv_nd_points)
csv_xd_vtk_mesh = np.expand_dims(df['X [ m ]'].to_numpy(),1)
csv_yd_vtk_mesh = np.expand_dims(df[' Y [ m ]'].to_numpy(),1)
csv_zd_vtk_mesh = np.expand_dims(df[' Z [ m ]'].to_numpy(),1)
x_data = np.reshape(csv_xd_vtk_mesh, (np.size(csv_xd_vtk_mesh[:]),1)) / X_scale
y_data = np.reshape(csv_yd_vtk_mesh, (np.size(csv_yd_vtk_mesh[:]),1)) / YZ_scale
z_data = np.reshape(csv_zd_vtk_mesh, (np.size(csv_zd_vtk_mesh[:]),1)) / YZ_scale

data_vel = np.array([df[' Velocity u [ m s^-1 ]'][::N_sample].to_numpy(),df[' Velocity v [ m s^-1 ]'][::N_sample].to_numpy(),df[' Velocity w [ m s^-1 ]'][::N_sample].to_numpy()])
data_vel_u = data_vel[0,:] / U_scale
data_vel_v = data_vel[1,:] / U_scale
data_vel_w = data_vel[2,:] / U_scale

xd= x_data[::N_sample].reshape(-1, 1) #need to reshape to get 2D array
yd= y_data[::N_sample].reshape(-1, 1) #need to reshape to get 2D array
zd= z_data[::N_sample].reshape(-1, 1) #need to reshape to get 2D array
ud= data_vel_u.reshape(-1, 1) #need to reshape to get 2D array
vd= data_vel_v.reshape(-1, 1) #need to reshape to get 2D array
wd= data_vel_w.reshape(-1, 1) #need to reshape to get 2D array


geo_train(device,x,y,z,xb,yb,zb,ub,vb,wb,xd,yd,zd,ud,vd,wd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC)


 








