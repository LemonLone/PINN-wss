import torch
import numpy as np
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

import pandas as pd
import numpy as np
#plot the loss on CPU (first load the net)



def create_csv(x,y,z):

	x = torch.Tensor(x).to(device)
	y = torch.Tensor(y).to(device)
	z = torch.Tensor(z).to(device)
	h_nD = 64  #for BC net
	h_D = 200 #128 # for distance net
	h_n = 200 #128 #for u,v,p
	input_n = 3 # this is what our answer is a function of. In the original example 3 : x,y,scale 



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

	class Net1_dist(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1_dist, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n ,h_D),
				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),
				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				
				nn.Linear(h_D,h_D),
				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			

				nn.Linear(h_D,h_D),

				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			

				nn.Linear(h_D,h_D),

				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			

				nn.Linear(h_D,h_D),

				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),

				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),

				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),



				nn.Linear(h_D,1),

				#nn.ReLU(), # make sure output is positive (does not work with PINN!)
				#nn.Sigmoid(), # make sure output is positive
				MySquared(),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net1_bc_u(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1_bc_u, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n ,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),


				Swish(),
				nn.Linear(h_nD,h_nD),


				#nn.ReLU(),
				Swish(),

				nn.Linear(h_nD,1),

				nn.ReLU(), # make sure output is positive
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net1_bc_v(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1_bc_v, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n ,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,1),

				nn.ReLU(), # make sure output is positive
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output




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


	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_w = Net2_w().to(device)
	net2_p = Net2_p().to(device)
	net1_dist = Net1_dist().to(device)
	net1_bc_u = Net1_bc_u().to(device)
	net1_bc_v = Net1_bc_v().to(device)


	print('load the network')
	net2_u.load_state_dict(torch.load(path+ NN_filename_prefix + "u.pt"))
	net2_v.load_state_dict(torch.load(path+ NN_filename_prefix + "v.pt"))
	net2_w.load_state_dict(torch.load(path+ NN_filename_prefix + "w.pt"))
	net2_p.load_state_dict(torch.load(path+ NN_filename_prefix + "p.pt"))


	net2_u.eval()
	net2_v.eval()
	net2_w.eval()
	net2_p.eval()
 
	net_in = torch.cat((x,y,z),1)
 
	output_u = net2_u(net_in)  #evaluate model
	output_u = output_u.data.numpy() 
	output_v = net2_v(net_in)  #evaluate model
	output_v= output_v.data.numpy() 
	output_w = net2_w(net_in)  #evaluate model
	output_w = output_w.data.numpy() 

	Velocity = np.zeros((csv_nd_points, 3)) #Velocity vector
	Velocity[:,0] = output_u[:,0] * U_scale
	Velocity[:,1] = output_v[:,0] * U_scale
	Velocity[:,2] = output_w[:,0] * U_scale

	# 创建一个DataFrame
	df = pd.read_csv(output_filename_xyz)
	df[' Velocity [ m s^-1 ]'] = np.sqrt(Velocity[:,0]**2 + Velocity[:,1]**2 + Velocity[:,2]**2)
	df[' Velocity u [ m s^-1 ]'] = Velocity[:,0]
	df[' Velocity v [ m s^-1 ]'] = Velocity[:,1]
	df[' Velocity w [ m s^-1 ]'] = Velocity[:,2]
	df.to_csv(output_filename_uvw, index=False)
	print(df)
	header = "[Name]\nVelocity\n\n[Data]\n"
	# 读取CSV文件
	with open(output_filename_uvw, 'r') as file:
		csv_data = file.read()
	# 将新的行和原始的CSV文件内容写入文件
	with open(output_filename_uvw, 'w') as file:
		file.write(header + csv_data)
	print ('Done!' )


############## Set parameters here (make sure you are calling the appropriate network in the code. Network code needs to be compied here)
Directory = "/home/Cuihaitao/PINN-wss/3D/"
device = torch.device("cpu")
outer_wall_location_csv = Directory+"velocity_1.csv" 
output_filename_xyz = Directory + "Results/output_xyz.csv"
output_filename_uvw = Directory + "Results/output_v.csv"
NN_filename_prefix = "IA3D_data3vel_" 

X_scale = 0.037723254 #2
Y_scale = 0.037723254 #1.
U_scale = 1.0
U_BC_in = 1.0 #0.5

Flag_BC_exact = False


path = Directory + "Results/results_tmp/"


print ('Loading', outer_wall_location_csv)
df = pd.read_csv(outer_wall_location_csv)
csv_nd_points = len(df)
print ('n_points of at wall' ,csv_nd_points)
csv_xd_vtk_mesh = np.expand_dims(df['X [ m ]'].to_numpy(),1)
csv_yd_vtk_mesh = np.expand_dims(df[' Y [ m ]'].to_numpy(),1)
csv_zd_vtk_mesh = np.expand_dims(df[' Z [ m ]'].to_numpy(),1)
x = np.reshape(csv_xd_vtk_mesh, (np.size(csv_xd_vtk_mesh[:]),1)) / X_scale
y = np.reshape(csv_yd_vtk_mesh, (np.size(csv_yd_vtk_mesh[:]),1)) / Y_scale
z = np.reshape(csv_zd_vtk_mesh, (np.size(csv_zd_vtk_mesh[:]),1)) / Y_scale

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(-1, 1)

print(x)

df = pd.DataFrame({
	'X [ m ]': df['X [ m ]'],
	' Y [ m ]': df[' Y [ m ]'],
	' Z [ m ]': df[' Z [ m ]']
})

# 保存为CSV文件
df.to_csv(output_filename_xyz, index=False)

create_csv(x,y,z)