import pandas as pd
import numpy as np
import vtk
from vtk.util import numpy_support as VN

# 原理：读取stl文件转成vtu文件，然后读取csv文件，将csv文件的数据写入vtk文件中

# 创建一个STL阅读器并设置文件路径
reader = vtk.vtkSTLReader()
reader.SetFileName('export.stl')

# 创建一个从STL到UnstructuredGrid的转换器
geometry_filter = vtk.vtkGeometryFilter()
geometry_filter.SetInputConnection(reader.GetOutputPort())
geometry_filter.Update()

# 创建一个vtkAppendFilter，将vtkPolyData转换为vtkUnstructuredGrid
appendFilter = vtk.vtkAppendFilter()
appendFilter.AddInputData(geometry_filter.GetOutput())
appendFilter.Update()

# 创建一个vtu写入器并设置输出文件路径
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName('output.vtu')
writer.SetInputData(appendFilter.GetOutput())
writer.Write()


# 读取CSV文件
df = pd.read_csv('export.csv')

# 输出列索引
print(df.columns)
x = df['X [ m ]'].to_numpy()
y = df[' Y [ m ]'].to_numpy()
z = df[' Z [ m ]'].to_numpy()
u = df[' Velocity u [ m s^-1 ]'].to_numpy()
v = df[' Velocity v [ m s^-1 ]'].to_numpy()
w = df[' Velocity w [ m s^-1 ]'].to_numpy()
wss = df[' Wall Shear [ Pa ]'].to_numpy()


# 读取vtu文件
mesh_file = 'output.vtu'
output_filename = 'test.vtk'
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)

# 初始化
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
z_vtk_mesh = np.zeros((n_points,1))

Velocity = np.zeros((n_points, 3))

WallShearStress = np.zeros((n_points, 1))


WallShearStress = wss[0:n_points]
Velocity[:,0] = u[0:n_points]
Velocity[:,1] = v[0:n_points]
Velocity[:,2] = w[0:n_points]

print(u)

# 测试坐标顺序是否正确
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
 
print(x_vtk_mesh)

# point_data = vtk.vtkUnstructuredGrid()
# point_data.SetPoints(VTKpoints)

#将速度、应力信息添加到vtk中

# theta_vtk = VN.numpy_to_vtk(x)
# theta_vtk.SetName('x')   #x vector
# data_vtk.GetPointData().AddArray(theta_vtk)

# theta_vtk = VN.numpy_to_vtk(y)
# theta_vtk.SetName('y')   #y vector
# data_vtk.GetPointData().AddArray(theta_vtk)

# theta_vtk = VN.numpy_to_vtk(z)
# theta_vtk.SetName('z')   #z vector
# data_vtk.GetPointData().AddArray(theta_vtk)

theta_vtk = VN.numpy_to_vtk(Velocity[:,0])
theta_vtk.SetName('u')   #u vector
data_vtk.GetPointData().AddArray(theta_vtk)

theta_vtk = VN.numpy_to_vtk(Velocity[:,1])
theta_vtk.SetName('v')   #v vector
data_vtk.GetPointData().AddArray(theta_vtk)

theta_vtk = VN.numpy_to_vtk(Velocity[:,2])
theta_vtk.SetName('w')   #w vector
data_vtk.GetPointData().AddArray(theta_vtk)

theta_vtk = VN.numpy_to_vtk(Velocity)
theta_vtk.SetName('Velocity')   #Velocity vector
data_vtk.GetPointData().AddArray(theta_vtk)

theta_vtk = VN.numpy_to_vtk(WallShearStress)
theta_vtk.SetName('WallShearStress')   #wss vector
data_vtk.GetPointData().AddArray(theta_vtk)

myoutput = vtk.vtkDataSetWriter()
myoutput.SetInputData(data_vtk)
myoutput.SetFileName(output_filename)
myoutput.Write()

myoutput = vtk.vtkXMLUnstructuredGridWriter()
myoutput.SetInputData(data_vtk)
myoutput.SetFileName('test.vtu')
myoutput.Write()








# Warning: In vtkDataSet.cxx, line 736
# vtkUnstructuredGrid (000001B84694FD70): Point array u with 1 components, has 7778 tuples but there are only 7351 points

# Warning: In vtkDataSet.cxx, line 736
# vtkUnstructuredGrid (000001B84694FD70): Point array v with 1 components, has 7778 tuples but there are only 7351 points

# Warning: In vtkDataSet.cxx, line 736
# vtkUnstructuredGrid (000001B84694FD70): Point array w with 1 components, has 7778 tuples but there are only 7351 points

















