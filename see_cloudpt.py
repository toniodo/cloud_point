import open3d as o3d
import numpy as np
import os
import struct
from numpy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Select file
path = os.path.join(os.path.expanduser('~'), 'Documents', 'kitti', 'dataset','sequences','00','velodyne')
filename= "000000.bin"

size_float = 4
list_pcd = []
with open (path+'/'+filename, "rb") as f:
    byte = f.read(size_float*4)
    while byte:
        x,y,z,intensity = struct.unpack("ffff", byte)
        list_pcd.append([x, y, z])
        byte = f.read(size_float*4)

# Array of all 3d points
np_pcd = np.asarray(list_pcd)

# Write in a pcd file
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector
pcd.points = v3d(np_pcd)
o3d.io.write_point_cloud("pcd_files/cloud_point.pcd", pcd)


# Normal is calculated using PCA
def cal_normal(index_pt,points):
    neighboors = np.concatenate((points[0:index_pt],points[(index_pt+1):]))
    # Calculate the covariance matrix
    M = np.cov(np.transpose(neighboors))
    l,w = eig(M)
    # Normal vector correspounds to the smallest eigenvalue
    l = list(l)
    index_min = l.index(min(l))
    normal_raw = w[index_min]
    return normal_raw/np.linalg.norm(normal_raw)

def normal_cloud(cloud_points, nb_neighbor):
    normal_vects = []
    for i in range(len(np_pcd)-nb_neighbor-1):
        normal_vects.append(cal_normal(0, cloud_points[i:i+nb_neighbor+1]))
    return normal_vects

# Define the list of normal vectors
list_normal = normal_cloud(np_pcd,6)
U,V,W= zip(*list_normal)

# Define the origin of points
X,Y,Z = zip(*np_pcd[:-6])

fig = plt.figure().add_subplot(projection='3d')
plt.quiver(X, Y, Z, U, V, W)
plt.show()


    




