import open3d as o3d
import numpy as np
import os
import struct
from numpy.linalg import eig

#Select file
path = os.path.join(os.path.expanduser('~'), 'Documents', 'kitti', 'dataset','sequences','03','velodyne')
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

"""
# Normal is calculated using PCA
def cal_normal(index_pt,points):
    neighboors = np.concatenate((points[0:index_pt],points[(index_pt+1):]))
    # Calculate the covariance matrix
    M = np.cov(neighboors)
    l,w = eig(M)
    # Normal vector correspounds to the smallest eigenvalue
    index_min = l.index(min(l))
    return w[index_min]

normal_vect = []
for i in range(len(np_pcd)):
    normal_vect.append(cal_normal(i,np_pcd))
print("OK")

"""



