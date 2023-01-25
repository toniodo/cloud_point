import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from operator import itemgetter
import os
import struct
from numpy.linalg import eig
import plotly.graph_objects as go
import time

st = time.time()

#Select file
path = os.path.join(os.path.expanduser('~'), 'Documents', 'kitti', 'dataset','sequences','00','velodyne')
filename= "000004.bin"

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

# Build a KDTree
tree = KDTree(np_pcd)

# Normal is calculated using PCA
def cal_normal(index_pt,points,index_neigh):
    current_point = points[index_pt]
    neighboors = list(itemgetter(*index_neigh)(points))
    # Calculate the covariance matrix
    M = np.cov(np.transpose(neighboors))
    l,w = eig(M)
    # Normal vector correspounds to the smallest eigenvalue
    l = list(l)
    index_min = l.index(min(l))
    normal_raw = w[index_min]
    # To have outgoing normal (from object)
    if np.dot(normal_raw, current_point)>0:
        return -normal_raw/np.linalg.norm(normal_raw)
    else:
        return normal_raw/np.linalg.norm(normal_raw)

def normal_cloud(cloud_points,list_index_neigh):
    normal_vects = []
    for i in range(len(cloud_points)):
        index_neigh = list_index_neigh[i]
        normal_vects.append(cal_normal(i,cloud_points,index_neigh))
    return normal_vects

# Define the number of neighboor to look at
nb_neighboor = 8
# Query in the tree the nearest neighboors
list_index_neigh=tree.query(np_pcd,k=nb_neighboor, return_distance=False)
list_normal = normal_cloud(np_pcd,list_index_neigh)
U,V,W= zip(*list_normal)

# Define the origin of points
X,Y,Z = zip(*np_pcd)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode='markers',marker=dict(size=0.5,color="blue")))

# Vectors are here respresented with two points : origin and direction
fig.add_trace(go.Scatter3d(x=np.add(X,U), y=np.add(Y,V), z=np.add(Z,W), mode='markers',marker=dict(size=0.5,color="red")))

# To set an uniform scale on all axes
fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[-80,80],),
        yaxis = dict(nticks=4, range=[-80,80],),
        zaxis = dict(nticks=4, range=[-80,80],)))

fig.show()

et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


