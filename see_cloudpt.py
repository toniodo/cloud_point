import open3d as o3d
import numpy as np
import os
import struct
from numpy.linalg import eig
import plotly.graph_objects as go
import time

st = time.time()

#Select file
path = os.path.join(os.path.expanduser('~'), 'Documents', 'kitti', 'dataset','sequences','04','velodyne')
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
    current_point = points[index_pt]
    neighboors = np.concatenate((points[0:index_pt],points[(index_pt+1):]))
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

def normal_cloud(cloud_points, nb_neighbor):
    normal_vects = []
    for i in range(len(np_pcd)-nb_neighbor-1):
        normal_vects.append(cal_normal(0, cloud_points[i:i+nb_neighbor+1]))
    return normal_vects

# Define the list of normal vector
nb_neighbor = 8
list_normal = normal_cloud(np_pcd,nb_neighbor)
U,V,W= zip(*list_normal)

# Define the origin of points
X,Y,Z = zip(*np_pcd[:-(nb_neighbor+1)])

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


