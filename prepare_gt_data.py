import os
import numpy as np
from pyquaternion.quaternion import Quaternion as Q
import pickle
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import kinematics as K
from visualization import draw_obj

# prepare ground truth data
hoi_data = pickle.load(open('sit_data_2.pkl', 'rb'))
total = len(hoi_data)

# load bbox
bbox_dir = os.path.join('OBB')
bbox_dict = {}
for d in os.listdir(bbox_dir):
    if '.' in d or '_' in d:
        continue
    for f in os.listdir(os.path.join(bbox_dir, d)):
        if len(f) > 12:
            continue
        bboxes = pickle.load(open(os.path.join(bbox_dir, d, f), 'rb'))[0]
        bbox_dict[f[:-4]] = bboxes

for i_data, data_dict in enumerate(hoi_data):
    ax = plt.subplot(111, projection='3d')
    if i_data in [65, 67, 78, 97, 103, 131, 144, 149, 150, 151, 152, 153, 160, 173, 177, 178]:   # incorrect sitting
        continue
    if i_data in [14, 68, 69, 70, 71, 129, 130, 131, 136, 137, 138, 139, 140, 141, 169]:   # incorrect labeling -> incorrect relationship 130? 169?
        continue
    print('\r%d/%d'%(i_data, total), end='')
    # extract obj bboxes
    obj = data_dict['object']
    if obj['hash'] not in bbox_dict:
        continue
    # extract skeleton
    skeleton = data_dict['person']['skeleton']
    skeleton = np.array([skeleton[x] for x in K.joint_names]).astype(np.float32)
    center = np.array(data_dict['person']['root_position']).astype(np.float32)
    skeleton -= center
    rotation = np.array(data_dict['person']['root_rotation']).astype(np.float32)[[3,0,1,2]]
    skeleton = np.matmul(Q(rotation).inverse.rotation_matrix, skeleton.T).T
    obj_config = []
    obj_cat_id = []
    for part_id in bbox_dict[obj['hash']]:
        for bbox in bbox_dict[obj['hash']][part_id]:
            if bbox is None:
                continue
            T1,T2,T3,D,W,H,R = bbox
            R2 = Q(rotation).inverse * Q(np.array(obj['root_rotation'])[[3,0,1,2]])
            dx, dy, dz = (float(x) for x in obj['root_position'])
            dx -= center[0]
            dy -= center[1]
            dz -= center[2]
            obj_config.append([T3,T2,T1,D,W,H,*R.rotation_matrix[:,:2].reshape([6]),dx,dy,dz,*R2.rotation_matrix[:,:2].reshape([6])])
            obj_cat_id.append(part_id)
    draw_obj(obj_config, obj_cat_id, ax)
    ax.scatter(skeleton[:,0], skeleton[:,1], skeleton[:,2])
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.view_init(azim=60)
    plt.show()

