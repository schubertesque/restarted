import numpy as np
from pyquaternion.quaternion import Quaternion as Q

def obb_get_verts_edges(T1,T2,T3,D,W,H,R,dx=0,dy=0,dz=0,R2=Q()):
  R = R2 * R
  verts = np.array([[T3,T2,T1],[T3,T2,H+T1],[T3,W+T2,T1],[T3,W+T2,H+T1],[D+T3,T2,T1],[D+T3,T2,H+T1],[D+T3,W+T2,T1],[D+T3,W+T2,H+T1]])
  verts = np.matmul(R.rotation_matrix, verts.T).T
  verts += np.array([[dx,dy,dz]])
  edges = [[0,1,3,2,0],[4,5,7,6,4],[0,4],[1,5],[2,6],[3,7]]
  return verts, edges

def rotation_matrix(r1,r2,r3,r4,r5,r6):
    R = np.array([[r1,r2], [r3,r4], [r5,r6]])
    a1 = R[:,0]
    a2 = R[:,1]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

def draw_obj(obj_config, obj_cat, ax):
    for i in range(len(obj_config)):
        T3, T2, T1, D, W, H, R10,R11,R12,R13,R14,R15, dx, dy, dz, R20,R21,R22,R23,R24,R25 = obj_config[i]

        R = rotation_matrix(R10,R11,R12,R13,R14,R15)
        R2 = rotation_matrix(R20,R21,R22,R23,R24,R25)
        R = Q._from_matrix(R2) * Q._from_matrix(R)
        verts = np.array([[T3,T2,T1],[T3,T2,H+T1],[T3,W+T2,T1],[T3,W+T2,H+T1],[D+T3,T2,T1],[D+T3,T2,H+T1],[D+T3,W+T2,T1],[D+T3,W+T2,H+T1]])
        verts = np.matmul(R.rotation_matrix, verts.T).T
        verts += np.array([[dx,dy,dz]])
        edges = [[0,1,3,2,0],[4,5,7,6,4],[0,4],[1,5],[2,6],[3,7]]
        
        for e in edges:
            ax.plot(verts[e,0], verts[e,1], verts[e,2], color='red')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pickle

    syn_skel, syn_obj = pickle.load(open('figs/17.pkl', 'rb'))
    for i in range(len(syn_skel)):
        ax = plt.subplot(111, projection='3d')
        draw_skel(syn_skel[i], ax)
        draw_obj(syn_obj[i,:-7], ax)
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
