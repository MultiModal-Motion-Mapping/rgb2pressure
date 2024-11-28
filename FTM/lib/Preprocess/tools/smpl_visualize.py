'''
本代码编写自MI 1353140938@qq.com
'''
import smplx
import torch
import trimesh
import numpy as np

# Load SMPL model (A body model)
smpl_model = smplx.create('./essentials/SMPL_NEUTRAL.pkl', 'smpl')

# Create a batch of parameters
batch_size = 1

# Read file
data = np.load('/nasdata/jiayi/MMVP/annotations/20230422/smpl_pose/S01/MoCap_20230422_093018/smpl_027.npz', allow_pickle=True)


body_pose = data['body_pose']
global_rot = data['global_rot']
transl = data['transl']

body_pose = torch.Tensor(np.stack([body_pose], axis=0).reshape((1,23,3)))
global_rot = torch.Tensor(np.stack([global_rot],axis=0))
transl = torch.Tensor(transl)

assert type(body_pose)==torch.Tensor and type(global_rot)==torch.Tensor and type(transl)==torch.Tensor

# Create a batch of parameters
result = smpl_model(betas=torch.zeros(batch_size, 10), # shape parameters
                    body_pose=body_pose, # pose parameters
                    global_orient=global_rot, # global orientation
                    transl=transl )# global translation

# joint names are shown in SMPL_joint_names.json
print(result.joints.shape) # (1, 45, 3)
print(result.vertices.shape) # (1, 6890, 3)


# Create a mesh from the vertices and faces
mesh = trimesh.Trimesh(vertices=result.vertices[0].cpu().numpy(), 
                       faces=smpl_model.faces)
mesh.export('./visual/smpl_mesh.obj')

# Visualization using blender
# Please refer to doc\Blender OBJ Sequence.md
# Also, available in the following link:
# https://blog.csdn.net/wjrzm2001/article/details/136676240