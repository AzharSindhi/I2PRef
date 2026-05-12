# coding=utf-8
import numpy as np
import transforms3d
import random
import math
from PIL import Image
import glob
import os
import shutil
import re
import open3d as o3d
import torch

def augment_cloud(Ps, args, return_augmentation_params=False):
    """" Augmentation on XYZ and jittering of everything """
    # Ps is a list of point clouds

    M = transforms3d.zooms.zfdir2mat(1) # M is 3*3 identity matrix
    # scale
    if args['pc_augm_scale'] > 1:
        s = random.uniform(1/args['pc_augm_scale'], args['pc_augm_scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)

    # rotation
    if args['pc_augm_rot']:
        scale = args['pc_rot_scale'] # we assume the scale is given in degrees
        # should range from 0 to 180
        if scale > 0:
            angle = random.uniform(-math.pi, math.pi) * scale / 180.0
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
            # we have verified that shapes from mvp data, the upright direction is along the y axis positive direction
    
    # mirror
    if args['pc_augm_mirror_prob'] > 0: # mirroring x&z, not y
        if random.random() < args['pc_augm_mirror_prob']/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args['pc_augm_mirror_prob']/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)

    # translation
    translation_sigma = args.get('translation_magnitude', 0)
    translation_sigma = max(args['pc_augm_scale'], 1) * translation_sigma
    if translation_sigma > 0:
        noise = np.random.normal(scale=translation_sigma, size=(1, 3))
        noise = noise.astype(Ps[0].dtype)
        
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)
        if translation_sigma > 0:
            P[:,:3] = P[:,:3] + noise
        if args['pc_augm_jitter']:
            sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
        result.append(P)

    if return_augmentation_params:
        augmentation_params = {}
        augmentation_params['M_inv'] = np.linalg.inv(M.T).astype(Ps[0].dtype)
        if translation_sigma > 0:
            augmentation_params['translation'] = noise
        else:
            augmentation_params['translation'] = np.zeros((1, 3)).astype(Ps[0].dtype)
        
        return result, augmentation_params

    return result

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)

def get_file_num(file_path):
    file_num = glob.glob(file_path)
    return len(file_num)

def get_folder_size(folder_path="/DISK/qwt"):
    '''
        getting the file size of target folder(B).
        1KB = 1024B
        1MB = 1024KB
        1GB = 1024MB
        1TB = 1024GB
    '''
    file_size = 0
    file_num = 0
    if(os.path.isdir(folder_path)):
        folders = os.listdir(folder_path)
        for folder in folders:
            folder = os.path.join(folder_path,folder)
            results = get_folder_size(folder)
            file_size += results[0]
            file_num += results[1]
    else:
        file_size += os.path.getsize(folder_path)
        file_num += 1
    return file_size, file_num

def bin2xyz(bin_path,xyz_path):
    '''
        convert *.bin to *.xyz (for KITTI)
    :param bin_path: *.bin file path
    :param xyz_path: saving path for *.xyz
    '''
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    xyz_folder = xyz_path[:xyz_path.rfind("/")]
    if(not os.path.exists(xyz_folder)):
        os.makedirs(xyz_folder)
    np.savetxt(fname=xyz_path, X=points)
    print(f"Converting {bin_path} to {xyz_path}")


def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0]) 
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    p_part = torch.tensor(np.array(pcd_part.points))
    
    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint] 
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

    
def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses


if __name__ == '__main__':
    import pdb
    args = {'pc_augm_scale':0, 'pc_augm_rot':False, 'pc_rot_scale':30.0, 'pc_augm_mirror_prob':0.5, 'pc_augm_jitter':False}
    N = 2048
    C = 6
    num_of_clouds = 2
    pc = []
    for _ in range(num_of_clouds):
        pc.append(np.random.rand(N,C)-0.5)
    
    result = augment_cloud(pc, args)
    pdb.set_trace()


