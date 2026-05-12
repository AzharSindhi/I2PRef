import torch
import numpy as np
import torch.utils.data as data
import os
import copy
import math
import random
from collections import defaultdict
from PIL import Image
from torchvision import transforms


class ViPCDataLoader(data.Dataset):
    def __init__(self, data_path, status, pc_input_num=3500, image_size=224,
                 view_align=False, category='plane', mini=True):
        super(ViPCDataLoader, self).__init__()
        self.npoints = pc_input_num
        self.filelist = []
        self.key = []
        self.view_align = view_align
        if category == 'train_all':
            self.cat_map = {
                'plane':'02691156',
                # 'bench': '02828884', 
                'cabinet':'02933112', 
                'car':'02958343',
                'chair':'03001627',
                # 'monitor': '03211117',
                'lamp':'03636649',
                # 'speaker': '03691459', 
                # 'firearm': '04090263', 
                'couch':'04256520',
                'table':'04379243',
                # 'cellphone': '04401088', 
                'watercraft':'04530566'
            }
            category = "all"
        elif category == 'unseen_all':
            self.cat_map = {
            # 'plane':'02691156',
            'bench': '02828884', 
            # 'cabinet':'02933112', 
            # 'car':'02958343',
            # 'chair':'03001627',
            'monitor': '03211117',
            # 'lamp':'03636649',
            'speaker': '03691459', 
            # 'firearm': '04090263', 
            # 'couch':'04256520',
            # 'table':'04379243',
            'cellphone': '04401088', 
            # 'watercraft':'04530566'
        }
            category = "all"
        else:
            self.cat_map = {
                'plane':'02691156',
                'bench': '02828884', 
                'cabinet':'02933112', 
                'car':'02958343',
                'chair':'03001627',
                'monitor': '03211117',
                'lamp':'03636649',
                'speaker': '03691459', 
                'firearm': '04090263', 
                'couch':'04256520',
                'table':'04379243',
                'cellphone': '04401088', 
                'watercraft':'04530566'
            }
        filename = f"{status}_list.txt"
        with open(os.path.join(data_path, filename),'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        all_category_ids = list(self.cat_map.values())
        self.cat2keys = defaultdict(list)
        for key in self.filelist:
            if category != 'all':
                if key.split('/')[0] != self.cat_map[category]:
                    continue
            if key.split('/')[0] not in all_category_ids:
                continue
            self.cat2keys[key.split('/')[0]].append(key)

        self.key = []
        for cat_id in self.cat2keys:
            all_values = self.cat2keys[cat_id]
            if mini and status == "train":
                nsamples = int(len(all_values) * 0.33)
                all_values = random.sample(all_values, nsamples)
            self.key.extend(all_values)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


        print(f'Dataset loaded: {len(self.key)} samples ({status})')


    def rotation_y(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                    [0.0, 1.0, 0.0],
                                    [sin_theta, 0.0, cos_theta]])
        return pts @ rotation_matrix.T


    def rotation_x(self, pts, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, cos_theta, -sin_theta],
                                    [0.0, sin_theta, cos_theta]])
        return pts @ rotation_matrix.T

    def __getitem__(self, idx):
        key = self.key[idx]
        pc_part_path = os.path.join(self.imcomplete_path, key.split('/')[0] + '/' + key.split('/')[1] + '/' + key.split('/')[-1].replace('\n', '') + '.npy')
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
       
        pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ ran_key.split('/')[1]+'/'+ran_key.split('/')[-1].replace('\n', '')+'.npy')
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')
        
        #Inserted to correct a bug in the splitting for some lines 
        if(len(ran_key.split('/')[-1])>3):
            print("bug")
            print(ran_key.split('/')[-1])
            fin = ran_key.split('/')[-1][-2:]
            interm = ran_key.split('/')[-1][:-2]
            
            pc_path = os.path.join(self.gt_path, ran_key.split('/')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.npy')
            view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(Image.open(view_path).convert('RGB'))
        views = views[:3, :, :]
        with open(pc_path, 'rb') as f:
            pc = np.load(f, allow_pickle=True).astype(np.float32)
        with open(pc_part_path, 'rb') as f:
            pc_part = np.load(f, allow_pickle=True).astype(np.float32)
        if pc_part.shape[0] < self.npoints:
            pc_part = np.repeat(pc_part, (self.npoints // pc_part.shape[0]) + 1, axis=0)[:self.npoints]
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = self.rotation_y(self.rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = self.rotation_x(self.rotation_y(pc_part, np.pi - theta_img), phi_img)

        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        result = {}
        result['partial'] = pc_part.astype(np.float32)
        result['complete'] = pc.astype(np.float32)
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])

        result['name'] = copy.deepcopy(self.key[idx]).replace("/", "_")
        result['image_tensor'] = views.numpy().astype(np.float32)

        return result

    def __len__(self):
        return len(self.key)
