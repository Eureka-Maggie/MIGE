import os
import random
from PIL import Image
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from diffusion.data.builder import get_data_path, DATASETS
from diffusion.utils.logger import get_root_logger

import json


from lavis.datasets.datasets.base_dataset import BaseDataset
from transformers import AutoImageProcessor, AutoModel



@DATASETS.register_module()
class MixData(BaseDataset):
    def __init__(self,
                 root,
                 vis_processor, 
                 text_processor, 
                 image_list_json=None,
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 load_mask_index=False,
                 max_length=120,
                 config=None,
                 **kwargs):
        super().__init__(vis_processor, text_processor) 
        self.root = get_data_path(root) 
        self.transform = transform
        self.load_vae_feat = load_vae_feat # true
        self.ori_imgs_nums = 0
        self.resolution = resolution # 512
        self.N = int(resolution // (input_size // patch_size)) # 16
        self.mask_ratio = mask_ratio # 0.0
        self.load_mask_index = load_mask_index #false
        self.max_lenth = max_length #256
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.prompt_samples = []
        self.data_types =[] 

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(os.path.dirname(self.root),json_file)) 
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['width']/item['height'] <= 4]
            self.meta_data_clean.extend(meta_data_clean) 
            self.img_samples.extend([item['image'] for item in meta_data_clean]) 
            for item in meta_data_clean: #target
                target = item['image']
                file_name = os.path.splitext(target.rsplit('/', 1)[1])[0] + '.npy'
                if 'Subjects200K' in target:
                    self.vae_feat_samples.extend([os.path.join(self.root, 'vae', f'vae_subject200k/{resolution}resolution/noflip', file_name)])
                    self.data_types.append('subject')
                elif 'instructpix2pix' in target:
                    self.vae_feat_samples.extend([os.path.join(self.root, 'vae', f'vae_instructpix/{resolution}resolution/noflip', file_name)])
                    self.data_types.append('edit')
                elif 'replace' in target:
                    self.vae_feat_samples.extend([os.path.join(self.root, 'vae', f'vae_replace/{resolution}resolution/noflip', file_name)])
                    self.data_types.append('subject_edit')
                elif 'add' in target:
                    self.vae_feat_samples.extend([os.path.join(self.root, 'vae', f'vae_add/{resolution}resolution/noflip', file_name)])
                    self.data_types.append('subject_edit')                    

        # Set loader and extensions
        if load_vae_feat: #true
            self.transform = None
            self.loader = self.vae_feat_loader #
        else:
            self.loader = default_loader

        if sample_subset is not None: #no
            self.sample_subset(sample_subset)  # sample dataset for local debug
        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.info(f"T5 max token length: {self.max_lenth}") #256

    def getdata(self, index):
        img_path = self.img_samples[index]

        npy_path = self.vae_feat_samples[index]
        data_info = {
            'img_hw': torch.tensor([torch.tensor(self.resolution), torch.tensor(self.resolution)], dtype=torch.float32),
            'aspect_ratio': torch.tensor(1.)
        }

        img = self.loader(npy_path) if self.load_vae_feat else self.loader(img_path) 
        
        ## source image
        caption = self.text_processor(self.meta_data_clean[index]["multi_prompt"])
        if self.data_types[index] == 'edit':
            caption = "Follow the instruction to edit the image:\n " + caption 
            source_image_path = self.meta_data_clean[index]['source_path'][0]
            file_name = os.path.splitext(image_path.rsplit('/', 1)[1])[0] + '.npy'

            if 'instructpix2pix' in image_path:
                source = self.loader(os.path.join(self.root, 'vae_entity', f'vae_instructpix/{self.resolution}resolution/noflip', file_name))


        elif self.data_types[index] == 'subject':
            caption = "Follow the text instruction to generate an image with the given objects, preserving the subject's identity:\n " + caption
            source = torch.zeros(4,64,64)

        elif self.data_types[index] == 'subject_edit':
            caption = "Subject-driven editing:\n " + caption
            source_image_path = self.meta_data_clean[index]['source_path'][0]
            image_path = source_image_path
            file_name = os.path.splitext(image_path.rsplit('/', 1)[1])[0] + '.npy'
            if 'replace' in image_path:
                source = self.loader(os.path.join(self.root, 'vae_source', f'vae_replace/{self.resolution}resolution/noflip', file_name))
            elif 'add' in image_path:
                source = self.loader(os.path.join(self.root, 'vae_source', f'vae_add/{self.resolution}resolution/noflip', file_name))
        
        ref_list = self.meta_data_clean[index]['entity_path']

        ref_paths = []
        refs_clip = []
        refs_vae = []
        for entity_path in ref_list:
            ref_paths.append(entity_path)
        for image_path in ref_paths:
            image = Image.open(image_path).convert("RGB")
            image_blip2 = self.vis_processor(image) #[3,224,224]
            refs_clip.append(image_blip2)

            file_name = os.path.splitext(image_path.rsplit('/', 1)[1])[0] + '.npy'
            if 'Subjects200K' in image_path:
                refs_vae.append(self.loader(os.path.join(self.root, 'vae_entity', f'vae_subject200k/{self.resolution}resolution/noflip', file_name)))

            elif 'instructpix2pix' in image_path:
                refs_vae.append(self.loader(os.path.join(self.root, 'vae_entity', f'vae_instructpix/{self.resolution}resolution/noflip', file_name)))

            elif 'replace' in image_path:
                refs_vae.append(self.loader(os.path.join(self.root, 'vae_entity', f'vae_replace/{self.resolution}resolution/noflip', file_name)))

            elif 'add' in image_path:
                refs_vae.append(self.loader(os.path.join(self.root, 'vae_entity', f'vae_add/{self.resolution}resolution/noflip', file_name)))


        while len(refs_clip) < 4:
            refs_clip.append(torch.zeros(3, 224, 224))
            refs_vae.append(torch.zeros(4, 64, 64))
            
        if self.transform: #no
            img = self.transform(img)

        refs_clip = torch.stack(refs_clip, dim=0)
        refs_vae = torch.stack(refs_vae, dim=0)
        
        return img, refs_clip, refs_vae, caption, source, data_info 

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def get_data_type(self, idx):
        return self.data_types[idx]

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        #print((mean+std*sample).size())
        return mean + std * sample

    def load_ori_img(self, img_path):
        transform = T.Compose([
            T.Resize(256),  # Image.BICUBIC
            T.CenterCrop(256),
            T.ToTensor(),
        ])
        return transform(Image.open(img_path))

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

