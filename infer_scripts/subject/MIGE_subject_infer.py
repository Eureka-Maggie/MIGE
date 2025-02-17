import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion import DPMS
from diffusion.model.nets import MIGE_XL_2
import random
import re
import json
from lavis.processors.blip_processors import BlipCaptionProcessor
from lavis.processors.blip_processors import BlipImageEvalProcessor
from PIL import Image
from transformers import AutoImageProcessor
from torchvision import transforms as T
from PIL import Image, ImageOps
from diffusers.utils.torch_utils import randn_tensor

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)

    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--txt_file', default='asset/samples.txt', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/model.pth', type=str)
    parser.add_argument('--bs', default=5, type=int)
    parser.add_argument('--cfg_scale_y', default=7, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['dpm-solver'])
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--multi_folder_path', default='infer_scripts/subject/example_dreambooth.json', type=str)
    parser.add_argument('--save_root', default='infer_scripts/subject/example_db_results', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    return parser.parse_args()


def set_env(seed):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

def safe_filename(caption):
    caption = caption.replace('/', ' or ')
    safe_caption = re.sub(r'[\\/*?:"<>|]', '_', caption)
    return safe_caption[:100]


@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale_y, cfg_scale_s=1.0):
    
    idx = 0
    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):
        
        cond_prompts = []
        if bs == 1:
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            cond_prompts.append({'prompt':prompt['prompt'],'source':prompt['source'],'ref_clip':prompt['ref_clip'],'ref_vae':prompt['ref_vae']})
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)

            for prompt in chunk:
                cond_prompts.append({'prompt':prompt['prompt'],'source':prompt['source'],'ref_clip':prompt['ref_clip'],'ref_vae':prompt['ref_vae']})
            latent_size_h, latent_size_w = latent_size, latent_size #64

        with torch.no_grad():

            if args.sampling_algo == 'dpm-solver':
                n = len(cond_prompts) #bs
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device) #[bs,4,64,64]
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=None)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition = cond_prompts,
                                  uncondition = "subject",
                                  cfg_scale_y = cfg_scale_y,
                                  model_kwargs = model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )

        samples = vae.decode(samples / 0.18215).sample #[1,4,32,32]->[1,3,256,256]

        # Save images:
        os.umask(0o000)
        for i, sample in enumerate(samples):
            entity_path = items[idx]['entity_path'][0]
            multi_prompt = items[idx]['multi_prompt']
            seed = args.seed
            base_name = os.path.splitext(os.path.basename(entity_path))[0]
            safe_prompt = safe_filename(multi_prompt)
            img_name = f"{base_name}_{safe_prompt}_{seed}"
            save_path = os.path.join(args.save_root, f"{img_name}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
            idx += 1


def resize_with_padding(img, target_size):
    width, height = img.size
    if width > height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)

    img = img.resize((new_width, new_height), Image.BICUBIC)

    padding_left = (target_size - new_width) // 2
    padding_top = (target_size - new_height) // 2
    padding_right = target_size - new_width - padding_left
    padding_bottom = target_size - new_height - padding_top

    img = ImageOps.expand(img, (padding_left, padding_top, padding_right, padding_bottom), (0, 0, 0))

    return img


def vae_encode(vae,image):
    vae_transform = T.Compose([
        T.Lambda(lambda img: resize_with_padding(img, 512)),
        T.ToTensor(), 
        T.Normalize([.5], [.5]), 
    ])
    img = vae_transform(image)[None]
    img = img.to(device)
    with torch.no_grad():
        posterior = vae.encode(img).latent_dist
        mean = posterior.mean.detach().cpu().squeeze()
        std = posterior.std.detach().cpu().squeeze()
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
    return mean + std * sample


if __name__ == '__main__':
    args = get_args()
    device = args.device
    seed = args.seed
    set_env(seed)
    print(seed)
    
    assert args.sampling_algo in ['dpm-solver']

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)

    latent_size = args.image_size // 8
    lewei_scale = {512: 1} 
    sample_steps_dict = {'dpm-solver': 50}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.bfloat16
    print(f"Inference with {weight_dtype}")

    model = MIGE_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size])
    
    print(f"Generating sample from ckpt: {args.model_path}")

    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    missing,unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)

    model = model.to(device)
    model.eval()
    model.to(weight_dtype) #torch.float16

    
    with open(args.multi_folder_path, 'r') as file:
        data_list = json.load(file)

    print('loading file_contents done')
    print('file_contents\'s length',len(data_list))
    
    # 处理prompt
    text_processor = BlipCaptionProcessor.from_config(None)
    cfg_vis  = {'name': 'blip_image_eval', 'image_size': 224}
    vis_processor = BlipImageEvalProcessor.from_config(cfg_vis)
    
    ref_list = []
    for idx, multi_prompt in enumerate(data_list):
        refs_clip = []
        refs_vae = []
        
        prefix = 'Follow the text instruction to generate an image with the given objects\' images, preserving the subject\'s identity:\n '
        text = text_processor(prefix + multi_prompt['multi_prompt'])
        ref_paths = []
        for entity_path in multi_prompt['entity_path']:
            ref_paths.append(entity_path)

        try:
            for image_path in ref_paths:
                image = Image.open(image_path).convert("RGB")
                image_blip2 = vis_processor(image) #[3,224,224]
                refs_clip.append(image_blip2)
                refs_vae.append(vae_encode(vae, image))
        except:
            print('wrong path')
            continue

        while len(refs_clip)<4:
            refs_clip.append(torch.zeros(3, 224, 224))
            refs_vae.append(torch.zeros(4, 64, 64))

        refs_clip = torch.stack(refs_clip, dim=0).to(weight_dtype).to(device)
        refs_vae = torch.stack(refs_vae, dim=0).to(weight_dtype).to(device)
        source = torch.zeros(4,64,64)

        ref_list.append({'prompt': text, 'source':source, 'ref_clip': refs_clip, 'ref_vae': refs_vae, 'index': idx, 'entity_path': multi_prompt['entity_path'], 'multi_prompt': multi_prompt['multi_prompt']})
    
    items = ref_list

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*.pth', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*.pth', args.model_path).group(1)
    except Exception:
        epoch_name = 'unknown'
        step_name = 'unknown'
        
    os.umask(0o000)

    os.makedirs(args.save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale_y)



