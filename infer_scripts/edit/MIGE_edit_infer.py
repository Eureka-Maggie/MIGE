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
from diffusers.utils.torch_utils import randn_tensor

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)

    parser.add_argument('--tokenizer_path', default='output/pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/model.pth', type=str)
    parser.add_argument('--bs', default=25, type=int)
    parser.add_argument('--cfg_scale_y', default=5, type=float)
    parser.add_argument('--cfg_scale_s', default=1, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['dpm-solver'])
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--multi_folder_path', default='infer_scripts/edit/emu_edit.json', type=str)
    parser.add_argument('--save_root', default='infer_scripts/edit/example_edit_results', type=str)
    parser.add_argument('--device', default='cuda:2', type=str)

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
def visualize(items, bs, sample_steps, cfg_scale_y, cfg_scale_s):
    
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
            cond_prompts.append({'prompt':chunk[0]['prompt'],'ref':chunk[0]['entity']})
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)

            for prompt in chunk:
                cond_prompts.append({'prompt':prompt['prompt'],'ref_clip':prompt['ref_clip'],'ref_vae':prompt['ref_vae'],'source':prompt['source']})
            latent_size_h, latent_size_w = latent_size, latent_size 
        with torch.no_grad():
            if args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(cond_prompts) #bs
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device) #[bs,4,64,64]
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=None)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition = cond_prompts,
                                  uncondition = "edit",
                                  cfg_scale_y = cfg_scale_y,
                                  cfg_scale_s = cfg_scale_s,
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
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            img_name = f"{items[idx]['index']}" 
            save_path = os.path.join(args.save_root, f"{img_name}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
            idx += 1

def vae_encode(vae,image):
    vae_transform = T.Compose([
        #T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(512),
        T.CenterCrop(512), 
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


def update_instruction(instruction, task):
    for keyword in replace_keywords:
        instruction = re.sub(rf"\s*{keyword.strip()}\s*", " <imagehere> ", instruction, flags=re.IGNORECASE)

    if "<imagehere>" not in instruction:
        if task == "global":
            instruction = instruction.rstrip(".") + " of<imagehere>"
        else:
            instruction = instruction.rstrip(".") + " in<imagehere>"

    return instruction


if __name__ == '__main__':
    args = get_args()
    device = args.device
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    print(seed)
    
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    lewei_scale = {256:1, 512: 1, 1024: 2}     # trick for positional embedding interpolation
    sample_steps_dict = {'iddpm': 20, 'dpm-solver': 60, 'sa-solver': 25} # 100,20,25
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.bfloat16
    print(f"Inference with {weight_dtype}")

    model = MIGE_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    
    print(f"Generating sample from ckpt: {args.model_path}")
    
    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    missing,unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print(missing)
    model.eval()
    model.to(weight_dtype) #torch.float16

    # 文件夹路径
    
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
        idx = data_list[idx]['idx']
        refs_clip = []
        refs_vae = []
         
        replace_keywords = ["the image", "this image", "this photo", "the photo", "the picture", "this picture"]
        
        text = 'Follow the instruction to edit the image:\n '+ text_processor(update_instruction(multi_prompt['instruction'],multi_prompt['task'])) #'a man is sitting at a table with a<imagehere>in front of him he is holding a cell phone in his hand, possibly texting or browsing'

        print(text)
        ref_paths = []
        #for entity_path in multi_prompt['image']:
        ref_paths.append(multi_prompt['image'])
            #print(entity_path)
        try:
            for image_path in ref_paths:
                image = Image.open(image_path).convert("RGB")
                image_blip2 = vis_processor(image) #[3,224,224]
                refs_clip.append(image_blip2)
                vae_fea = vae_encode(vae, image)
                source = vae_fea * 0.18215
                refs_vae.append(vae_fea)

        except:
            print('wrong path')
            continue

        while len(refs_clip)<4:
            refs_clip.append(torch.zeros(3, 224, 224))
            refs_vae.append(torch.zeros(4, 64, 64))

        refs_clip = torch.stack(refs_clip, dim=0).to(weight_dtype).to(device)
        refs_vae = torch.stack(refs_vae, dim=0).to(weight_dtype).to(device)

        ref_list.append({'prompt': text, 'source': source, 'ref_clip': refs_clip, 'ref_vae': refs_vae, 'index': idx, 'entity_path': multi_prompt['image'], 'multi_prompt': multi_prompt['instruction']})
    
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
    visualize(items, args.bs, sample_steps, args.cfg_scale_y, args.cfg_scale_s)

