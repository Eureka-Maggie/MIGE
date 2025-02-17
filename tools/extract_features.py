import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
from pathlib import Path

from diffusers.models import AutoencoderKL
from diffusion.utils.misc import SimpleTimer


def extract_img_vae_do(q):
    while not q.empty():
        item = q.get()
        extract_img_vae_job(item)
        q.task_done()


def extract_img_vae_job(item):
    return


def extract_img_vae():
    vae = AutoencoderKL.from_pretrained(f'{args.pretrained_models_dir}/sd-vae-ft-ema').to(device)

    train_data_json = json.load(open(args.json_path, 'r'))
    image_names = set()

    vae_save_root = f'{args.vae_save_root}/{image_resize}resolution'
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(vae_save_root, exist_ok=True)

    vae_save_dir = os.path.join(vae_save_root, 'noflip')
    os.makedirs(vae_save_dir, exist_ok=True)

    for items in train_data_json:
        for item in items['source_path']:
            image_name = item
            if image_name in image_names:
                continue

    lines = sorted(image_names)
    print(len(lines))
    lines = lines[args.start_index: args.end_index]

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize), 
        T.CenterCrop(image_resize), 
        T.ToTensor(), 
        T.Normalize([.5], [.5]), 
    ])

    os.umask(0o000)  # file permission: 666; dir permission: 777
    for image_name in tqdm(lines): #'images/0.jpg'
        save_path = os.path.join(vae_save_dir, Path(image_name).stem)
        if os.path.exists(f"{save_path}.npy"):
            continue
        try:
            img = Image.open(image_name)
            img = transform(img).to(device)[None] #[1,3,256,256]的tensor

            with torch.no_grad():
                posterior = vae.encode(img).latent_dist
                z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze() #[8,32,32]

            np.save(save_path, z)
        except Exception as e:
            print(e)
            print(image_name)


def save_results(results, paths, signature, work_dir):
    timer = SimpleTimer(len(results), log_interval=100, desc="Saving Results")
    # save to npy
    new_paths = []
    os.umask(0o000)  # file permission: 666; dir permission: 777
    for res, p in zip(results, paths):
        file_name = p.split('.')[0] + '.npy'
        new_folder = signature
        save_folder = os.path.join(work_dir, new_folder)
        if os.path.exists(save_folder):
            raise FileExistsError(f"{save_folder} exists. BE careful not to overwrite your files. Comment this error raising for overwriting!!")
        os.makedirs(save_folder, exist_ok=True)
        new_paths.append(os.path.join(new_folder, file_name))
        np.save(os.path.join(save_folder, file_name), res)
        timer.log()
    # save paths
    with open(os.path.join(work_dir, f"VAE-{signature}.txt"), 'w') as f:
        f.write('\n'.join(new_paths))


def inference(vae, dataloader, signature, work_dir):
    timer = SimpleTimer(len(dataloader), log_interval=100, desc="VAE-Inference")

    for batch in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                posterior = vae.encode(batch[0]).latent_dist
                results = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        path = batch[1]
        save_results(results, path, signature=signature, work_dir=work_dir)
        timer.log()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=512, type=int, help="image scale for multi-scale feature extraction")
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=4000000, type=int)
    parser.add_argument('--json_path', default='train_scripts/test_data.json',type=str)
    parser.add_argument('--vae_save_root', default='example_data/vae', type=str)
    parser.add_argument('--pretrained_models_dir', default='output/pretrained_models', type=str)
    parser.add_argument('--json_file', type=str)
    
    return parser.parse_args()

# vae_entity, source_image, and target_image should each have a separate folder to store the corresponding VAE features.
# You can modify the for item in items['source_path']:(line 44) loop to determine the type of image you are processing.
# It is recommended to store data for different tasks in separate JSON files. The example_data.json, which mixes all types of data, is primarily for demonstration purposes.


if __name__ == '__main__':
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_resize = args.img_size
    print(args.start_index)
    print(args.end_index)
    print(args.json_path)

    print(f'Extracting Single Image Resolution {image_resize}')
    extract_img_vae()