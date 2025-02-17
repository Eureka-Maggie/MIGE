import os
import torch
from PIL import Image
from tqdm import tqdm
from segment_anything import SamPredictor, build_sam
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import load_image, predict
from huggingface_hub import hf_hub_download
import json
import shutil
import numpy as np
import argparse


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

@torch.no_grad()
def get_bbox(object, image, groundingdino_model, BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE):
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=object,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE
    )
    combined_list = [{
        "box": box,
        "logit": logit.item(), 
        "phrase": phrase
    } for box, logit, phrase in zip(boxes, logits, phrases)]
    
    return combined_list

@torch.no_grad()
def get_masks(image_source, entity, sam_predictor, DEVICE):
    H, W, _ = image_source.shape
    sam_predictor.set_image(image_source)
    
    box = torch.tensor(entity['box']).round(decimals=4)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    mask, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    entity['mask'] = mask
            
    return entity

def crop_image(img, bbox, output_dir, idx):
    left, top, right, bottom = bbox
    cropped_img = img.crop((left, top, right, bottom)).convert('RGB')
    
    entity_folder = os.path.join(args.entity_output_dir, 'entity')
    os.makedirs(entity_folder, exist_ok=True)
    entity_path = os.path.join(entity_folder, f"{idx}_entity.jpg")
    cropped_img.save(entity_path)
    
    return entity_path, cropped_img

def process_image(item, groundingdino_model, BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE, sam_predictor, entity_output_dir):
    try:
        idx = item['idx']
        image_path = os.path.join(entity_output_dir, f"{idx}.jpg")
        object = item['object']

        image_source, image = load_image(image_path)

        combined_list = get_bbox(object, image, groundingdino_model, BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE)
        
        if not combined_list:
            entity_folder = os.path.join(entity_output_dir, 'entity')
            os.makedirs(entity_folder, exist_ok=True)
            shutil.copy(image_path, os.path.join(entity_output_dir, 'entity', f"{idx}_entity.jpg"))
            return None 

        filtered_entities = max(combined_list, key=lambda x: x['logit'])

        filtered_entities = get_masks(image_source, filtered_entities, sam_predictor, DEVICE)

        H, W, _ = image_source.shape        
        x_c,y_c,w,h = filtered_entities['box']
        bbox = [((x_c - 0.5 * w)*W).item(), ((y_c - 0.5 * h)*H).item(), ((x_c + 0.5 * w)*W).item(), ((y_c + 0.5 * h)*H).item()]
        mask = filtered_entities['mask'].squeeze().cpu().numpy()

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_array[~mask] = 0 
        combined_img = Image.fromarray(img_array)
        
        entity_path, _ = crop_image(combined_img, bbox, entity_output_dir, idx)
        
        return [entity_path]
    
    except Exception as e:
        print(f"Error processing {item['idx']}: {e}")
        return []

def process_all_images(data, groundingdino_model, BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE, sam_predictor, entity_output_dir):
    all_entity_paths = []
    
    for item in tqdm(data):
        entity_paths = process_image(item, groundingdino_model, BOX_TRESHOLD, TEXT_TRESHOLD, DEVICE, sam_predictor, entity_output_dir)
        if entity_paths:
            all_entity_paths.extend(entity_paths)
    
    return all_entity_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Process images and captions.")
    parser.add_argument('--caption_file_path', default='/path/to/MIGEbench/add_bench.json',type=str, help='Path to the caption file.')
    parser.add_argument('--entity_output_dir', default='/path/to/your/generated/images/folder/on/MIGEbench',type=str, help='Directory to save the entity images.')
    parser.add_argument('--device_idx', type=int, default=0, help='CUDA device index')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args.entity_output_dir)
    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{args.device_idx}')
        torch.cuda.set_device(args.device_idx)
    else:
        DEVICE = torch.device('cpu')

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, DEVICE).to(DEVICE)

    # Download SAM: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    sam_checkpoint = '/path/to/sam/model'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    with open(args.caption_file_path, 'r') as f:
        data = json.load(f)


    entity_output_dir = args.entity_output_dir
    print(entity_output_dir)
    all_entity_paths = process_all_images(data, groundingdino_model, 0.3, 0.25, DEVICE, sam_predictor, entity_output_dir)

    print(f"Processed {len(all_entity_paths)} entity images.")
