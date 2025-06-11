import os
import cv2
import json
import torch
import numpy as np
from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config
from pycocotools import mask as coco_mask
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def parse_argments():
    import argparse
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-r', '--record-path', default='data/SAM_1B', type=str, help='record path')
    parser.add_argument('-s', '--save-path', default='dir_path/to/save', type=str, help='save path')
    parser.add_argument('-d', '--device', default='cuda:0', type=str, help='load model device')
    args = parser.parse_args()
    return args

# 扩大掩码区域
def enlarge_mask(mask, enlarge_val):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * enlarge_val + 1, 2 * enlarge_val + 1))
    enlarged_mask = cv2.dilate(mask, kernel)
    return enlarged_mask

# 获取掩码的边界框大小
def get_bbox_size(mask):
    rows, cols = np.where(mask != 0)[:2]
    if len(rows) == 0:
        row_min, row_max = 0, mask.shape[0]
    else:
        row_min, row_max = rows.min(), rows.max()
    if len(cols) == 0:
        col_min, col_max = 0, mask.shape[1]
    else:
        col_min, col_max = cols.min(), cols.max()
    w = col_max - col_min
    h = row_max - col_min
    return col_min, row_min, w, h

# 应用高斯模糊
def apply_gaussian_blur(image, sigma=1):
    size = int(2 * round(3 * sigma) + 1)
    blurred_image = cv2.GaussianBlur(image, (size, size), sigma)
    return blurred_image

# 调整图像对比度
def adjust_contrast(image, gamma=1.0):
    contrast = gamma
    brightness = 0
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image

# 处理前景图像，调整到目标尺寸并居中
def process_fg(fg, target_size):
    h, w = fg.shape[:2]
    if h > w:
        new_h, new_w = target_size, int(target_size * w / h)
    else:
        new_h, new_w = int(target_size * h / w), target_size
    white = np.full((target_size, target_size, 3), fill_value=255, dtype=np.uint8)
    fg = cv2.resize(fg, (new_w, new_h))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    white[top:top + new_h, left:left + new_w] = fg
    return white

# 从图像和掩码提取前景
def get_fg(image, bi_mask):
    fg_mask = np.array(bi_mask)
    fg = np.array(image)
    fg[bi_mask == 0] = 255
    fg_mask = apply_gaussian_blur(fg_mask)
    fg_mask = adjust_contrast(fg_mask)
    x, y, w, h = get_bbox_size(fg_mask)
    fg = fg[y:y + h, x:x + w]
    fg_mask = fg_mask.astype(np.uint8)
    return fg, fg_mask, [int(x), int(y), int(w), int(h)]

# 生成背景掩码
def get_bgmask(bi_mask):
    return enlarge_mask(bi_mask, 20)

def load_qwen_model_and_processor(device):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "models/qwen2vl",
        torch_dtype= torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained("models/qwen2vl")
    model.to(device)
    return model, processor

def is_valid_foreground(image_path, qwen_model, qwen_processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Determine if the subject in the image is complete, if it is complete and not an abstract object such as background, grass, sky, tree, stone and is not part of another item, please return True, otherwise return False."},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    result = output_text[0].strip().lower()
    return result == "true"

# 根据标注文件修复背景
def inpaint_by_ann(image_path, ann_path, inpaint_model, qwen_model, qwen_processor, device):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fg_list, bg_list, gt_list, bbox_list, fg_mask_list, ng_fg_list = [], [], [], [], [], []

    with open(ann_path, 'r') as f:
        anns = json.load(f)

    for idx, ann in enumerate(anns['annotations']):
        seg = ann['segmentation']
        bi_mask = coco_mask.decode(seg)  # 解码掩码 (前景为255，背景为0)
        bi_mask[bi_mask == 1] = 255
        # 计算比例
        height = seg['size'][0]
        width = seg['size'][1]
        total_pixel = height * width
        foreground_pixels = (bi_mask == 255).sum()
        ratio = foreground_pixels / total_pixel
        if ratio < 0.1 or ratio > 0.5:
            continue
        fg, fg_mask, bbox = get_fg(image, bi_mask)

        # 保存临时前景图像路径
        temp_fg_path = f"/tmp/temp_fg_{idx}.jpg"
        try:
            cv2.imwrite(temp_fg_path, fg)

            # 判断前景有效性
            if not is_valid_foreground(temp_fg_path, qwen_model, qwen_processor, device):
                fg = process_fg(fg, target_size=224)
                fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
                ng_fg_list.append(fg)
                continue

            fg = process_fg(fg, target_size=224)
            bg_mask = get_bgmask(bi_mask)
            h, w = bg_mask.shape[:2]

            tar_size = 720
            if max(h, w) == h:
                new_h, new_w = tar_size, int(w * tar_size / h)
            else:
                new_h, new_w = int(h * tar_size / w), tar_size
            bg_mask = cv2.resize(bg_mask, (new_w, new_h))

            inpainted_bg = inpaint_model(cv2.resize(image, (new_w, new_h)), 
                                        bg_mask, 
                                        Config(hd_strategy="Original", 
                                                ldm_steps=20,
                                                hd_strategy_crop_margin=128,
                                                hd_strategy_crop_trigger_size=800,
                                                hd_strategy_resize_limit=800))
            fg_list.append(cv2.cvtColor(fg, cv2.COLOR_RGB2BGR))
            bg_list.append(inpainted_bg)
            fg_mask_list.append(fg_mask)
            gt_list.append(image)
            bbox_list.append(bbox)
        except:
            continue

    return fg_list, bg_list, bbox_list, fg_mask_list, ng_fg_list, gt_list, height, width

def main():
    args = parse_argments()
    record_path, save_path, device = args.record_path, args.save_path, args.device
    print('save_path:', save_path)
    foreground_path = os.path.join(save_path, 'foreground')
    background_path = os.path.join(save_path, 'background')
    groundtruth_path = os.path.join(save_path, 'groundtruth')
    prompt_path = os.path.join(save_path, 'prompt')
    mask_path = os.path.join(save_path, 'mask')
    os.makedirs(foreground_path, exist_ok=True)
    os.makedirs(background_path, exist_ok=True)
    os.makedirs(prompt_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(groundtruth_path, exist_ok=True)

    record_list = []
    for file_name in os.listdir(record_path):
        if file_name.endswith('.json'):
            with open(os.path.join(record_path, file_name), 'r') as f:
                record_list.extend(json.load(f))

    inpaint_model = LaMa(device)
    qwen_model, qwen_processor = load_qwen_model_and_processor(device)
    pos_prompt = []
    print(len(record_list))
    quater = int(len(record_list)/4)
    for fig_idx, record in enumerate(tqdm(record_list[:quater])):
        image_path = record['image']
        ann_path = record['ann']
        fg_list, bg_list, bbox_list, fg_mask_list, ng_fg_list, gt_list, height, width = \
            inpaint_by_ann(image_path, ann_path, inpaint_model, qwen_model, qwen_processor, device)
        image_name = os.path.basename(image_path)[:-4]
        print('saving fig ', fig_idx)
        for idx, (fg, bg, gt, fg_mask, bbox) in \
                enumerate(zip(fg_list, bg_list, gt_list, fg_mask_list, bbox_list)):
            save_name = f'{image_name}_{idx}.jpg'
            cv2.imwrite(os.path.join(foreground_path, save_name), fg)
            cv2.imwrite(os.path.join(background_path, save_name), bg)
            cv2.imwrite(os.path.join(mask_path, save_name), fg_mask)
            cv2.imwrite(os.path.join(groundtruth_path, save_name), gt)
            pos_prompt.append({
                'image':os.path.join(groundtruth_path, save_name),
                'width':width,
                'height':height,
                'entity_path' : [os.path.join(foreground_path, save_name)],
                'source_path' :[os.path.join(background_path, save_name)],
                'bbox': bbox,
                'mask_path': [os.path.join(mask_path, save_name)]
                })
    with open(os.path.join(prompt_path, 'prompt_1.json'), 'w') as f:
        json.dump(pos_prompt, f)

if __name__ == "__main__":
    main()
