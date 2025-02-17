# from __future__ import all_feature_names
from transformers import CLIPModel, CLIPProcessor
import torch
from tqdm.auto import tqdm
from PIL import Image
import math

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
import torch.nn as nn
import json

def batchify(data, batch_size=16):
    one_batch = []
    for example in data:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            yield one_batch
            one_batch = []
    if one_batch:
        yield one_batch


def compute_cosine_distance(image_features, image_features2):
    # normalized features
    image_features = image_features / np.linalg.norm(image_features, ord=2)
    image_features2 = image_features2 / np.linalg.norm(image_features2, ord=2)
    return np.dot(image_features, image_features2)

class clip_dino:
    def __init__(self,device) -> None:
        
        self.model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        self.model_dino.eval()
        self.device = device

        self.model =  CLIPModel.from_pretrained("/path/to/clip-vit-base-patch32").to(self.device)
        self.model.eval()
        #self.processor = CLIPProcessor.from_pretrained(clip_model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained("/path/to/clip-vit-base-patch32")
        self.processor.tokenizer.pad_token_id = 0

    def get_transform(self):
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return val_transform

    def get_embeddings(self, tensor_image):
        output = self.model_dino(tensor_image.to(self.device))
        return output


    def compute_dino(self, images_path2, images_path):

        real_image = Image.open(images_path2).convert('RGB')
        generated_image = Image.open(images_path).convert('RGB')
        preprocess = self.get_transform()
        tensor_image_1 = preprocess(real_image).unsqueeze(0) #[1,3,224,224]
        tensor_image_2 = preprocess(generated_image).unsqueeze(0)
        emb_1 = self.get_embeddings(tensor_image_1.float().to(self.device)) #[1,384]
        emb_2 = self.get_embeddings(tensor_image_2.float().to(self.device))
        assert emb_1.shape == emb_2.shape
        score = compute_cosine_distance(emb_1.detach().cpu().numpy(), emb_2.detach().permute(1, 0).cpu().numpy())
        return score[0][0]

    @torch.no_grad()
    def compute_clip_T_from_folder(self, image_folder, json_file, batch_size=64):
        """
        Compute CLIP-T scores for images in a folder and captions in a JSON file.

        Args:
            image_folder (str): Path to the folder containing images.
            json_file (str): Path to the JSON file with captions and metadata.
            batch_size (int): Number of images to process in a batch.

        Returns:
            float: Average CLIP-T score for the dataset.
        """
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Prepare paths and captions
        images_path = []
        captions = []
        for entry in data:
            idx = entry['idx']
            image_path = os.path.join(image_folder, f"{idx}.jpg")
            if entry['input_caption'] == entry['output_caption']:
                continue
            if os.path.exists(image_path):
                images_path.append(image_path)
                captions.append(entry['output_caption'])
        print(len(captions))

        all_text_embeds = []
        all_image_embeds = []

        # Process in batches
        for captions_batch, images_batch in tqdm(
            zip(batchify(captions, batch_size), batchify(images_path, batch_size)), 
            total=math.ceil(len(captions) / batch_size)
        ):
            # Prepare inputs for CLIP processor
            batch_inputs = self.processor(
                text=captions_batch,
                images=[Image.open(image).convert('RGB') for image in images_batch],
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Compute text and image embeddings
            text_embeds = self.model.get_text_features(input_ids=batch_inputs["input_ids"])
            image_embeds = self.model.get_image_features(pixel_values=batch_inputs["pixel_values"])

            all_text_embeds.append(text_embeds)
            all_image_embeds.append(image_embeds)

        # Concatenate all embeddings
        all_text_embeds = torch.cat(all_text_embeds)
        all_image_embeds = torch.cat(all_image_embeds)

        # Normalize embeddings
        all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)
        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity scores
        clip_scores = (all_image_embeds * all_text_embeds).sum(dim=-1)

        return clip_scores.mean().item()

    @torch.no_grad()
    def compute_clip_T_from_folder_my(self, image_folder, json_file, batch_size=64):
        """
        Compute CLIP-T scores for images in a folder and captions in a JSON file.

        Args:
            image_folder (str): Path to the folder containing images.
            json_file (str): Path to the JSON file with captions and metadata.
            batch_size (int): Number of images to process in a batch.

        Returns:
            float: Average CLIP-T score for the dataset.
        """
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Prepare paths and captions
        images_path = []
        captions = []
        for entry in data:
            idx = entry['idx']
            image_path = os.path.join(image_folder, f"{idx}.jpg")
            if os.path.exists(image_path):
                images_path.append(image_path)
                captions.append(entry['target_caption'])
        print(len(captions))

        all_text_embeds = []
        all_image_embeds = []

        # Process in batches
        for captions_batch, images_batch in tqdm(
            zip(batchify(captions, batch_size), batchify(images_path, batch_size)), 
            total=math.ceil(len(captions) / batch_size)
        ):
            # Prepare inputs for CLIP processor
            batch_inputs = self.processor(
                text=captions_batch,
                images=[Image.open(image).convert('RGB') for image in images_batch],
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Compute text and image embeddings
            text_embeds = self.model.get_text_features(input_ids=batch_inputs["input_ids"])
            image_embeds = self.model.get_image_features(pixel_values=batch_inputs["pixel_values"])

            all_text_embeds.append(text_embeds)
            all_image_embeds.append(image_embeds)

        # Concatenate all embeddings
        all_text_embeds = torch.cat(all_text_embeds)
        all_image_embeds = torch.cat(all_image_embeds)

        # Normalize embeddings
        all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)
        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity scores
        clip_scores = (all_image_embeds * all_text_embeds).sum(dim=-1)

        return clip_scores.mean().item()



    @torch.no_grad()
    def compute_clip_I(self, images_path2, images_path, batch_size=64):
        all_image2_embeds = []
        all_image_embeds = []
        for image_path2, image_path in tqdm(
            zip(batchify(images_path2, batch_size), batchify(images_path, batch_size)), total=math.ceil(len(images_path2) / batch_size)
        ):
            assert len(image_path2) == len(image_path)
            batch_inputs2 = self.processor(
                text="",
                images=[Image.open(image) for image in image_path2],
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)
            batch_inputs = self.processor(
                text="",
                images=[Image.open(image) for image in image_path],
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)
            image2_embeds = self.model.get_image_features(pixel_values=batch_inputs2["pixel_values"])
            image_embeds = self.model.get_image_features(pixel_values=batch_inputs["pixel_values"])
            all_image2_embeds.append(image2_embeds)
            all_image_embeds.append(image_embeds)

        all_image2_embeds = torch.concat(all_image2_embeds) #[3000,512]
        all_image_embeds = torch.concat(all_image_embeds)
        all_image2_embeds = all_image2_embeds / all_image2_embeds.norm(dim=-1, keepdim=True)
        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
        clip_score = (all_image_embeds * all_image2_embeds).sum(-1)
        return clip_score.mean().item()
    
    @torch.no_grad()
    def eval_distance(self,image_files, image_files2, metric='l1'):
        """
        Evaluate L1 or L2 distance between image pairs one by one.
        
        Args:
            image_files (list): List of generated image file paths.
            image_files2 (list): List of ground truth image file paths.
            metric (str): Distance metric, either 'l1' or 'l2'.
            
        Returns:
            float: Average distance over all image pairs.
        """
        if metric == 'l1':
            criterion = nn.L1Loss()
        elif metric == 'l2':
            criterion = nn.MSELoss()
        else:
            raise ValueError("Unsupported metric. Use 'l1' or 'l2'.")

        eval_score = 0
        total_pairs = len(image_files)

        for gen_path, gt_path in tqdm(zip(image_files, image_files2), total=total_pairs):
            gen_img = Image.open(gen_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            # Resize generated image to match ground truth size
            gen_img = gen_img.resize(gt_img.size)
            
            # Convert to tensor
            gen_img = transforms.ToTensor()(gen_img)
            gt_img = transforms.ToTensor()(gt_img)

            # Calculate distance and accumulate score
            per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
            eval_score += per_score

        return eval_score / total_pairs

    
    @torch.no_grad()
    def compute_clip_dir(self, images_path, images_path2, input_captions, output_captions, batch_size=64):
        """
        Compute CLIPdir score.

        Args:
            images_path (list): List of paths to input images.
            images_path2 (list): List of paths to generated images.
            input_captions (list): List of input captions.
            output_captions (list): List of output captions.
            batch_size (int): Batch size for processing.

        Returns:
            float: CLIPdir score.
        """
        all_text1_embeds = []
        all_text2_embeds = []
        all_image1_embeds = []
        all_image2_embeds = []

        for image_path, image_path2, input_caption, output_caption in tqdm(
            zip(
                batchify(images_path, batch_size),
                batchify(images_path2, batch_size),
                batchify(input_captions, batch_size),
                batchify(output_captions, batch_size),
            ),
            total=math.ceil(len(images_path2) / batch_size),
        ):
            assert len(image_path2) == len(image_path) == len(input_caption) == len(output_caption)

            # Prepare image inputs
            batch_inputs1 = self.processor(
                text="",
                images=[Image.open(image) for image in image_path],
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            batch_inputs2 = self.processor(
                text="",
                images=[Image.open(image) for image in image_path2],
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Prepare text inputs
            batch_text1 = self.processor(
                text=input_caption,
                images=None,
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            batch_text2 = self.processor(
                text=output_caption,
                images=None,
                return_tensors="pt",
                max_length=self.processor.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Extract embeddings
            image1_embeds = self.model.get_image_features(pixel_values=batch_inputs1["pixel_values"]) #[64,512]
            image2_embeds = self.model.get_image_features(pixel_values=batch_inputs2["pixel_values"])
            text1_embeds = self.model.get_text_features(input_ids=batch_text1["input_ids"])
            text2_embeds = self.model.get_text_features(input_ids=batch_text2["input_ids"])

            # Normalize embeddings
            image1_embeds = image1_embeds / image1_embeds.norm(dim=-1, keepdim=True)
            image2_embeds = image2_embeds / image2_embeds.norm(dim=-1, keepdim=True)
            text1_embeds = text1_embeds / text1_embeds.norm(dim=-1, keepdim=True)
            text2_embeds = text2_embeds / text2_embeds.norm(dim=-1, keepdim=True)

            all_image1_embeds.append(image1_embeds)
            all_image2_embeds.append(image2_embeds)
            all_text1_embeds.append(text1_embeds)
            all_text2_embeds.append(text2_embeds)

        # Concatenate all embeddings
        all_image1_embeds = torch.cat(all_image1_embeds)
        all_image2_embeds = torch.cat(all_image2_embeds)
        all_text1_embeds = torch.cat(all_text1_embeds)
        all_text2_embeds = torch.cat(all_text2_embeds)

        # Compute directional consistency
        image_delta = all_image2_embeds - all_image1_embeds
        text_delta = all_text2_embeds - all_text1_embeds

        image_delta = image_delta / image_delta.norm(dim=-1, keepdim=True)
        text_delta = text_delta / text_delta.norm(dim=-1, keepdim=True)

        clip_dir_score = (image_delta * text_delta).sum(dim=-1).mean().item()

        return clip_dir_score