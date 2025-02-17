_base_ = ['basic.py']
data_root = 'example_data'
# EXAMPLE:
# - data_root
#     - test_data
#     - vae
#         - vae_test_data(extracted vae feature by tools/extract_features.py)
image_list_json = ['train_scripts/test_data.json']

data = dict(type='MixData', root='', image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 512

# model setting
window_block_indexes=[]
window_size = 0
use_rel_pos = False 
model = 'MIGE_XL_2'
fp32_attention = False 
load_from = None
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"

# training setting
num_workers = 8 
train_batch_size = 8 
num_epochs = 20
gradient_accumulation_steps = 1 
grad_checkpointing = True 
gradient_clip = 0.01 
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps = 1000)

log_interval = 10 
save_model_epochs = 1 

model_max_length = 256

load_from = True
mixed_precision = 'bf16'

vit_path = 'output/pretrained_models/eva_vit_g.pth'
blip2_path = 'output/pretrained_models/blip2_pretrained_flant5xxl.pth'
t5_path = 'output/pretrained_models/t5-v1_1-xxl'