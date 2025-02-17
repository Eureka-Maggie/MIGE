import argparse
import datetime
import os
import sys
import time
import types
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from diffusion import IDDPM
from diffusion.data.builder import build_dataset_multi, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler, CustomBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from lavis.models.blip2_models.modeling_t5 import T5EncoderModel

warnings.filterwarnings("ignore")  # ignore warning
from lavis.models.eva_vit import interpolate_pos_embed

def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader) #0
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs # 809*100


    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False) #True

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        model.train()
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start

            if load_vae_feat:
                z = batch[0] 

            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'): #True
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()
            clean_images = z * config.scale_factor #[bs,4,64,64] float32
            ref_clip = batch[1] 
            ref_vae = batch[2]
            y = batch[3] # list:bs multimodal_prompt
            source = batch[4] * config.scale_factor
            data_info = batch[5]
            
            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(model, clean_images, timesteps, model_kwargs=dict(y=y, source = source, ref_clip = ref_clip, ref_vae = ref_vae, data_info=data_info))
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)         
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None: 
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()
                info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({model.module.h}, {model.module.w}), "
                info += ', '.join([f"{k}:{v:.6f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            global_step += 1
            data_time_start= time.time()

            synchronize()
            if accelerator.is_main_process: #yes
                if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
            synchronize()

        synchronize()
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config", default='configs/config_mige.py',type=str,help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', default='output/test_mige',help='the dir to save logs and models')
    parser.add_argument('--load-from', default='output/pretrained_models/PixArt-XL-2-512x512.pth', help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 8
        config.valid_num = 100

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp: #no
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    if accelerator.state.deepspeed_plugin:
        print('accelerator.state.deepspeed_plugin:', accelerator.state.deepspeed_plugin)
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.train_batch_size

    logger = get_root_logger(os.path.join(config.work_dir, 'train_main_lf.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process: #yes
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"vit_path":config.vit_path,"blip2":config.blip2_path,"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length}

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.load_from is not None: 
        if args.load_from is not None:
            config.load_from = args.load_from
        
        #load pixart
        missing, unexpected = load_checkpoint(config.load_from, model, load_ema=config.get('load_ema', False))
        print('finish loading pixart')

        #load vit
        vit_weights = torch.load(config.vit_path, map_location='cpu')
        interpolate_pos_embed(model.visual_encoder, vit_weights)
        model.visual_encoder.load_state_dict(vit_weights, strict=False)
        print('finish loading vit')
        
        #load blip2
        blip2_weights = torch.load(config.blip2_path, map_location='cpu')
        state_dict = blip2_weights['model']
        model.load_state_dict(state_dict,strict=False)
        print('finish loading q-former')

        #load t5
        model.t5_model = T5EncoderModel.from_pretrained(config.t5_path)
        print('finish loading T5')
        
        
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).cuda()

    model.to(torch.bfloat16)

    keywords = ["visual_encoder"]

    for name, param in model.named_parameters():
        if not any(keyword in name for keyword in keywords):
            param.requires_grad = True

        else:
            param.requires_grad = False


    # prepare for FSDP clip grad norm calculation
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{model.__class__.__name__} 可训练参数总量: {total_trainable_params:,}")
    
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset_multi(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type) 
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
    else:
        #batch_sampler = CustomBatchSampler(dataset, batch_size=config.train_batch_size, drop_last=True)
        #train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers, pin_memory=True)
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None): #yes
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
        
    
    optimizer = build_optimizer(model, config.optimizer)

    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))#yes
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}") #yes

    start_epoch = 0

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    print('prepare')
    
    train()
