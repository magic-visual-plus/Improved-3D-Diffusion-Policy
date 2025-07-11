from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.common.pytorch_util import dict_apply
from termcolor import cprint
from diffusion_policy_3d.model.vision.timm_obs_encoder import TimmObsEncoder
import numpy as np
from loguru import logger


class DiffusionImagePolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type='film',
            use_depth=False,
            use_wrist=False,
            use_depth_only=False,
            obs_encoder: TimmObsEncoder = None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.use_depth = use_depth
        self.use_depth_only = use_depth_only
        self.use_wrist = use_wrist
        cprint(f"use_depth: {use_depth}, use_depth_only: {use_depth_only}, use_wrist: {use_wrist}", 'red')
        # parse shape_meta
        logger.info("n_obs_steps {}", n_obs_steps)
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        
        if use_depth and not use_depth_only:
            obs_shape_meta['image']['shape'][0] = 4 # 3,H,W -> 4,H,W
        
        if use_depth and use_depth_only:
            obs_shape_meta['image']['shape'][0] = 1


        obs_feature_dim = np.prod(obs_encoder.output_shape())
       
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict = obs_dict.copy()
            
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        nobs['image'] /= 255.0
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
        if self.use_wrist:
            nobs['wrist_image'] /= 255.0
            if nobs['wrist_image'].shape[-1] == 3:
                if len(nobs['wrist_image'].shape) == 5:
                    nobs['wrist_image'] = nobs['wrist_image'].permute(0, 1, 4, 2, 3)
                if len(nobs['wrist_image'].shape) == 4:
                    nobs['wrist_image'] = nobs['wrist_image'].permute(0, 3, 1, 2)
        if self.use_depth and not self.use_depth_only:
            nobs['image'] = torch.cat([nobs['image'], nobs['depth'].unsqueeze(-3)], dim=-3)
        if self.use_depth and self.use_depth_only:
            nobs['image'] = nobs['depth'].unsqueeze(-3)

        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

            
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...])
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
     
            
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        return action


    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                # **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # normalize image by hand
        nobs['image'] /= 255.0
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
                
        if self.use_wrist:
            nobs['wrist_image'] /= 255.0
            if nobs['wrist_image'].shape[-1] == 3:
                if len(nobs['wrist_image'].shape) == 5:
                    nobs['wrist_image'] = nobs['wrist_image'].permute(0, 1, 4, 2, 3)
                if len(nobs['wrist_image'].shape) == 4:
                    nobs['wrist_image'] = nobs['wrist_image'].permute(0, 3, 1, 2)
        # 
        if self.use_depth and not self.use_depth_only:
            nobs['image'] = torch.cat([nobs['image'], nobs['depth'].unsqueeze(-3)], dim=-3)
        if self.use_depth and self.use_depth_only:
            nobs['image'] = nobs['depth'].unsqueeze(-3)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        # start = To - 1
        start = 0
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # normalize image by hand
        nobs['image'] /= 255.0
        if nobs['image'].shape[-1] == 3:
            if len(nobs['image'].shape) == 5:
                nobs['image'] = nobs['image'].permute(0, 1, 4, 2, 3)
            if len(nobs['image'].shape) == 4:
                nobs['image'] = nobs['image'].permute(0, 3, 1, 2)
                
        if self.use_wrist:
            nobs['wrist_image'] /= 255.0
            if nobs['wrist_image'].shape[-1] == 3:
                if len(nobs['wrist_image'].shape) == 5:
                    nobs['wrist_image'] = nobs['wrist_image'].permute(0, 1, 4, 2, 3)
                if len(nobs['wrist_image'].shape) == 4:
                    nobs['wrist_image'] = nobs['wrist_image'].permute(0, 3, 1, 2)
            
        if self.use_depth and not self.use_depth_only:
            nobs['image'] = torch.cat([nobs['image'], nobs['depth'].unsqueeze(-3)], dim=-3)
        if self.use_depth and self.use_depth_only:
            nobs['image'] = nobs['depth'].unsqueeze(-3)
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x)
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
