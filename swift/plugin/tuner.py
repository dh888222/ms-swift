# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from peft import IA3Config, PeftModel, get_peft_model

from swift.llm import ModelKeys
from swift.utils import find_all_linears

if TYPE_CHECKING:
    from swift.llm import TrainArguments


class Tuner:

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        """Prepare a new model with a tuner

        Args:
            args: The training arguments
            model: The model instance

        Returns:
            The wrapped model
        """
        raise NotImplementedError

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save when save_steps reaches

        Args:
            model: The wrapped model by `prepare_model`
            save_directory: The directory to save
            safe_serialization: Use safetensors or not
        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        """Load the ckpt_dir

        Args:
            model: The original model instance.
            model_id: The model id or ckpt_dir to load
        Returns:
            The wrapped model instance
        """
        raise NotImplementedError


class PeftTuner(Tuner):

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        return PeftModel.from_pretrained(model, model_id, **kwargs)


# Here gives a simple example of IA3
class IA3(PeftTuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch: ModelKeys = model.model_meta.model_arch
        ia3_config = IA3Config(
            target_modules=find_all_linears(model), feedforward_modules='.*' + model_arch.mlp.split('{}.')[1] + '.*')
        return get_peft_model(model, ia3_config)


class DummyTuner(PeftTuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        return model


import os, safetensors
from collections import OrderedDict
import copy

class ImplicitRewardTuner:
    @staticmethod
    def load_mlp(model:torch.nn.Module, model_id) -> torch.nn.Module:
        print("start to load mlp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, path:", model_id)
        # 尝试加载额外的MLP权重

        # 尝试加载额外的MLP权重
        mlp_path = os.path.join(model_id, 'additional_mlp.safetensors')
        if os.path.exists(mlp_path):
            try:
                # 使用safetensors加载
                state_dict = safetensors.torch.load_file(mlp_path)
                
                # 创建新的状态字典，移除"additional_mlp."前缀
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    # 移除"additional_mlp."前缀
                    if key.startswith('additional_mlp.'):
                        new_key = key[len('additional_mlp.'):]
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                
                # 加载到MLP层
                model.additional_mlp.load_state_dict(new_state_dict)
                print(f"Loaded additional MLP weights from {mlp_path}")
            except Exception as e:
                print(f"Failed to load additional MLP weights: {str(e)}")
                # 打印状态字典键名以帮助调试
                print(f"Expected keys: {list(model.additional_mlp.state_dict().keys())}")
                print(f"Loaded keys: {list(state_dict.keys())}")
        else:
            print("No additional MLP weights found. Using initialized weights.")

        return model

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        """Prepare a new model with a tuner

        Args:
            args: The training arguments
            model: The model instance

        Returns:
            The wrapped model
        """
        # print("Initiate Implicit Reward Tuner, args:", args)
        if (args.freeze_llm == True):
            for name, param in model.named_parameters():
                # 精确匹配additional_mlp及其子模块
                if name.startswith("additional_mlp.") or name == "additional_mlp":
                    continue
                param.requires_grad = False

        # 检查模型是否已经被修改
        if hasattr(model, 'additional_mlp'):
            return model  # 如果已经修改过，直接返回
        

        # 定义新的 MLP
        class AdditionalMLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, output_size)
                self.act = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.act(self.linear1(x))
                return self.sigmoid(self.linear2(x))

        # 创建并添加新的 MLP
        hidden_size = model.config.hidden_size
        additional_mlp = AdditionalMLP(hidden_size, hidden_size // 2, 1)
        setattr(model, 'additional_mlp', additional_mlp)

        # 保存原始的 forward 方法
        original_forward = model.forward

        # 定义新的 forward 方法
        def new_forward(self, input_ids, attention_mask=None, **kwargs):
            # 禁用中间层输出
            outputs = self.model(input_ids)  # 仅获取基础Transformer输出
            # print('model outputs:', outputs)
            last_hidden_state = outputs.last_hidden_state
            # print('last_hidden_state shape: ', last_hidden_state.shape)
            
            # 计算logits
            lm_logits = self.lm_head(last_hidden_state)
            
            # 通过额外的 MLP
            additional_output = self.additional_mlp(last_hidden_state.detach())

            # 拼接原始 lm_head 输出和新 MLP 的输出
            combined_logits = torch.cat([lm_logits, additional_output], dim=-1)

            # 创建新的输出对象，保持与原始模型输出格式一致
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=combined_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None
            )

        # 保存原始的 forward 方法
        setattr(model, '_original_forward', model.forward)
        # 使用 setattr 替换原始的 forward 方法
        setattr(model, '_new_forward', new_forward.__get__(model))
        setattr(model, 'forward', model._new_forward)

        # 初始化新添加的 MLP 权重
        ImplicitRewardTuner._init_mlp_weights(model.additional_mlp)

        # 尝试载入
        model = ImplicitRewardTuner.load_mlp(model, args.model)

        return model

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        """Load the ckpt_dir

        Args:
            model: The original model instance.
            model_id: The model id or ckpt_dir to load
        Returns:
            The wrapped model instance
        """
        print("Loading model in from_pretrained function")
        # 先加载主模型（包括Swift适配器）
        model = Swift.from_pretrained(model, model_id, **kwargs)
        
        # 检查模型是否已经包含MLP层
        if not hasattr(model, 'additional_mlp'):
            # 如果模型没有MLP层，添加它
            model = ImplicitRewardTuner.prepare_model(None, model)
            print("Added additional MLP layer to the model")

        model = self.load_mlp(model, model_id)
        
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        # 创建模型的深度副本（不包含MLP）
        clean_model = copy.deepcopy(model)
        
        # 从副本中移除所有MLP相关属性
        if hasattr(clean_model, 'additional_mlp'):
            delattr(clean_model, 'additional_mlp')
        
        if hasattr(clean_model, '_original_forward'):
            # 恢复原始forward方法
            setattr(clean_model, 'forward', clean_model._original_forward)
            delattr(clean_model, '_original_forward')
            delattr(clean_model, '_new_forward')
        
        # 创建过滤后的状态字典
        clean_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'additional_mlp' not in key:
                clean_state_dict[key] = value
        
        # 保存干净的模型（不包含任何MLP痕迹）
        clean_model.save_pretrained(
            save_directory,
            state_dict=clean_state_dict,
            safe_serialization=safe_serialization,
            **kwargs
        )

        print("Saved clean model: ",clean_model)
        
        # 单独保存MLP层权重
        if hasattr(model, 'additional_mlp'):
            # 提取MLP权重
            mlp_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'additional_mlp' in key:
                    mlp_state_dict[key] = value
            
            # 保存到单独的文件
            if mlp_state_dict:
                mlp_path = os.path.join(save_directory, 'additional_mlp.safetensors')
                safetensors.torch.save_file(mlp_state_dict, mlp_path)
                print(f"Saved additional MLP weights to {mlp_path}")

    @staticmethod
    def _init_mlp_weights(module):
        """Initialize the weights of the MLP"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Add your own tuner here, use --train_type xxx to begin
extra_tuners = {'ia3': IA3, 'dummy': DummyTuner, 'implicit_reward': ImplicitRewardTuner}
