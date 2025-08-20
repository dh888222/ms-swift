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

class ImplicitRewardTuner:
    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        """Prepare a new model with a tuner

        Args:
            args: The training arguments
            model: The model instance

        Returns:
            The wrapped model
        """
        # assert isinstance(model, Qwen2ForCausalLM), "Model must be an instance of Qwen2ForCausalLM"
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

            def forward(self, x):
                x = self.act(self.linear1(x))
                return self.linear2(x)

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
            additional_output = self.additional_mlp(last_hidden_state)

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

        # 使用 setattr 替换原始的 forward 方法
        setattr(model, 'forward', new_forward.__get__(model))

        # 初始化新添加的 MLP 权重
        ImplicitRewardTuner._init_mlp_weights(model.additional_mlp)

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
            logger.info("Added additional MLP layer to the model")
        
        # 尝试加载额外的MLP权重
        mlp_path = os.path.join(model_id, 'additional_mlp.safetensors')
        if os.path.exists(mlp_path):
            try:
                # 使用safetensors加载
                state_dict = safetensors.torch.load_file(mlp_path)
                model.additional_mlp.load_state_dict(state_dict)
                print(f"Loaded additional MLP weights from {mlp_path}")
            except Exception as e:
                print(f"Failed to load additional MLP weights: {str(e)}")
        else:
            print("No additional MLP weights found. Using initialized weights.")
        
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        import os, safetensors
        # 2. 分离 additional_mlp 参数
        mlp_state_dict = {}
        for key in list(state_dict.keys()):
            if "additional_mlp" in key:
                mlp_state_dict[key] = state_dict.pop(key)
        
        # 3. 保存主模型（不包含 additional_mlp）
        model.save_pretrained(
            save_directory,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
            **kwargs
        )
        
        # 4. 单独保存 additional_mlp 参数
        if mlp_state_dict:
            mlp_path = os.path.join(save_directory, 'additional_mlp.safetensors')
            safetensors.torch.save_file(mlp_state_dict, mlp_path, metadata={'format': 'pt'})
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
