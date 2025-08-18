# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING, Optional

import torch
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

class TestTuner:
    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        """Prepare a new model with a tuner

        Args:
            args: The training arguments
            model: The model instance

        Returns:
            The wrapped model
        """
        assert isinstance(model, Qwen2ForCausalLM), "Model must be an instance of Qwen2ForCausalLM"

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
            outputs = original_forward(input_ids, attention_mask=attention_mask, **kwargs)
            lm_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]  # 使用最后一层的隐藏状态

            # 通过额外的 MLP
            additional_output = self.additional_mlp(hidden_states)

            # 拼接原始 lm_head 输出和新 MLP 的输出
            combined_logits = torch.cat([lm_logits, additional_output], dim=-1)

            # 创建新的输出对象，保持与原始模型输出格式一致
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=combined_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )

        # 使用 setattr 替换原始的 forward 方法
        setattr(model, 'forward', new_forward.__get__(model))

        # 初始化新添加的 MLP 权重
        TestTuner._init_mlp_weights(model.additional_mlp)

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
        assert isinstance(model, Qwen2ForCausalLM), "Model must be an instance of Qwen2ForCausalLM"

        model = Swift.from_pretrained(model, model_id, **kwargs)

        return model

    @staticmethod
    def _init_mlp_weights(module):
        """Initialize the weights of the MLP"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Add your own tuner here, use --train_type xxx to begin
extra_tuners = {'ia3': IA3, 'dummy': DummyTuner, 'TestTuner': TestTuner}
