import os
import math
from typing import Optional, List

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


class FairGenLayer(nn.Module):
    """
    1-D adapter layer: replaces forward method of the original Linear/Conv2d.
    """

    def __init__(
        self,
        name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        super().__init__()
        self.name = name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, dim, bias=False)
            self.lora_up = nn.Linear(dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class FairGenNetwork(nn.Module):
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    UNET_TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    FAIRGEN_PREFIX_UNET = "lora_unet"
    DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module=FairGenLayer,
        module_kwargs=None,
        cross_attention_only: bool = False,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha

        self.module = module
        self.module_kwargs = module_kwargs or {}
        self.cross_attention_only = cross_attention_only

        self.unet_layers = self.create_modules(
            FairGenNetwork.FAIRGEN_PREFIX_UNET,
            unet,
            FairGenNetwork.DEFAULT_TARGET_REPLACE,
            self.dim,
            self.multiplier,
        )
        print(f"Create FairGen adapters for U-Net: {len(self.unet_layers)} modules.")

        names = set()
        for layer in self.unet_layers:
            assert layer.name not in names, f"duplicated adapter name: {layer.name}. {names}"
            names.add(layer.name)

        for layer in self.unet_layers:
            layer.apply_to()
            self.add_module(
                layer.name,
                layer,
            )

        del unet
        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
                        full_child_path = name + "." + child_name if name else child_name
                        if self.cross_attention_only and ("attn2" not in full_child_path):
                            continue
                        adapter_name = prefix + "." + full_child_path
                        adapter_name = adapter_name.replace(".", "_")
                        print(f"{adapter_name}")
                        layer = self.module(
                            adapter_name, child_module, multiplier, rank, self.alpha, **self.module_kwargs
                        )
                        layers.append(layer)

        return layers

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        all_params = []

        if self.unet_layers:
            params = []
            [params.extend(layer.parameters()) for layer in self.unet_layers]
            param_data = {"params": params}
            if default_lr is not None:
                param_data["lr"] = default_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for layer in self.unet_layers:
            layer.multiplier = self.multiplier

    def __exit__(self, exc_type, exc_value, tb):
        for layer in self.unet_layers:
            layer.multiplier = 0


