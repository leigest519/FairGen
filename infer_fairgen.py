import argparse
import gc
from pathlib import Path

import torch
from typing import Literal

from src.configs.generation_config import load_config_from_yaml, GenerationConfig
from src.configs.config import parse_precision
from src.engine import train_util
from src.models import model_util
from src.models.fairgen_adapter import FairGenLayer, FairGenNetwork
from src.models.io import load_state_dict

DEVICE_CUDA = torch.device("cuda:0")
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def infer_with_fairgen(
        adapter_paths: list[str],
        config: GenerationConfig,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
        categories: int = 0,
        uniform: bool = True,
    ):

    adapter_model_paths = [lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in adapter_paths]

    weight_dtype = parse_precision(precision)
    
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    adapters, metadatas = zip(*[
        load_state_dict(path, weight_dtype) for path in adapter_model_paths
    ])

    network = FairGenNetwork(
        unet,
        rank=int(float(metadatas[0]["rank"])),
        alpha=float(metadatas[0]["alpha"]),
        module=FairGenLayer,
        cross_attention_only=True,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    with torch.no_grad():
        for prompt in config.prompts:
            prompt += config.unconditional_prompt
            print(f"Generating for prompt: {prompt}")
            prompt_embeds = train_util.encode_prompts(
                tokenizer, text_encoder, [prompt], return_tokens=False
                )

            if categories and categories > 0:
                num_cats = min(categories, len(adapters))
                if not uniform and config.pmf is not None:
                    probs = torch.tensor(config.pmf[:num_cats], dtype=torch.float64)
                    probs = probs / probs.sum()
                    k = torch.multinomial(probs, 1).item()
                else:
                    k = torch.randint(0, num_cats, (1,)).item()
                used_multipliers = [1.0 if i == k else 0.0 for i in range(len(adapters))]
                weighted = {}
                for adapter, w in zip(adapters, used_multipliers):
                    for key, value in adapter.items():
                        weighted[key] = weighted.get(key, 0) + value * w
                network.load_state_dict(weighted)
                # scale adapter strength if alpha provided
                network.multiplier = getattr(config, 'alpha', 1.0)
                with network:
                    images = pipe(
                        negative_prompt=config.negative_prompt,
                        width=config.width,
                        height=config.height,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        generator=torch.cuda.manual_seed(config.seed),
                        num_images_per_prompt=config.generate_num,
                        prompt_embeds=prompt_embeds,
                    ).images
                folder = Path(config.save_path.format(prompt.replace(" ", "_"), "0")).parent
                if not folder.exists():
                    folder.mkdir(parents=True, exist_ok=True)
                for i, image in enumerate(images):
                    image.save(
                        config.save_path.format(
                            prompt.replace(" ", "_"), i
                        )
                    )
                continue


def main(args):
    adapter_paths = [Path(lp) for lp in args.adapter_paths]
    generation_config = load_config_from_yaml(args.config)
            
    infer_with_fairgen(
        adapter_paths,
        generation_config,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
        categories=args.categories,
        uniform=args.uniform,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation.yaml",
        help="Base configs for image generation.",
    )
    parser.add_argument(
        "--adapter_paths",
        required=True,
        nargs="*",
        help="FairGen adapters to use.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )
    parser.add_argument(
        "--categories",
        type=int,
        default=0,
        help="If >0, enable FairGen gating with this many categories (one-hot).",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Use uniform sampling over categories for FairGen gating.",
    )

    args = parser.parse_args()

    main(args)


