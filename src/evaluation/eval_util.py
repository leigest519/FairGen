# ref:
# - https://github.com/jmhessel/clipscore/blob/main/clipscore.py
# - https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

import torch
import clip
import numpy as np
from typing import List, Union, Callable, Sequence
from PIL import Image
import random

from src.engine.train_util import text2img
from src.configs.config import RootConfig
from src.misc.clip_templates import imagenet_templates

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from diffusers.pipelines import DiffusionPipeline
try:
    import piq
except Exception:
    piq = None


def get_clip_preprocess(n_px=224):
    def Convert(image):
        return image.convert("RGB")

    image_preprocess = Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def text_preprocess(text):
        return clip.tokenize(text, truncate=True)

    return image_preprocess, text_preprocess


@torch.no_grad()
def clip_score(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    texts: str,
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
    cross_matching: bool = False,
):
    """
    Compute CLIPScore (https://arxiv.org/abs/2104.08718) for generated images according to their prompts.
    *Important*: same as the official implementation, we take *SUM* of the similarity scores across all the
        reference texts. If you are evaluating on the Concept Erasing task, it might should be modified to *MEAN*,
        or only one reference text should be given.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        texts (str): A list of prompts.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.
        cross_matching (bool, optional): Whether to compute the similarity between images and texts in cross-matching manner.

    Returns:
        score (np.ndarray): The CLIPScore of generated images.
            size: (len(images), )
    """
    if isinstance(texts, str):
        texts = [texts]
    if not cross_matching:
        assert len(images) == len(
            texts
        ), "The length of images and texts should be the same if cross_matching is False."

    if isinstance(images[0], str):
        images = [Image.open(img) for img in images]
    elif isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    elif isinstance(images[0], torch.Tensor):
        images = [Image.fromarray(img.cpu().numpy()) for img in images]
    else:
        assert isinstance(images[0], Image.Image), "Invalid image type."

    model, _ = clip.load(clip_model, device="cuda")
    image_preprocess, text_preprocess = get_clip_preprocess(
        n_px
    )  # following the official implementation, rather than using the default CLIP preprocess

    # extract all texts
    texts_feats = text_preprocess(texts).cuda()
    texts_feats = model.encode_text(texts_feats)

    # extract all images
    images_feats = [image_preprocess(img) for img in images]
    images_feats = torch.stack(images_feats, dim=0).cuda()
    images_feats = model.encode_image(images_feats)

    # compute the similarity
    images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
    texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
    if cross_matching:
        score = w * images_feats @ texts_feats.T
        # TODO: the *SUM* here remains to be verified
        return score.sum(dim=1).clamp(min=0).cpu().numpy()
    else:
        score = w * images_feats * texts_feats
        return score.sum(dim=1).clamp(min=0).cpu().numpy()


@torch.no_grad()
def clip_accuracy(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    ablated_texts: Union[List[str], str],
    anchor_texts: Union[List[str], str],
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
):
    """
    Compute CLIPAccuracy according to CLIPScore.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        ablated_texts (Union[List[str], str]): A list of prompts that are ablated from the anchor texts.
        anchor_texts (Union[List[str], str]): A list of prompts that the ablated concepts fall back to.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.

    Returns:
        accuracy (float): The CLIPAccuracy of generated images. size: (len(images), )
    """
    if isinstance(ablated_texts, str):
        ablated_texts = [ablated_texts]
    if isinstance(anchor_texts, str):
        anchor_texts = [anchor_texts]

    assert len(ablated_texts) == len(
        anchor_texts
    ), "The length of ablated_texts and anchor_texts should be the same."

    ablated_clip_score = clip_score(images, ablated_texts, w, clip_model, n_px)
    anchor_clip_score = clip_score(images, anchor_texts, w, clip_model, n_px)
    accuracy = np.mean(anchor_clip_score < ablated_clip_score).item()

    return accuracy


def clip_eval_by_image(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    ablated_texts: Union[List[str], str],
    anchor_texts: Union[List[str], str],
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
):
    """
    Compute CLIPScore and CLIPAccuracy with generated images.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        ablated_texts (Union[List[str], str]): A list of prompts that are ablated from the anchor texts.
        anchor_texts (Union[List[str], str]): A list of prompts that the ablated concepts fall back to.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.

    Returns:
        score (float): The CLIPScore of generated images.
        accuracy (float): The CLIPAccuracy of generated images.
    """
    ablated_clip_score = clip_score(images, ablated_texts, w, clip_model, n_px)
    anchor_clip_score = clip_score(images, anchor_texts, w, clip_model, n_px)
    accuracy = np.mean(anchor_clip_score < ablated_clip_score).item()
    score = np.mean(ablated_clip_score).item()

    return score, accuracy


def clip_eval(
    pipe: DiffusionPipeline,
    config: RootConfig,
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
):
    """
    Compute CLIPScore and CLIPAccuracy.
    For each given prompt in config.logging.prompts, we:
        1. sample config.logging.eval_num templates
        2. generate images with the sampled templates
        3. compute CLIPScore and CLIPAccuracy between each generated image and the *corresponding* template
    to get the final CLIPScore and CLIPAccuracy for each prompt.

    Args:
        pipe (DiffusionPipeline): The diffusion pipeline.
        config (RootConfig): The root config.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.

    Returns:
        score (list[float]): The CLIPScore of each concept to evaluate.
        accuracy (list[float]): The CLIPAccuracy of each concept to evaluate.
    """
    scores, accs = [], []
    for prompt in config.logging.prompts:
        templates = random.choices(imagenet_templates, k=config.logging.eval_num)
        templated_prompts = [template.format(prompt) for template in templates]
        samples = text2img(
            pipe,
            templated_prompts,
            negative_prompt=config.logging.negative_prompt,
            width=config.logging.width,
            height=config.logging.height,
            num_inference_steps=config.logging.num_inference_steps,
            guidance_scale=config.logging.guidance_scale,
            seed=config.logging.seed,
        )
        images = [sample[1] for sample in samples]
        score, acc = clip_eval_by_image(
            images,
            templated_prompts,
            [config.logging.anchor_prompt] * config.logging.eval_num,
            w,
            clip_model,
            n_px,
        )
        scores.append(score)
        accs.append(acc)
    return scores, accs


# ======================= FairGen Metrics =======================

@torch.no_grad()
def fairness_discrepancy_from_probs(
    probs: torch.Tensor,
    target_pmf: Sequence[float],
) -> float:
    """
    probs: (N, C) softmax probabilities per image over C categories
    target_pmf: length-C target distribution (e.g., uniform)
    returns L2 distance || target - E_x[y] ||_2
    """
    assert probs.ndim == 2, "probs must be (N, C)"
    pmf = torch.tensor(target_pmf, dtype=probs.dtype, device=probs.device)
    pmf = pmf / pmf.sum()
    mean_probs = probs.mean(dim=0)
    return torch.norm(pmf - mean_probs, p=2).item()


@torch.no_grad()
def fairness_discrepancy_with_classifier(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    classifier: Callable[[Image.Image], torch.Tensor],  # returns logits (C,)
    target_pmf: Sequence[float],
) -> float:
    logits = []
    for img in images:
        if isinstance(img, str):
            pil = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            arr = (img.detach().cpu().numpy() * 255).astype(np.uint8) if img.max() <= 1.0 else img.detach().cpu().numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = img
        logits.append(classifier(pil).detach().cpu())
    logits = torch.stack(logits, dim=0)
    probs = torch.softmax(logits, dim=-1)
    return fairness_discrepancy_from_probs(probs, target_pmf)


@torch.no_grad()
def fairness_discrepancy_with_clip(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    attribute_texts: List[str],
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
    target_pmf: Sequence[float] = None,
) -> float:
    """
    Approximate FD using CLIP similarities to attribute texts.
    """
    if target_pmf is None:
        target_pmf = [1.0 / len(attribute_texts)] * len(attribute_texts)

    model, _ = clip.load(clip_model, device="cuda")
    image_preprocess, text_preprocess = get_clip_preprocess(n_px)
    text_tokens = text_preprocess(attribute_texts).cuda()
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

    image_list = []
    for img in images:
        if isinstance(img, str):
            pil = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            arr = (img.detach().cpu().numpy() * 255).astype(np.uint8) if img.max() <= 1.0 else img.detach().cpu().numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = img
        image_list.append(image_preprocess(pil))
    image_batch = torch.stack(image_list, dim=0).cuda()
    image_feats = model.encode_image(image_batch)
    image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)

    sims = image_feats @ text_feats.T  # (N, C)
    probs = torch.softmax(sims, dim=-1)
    return fairness_discrepancy_from_probs(probs, target_pmf)


@torch.no_grad()
def clip_similarity_mean(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    texts: List[str],
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
) -> float:
    """Mean cosine similarity between each image and its prompt."""
    assert len(images) == len(texts)
    model, _ = clip.load(clip_model, device="cuda")
    image_preprocess, text_preprocess = get_clip_preprocess(n_px)
    # images
    image_list = []
    for img in images:
        if isinstance(img, str):
            pil = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            arr = (img.detach().cpu().numpy() * 255).astype(np.uint8) if img.max() <= 1.0 else img.detach().cpu().numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = img
        image_list.append(image_preprocess(pil))
    image_batch = torch.stack(image_list, dim=0).cuda()
    image_feats = model.encode_image(image_batch)
    image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
    # texts
    text_tokens = text_preprocess(texts).cuda()
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
    sims = (image_feats * text_feats).sum(dim=1)
    return sims.mean().item()


@torch.no_grad()
def brisque_mean(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
) -> float:
    """Mean BRISQUE score (lower is better). Requires `piq`."""
    if piq is None:
        raise ImportError("piq is required for BRISQUE. Please `pip install piq`." )
    tensors = []
    for img in images:
        if isinstance(img, str):
            pil = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                t = img.float()
                if t.max() > 1.0:
                    t = t / 255.0
                tensors.append(t)
                continue
            arr = (img.detach().cpu().numpy() * 255).astype(np.uint8) if img.max() <= 1.0 else img.detach().cpu().numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = img
        t = ToTensor()(pil).float()
        tensors.append(t)
    batch = torch.stack(tensors, dim=0).cuda()
    score = piq.brisque(batch, data_range=1.0)
    return score.mean().item()
