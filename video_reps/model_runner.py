"""Model loading and inference for quantized InternVL3-8B."""

import logging
import re
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("video_reps")

DEFAULT_HF_REPO = "OpenGVLab/InternVL3-8B"
DEFAULT_LOCAL_MODEL_DIR = Path("models/InternVL3-8B")


def _looks_like_hub_id(model_path: str) -> bool:
    """True for 'org/name' Hub IDs; False for paths like 'models/InternVL3-8B'."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_path):
        return False
    if model_path.startswith("models/") or model_path.startswith("./"):
        return False
    return True


def ensure_local_model(model_path: str) -> str:
    """Resolve to a local directory with model files; download into the project if needed.

    Uses ``huggingface_hub.snapshot_download(..., local_dir=...)`` so weights land under
    the requested directory (see ``video_reps.project_paths`` for HF cache under the repo).
    """
    from huggingface_hub import snapshot_download

    path = Path(model_path)
    resolved = path.resolve()

    def has_config(p: Path) -> bool:
        return p.is_dir() and (p / "config.json").is_file()

    if has_config(resolved):
        return str(resolved)

    if _looks_like_hub_id(model_path):
        hub_id = model_path
        target_dir = DEFAULT_LOCAL_MODEL_DIR.resolve()
    else:
        hub_id = DEFAULT_HF_REPO
        target_dir = resolved

    if not has_config(target_dir):
        logger.info("Downloading %s to %s", hub_id, target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            hub_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )

    return str(target_dir)


# InternVL3 image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448) -> T.Compose:
    """Build the image transform pipeline matching InternVL3 expectations."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from the target set."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Split an image into tiles following InternVL3's dynamic resolution strategy.

    Returns a list of PIL Image tiles.
    """
    width, height = image.size
    aspect_ratio = width / height

    # Generate target aspect ratios
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )

    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        thumbnail = image.resize((image_size, image_size))
        processed.append(thumbnail)

    return processed


class ModelRunner:
    """Handles loading and inference with quantized InternVL3-8B."""

    def __init__(self, model_path: str = "models/InternVL3-8B", device: str = "cuda"):
        """Load the quantized model and tokenizer.

        Args:
            model_path: HuggingFace model ID or local path. Default Hub ID downloads to
                ``models/InternVL3-8B`` under the current working directory.
            device: Device string ('cuda' or 'cpu'). When using 4-bit quantization,
                    device_map='auto' is used regardless.
        """
        local_path = ensure_local_model(model_path)
        logger.info("Loading tokenizer from %s", local_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True
        )

        logger.info("Loading model with 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self._transform = _build_transform(448)
        logger.info("Model loaded successfully")

    def _prepare_pixel_values(self, images: list[Image.Image]) -> tuple[torch.Tensor, list[int]]:
        """Process images into pixel value tensors with tile tracking.

        Args:
            images: List of PIL images.

        Returns:
            Tuple of (pixel_values tensor, num_patches_list).
        """
        all_tiles = []
        num_patches_list = []

        for img in images:
            tiles = _dynamic_preprocess(img, min_num=1, max_num=12, image_size=448)
            tile_tensors = [self._transform(tile) for tile in tiles]
            all_tiles.extend(tile_tensors)
            num_patches_list.append(len(tiles))

        pixel_values = torch.stack(all_tiles).to(
            device=self.model.device, dtype=torch.bfloat16
        )
        return pixel_values, num_patches_list

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        """Run inference with multi-image input.

        Args:
            images: List of PIL images (frames).
            prompt: The prompt string with <image> tags.

        Returns:
            The model's text response.
        """
        pixel_values, num_patches_list = self._prepare_pixel_values(images)

        generation_config = {
            "max_new_tokens": 256,
            "do_sample": False,
        }

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
        )

        logger.debug("Raw model output: %s", response)
        return response
