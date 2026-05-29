from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_transforms(img_size: int = 320, color_jitter: bool = True, random_erasing: bool = True):
    ops = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if color_jitter:
        ops.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        )

    ops.append(transforms.ToTensor())

    if random_erasing:
        ops.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random")
        )

    return transforms.Compose(ops)


def build_val_transforms(img_size: int = 320):
    ops = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]
    return transforms.Compose(ops)
