"""
Image transforms for training and evaluation.
Uses ImageNet normalization since all models are pretrained on ImageNet.
"""

from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    """
    Training transforms with data augmentation.
    Augmentation is important for skin lesion classification because:
    - Lesions can appear in any orientation → flips + rotation
    - Lighting varies across dermatoscopes → color jitter
    - Scale varies → random resized crop
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size=224):
    """
    Evaluation transforms — no augmentation, just resize and normalize.
    Used for test set evaluation and uncertainty scoring.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
