from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import Brain_Segmentation_Dataset
from Data_Augmentation import Scale, Rotate, HorizontalFlip


def transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return transforms.Compose(transform_list)


def datasets(images, image_size, aug_scale, aug_angle):
    train = Brain_Segmentation_Dataset(
        images_dir=images,
        subset="train",
        image_size=image_size,
        transform=transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )
    valid = Brain_Segmentation_Dataset(
        images_dir=images,
        subset="validation",
        image_size=image_size,
        random_sampling=False,
    )
    return train, valid

def data_loader(batch_size, workers, image_size, aug_scale, aug_angle):
    train_dataset, valid_dataset = datasets('./data', image_size, aug_scale, aug_angle)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)

    return train_loader, valid_loader
