import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

def get_augmentations_train():
    AUGMENTATIONS_TRAIN = Compose([ HorizontalFlip(p=0.5), OneOf([RandomContrast(),RandomGamma(),RandomBrightness(),], p=0.3),
                            OneOf([ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),GridDistortion(),OpticalDistortion(distort_limit=2, shift_limit=0.5),], p=0.3),
                            RandomSizedCrop(min_max_height=(176, 256), height=h, width=w,p=0.25),
                            ToFloat(max_value=1)],p=1)
    return AUGMENTATIONS_TRAIN


def get_augmentations_test():
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=1)],p=1)
    return AUGMENTATIONS_TEST


def get_augmentations_test_flipped():
    AUGMENTATIONS_TEST_FLIPPED = Compose([HorizontalFlip(p=1),ToFloat(max_value=1)],p=1)
    return AUGMENTATIONS_TEST_FLIPPED
