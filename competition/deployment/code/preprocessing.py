"""Same PIL pixel operations with redundant full-resolution copies removed."""
import math
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


def open_rgb_fast(path):
    with Image.open(path) as source:
        source.load()
        ImageOps.exif_transpose(source, in_place=True)
        image = source if source.mode == 'RGB' else source.convert('RGB')
        image.load()
    return image


def valid_bbox_fast(image):
    # Match thumbnail((160,160), BICUBIC, reducing_gap=2), without image.copy().
    if image.width <= 160 and image.height <= 160:
        thumb = image
    else:
        aspect = image.width / image.height
        if aspect <= 1:
            choices = [math.floor(160*aspect), math.ceil(160*aspect)]
            width = max(1, min(choices, key=lambda n: abs(aspect - n/160)))
            target = (width, 160)
        else:
            choices = [math.floor(160/aspect), math.ceil(160/aspect)]
            height = max(1, min(choices, key=lambda n: 0 if n == 0 else abs(aspect - 160/n)))
            target = (160, height)
        thumb = image.resize(target, Image.Resampling.BICUBIC, reducing_gap=2.0)
    array = np.asarray(thumb)
    ys, xs = np.nonzero(array.max(axis=2) > 8)
    if len(xs) < 20:
        return (0, 0, image.width, image.height)
    sx, sy = image.width/thumb.width, image.height/thumb.height
    pad = max(2, int(min(image.size)*.025))
    return (max(0,int(xs.min()*sx)-pad), max(0,int(ys.min()*sy)-pad),
            min(image.width,int((xs.max()+1)*sx)+pad), min(image.height,int((ys.max()+1)*sy)+pad))


def preprocess(path):
    image = open_rgb_fast(path)
    box = valid_bbox_fast(image)
    if box != (0,0,image.width,image.height):
        image = image.crop(box)
    if image.size != (384,384):
        image = ImageOps.pad(image, (384,384), method=Image.Resampling.BILINEAR, color=(0,0,0))
    return TF.normalize(TF.to_tensor(image), [.485,.456,.406], [.229,.224,.225])
