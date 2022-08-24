import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn

from models.bisenetv1 import BiSeNetV1
from models.torch_models import BiSeNetV1 as torch_BiSeNetV1

def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert(torch_model, torch_dict):
    new_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        new_dict[k] = data
    return new_dict


def main(torch_name, torch_path, num_classes):
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    torch_model = torch_BiSeNetV1(n_classes=num_classes)

    new_dict = convert(torch_model, torch_state_dict)

    model = BiSeNetV1(n_classes=num_classes)
    model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='cityscapes-bisenetv1',
        help=f"The name of converted model to save",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        default="./model_final_v1_city_new.pth",
        help=f"path to torch trained model",
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        type=int,
        default=19,
        help=f"nums of classes in the model, default: 19",
    )
    args = parser.parse_args()
    main(args.model, args.ckpt, args.num_classes)