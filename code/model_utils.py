import os
import torch

import matplotlib.pyplot as plt


def merge_weight(model_path1, model_path2, save_path):
    model1_weights = torch.load(
        f"{model_path1}/mm_projector.bin",
        map_location='cpu'
    )

    model2_weights = torch.load(
        f"{model_path2}/mm_projector.bin",
        map_location='cpu'
    )

    assert model1_weights.keys() == model2_weights.keys()

    weight_to_save = {}
    for k in model1_weights.keys():
        weight_to_save[k] = 0.5 * (model1_weights[k] + model2_weights[k])
        # plt.hist(model1_weights[k].view(-1, 1).float().numpy(), bins=1000, density=True, alpha=0.6, color='b')
        # plt.show()
        # plt.hist(model2_weights[k].view(-1, 1).float().numpy(), bins=1000, density=True, alpha=0.6, color='b')
        # plt.show()

    os.makedirs(save_path, exist_ok=True)
    torch.save(weight_to_save,  f'{save_path}/mm_projector.bin')

# 597/1000 |203/1000 | 200/1000

if __name__ == "__main__":
    model_path = "playground/model"
    merge_save_dir = f"{model_path}/merged_corvid/s1_s2_gate_mixer_hpre24_loss1"

    merge_weight(
        f"{model_path}/merged_corvid/plain_mga_1001k-llama31_siglip_convnext-gate_mixer_hpre24_loss1",
        f"{model_path}/merged_corvid/s2_ref_1190k-llama31_siglip_convnext-gate_mixer_hpre24_loss1",
        merge_save_dir
    )
