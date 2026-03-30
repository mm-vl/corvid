import matplotlib.pyplot as plt
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig

from safetensors.torch import load_file, save_file



def weight_vis():
    siglip = SiglipVisionModel.from_pretrained(
        "playground/model/google/siglip-so400m-patch14-384"
    )
    plt.hist(
        siglip.vision_model.encoder.layers[-1].mlp.fc1.weight.view(-1, 1).detach().cpu().numpy(),
        bins=1000, density=True, alpha=0.6, color='b')
    plt.title('Histogram of vec')
    plt.show()

    # model = LlamaForCausalLM.from_pretrained(
    #     "experiments/vilamr-llama3-8b-siglip",
    #     low_cpu_mem_usage=True,
    # )
    # plt.hist(model.base_model.layers[10].mlp.up_proj.weight.view(-1, 1).detach().cpu().numpy(), bins=1000, density=True, alpha=0.6, color='b')
    # plt.title('Histogram of vec')
    # plt.show()

    # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # mm_projector = build_vision_projector(model_args)

    # plt.hist(mm_projector_weights['model.mm_projector.proj_dn1.weight'].view(-1, 1).detach().cpu().numpy(), bins=1000, density=True, alpha=0.6, color='b')
    # plt.title('Histogram of vec')
    # plt.show()


def weight_merging(num_safetensors=4, ):
    total = num_safetensors

    # model_1_all = {}
    # model_2_all = {}

    math_wo_weight_name = []

    for i in range(1, total + 1):
        model_1 = load_file(
            f'/data/mllm_ws/playground/checkpoints/llava-v1.6-vicuna-13b/model-{i:05d}-of-{total:05d}.safetensors')
        model_2 = load_file(
            f'/data/mllm_ws/playground/checkpoints/Math-LLaVA/model-{i:05d}-of-{total:05d}.safetensors')

        for k in model_1.keys():
            if k not in model_2:
                math_wo_weight_name.append(k)

        # model_1_all.update(model_1)
        # model_2_all.update(model_2)
        # assert model_1.keys() == model_2.keys()

        avg = {}
        for k in model_1.keys():
            if k in math_wo_weight_name:
                avg[k] = model_1[k]
            else:
                avg[k] = model_1[k] * 0.5 + model_2[k] * 0.5
                # avg[k] = model_1[k] * 0.58 + model_2[k] * 0.42

        save_file(
            avg,
            f'/data/mllm_ws/playground/checkpoints/merged_llava-v1.6_math/model-{i:05d}-of-{total:05d}.safetensors',
            {'format': 'pt'}
        )
    # assert model_1_all.keys() == model_2_all.keys()



def evaluate():
    pass



if __name__ == "__main__":
    weight_merging(num_safetensors=6)
