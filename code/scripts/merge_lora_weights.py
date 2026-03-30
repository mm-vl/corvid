import argparse
from vilamr.model.builder import load_pretrained_model
from vilamr.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map='cpu'
    )

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base", type=str,
        default="playground/model/vicuna-13b-v1.5")
    parser.add_argument(
        "--model-path", type=str,
        default="playground/model/vilamr-v1.5-13b-lora")
    parser.add_argument(
        "--save-model-path", type=str,
        default="playground/model/vilamr-v1.5-13b-merge")

    args = parser.parse_args()

    merge_lora(args)
