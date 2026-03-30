import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip


class ConvNeXTVisionTower(nn.Module):
    def __init__(
            self,
            vision_tower='convnext_xxlarge', args=None,  delay_load=True,
            model_path="playground/model/openai/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
    ):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.model_path = model_path

        self.args = args
        self._image_size = args.image_size_mix
        self.select_layer = -1

        self.is_multi_stage = True
        self.num_patches_per_side = 27

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            if self.is_multi_stage:
                self._hidden_size = sum([384, 768, 1536, 3072])
            else:
                if self.select_layer == -2:
                    self._hidden_size = 1536
                else:
                    self._hidden_size = 3072

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, '
                  'skipping.'.format(self.vision_tower_name))
            return

        clip_model, processor = open_clip.create_model_from_pretrained(
            model_name=self.vision_tower_name,
            pretrained=f"{self.model_path}/open_clip_pytorch_model.bin",
            # device="cpu"
        )

        processor.transforms[0].size = self._image_size
        processor.transforms[1].size = (self._image_size, self._image_size)
        self.image_processor = ProcessorWrapper(
            processor, height=self._image_size, width=self._image_size)

        self.vision_tower = clip_model.visual.trunk
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def extract_region_features(self, image):
        x = self.vision_tower.stem(image.to(device=self.device, dtype=self.dtype))  # [bs, 384,
        for stage in self.vision_tower.stages[:self.select_layer + 5]:
            x = stage(x)
        return x.flatten(2, 3).permute(0, 2, 1).contiguous().to(image.dtype)

    @torch.no_grad()
    def forward(self, images):
        x = self.vision_tower.stem(images.to(device=self.device, dtype=self.dtype))

        image_features_stages = []
        for stage in self.vision_tower.stages:
            x = stage(x)
            image_features_stages.append(x)

        if not self.is_multi_stage:
            image_features_stages = image_features_stages[self.select_layer]

        image_features_stages_rescaled = [self.interpolate(feat) for feat in image_features_stages]
        return torch.cat(image_features_stages_rescaled, -1).to(images.dtype)

    def interpolate(self, image_forward_outs):
        """
        Interpolate the image features to the desired number of patches.
        """
        image_features = F.interpolate(
            image_forward_outs.float(),
            size=(self.num_patches_per_side, self.num_patches_per_side),
            mode='bilinear',
        ).to(dtype=image_forward_outs.dtype)
        image_features = image_features.flatten(2, 3).permute(0, 2, 1).contiguous()
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, 'dtype'):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, 'device'):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")  # Default to CPU if no parameters

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size


class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378,
                 image_mean = [0.48145466, 0.4578275, 0.40821073]):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = {}
        output['pixel_values'] = [self._transforms(image)]
        return output


class CLIPCNN(nn.Module):
    def __init__(
            self,
            vision_tower='convnext_xxlarge',
            delay_load=True,
            model_path="playground/model/openai/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
    ):
        super().__init__()
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.model_path = model_path
        self.visual = None

        if not delay_load:
            self.load_model()

    def load_model(self, device=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, '
                  'skipping.'.format(self.vision_tower_name))
            return

        from open_clip.model import _build_vision_tower
        config = open_clip.get_model_config(self.vision_tower_name)

        self.visual = _build_vision_tower(
            embed_dim=config['embed_dim'],
            vision_cfg=config['vision_cfg'],
        )
        # pretrained weight
        weights = torch.load(f"{self.model_path}/open_clip_pytorch_model.bin")
        # print(weights.keys())

        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items()
                    if keyword in k}

        self.visual.load_state_dict(get_w(weights, 'visual'), strict=False)
        # print(self.visual.state_dict())

        self.visual.trunk.head.global_pool = nn.Identity()
        self.visual.trunk.head.flatten = nn.Identity()
        if device is not None:
            self.visual = self.visual.to(device)
        self.visual.requires_grad_(False)

        self.is_loaded = True

    def extract_cnn_feat(self, image):
        # [bs, 192, 128, 128]
        x = self.visual.trunk.stem(
            image.to(device=self.device, dtype=self.dtype)
        ).contiguous()

        for i in range(4 - 1):  # 倒数第二
            x = self.visual.trunk.stages[i](x)

        vis_loc = x.reshape(*x.shape[:2], -1).permute(0, 2, 1)
        vis_loc = vis_loc.contiguous()
        return vis_loc.to(image.dtype)

    @torch.no_grad()
    def forward(self, image):
        # self.eval()
        return self.extract_cnn_feat(image)

    @property
    def dtype(self):
        return self.visual.trunk.stem[0].weight.dtype

    @property
    def device(self):
        return self.visual.trunk.stem[0].weight.device

