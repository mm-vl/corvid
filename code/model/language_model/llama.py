#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from corvid.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from corvid.constants import IGNORE_INDEX
from corvid.model.multimodal_projector.decision_nce import DecisionNCELoss


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    temperature: float = 0.0
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_llama"
        # config.rope_scaling = None

        self.config = config

        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            images_mix: Optional[torch.FloatTensor] = None,
            bbox_cap_mask: Optional[torch.BoolTensor] = None,  # for align loss in PT-S2
            return_dict: Optional[bool] = None,
            dpo_forward: Optional[bool] = None,
            cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values,
             inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values,
                labels, images, image_sizes, images_mix
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
                output_hidden_states=True if self.config.using_align_loss_s2 else False,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            # FIXME: align loss in pretraining stage
            # In 1-th stage, label_mask == caption_position, requiring image_mask
            if self.config.using_align_loss_s1 and self.training:
                text_reps = []
                vis_reps = []
                for i in range(labels.size(0)):
                    text_reps.append(hidden_states[i, labels[i] != IGNORE_INDEX].mean(0))
                    vis_reps.append(inputs_embeds[i, :self.config.num_image_patches].mean(0))
                text_reps = torch.stack(text_reps)  # [bs, dim]
                vis_reps = torch.stack(vis_reps)  # [bs, dim]

                align_loss_s1 = self.clip_loss(vis_reps, text_reps)
                loss += align_loss_s1

            # In 2-th stage, given bbox_feature, requiring caption_mask
            if self.config.using_align_loss_s2 and self.training:
                # [bs, num_patch, 3072]
                bbox_feats = self.model.convnext_xxl.extract_region_features(images_mix)
                bbox_feats = bbox_feats.mean(1)  # [bs, 3072]

                text_reps = []
                for i in range(labels.size(0)):
                    bcap_mask = [0] * (self.config.num_image_patches - 1) + bbox_cap_mask[i].tolist()
                    bcap_reps_layer = [h[i][bcap_mask].mean(0) for h in outputs.hidden_states]

                    text_reps.append(torch.stack(bcap_reps_layer))
                text_reps = torch.stack(text_reps)  # [bs, n_layer, 4096]

                # align_loss_fct = DecisionNCELoss(logit_scale=100, loss_type="DecisionNCE-T")
                align_loss = self.model.loss_estimator(text_reps, bbox_feats)

                loss += align_loss

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @staticmethod
    def contrastive_loss(ht, hv, temperature=1.0):
        # https://github.com/moein-shariatnia/OpenAI-CLIP
        ht = ht / ht.norm(dim=1, keepdim=True)
        hv = hv / hv.norm(dim=1, keepdim=True)

        logits = (ht @ hv.T) / temperature

        img_similarity = hv @ hv.T
        seq_similarity = ht @ ht.T

        targets = F.softmax((img_similarity + seq_similarity) / 2 * temperature, dim=-1)
        seq_loss = (-targets * nn.LogSoftmax(dim=-1)(logits)).sum(1)
        img_loss = (-targets.T * nn.LogSoftmax(dim=-1)(logits.T)).sum(1)

        loss = (img_loss + seq_loss) / 2.0  # shape: (batch_size)

        return loss.mean()

    @staticmethod
    def clip_loss(hv, ht, logit_scale=14.3):
        ht = ht / ht.norm(dim=-1, keepdim=True)
        hv = hv / hv.norm(dim=-1, keepdim=True)

        logits =  logit_scale * ht @ hv.T  # [b, b]

        labels = torch.arange(logits.shape[1] ).to(logits.device)

        loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
        loss_t = F.cross_entropy(logits, labels, reduction="mean")

        loss = (loss_i + loss_t) / 2

        return loss

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            images_mix: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds,
             _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images,
                image_sizes=image_sizes, images_mix=images_mix
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        images_mix = kwargs.pop("images_mix", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if images_mix is not None:
            inputs["images_mix"] = images_mix
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
