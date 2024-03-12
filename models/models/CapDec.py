import torch
import torch.nn as nn
from ClipContextual import clip
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.BaseModel import BaseModel, MLP, TransformerEncoder, LayerNorm, TransformerEncoderLayer, \
    TransformerDecoder, TransformerDecoderLayer
import math

class DeCap(BaseModel):

    def __init__(self, clip_model, llm, tokenizer, args=None, prefix_size: int = 512):
        super(DeCap, self).__init__(clip_model, llm, tokenizer, args)

        self.align_proj = MLP(prefix_size, prefix_size, self.vocab_dim, 3)
        self.variance = args.noise_variance
        self.num_query = 1
        # self.query_fusion = nn.Embedding(self.num_query, prefix_size)

        # encoder_layer = TransformerEncoderLayer(prefix_size, 8)
        # encoder_norm = LayerNorm(prefix_size)
        # self.aligner = TransformerEncoder(encoder_layer, 4, encoder_norm)

        # decoder_layer = TransformerDecoderLayer(prefix_size, 8)
        # decoder_norm = LayerNorm(prefix_size)
        # self.aligner = TransformerDecoder(decoder_layer, 4, decoder_norm)
        # self.variance = args.noise_variance

    @torch.no_grad()
    def embed_img(self, img):
        cls_features, img_context, img_proj = self.clip_model.visual(img.half())
        cls_features /= cls_features.norm(dim=-1, keepdim=True)
        queries = self.query_fusion.weight.unsqueeze(0).repeat(1, 1, 1)
        inter_hs = self.aligner(queries.permute(1, 0, 2),
                                cls_features.unsqueeze(0).to(torch.float32), pos=None)
        embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2))
        return embedding_clip

    @torch.no_grad()
    def embed_text(self, clip_tokens):
        bs = clip_tokens.shape[0]
        clip_features, contex_text, text_proj = self.clip_model.encode_text(clip_tokens)
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        queries = self.query_fusion.weight.unsqueeze(0).repeat(bs, 1, 1)
        inter_hs = self.aligner(queries.permute(1, 0, 2),
                                clip_features.unsqueeze(0).to(torch.float32), pos=None)
        embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2))
        return embedding_clip

    def get_uniform_ball_noise(self, input_shape, radius=0.1):
        uniform_noise_ball = torch.randn(input_shape)  # normal distribution
        uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)
        u = torch.rand(input_shape[0])  # unified distribution
        u = u ** (1. / input_shape[1])
        uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
        return uniform_noise_ball

    def noise_injection(self, x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
        device = x.device
        if variance == 0.0:
            return x
        std = math.sqrt(variance)
        if not dont_norm:
            x = F.normalize(x, dim=1)
        if uniform_noise:
            x = x + self.get_uniform_ball_noise(x.shape, radius=std).to(device)
        else:
            x = x + (torch.randn(x.shape,
                                 device=device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
        if modality_offset is not None:
            x = x + modality_offset
        return F.normalize(x, dim=1)

    def generate(self, img, gen_strategy):
        bs = img.shape[0]
        cls_features, img_context, img_proj = self.clip_model.visual(img.half())
        cls_features /= cls_features.norm(dim=-1, keepdim=True)
        # clip_conx = img_context @ img_proj
        # clip_conx /= clip_conx.norm(dim=-1, keepdim=True)
        attn_mask = None
        embedding_clip = self.align_proj(cls_features.to(torch.float32)).unsqueeze(1)
        return self.generate_by_strategy(embedding_clip, attn_mask, gen_strategy, cls_features, self.num_query)

    def forward(self, clip_tokens, gpt_tokens):
        bs, token_len = gpt_tokens.shape
        clip_token_len = 1
        device = clip_tokens.device
        with torch.no_grad():
            clip_features, contex_text, text_proj = self.clip_model.encode_text(clip_tokens)
            # clip_conx = contex_text @ text_proj
            # clip_conx /= clip_conx.norm(dim=-1, keepdim=True)

            clip_features /= clip_features.norm(dim=-1, keepdim=True)

        attn_mask = torch.ones(bs, token_len + clip_token_len).to(device)
        for idx, (i, j) in enumerate(zip(clip_tokens, gpt_tokens)):
            # valid_clip_len = i.count_nonzero().item()
            # attn_mask[idx][valid_clip_len:clip_token_len] = 0

            valid_llm_len = (j - 1).count_nonzero().item()
            attn_mask[idx][clip_token_len + valid_llm_len:] = 0

        embedding_text = self.llm_vocab(gpt_tokens)

        content_feature = self.noise_injection(clip_features, self.variance)
        embedding_clip = self.align_proj(content_feature).unsqueeze(1)
        embedding_cls_clip = None

        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1).to(torch.float16)
        label_mask = torch.zeros(bs, clip_token_len, dtype=torch.int64).to(device) - 100
        labels = torch.cat([label_mask, gpt_tokens], dim=1)
        inputs = {'inputs_embeds': embedding_cat, 'attention_mask': attn_mask, 'labels': labels,
                  'output_hidden_states': True}

        out = self.model(**inputs)

        return self.compute_loss(out, clip_token_len, attn_mask, embedding_cls_clip, gpt_tokens)


def build_model(args):
    clip_model, preprocess = clip.load(args.clip_model)
    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    llm = AutoModelForCausalLM.from_pretrained(args.language_model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=False)
    for p in llm.parameters():
        p.requires_grad = False
    # tokenizer_llm = AutoTokenizer.from_pretrained("Manuel030/alpaca-opt-6.7b", use_fast=False)

    model = DeCap(clip_model, llm, tokenizer, args)
    return model
