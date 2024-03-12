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


class DeCap(BaseModel):

    def __init__(self, clip_model, llm, tokenizer, args=None):
        """
        MAGIC is a zero-shot caption method
        """
        super(DeCap, self).__init__(clip_model, llm, tokenizer, args)

    def generate(self, img, gen_strategy):
        cls_features, img_context, img_proj = self.clip_model.visual(img.half())
        cls_features /= cls_features.norm(dim=-1, keepdim=True)
        clip_conx = img_context @ img_proj
        clip_conx /= clip_conx.norm(dim=-1, keepdim=True)
        attn_mask = None

        # embedding_clip = torch.zeros(cls_features.shape[0], 32, self.vocab_dim).to(clip_conx.device)
        embedding_clip = None
        return self.generate_by_strategy(embedding_clip, attn_mask, gen_strategy, cls_features)

    def forward(self, clip_tokens, gpt_tokens, img=None):
        raise NotImplementedError


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
