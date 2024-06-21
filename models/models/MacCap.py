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


class MacCap(BaseModel):

    def __init__(self, clip_model, llm, tokenizer, args=None, prefix_size: int = 512):
        super(MacCap, self).__init__(clip_model, llm, tokenizer, args)

        self.align_proj = MLP(prefix_size, prefix_size, self.vocab_dim, 3)
        # self.align_proj = nn.Linear(prefix_size, self.vocab_dim)
        self.num_query = 32
        self.query_fusion = nn.Embedding(self.num_query, prefix_size)

        # encoder_layer = TransformerEncoderLayer(prefix_size, 8)
        # encoder_norm = LayerNorm(prefix_size)
        # self.aligner = TransformerEncoder(encoder_layer, 4, encoder_norm)

        decoder_layer = TransformerDecoderLayer(prefix_size, 8)
        decoder_norm = LayerNorm(prefix_size)
        self.aligner = TransformerDecoder(decoder_layer, args.num_decoder_layer, decoder_norm)

        self.num_noise = args.num_noise
        self.eval_variance = args.noise_k
        self.sampling_type = args.sampling_type

        if self.sampling_type == 'reconstruction' or self.sampling_type == 'reconstruction_rank':
            self.noise_N_var = args.noise_N_var

        elif self.sampling_type == 'reconstruction_repeat' or self.sampling_type == 'reconstruction_concat' or \
                self.sampling_type == 'reconstruction_concat_wo_token_noise':
            self.noise_N_var = args.noise_N_var
            if len(self.noise_N_var) != 1:
                raise ValueError('unrecognizable args.noise_N_var!')
            self.noise_N_var = self.noise_N_var * args.num_reconstruction

    def generate(self, img, gen_strategy, clip_tokens=None):
        bs = img.shape[0]
        cls_features, img_context, img_proj, attn_weight = self.clip_model.visual(img.half(), require_attn=True)
        if clip_tokens is not None:
            text_cls, _, _ = self.clip_model.encode_text(clip_tokens)
            text_cls /= text_cls.norm(dim=-1, keepdim=True)

        cls_features /= cls_features.norm(dim=-1, keepdim=True)
        clip_conx = img_context @ img_proj
        clip_conx /= clip_conx.norm(dim=-1, keepdim=True)
        attn_mask = None

        mixed_patch_feature = []

        # attn_weight: bs, 50, 50 (patch number)
        cls_weight = attn_weight[:, 0, :]
        top_cls_patch_ids = cls_weight.topk(self.train_seq_length).indices
        for idx in range(bs):
            tp_idx = top_cls_patch_ids[idx]
            top_weight = attn_weight[idx, tp_idx].softmax(dim=-1)
            # top_weight = attn_weight[idx, tp_idx]
            top_features = top_weight @ clip_conx[idx]
            mixed_patch_feature.append(top_features.unsqueeze(0))
        mixed_patch_feature = torch.cat(mixed_patch_feature, dim=0)
        mixed_patch_feature = F.normalize(mixed_patch_feature, dim=-1)

        cls_features = cls_features.unsqueeze(1) + mixed_patch_feature * self.infer_patch_weight
        noisy_cls_features = self.noise_injection(cls_features, self.eval_variance)

        noisy_cls_features /= noisy_cls_features.norm(dim=-1, keepdim=True)

        queries = self.query_fusion.weight.unsqueeze(0).repeat(bs * 1, 1, 1)
        inter_hs = self.aligner(queries.permute(1, 0, 2),
                                noisy_cls_features.permute(1, 0, 2).to(torch.float32), pos=None)
        embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2)).view(1, bs, self.num_query, -1)
        embedding_clip = embedding_clip.permute(1, 0, 2, 3).reshape(bs, 1 * self.num_query, -1)
        return self.generate_by_strategy(embedding_clip, attn_mask, gen_strategy, cls_features, self.num_query)

    def forward(self, clip_tokens, gpt_tokens):
        bs, token_len = gpt_tokens.shape
        clip_token_len = self.num_query
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

        queries = self.query_fusion.weight.unsqueeze(0).repeat(bs, 1, 1)
        # repeat cls feature 50 (#image patch)
        content_feature = self.noise_injection(
            clip_features.unsqueeze(1).repeat(1, self.train_seq_length, 1).to(torch.float32),
            self.variance)
        inter_hs = self.aligner(queries.permute(1, 0, 2),
                                content_feature.permute(1, 0, 2), pos=None)
        embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2))
        # embedding_cls_clip = self.align_proj(clip_features.to(torch.float32))

        # llm_dtype = self.model.model.decoder.embed_tokens.weight.dtype
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1).to(self.llm_dtype)
        label_mask = torch.zeros(bs, clip_token_len, dtype=torch.int64).to(device) - 100
        labels = torch.cat([label_mask, gpt_tokens], dim=1)
        inputs = {'inputs_embeds': embedding_cat, 'attention_mask': attn_mask, 'labels': labels,
                  'output_hidden_states': True}

        out = self.model(**inputs)

        return self.compute_loss(out, clip_token_len, attn_mask, None, gpt_tokens, embedding_clip)


def build_model(args):
    clip_model, preprocess = clip.load(args.clip_model)
    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    llm = AutoModelForCausalLM.from_pretrained(args.language_model, torch_dtype=torch.float16)
    # llm = AutoModelForCausalLM.from_pretrained(args.language_model, load_in_8bit=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=False)
    if not args.ft_llm:
        for p in llm.parameters():
            p.requires_grad = False
    else:
        llm.float()
    # tokenizer_llm = AutoTokenizer.from_pretrained("Manuel030/alpaca-opt-6.7b", use_fast=False)

    model = MacCap(clip_model, llm, tokenizer, args, clip_model.text_projection.shape[0])
    return model
