import torch
import torch.nn as nn
from ClipContextual import clip
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.utlis import PlugAndPlayContrastiveDecodingOneStepFast
import datetime
import re
import math
import os
import json


class BaseModel(nn.Module):
    def __init__(self, clip, llm, tokenizer, args):
        super(BaseModel, self).__init__()

        self.model = llm
        self.tokenizer = tokenizer
        self.clip_model = clip

        self.llm_vocab = llm.get_input_embeddings()
        self.vocab_size, self.vocab_dim = self.llm_vocab.weight.shape

        self.beam_width = args.k
        self.alpha = args.alpha
        self.beta = args.beta
        self.decoding_len = args.decoding_len
        desc = args.eval_desc
        self.desc = ''
        self.question_ids = torch.tensor(self.tokenizer.encode(desc))
        question_embed = self.llm_vocab(self.question_ids)
        self.demo_feature = None
        self.demo_mask = None
        self.answer_ids = []
        self.answer_text = []

        self.register_buffer('question_embed', question_embed)

        # fixed vqa config from BLIP2
        self.prompt = 'Question: {} Short answer:'
        # self.prompt = '{} '

        self.max_len = 10
        self.min_len = 1
        self.num_beams = 5
        # self.num_ans_candidates = 128
        self.variance = args.noise_variance
        self.noise_type = args.noise_type
        # inference parameters for dcn_seq
        self.infer_patch_weight = args.infer_patch_weight
        self.train_seq_length = args.train_seq_length
        self.infer_variance_ratio = args.infer_variance_ratio
        self.infer_window_size = args.infer_window_size
        self.mask_ratio = args.mask_ratio
        self.infer_multi_cls = args.infer_multi_cls

        # default: use sentence embedding
        memory_bank = torch.load(args.memory_path, map_location='cpu').half()
        self.register_buffer('memory_bank', memory_bank)
        self.temperature = args.temperature
        self.inference_mb_type = args.inference_mb_type
        # contrastive loss

        self.ce_loss = nn.CrossEntropyLoss()
        self.llm_weight = []

        try:
            self.llm_dtype = self.model.model.embed_tokens.weight.dtype
        except Exception as e:
            self.llm_dtype = self.model.model.decoder.embed_tokens.weight.dtype


    def change_generate_config(self, beam_width=None, alpha=None, beta=None, decoding_len=None, desc=None,
                               demo_feature=None, demo_mask=None, answer_ids=None, answer_text=None):
        if beam_width is not None:
            self.beam_width = beam_width
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if decoding_len is not None:
            self.decoding_len = decoding_len
        if desc is not None:
            if isinstance(desc, str):
                self.desc = desc
                device = self.question_embed.device

                res = self.tokenizer(desc, return_tensors="pt")
                self.question_mask = res['attention_mask'].to(device)
                self.question_embed = self.llm_vocab(res['input_ids'].to(device))
            else:
                self.desc = desc
                device = self.question_embed.device
                res = self.tokenizer([self.prompt.format(i) for i in desc], padding=True,
                                     return_tensors='pt')
                self.question_mask = res['attention_mask'].to(device)
                self.question_embed = self.llm_vocab(res['input_ids'].to(device))

        if demo_feature is not None:
            self.demo_feature = demo_feature

        if demo_mask is not None:
            self.demo_mask = demo_mask

        if answer_ids is not None:
            self.answer_ids = answer_ids

        if answer_text is not None:
            self.answer_text = answer_text

    @torch.no_grad()
    def magic_search(self, prefix_embed, start_token, image_instance, img_len=32, clip_text_max_len=60):
        prefix_len = prefix_embed.size()[1]
        past_key_values, last_hidden_states, logits = None, None, None
        input_ids_for_class = start_token.clone()
        image_embeds = image_instance

        start_time = datetime.datetime.now()
        decoding_len = self.decoding_len
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits, input_ids_for_class = \
                PlugAndPlayContrastiveDecodingOneStepFast(
                    self.model,
                    prefix_embed,
                    prefix_len,
                    self.beam_width,
                    self.alpha,
                    self.beta,
                    self.tokenizer,
                    image_embeds,
                    self.clip_model,
                    clip_text_max_len,
                    past_key_values,
                    last_hidden_states,
                    logits,
                    first_step=step == 0,
                    input_ids_for_class=input_ids_for_class,
                    img_len=img_len,
                    step=step
                )
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds() * 1000
        # a = self.clean_output(self.tokenizer.decode(input_ids_for_class[0], skip_special_tokens=True))
        return self.tokenizer.decode(input_ids_for_class[0], skip_special_tokens=True)

    def clean_func(self, s):
        if 'Answer:' in s:
            s = s.split('Answer:')[-1].strip()
        if '<PAD>' in s:
            s = re.sub('<PAD>', '', s).strip()
        # remove bad pattern
        s = re.sub('[^a-zA-Z\d ,.]', '', s)
        # remove multiple blank space
        s = re.sub('[ ]+', ' ', s)
        s = re.sub('[.]+', '.', s)
        s = re.sub('[0-9]{4}', '', s)
        return s.strip()

    def clean_output(self, in_str):
        if isinstance(in_str, list):
            s = [self.clean_output(i) for i in in_str]
        else:
            s = self.clean_func(in_str)
        return s

    def generate_by_strategy(self, embedding_clip, attn_mask=None, gen_strategy='naive', img_clip_cls=None,
                             num_fusion=32):
        bs = embedding_clip.shape[0]
        device = embedding_clip.device
        if len(self.question_embed.shape) == 2:
            start_token = self.question_embed.unsqueeze(0).repeat(bs, 1, 1)
        else:
            start_token = self.question_embed
        #
        # # for multi-noise
        # if gen_strategy == 'multi_noise':
        #     embedding_clip = embedding_clip.view()
        #

        if embedding_clip is not None:
            prefix_embed = torch.cat([embedding_clip.half(), start_token], dim=1)
        else:
            prefix_embed = start_token
            num_fusion = 0

        if gen_strategy == 'naive':

            inputs = {'inputs_embeds': prefix_embed, 'attention_mask': attn_mask}
            ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                      num_beams=4,
                                      max_new_tokens=self.decoding_len, output_scores=False)
            try:
                output_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
            except Exception as e:
                print(e)
                output_text = 'This is a photo of nothing.'
        elif gen_strategy == 'magic':
            output_text = self.magic_search(prefix_embed.half(),
                                            self.question_ids.repeat(bs, 1).to(device),
                                            img_clip_cls, num_fusion)
        elif gen_strategy == 'text':
            # self.question embedding is the input
            for desc in ['A child holding a flowered umbrella and petting a yak.',
                         "The bathroom wall needs to be resurfaced and painted.",
                         "Two stainless steel sinks with mirrors and a fire extinguisher.",
                         "Picture of a church and its tall steeple."
                         ]:
                self.desc = desc
                clip_tokens = self.clip_model.tokenizer.encode(self.desc)
                clip_tokens += [0] * (64 - len(clip_tokens))
                clip_tokens = torch.tensor(clip_tokens).unsqueeze(0).to(device)

                clip_features, contex_text, text_proj = self.clip_model.encode_text(clip_tokens)
                clip_conx = contex_text @ text_proj.T
                clip_conx /= clip_conx.norm(dim=-1, keepdim=True)

                clip_features /= clip_features.norm(dim=-1, keepdim=True)

                queries = self.query_fusion.weight.unsqueeze(0).repeat(bs, 1, 1)
                inter_hs = self.aligner(queries.permute(1, 0, 2),
                                        clip_conx.permute(1, 0, 2).to(torch.float32), pos=None)
                embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2))
                inputs = {'inputs_embeds': torch.cat(
                    [embedding_clip, self.llm_vocab(torch.tensor(self.tokenizer.encode('')).to(device)).unsqueeze(0)],
                    dim=1).to(torch.float16),
                          'attention_mask': None}
                ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                          num_beams=4,
                                          max_new_tokens=self.decoding_len, output_scores=False)
                print(self.tokenizer.batch_decode(ans['sequences']))

            return self.tokenizer.batch_decode(ans['sequences'])
        elif gen_strategy == 'multi_noise':
            inputs = {'inputs_embeds': prefix_embed, 'attention_mask': attn_mask}
            ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                      num_beams=self.contrastive_generation_beam_size,
                                      max_new_tokens=self.decoding_len, output_scores=False)
            output_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
        elif gen_strategy == 'vqa':

            inputs = {'inputs_embeds': prefix_embed,
                      'attention_mask': torch.cat(
                          [torch.ones(embedding_clip.shape[0], embedding_clip.shape[1]).to(device), self.question_mask],
                          dim=1)}
            stop = False
            decode_length = self.decoding_len
            while not stop:
                ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                          num_beams=self.num_beams,
                                          max_new_tokens=decode_length,
                                          min_length=self.min_len,
                                          length_penalty=-1, output_scores=True if self.answer_ids else False)
                if not self.answer_ids:
                    output_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
                else:
                    output_text = self.extract_answer_from_candidate(ans)
                if 'Answer:' in output_text[0] or decode_length > 48:
                    # print(decode_length)
                    stop = True
                else:
                    decode_length += 8

        elif gen_strategy == 'vqa_demo':
            inputs = {'inputs_embeds': torch.cat([self.demo_feature.half(), prefix_embed], dim=1),
                      'attention_mask': torch.cat(
                          [self.demo_mask, torch.ones(embedding_clip.shape[0], embedding_clip.shape[1]).to(device),
                           self.question_mask],
                          dim=1)}
            ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                      num_beams=self.num_beams,
                                      max_new_tokens=self.max_len,
                                      min_length=self.min_len,
                                      length_penalty=-1, output_scores=True if self.answer_ids else False)
            if not self.answer_ids:
                output_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
            else:
                output_text = self.extract_answer_from_candidate(ans)
        elif gen_strategy == 'contrastive':
            inputs = {'inputs_embeds': prefix_embed, 'attention_mask': attn_mask}
            ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                      num_beams=self.contrastive_generation_beam_size,
                                      num_return_sequences=self.contrastive_generation_beam_size, do_sample=False,
                                      early_stopping=True,
                                      num_beam_groups=self.contrastive_generation_beam_size,
                                      diversity_penalty=self.contrastive_generation_diversity,
                                      max_new_tokens=self.decoding_len, output_scores=False)
            output_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
            out_list = []
            for idx in range(bs):
                out_list.append(output_text[idx * self.contrastive_generation_beam_size:
                                            (idx + 1) * self.contrastive_generation_beam_size])

            output_text = out_list
        elif gen_strategy == 'visualization':

            inputs = {'inputs_embeds': prefix_embed, 'attention_mask': attn_mask}
            ans = self.model.generate(**inputs, return_dict_in_generate=True,
                                      num_beams=4, output_attentions=True,
                                      max_new_tokens=self.decoding_len, output_scores=True)
            beam = ans['beam_indices'][-1][-2]
            attn = ans['attentions'][-1][-1][beam].squeeze(1)
            self.llm_weight.append(attn.tolist())
            with open(os.path.join(self.vis_path, 'llm_weight'), 'w') as f:
                f.write(json.dumps(self.llm_weight))

            output_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
        else:
            raise NotImplementedError

        return self.clean_output(output_text)

    def extract_answer_from_candidate(self, ans):
        if not isinstance(ans, list):
            pred_text = self.tokenizer.batch_decode(ans['sequences'], skip_special_tokens=True)
        else:
            pred_text = ans
        filter_text = []
        for idx, p in enumerate(pred_text):
            filter_text.append([])
            wd_list = re.sub('[.,?!]', '', p).split(' ')
            for a in self.answer_text:
                if a in wd_list:
                    filter_text[-1].append(a)
        return filter_text

        pred_seq = ans['sequences']
        bs, pred_length = pred_seq.shape
        if 'beam_indices' in ans:
            # collect beam generation scores
            beam_indices = ans['beam_indices'][:, 1:]
            scores = torch.zeros(bs, pred_length)
            beam_scores = [v.sigmoid() for v in ans['scores']]
            for idx, score in enumerate(beam_scores):
                # assert score[beam_indices[0][idx]].topk(1).indices == pred_seq[0][idx + 1]
                scores[:, idx + 1] = score[beam_indices[:, idx]][torch.arange(bs), pred_seq[:, idx + 1]]
        elif 'scores' in ans:
            scores = ans['scores']
        else:
            raise NotImplementedError

        answer = [['', 0] for _ in range(bs)]
        for a_i, a_idx in enumerate(self.answer_ids):
            for idx in range(1, pred_length):
                for p_i, p_idx in enumerate(pred_seq):
                    for aa_idx in a_idx:
                        if idx + len(aa_idx) < pred_length and p_idx[idx:idx + len(aa_idx)] == aa_idx:
                            ans_s = scores[p_i][idx:idx + len(aa_idx)].mean()
                            if answer[p_i][1] < ans_s:
                                answer[p_i][0] = self.answer_text[a_i]

    def compute_loss(self, out, clip_token_len, attn_mask, embedding_cls_clip, gpt_tokens, embedding_clip=None):
        hidden_states = out.hidden_states[-1][:, clip_token_len:]

        logits = out.logits
        loss = out.loss

        # predicted logits is shift right for one position
        ac = ((logits[:, clip_token_len - 1:-1].argmax(2) == gpt_tokens) * (gpt_tokens > 1)).sum() / (
                gpt_tokens > 1).sum()
        return {'loss': loss}, ac

    def get_uniform_ball_noise(self, input_shape, radius=0.1):
        uniform_noise_ball = torch.randn(input_shape)  # normal distribution
        uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)
        u = torch.rand(input_shape[0])  # unified distribution
        u = u ** (1. / input_shape[1])
        uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
        return uniform_noise_ball

    def get_von_mesis_fisher_noise(self, feature, radius=10):
        generator = torch.distributions.von_mises.VonMises(feature, torch.tensor(radius).to(feature.device))
        return generator.sample()

    def noise_injection(self, x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
        device = x.device
        if variance == 0.0:
            return F.normalize(x, dim=-1)
        std = math.sqrt(variance)
        # if not dont_norm:
        #     x = F.normalize(x, dim=1)
        # if uniform_noise:
        #     x = x + self.get_uniform_ball_noise(x.shape, radius=std).to(device)
        if self.noise_type == 'gaussian':
            x = x + (torch.randn(x.shape,
                                 device=device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
        elif self.noise_type == 'vonMisesFisher':
            x = self.get_von_mesis_fisher_noise(x, int(variance))
        else:
            raise NotImplementedError

        if modality_offset is not None:
            x = x + modality_offset
        return F.normalize(x, dim=-1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayerWeight(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            if isinstance(output, tuple):
                output = (self.norm(output[0]), output[1])
            else:
                output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
