import os.path

import torch
from torch.utils.data import Dataset
from typing import Tuple
from ClipContextual import clip
import json
import random
from transformers import AutoTokenizer
from PIL import Image
from torch.utils.data.dataloader import default_collate


def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1][0])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids


class ClipCocoDataset(Dataset):
    def __init__(self, data_path: str, language_model: str, data_ids=None, train_max_length=25):
        self.clip_tokenizer = clip.tokenize
        self.max_seq_len = 25
        self.clip_visual_token_len = 77
        self.tokenizer_llm = AutoTokenizer.from_pretrained(language_model, use_fast=False)
        with open(data_path, 'r') as f:
            self.captions = json.load(f)
        random.shuffle(self.captions)
        self.is_vqa = False
        if 'vqa' in data_path:
            self.is_vqa = True

        self.cache_ids = [None for _ in range(len(self.captions))]
        if data_ids and os.path.exists(data_ids):
            print('\nLoading tokenized ids\n')
            with open(data_ids, 'r') as f:
                self.cache_ids = json.load(f)

        self.trigger = ''
        # self.trigger = ''
        # self.captions_tokens_clip = []
        # self.captions_tokens_llm = []
        # for caption in tqdm(self.captions[:]):
        #     try:
        #         self.captions_tokens_clip.append(self.clip_tokenizer(caption).to(torch.int64).squeeze(0).tolist())
        #         self.captions_tokens_llm.append(self.pad_tokens(
        #         self.tokenizer_llm.encode(self.trigger + caption, return_tensors="pt").to(torch.int64).squeeze(0)).tolist())
        #     except:
        #         continue

        print(f"Total {len(self.captions)} training texts")

    def __len__(self) -> int:
        return len(self.captions)

    def pad_tokens(self, item, max_len, padding_idx=0):
        tokens = item
        padding = max_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) + padding_idx))
        elif padding < 0:
            tokens = tokens[:max_len]
        return tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # tokens = self.captions_tokens[item]
        if self.cache_ids and self.cache_ids[item] is not None:
            clip_tokens, llm_tokens = self.cache_ids[item]
            clip_tokens = torch.tensor(clip_tokens).to(torch.int64)
            llm_tokens = torch.tensor(llm_tokens).to(torch.int64)
        else:
            text = self.captions[item]
            if self.is_vqa:
                text = text.split('.')[0] + '.'
            try:
                self.clip_tokenizer(text)
            except:
                text = 'A photo of apple.'
            clip_tokens = self.pad_tokens(self.clip_tokenizer(text).to(torch.int64).squeeze(0),
                                          77, padding_idx=0)

            llm_tokens = self.pad_tokens(
                self.tokenizer_llm.encode(self.trigger + self.captions[item] + '</s>', return_tensors="pt").to(
                    torch.int64).squeeze(0), self.max_seq_len, padding_idx=1)
            self.cache_ids[item] = [clip_tokens.tolist(), llm_tokens.tolist()]

        return clip_tokens, llm_tokens



class CaptionDataset(Dataset):
    def __init__(self, args, split='test'):
        if split == 'test':
            self.image_path = args.test_image_prefix_path
            with open(args.test_path) as f:
                self.captions = json.load(f)
        elif split == 'val':
            self.image_path = args.val_image_prefix_path
            with open(args.val_path) as f:
                self.captions = json.load(f)
        else:
            raise NotImplementedError
        _, self.preprocess = clip.load(args.clip_model)
        # self.desc = 'Describe the scene of people interacting with a objects.'

        print(f"Total {len(self.captions)} {split} caption pair")

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, item: int):
        one_test_dict = self.captions[item]

        image_full_path = self.image_path + '/' + one_test_dict['image_name']
        image_instance = Image.open(image_full_path)
        image_instance = self.preprocess(image_instance)

        return image_instance, one_test_dict['image_name']



