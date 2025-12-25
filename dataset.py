# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_default_device


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def train_tokenizer(conf, ds, lang):
    # tokenizer path
    tokenizer_path = Path(conf.tokenizer_file.format(lang))

    # train tokenizer vocab
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

    # save tokenizer
    tokenizer.save(str(tokenizer_path))

    return tokenizer


def get_tokenizers(conf, ds_train):
    tokenizer_src = train_tokenizer(conf, ds_train, conf.lang_src)
    tokenizer_tgt = train_tokenizer(conf, ds_train, conf.lang_tgt)
    return tokenizer_src, tokenizer_tgt


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # SOS (start of sentence)
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        # EOS (end of sentence)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        # PAD (padding token)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # transform text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # add SOS, EOS, & PAD
        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # we will add <s> and </s>
        # we will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure the number of padding tokens is not negative
        # if it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def get_ds_split(conf, split, tokenizer_src, tokenizer_tgt):
    ds_raw = load_dataset(
        f"{conf.corpus}", f"{conf.lang_src}-{conf.lang_tgt}", split=split
    )
    dataset = BilingualDataset(
        ds_raw, tokenizer_src, tokenizer_tgt, conf.src_lang, conf.tgt_lang, conf.seq_len
    )
    return dataset


class DeviceDataLoader:
    def __init__(self, dl, device=get_default_device()):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield b.to(self.device)

    def __len__(self):
        return len(self.dl)


def get_dataloaders(conf):
    # load train split for tokenizer
    ds_train_raw = load_dataset(
        f"{conf.corpus}", f"{conf.lang_src}-{conf.lang_tgt}", split="train"
    )

    # configurable subset
    ds_train_raw = ds_train_raw.train_test_split(
        train_size=conf.train_set_ratio, seed=42
    )["train"]

    # train tokenizers
    tokenizer_src, tokenizer_tgt = get_tokenizers(conf, ds_train_raw)

    # build datasets
    train_ds = get_ds_split(conf, "train", tokenizer_src, tokenizer_tgt)
    val_ds = get_ds_split(conf, "val", tokenizer_src, tokenizer_tgt)
    test_ds = get_ds_split(conf, "test", tokenizer_src, tokenizer_tgt)

    # data loaders
    train_dl = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1)
    test_dl = DataLoader(test_ds, batch_size=1)

    return (
        DeviceDataLoader(train_dl),
        DeviceDataLoader(val_dl),
        DeviceDataLoader(test_dl),
        tokenizer_src,
        tokenizer_tgt,
    )
