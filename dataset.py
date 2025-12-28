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


def get_or_train_tokenizer(conf, ds, lang, train=False):
    # tokenizer path
    tokenizer_path = Path(conf.tokenizer_file.format(lang))

    if not Path.exists(tokenizer_path) or train:
        print(f"tokenizing: {lang}")
        # train tokenizer if not exist
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # save tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"tokenizer exist, getting from: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_tokenizers(conf, ds_train, force_retrain_tokenizer=False):
    tokenizer_src = get_or_train_tokenizer(
        conf, ds_train, conf.lang_src, force_retrain_tokenizer
    )
    tokenizer_tgt = get_or_train_tokenizer(
        conf, ds_train, conf.lang_tgt, force_retrain_tokenizer
    )
    return tokenizer_src, tokenizer_tgt


def get_max_length_sentence(conf, ds_train, tokenizer_src, tokenizer_tgt):
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_train:
        src_ids = tokenizer_src.encode(item["translation"][conf.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][conf.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max length of source sentence: {max_len_src}")
    print(f"max length of target sentence: {max_len_tgt}")


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


def get_ds_split(conf, split, tokenizer_src, tokenizer_tgt, ds_raw=None):
    # load_dataset is used to get val & test, while ds_raw is the splitted train
    if ds_raw is None:
        ds = load_dataset(
            f"{conf.corpus}", f"{conf.lang_src}-{conf.lang_tgt}", split=split
        )
    else:
        ds = ds_raw

    dataset = BilingualDataset(
        ds, tokenizer_src, tokenizer_tgt, conf.lang_src, conf.lang_tgt, conf.seq_len
    )

    return dataset


class DeviceDataLoader:
    def __init__(self, dl, device=get_default_device()):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield self._to_device(batch)

    def __len__(self):
        return len(self.dl)

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        else:
            return batch  # leave other types untouched


def get_dataloaders(conf, force_retrain_tokenizer=False):
    # load train split for tokenizer
    ds_train_raw = load_dataset(
        f"{conf.corpus}", f"{conf.lang_src}-{conf.lang_tgt}", split="train"
    )

    # configurable subset
    if conf.train_set_ratio < 1.0:
        ds_train_raw = ds_train_raw.train_test_split(
            train_size=conf.train_set_ratio, seed=42
        )["train"]

    print("total sentence pair for training:", len(ds_train_raw))

    # train tokenizers
    tokenizer_src, tokenizer_tgt = get_tokenizers(
        conf, ds_train_raw, force_retrain_tokenizer
    )

    # show the optimal seq_len
    get_max_length_sentence(conf, ds_train_raw, tokenizer_src, tokenizer_tgt)

    # build datasets
    train_ds = get_ds_split(
        conf, "train", tokenizer_src, tokenizer_tgt, ds_raw=ds_train_raw
    )
    val_ds = get_ds_split(conf, "validation", tokenizer_src, tokenizer_tgt)
    test_ds = get_ds_split(conf, "test", tokenizer_src, tokenizer_tgt)

    # data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

    return (
        DeviceDataLoader(train_dl),
        DeviceDataLoader(val_dl),
        DeviceDataLoader(test_dl),
        tokenizer_src,
        tokenizer_tgt,
    )
