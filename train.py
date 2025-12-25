# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import time
import torch
import torch.nn as nn

from tqdm import tqdm

from dataset import get_dataloaders, causal_mask
from model import Transformer, initialize_parameters
from config import to_device, get_default_device
from utils import time_formatter


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len):
    device = get_default_device()

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # init the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def train_model(conf, callback):
    train_dl, val_dl, _tdl, tokenizer_src, tokenizer_tgt = get_dataloaders(conf)

    # model
    model = Transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=conf.seq_len,
        tgt_seq_len=conf.seq_len,
        d_model=conf.d_model,
        N=conf.num_layers,
        h=conf.num_heads,
        dropout=conf.dropout,
        d_ff=conf.ffn_dim,
    )
    initialize_parameters(model)
    model = to_device(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, eps=1e-9)

    # criterion
    loss_fn = to_device(
        nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )
    )

    callback.init()
    start_time = time.time()

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    init_epoch = 0

    for epoch in range(init_epoch, conf.num_epoch):
        epoch_start = time.time()
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0

        batch_iter = tqdm(train_dl, desc=f"processing epoch {epoch}")

        for batch in batch_iter:
            encoder_input = to_device(batch["encoder_input"])  # (B, seq_len)
            decoder_input = to_device(batch["decoder_input"])  # (B, seq_len)
            encoder_mask = to_device(batch["encoder_mask"])  # (B, 1, 1, seq_len)
            decoder_mask = to_device(batch["decoder_mask"])  # (B, 1, seq_len, seq_len)

            # forward
            proj_output = model(
                encoder_input, decoder_input, encoder_mask, decoder_mask
            )

            # compare the output with the label
            label = to_device(batch["label"])

            # compute loss
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            train_loss += loss.item()

            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

            # backward + update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dl:
                encoder_input = to_device(batch["encoder_input"])  # (B, seq_len)
                decoder_input = to_device(batch["decoder_input"])  # (B, seq_len)
                encoder_mask = to_device(batch["encoder_mask"])  # (B, 1, 1, seq_len)
                decoder_mask = to_device(
                    batch["decoder_mask"]
                )  # (B, 1, seq_len, seq_len)

                # forward
                proj_output = model(
                    encoder_input, decoder_input, encoder_mask, decoder_mask
                )

                # compare the output with the label
                label = to_device(batch["label"])

                # compute loss
                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
                )
                val_loss += loss.item()

        # avg
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)

        # history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1} - {time_formatter(epoch_time)} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}\n"
        )

        # callbacks
        model_state = {
            "model": model.state_dict(),
        }
        optimizer_state = {
            "optimizer": optimizer.state_dict(),
        }

        early_stop = callback.step(val_loss, model_state, optimizer_state)
        if early_stop:
            break

    end_time = time.time()
    print(f"elapsed time: {time_formatter(end_time - start_time)}")
    return history
