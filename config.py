# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device=get_default_device()):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


en_id_model = EasyDict(__name__="Config: En Id Model Translation")

# data
en_id_model.corpus = "opus100"
en_id_model.lang_src = "en"
en_id_model.lang_tgt = "id"
en_id_model.tokenizer_file = "tokenizer_en_id.json"
en_id_model.train_set_ratio = 0.5

# model
en_id_model.autocast = True
en_id_model.num_layers = 6
en_id_model.num_heads = 8
en_id_model.d_model = 512
en_id_model.ffn_dim = 2048
en_id_model.dropout = 0.1

# train
en_id_model.batch_size = 8
en_id_model.num_epochs = 20
en_id_model.lr = 0.0001
en_id_model.seq_len = 350
en_id_model.d_model = 512
en_id_model.basename = "tmodel_"
en_id_model.output_dir = ".output"
