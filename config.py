# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data, device=get_default_device()):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# model config
en_id_model = EasyDict(__name__="Config: En Id Model Translation")

# data
en_id_model.corpus = "opus100"
en_id_model.lang_src = "en"
en_id_model.lang_tgt = "id"
en_id_model.tokenizer_file = ".output/tokenizer_{}.json"
en_id_model.train_set_ratio = 0.05

# model
en_id_model.autocast = True
en_id_model.num_layers = 6
en_id_model.num_heads = 8
en_id_model.d_model = 512
en_id_model.ffn_dim = 2048
en_id_model.dropout = 0.1

# train
en_id_model.batch_size = 10
en_id_model.num_epochs = 20
en_id_model.lr = 0.0001
en_id_model.seq_len = 225
en_id_model.d_model = 512
en_id_model.model_output = ".output/opus100_en_id_mtmodel.pth"
