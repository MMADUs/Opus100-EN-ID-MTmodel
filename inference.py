import torch

from pathlib import Path
from config import en_id_model as mtconf, to_device, get_default_device
from tokenizers import Tokenizer

from model import build_model


def translate(sentence: str):
    device = get_default_device()

    # get tokenizers
    tokenizer_src = Tokenizer.from_file(
        str(Path(mtconf.tokenizer_file.format(mtconf.lang_src)))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(mtconf.tokenizer_file.format(mtconf.lang_tgt)))
    )

    # load model
    model = build_model(mtconf, tokenizer_src, tokenizer_tgt)
    checkpoint = torch.load(mtconf.model_output)
    model.load_state_dict(checkpoint["model"]["model"])
    model = to_device(model)
    model.eval()

    # special tokens
    pad = tokenizer_tgt.token_to_id("[PAD]")
    sos = tokenizer_tgt.token_to_id("[SOS]")
    eos = tokenizer_tgt.token_to_id("[EOS]")

    with torch.no_grad():
        # build input source
        src_ids = tokenizer_src.encode(sentence).ids
        source = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")]
            + src_ids
            + [tokenizer_src.token_to_id("[EOS]")]
            + [tokenizer_src.token_to_id("[PAD]")]
            * (mtconf.seq_len - len(src_ids) - 2),
            dtype=torch.long,
            device=device,
        )

        # make encoder mask
        source_mask = (
            (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0)
        )

        # encode input (making this as the context to understand)
        memory = model.encode(source, source_mask)

        # start of sentence
        decoder_input = torch.tensor([[sos]], device=device)

        # generate text gradually
        while decoder_input.size(1) < mtconf.seq_len:
            seq_len = decoder_input.size(1)

            # mask to ignore padding and stop looking at future words
            future_mask = torch.triu(
                torch.ones((1, seq_len, seq_len), device=device), diagonal=1
            ).bool()
            dec_mask = (decoder_input != pad).unsqueeze(1) & ~future_mask

            # predict the next word
            out = model.decode(memory, source_mask, decoder_input, dec_mask)
            logits = model.project(out[:, -1])
            next_word = torch.argmax(logits, dim=1)

            # append predicted word
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

            # if end of sentence, stop generating word
            if next_word.item() == eos:
                break

        # list of predicted tokens
        ids = decoder_input[0].tolist()

        # ignore padding and stop at eos
        cleaned_ids = [t for t in ids if t != pad and t != eos]

        # decode to text
        return tokenizer_tgt.decode(cleaned_ids)
