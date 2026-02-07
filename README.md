# Opus100 English-Indonesian Neural Machine Translation

A PyTorch implementation of a Transformer-based neural machine translation (NMT) model trained on the Opus100 English-Indonesian parallel corpus. This project implements the complete Transformer architecture from scratch, providing both educational value and practical translation capabilities.

## Features

- **Transformer Architecture**: Complete encoder-decoder implementation with multi-head attention
- **Built from Scratch**: All core components implemented without relying on high-level PyTorch Transformer API
- **Mixed Precision Training**: Uses autocast for efficient GPU training
- **Advanced Metrics**: Evaluates translation quality using CER, WER, and BLEU scores
- **Flexible Configuration**: Easy-to-customize hyperparameters for experimentation
- **Checkpoint Management**: Automatic model saving and early stopping
- **Learning Rate Scheduling**: Adaptive learning rate reduction on validation plateau
- **Data Preprocessing**: Automatic tokenization, padding, and causal masking

## Architecture

<img src="transformer.png" width="500">

The model consists of:

- **Encoder**: Stack of 4 identical layers with multi-head self-attention and feed-forward networks
- **Decoder**: Stack of 4 identical layers with masked self-attention, cross-attention, and feed-forward networks
- **Embeddings**: Learned token embeddings scaled by √d_model
- **Positional Encoding**: Sinusoidal positional encoding for sequence ordering
- **Layer Normalization**: Pre-normalization applied before each sub-layer
- **Residual Connections**: Skip connections around attention and feed-forward modules

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Embedding dimension |
| `num_layers` | 4 | Number of encoder/decoder layers |
| `num_heads` | 4 | Number of attention heads |
| `ffn_dim` | 1024 | Feed-forward network inner dimension |
| `seq_len` | 50 | Maximum sequence length |
| `dropout` | 0.1 | Dropout rate |

## Usage

### Training

See the training notebook here:  
[test.ipynb](https://github.com/MMADUs/Opus100-EN-ID-NMT/blob/main/train.ipynb)

```python
from train import train_model
from utils import TrainCheckpoint, EarlyStopping, ReduceLROnPlateau, TrainingCallback
from config import en_id_model

# Setup callbacks
tcp = TrainCheckpoint(en_id_model.model_output)
es = EarlyStopping(patience=5)
rlr = ReduceLROnPlateau(factor=0.5, patience=2, cooldown=2)
callback = TrainingCallback(checkpoint=tcp, early_stop=es, reduce_lr=rlr)

# Train model
history = train_model(en_id_model, callback, preload=False)
```

### Inference

See the inference test right here:  
[test.ipynb](https://github.com/MMADUs/Opus100-EN-ID-NMT/blob/main/test.ipynb)

```python
from inference import TranslationContext

context = TranslationContext()

sentence = "good morning, lets have a breakfast together"
translation = context.translate(sentence)
print(translation)
```

## Project Structure

```
Opus100-EN-ID-MTmodel/
├── config.py           # Model and training configuration
├── model.py            # Transformer architecture from scratch
├── pytorch.py          # Alternative PyTorch nn.Transformer implementation
├── dataset.py          # Data loading and preprocessing
├── train.py            # Training loop and inference utilities
├── utils.py            # Callbacks, checkpointing, and helpers
├── inference.py        # Inference utilities
├── plot.py             # Visualization functions
├── train.ipynb         # Training notebook
├── test.ipynb          # Testing and evaluation notebook
└── README.md           # This file
```

## Configuration

Edit `config.py` to customize the model:

```python
en_id_model.d_model = 256          # Model dimension
en_id_model.num_layers = 4         # Number of layers
en_id_model.num_heads = 4          # Attention heads
en_id_model.ffn_dim = 1024         # Feed-forward dimension
en_id_model.seq_len = 50           # Max sequence length
en_id_model.batch_size = 64        # Batch size
en_id_model.num_epochs = 30        # Training epochs
en_id_model.lr = 0.0001            # Learning rate
en_id_model.dropout = 0.1          # Dropout rate
```

## Evaluation Metrics

The model is evaluated using:

- **Character Error Rate (CER)**: Character-level accuracy
- **Word Error Rate (WER)**: Word-level accuracy  
- **BLEU Score**: Fluency and n-gram similarity with reference translations

## Training Details

- **Optimizer**: Adam (lr=0.0001, eps=1e-9)
- **Loss**: Cross-entropy with label smoothing (0.1)
- **Mixed Precision**: Automatic mixed precision for efficiency
- **Early Stopping**: Stops if validation loss doesn't improve for 5 epochs
- **Learning Rate Scheduling**: Reduces LR by 0.5x if validation loss plateaus

## Dataset

The model is trained on the [Opus100](https://opus.nlpl.eu/opus-100.php) English-Indonesian parallel corpus:

- **Training pairs**: ~1M sentence pairs (configurable)
- **Vocabulary**: Built using WordLevel tokenizer with min_frequency=2
- **Special tokens**: [UNK], [PAD], [SOS], [EOS]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{opus100_en_id_mt,
  title={Opus100 English-Indonesian Neural Machine Translation},
  author={Muhammad Nizwa},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/Opus100-EN-ID-MTmodel}}
}
```

## Acknowledgments

- [Opus100](https://opus.nlpl.eu/opus-100.php) for the parallel corpus
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) for easy dataset loading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper for the Transformer architecture

## Author

**Muhammad Nizwa** - 2025

---

For questions or issues, please open an issue on GitHub.
