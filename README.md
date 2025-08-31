# Sequence-to-Sequence LSTM vs Transformer

A comprehensive comparison of LSTM and Transformer architectures for English-to-French machine translation using Keras/TensorFlow.

## 📋 Project Overview

This project implements and compares two popular sequence-to-sequence architectures:
- **LSTM-based Encoder-Decoder** with attention mechanism
- **Transformer** with multi-head self-attention

Both models are trained on English-French sentence pairs to perform machine translation tasks.

**Note:** For detailed comparison results and model outputs, please consult the `Documentation/` folder which contains comprehensive translation examples from both models.

## 🏗️ Project Structure

```
├── Machine-Translation-Seq2Seq-Keras/
│   ├── LSTM.py                    # LSTM seq2seq implementation
│   ├── transformer.py             # Transformer implementation
│   └── tools/
│       ├── data_class.py          # Data preprocessing utilities
│       └── data/                  # Training data files
├── Documentation/
│   ├── LSTM/
│   │   ├── seq2seq_outputs.txt    # LSTM model translation examples
│   │   └── trained_model.keras    # Saved LSTM model
│   └── Transformer/
│       ├── transformer_outputs.txt # Transformer translation examples
│       └── transformer.keras      # Saved Transformer model
├── others/
│   ├── myseq2seq.py              # Simple seq2seq example
│   └── keras/
│       ├── lstm_seq2seq.py       # Character-level LSTM seq2seq
│       ├── fra.txt               # French-English dataset
│       └── ara.txt               # Arabic dataset
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
pip install numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdeljalil-Ounaceur/Sequence-to-Sequence-LSTM-vs-Transformer.git
cd Sequence-to-Sequence-LSTM-vs-Transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # Create this file with above packages
```

## 💻 Usage

### Training the LSTM Model

```bash
cd Machine-Translation-Seq2Seq-Keras
python LSTM.py
```

**Model Architecture:**
- Encoder: LSTM with 256 units, ReLU activation
- Decoder: LSTM with 256 units, teacher forcing
- Embedding: 200-dimensional word embeddings
- Training: 16 epochs, batch size 1024

### Training the Transformer Model

```bash
cd Machine-Translation-Seq2Seq-Keras
python transformer.py
```

**Model Architecture:**
- Layers: 2 encoder/decoder layers
- Model dimension: 128
- Attention heads: 4
- Feed-forward dimension: 256
- Training: 2 epochs, batch size 256

### Using Pre-trained Models

```python
# Load LSTM model
from keras.models import load_model
lstm_model = load_model('Documentation/LSTM/trained_model.keras')

# Load Transformer model
transformer_model = load_model('Documentation/Transformer/transformer.keras')
```

## 📊 Model Comparison

### Translation Quality Examples

**Input:** "the weather in paris during the fall is usually pleasant and sometimes rainy"

- **LSTM Output:** "californie est le enneigée pendant l' automne mais il est généralement pluvieux en hiver"
- **Transformer Output:** "le raisin est le paris l' automne mais il est parfois agréable en pluvieux en et il"

**Input:** "i like apples and oranges but my least favorite fruit is grapefruit"

- **LSTM Output:** "california les pommes pommes et pommes et il pommes pamplemousse"
- **Transformer Output:** "j'aime les pommes et le moins préféré est le moins préféré"

### Performance Analysis

| Model | Architecture | Parameters | Training Time | Translation Quality |
|-------|-------------|------------|---------------|-------------------|
| LSTM | Encoder-Decoder | ~2M | 16 epochs | Moderate accuracy, some semantic errors |
| Transformer | Multi-head Attention | ~1.5M | 2 epochs | Better semantic understanding, more fluent |

**Key Observations:**
- Transformer shows better semantic understanding despite fewer training epochs
- LSTM tends to repeat words and lose context in longer sentences
- Both models struggle with complex sentence structures due to limited vocabulary

## 🔧 Data Processing

The `Data_class` handles:
- Loading English-French sentence pairs
- Tokenization with start/end tokens (SOS/EOS)
- Sequence padding for batch processing
- Vocabulary size management

## 📁 Additional Examples

### Character-Level Translation
Run the character-level LSTM example:
```bash
cd others/keras
python lstm_seq2seq.py
```

### Simple Seq2Seq Demo
```bash
cd others
python myseq2seq.py
```

## 🎯 Key Features

- **Modular Design:** Separate implementations for easy comparison
- **Comprehensive Evaluation:** Translation examples and performance metrics
- **Multiple Approaches:** Word-level and character-level implementations
- **Pre-trained Models:** Ready-to-use saved models
- **Educational Value:** Well-commented code for learning purposes

## 📈 Future Improvements

- [ ] Implement BLEU score evaluation
- [ ] Add attention visualization
- [ ] Experiment with larger datasets
- [ ] Implement beam search decoding
- [ ] Add more language pairs

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
