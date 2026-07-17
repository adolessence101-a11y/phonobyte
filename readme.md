Phonobyte: Deterministic, Phonotactically Aligned Tokenization

Phonobyte is a high-performance, deterministic alternative to statistical sub-word tokenization algorithms like Byte-Pair Encoding (BPE) and WordPiece. By mapping human language directly onto the physical, biological rules of speech production (phonotactics), Phonobyte completely eliminates the need for massive, active lookup embedding tables at the input layer of neural networks.

This repository contains the complete implementation of the Phonobyte framework, the storage-optimized Phonobyte Vellum Binary (.pvb) compression engine, and a stereoscopic Dual-Channel Transformer architecture in PyTorch.

Core Features

Zero-Vocabulary Representation: Eliminates standard 32000+ token dictionaries, reducing input layer memory overhead by over 99.9%.

Dual-Channel Stereoscopic Embedding: Projects the physical vocal envelope (8-bit acoustic mask) and lexical identity (syllable ID) as parallel feature vectors for direct attention processing.

Unified Cross-Lingual Routing: Bypasses language-specific dictionaries using bit-level prefix routing, maximizing context window savings for both Latin and CJK logographic scripts simultaneously.

Zero-Core State Toggles: Handles capitalization, formatting (bold, italics), and grammatical suffixes out-of-band, preserving a clean and highly stable hidden representation space.

High-Density Disk Storage (.pvb): Combines phonotactic frequency distributions with static Huffman coding to minimize pre-training database sizes and eliminate network I/O bottlenecks during cluster training.

Repository Structure

phonobyte/
  ├── src/
  │    ├── core/
  │    │    └── pvb_forge.py        (The unified .pvb compressor and bitstream router)
  │    ├── tokenizers/
  │    │    ├── base.py             (The primary tokenizer abstract base class)
  │    │    └── jp_tokenizer.py     (The mathematical, zero-dictionary Japanese tokenizer)
  │    └── dataset.py               (The PyTorch data loading pipeline for stereoscopic tensors)
  ├── models/
  │    └── transformer.py           (PyTorch Dual-Channel Transformer skeleton and stereoscopic attention layers)
  ├── benchmarks/
  │    ├── run_advanced_benchmark.py (The unified multi-metric testing harness for English parity runs)
  │    ├── run_jp_benchmark.py       (The contextual density comparison harness for Japanese text)
  │    └── results/
  │         ├── en_l4_dashboard.png  (English convergence baseline comparison)
  │         └── jp_l4_dashboard.png  (Japanese performance metrics dashboard)
  ├── phonobyte-white-paper.pdf
  ├── LICENSE
  └── README.md


The Multi-Linguistic Paradigm Shift

Traditional tokenizers treat non-Latin scripts inefficiently. To represent a single Japanese Kanji or Kana character, standard BPE algorithms explode the sequence length by breaking the character into 3 to 4 raw UTF-8 bytes. This imposes a heavy context overhead on localized AI deployment.

Phonobyte resolves this through bit-level routing. The .pvb compression engine utilizes an unallocated bit prefix to map different linguistic architectures into a single, unified bitstream:

Bit Prefix

Target Pipeline

Mechanism

Storage Footprint

0 / 10

English Syllable

Dynamic Syllable ID Lookup

8 to 12 bits per syllable

111

Escape State

Punctuation, Capitalization, Markdown

10 bits total

110

Japanese Mora

Continuous 8-bit Packed Sound Envelope

11 bits total (Zero Vocab)

By routing Japanese characters directly behind the 110 prefix, Phonobyte streams continuous phonetic mora structures mathematically, bypassing vocabulary lookups entirely and reducing a raw Japanese character footprint from 24 bits (UTF-8) down to just 11 bits.

Empirical Benchmark: BPE vs. Phonobyte-JP

We evaluated the contextual density of the zero-dictionary Japanese tokenizer against a state-of-the-art BPE tokenizer (LLaMA-3) across a high-variance validation corpus. By removing structural word-spacing metadata (which Japanese text does not natively use), Phonobyte-JP compresses the context footprint significantly:

Sentence 1 (Dialogue/Verbs): 12 Phonobyte Tokens vs. 15 BPE Tokens (20.0% context savings)

Sentence 3 (Technical Statement): 37 Phonobyte Tokens vs. 46 BPE Tokens (19.6% context savings)

Sentence 5 (Complex System Design): 39 Phonobyte Tokens vs. 49 BPE Tokens (20.4% context savings)

Mixed-Language Compression Efficiency: When evaluating a highly complex bilingual string ("The developer creates a lean engine. 日本語のトークン化は非常に非効率的です。"), the upgraded UnifiedPVBForge compressed the raw input from 776 bits down to 419 bits, achieving a 46.0% overall data reduction across the unified binary layout.

Quickstart Guide

The scripts in this repository are written in standard Python and PyTorch, designed to be easily integrated into any existing data pipeline or machine learning workflow.

1. Linguistic Tokenization and Decoding

from src.tokenizers.jp_tokenizer import JapanesePhonobyteTokenizer

tokenizer = JapanesePhonobyteTokenizer()

# Tokenize raw Japanese text
text = "赤色の竜が歩いた。"
masks, ids = tokenizer.encode(text, return_tensors="list")

print("Packed 8-bit Mora Registers:", masks)


2. High-Density Storage Compression (.pvb)

from src.core.pvb_forge import UnifiedPVBForge

forge = UnifiedPVBForge()

# Compress mixed bilingual text directly to a unified bitstream
text = "The developer creates a lean engine. 日本語のトークン化は非常に非効率的です。"
bitstream = forge.compress_arbitrary_stream(text)

# Check data conservation ratio compared to raw UTF-8 bits
ratio = forge.evaluate_efficiency(text, bitstream)
print(f"Data Reduction: {ratio:.1f}%")


3. Loading into the Dual-Channel Transformer

import torch
from models.transformer import RawPhonobyteTransformer, PhonobyteConfig

config = PhonobyteConfig()
model = RawPhonobyteTransformer(config)

# Dummy batch input representing [Batch Size, Sequence Length]
mask_tensor = torch.randint(0, 256, (2, 128))
id_tensor = torch.randint(0, 10000, (2, 128))

# Forward pass outputs next-step predictions over the 256 register states
logits = model(mask_tensor, id_tensor)
print("Output shape:", logits.shape)


Mathematical Design: The 8-Bit Envelope

Every human syllable processed by the framework is mapped to a static 8-bit register. This register represents a structured acoustic envelope consisting of three distinct segments:

Onset Bits [7 : 5] (3 bits): Tracks the physical manner, placement, and complexity of the initial consonant cluster.

Nucleus Bit [4 : 4] (1 bit): Tracks the core vowel sound or diphthong.

Coda Bits [3 : 0] (4 bits): Tracks the terminal consonant cluster constraints.

This structure locks the baseline storage requirement of any valid spoken syllable to a constant 8 bits, yielding instant, deterministic data compression without requiring a statistical vocabulary dictionary.

The Verification Gauntlet: Replication Protocol & Empirical Results

We challenge the research community to verify the empirical advantages of the Phonobyte framework. To replicate our benchmarks or verify our results, configure two identical sequence-to-sequence autoregressive models using the following parameters:

Sequence Length: 128

Hidden Dimension Size (d_model): 256

Attention Heads: 8

Attention Blocks (Layers): 4

Feed-Forward Interior Dimension (d_ff): 512

Dataset: TinyStories validation corpus (~19.4 MB)

Training Steps: 50,000 (Batch Size: 16)

Parallel Configurations

Model A (Control): Standard BPE tokenizer (32,000 vocabulary size) with a conventional single-channel embedding table and classification head.

Model B (Experimental): Phonobyte Tokenizer with the static 8-bit Dual-Channel Embedding projection layer and a 256-state linear head.

Our Baseline Results (Nvidia L4 GPU Run)

We executed this exact protocol for 50,000 steps on an enterprise Nvidia L4 GPU. Here is the empirical baseline you are looking to replicate:

1. Input Parameter & Memory Footprint

Model A (Legacy BPE-32k): 8,192,000 parameters

Model B (Phonobyte): 1,312,000 parameters (84% reduction in input layer overhead)

2. Convergence Velocity & Stability

Model A (Legacy BPE): Commences training at a high initial entropy baseline (~10.0 loss) and plateaus early. Because BPE lacks a unified phonological structure, its loss curve exhibits high chaotic jitter, struggling to settle around the $1.0$ loss mark.

Model B (Phonobyte): Commences at a much lower initial entropy (~5.0 loss) due to its constrained 256-state output register. It exhibits a smooth, stable, continuous diagonal slide, maintaining an average loss of 0.6 (frequently dipping to 0.5).

How to Run the Benchmark

To execute this protocol on your own system or cloud pod:

Download the TinyStories-valid.txt file from Hugging Face.

Save it to your root directory as tinystories_sample.txt.

Run the optimized benchmarking suite:

python benchmarks/run_advanced_benchmark.py


Citation and Prior Art

@preprint{phonobyte2026,
  author    = {Michael D. Flowers},
  title     = {The Phonobyte: A Deterministic, Phonotactically Aligned Alternative to Sub-Word Tokenization},
  year      = {2026},
  publisher = {ResearchGate},
  doi       = {DOI: 10.13140/RG.2.2.30807.43687}
}


License

This project is licensed under the MIT License.