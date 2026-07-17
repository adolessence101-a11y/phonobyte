Phonobyte: Deterministic, Phonotactically Aligned Tokenization

A high-performance, zero-dictionary alternative to statistical sub-word tokenization. By mapping language directly to the physical, biological rules of speech production (phonotactics), Phonobyte completely eliminates massive lookup embedding tables at the input layer of neural networks.

🚀 Core Breakthroughs

Zero-Vocabulary Representation: Eliminates standard 32,000+ token dictionaries, slashing input layer memory overhead by 99.9%.

Dual-Channel Stereoscopic Embedding: Projects the physical vocal envelope (8-bit acoustic mask) and lexical identity (syllable ID) as parallel feature vectors.

Unified Cross-Lingual Routing: Bypasses language-specific dictionaries using bit-level prefix routing for Latin and CJK logographic scripts simultaneously.

High-Density Storage: The storage-optimized Phonobyte Vellum Binary (.pvb) format leverages static Huffman coding to eliminate pre-training database bottlenecks.

📂 Repository Structure

phonobyte/
├── src/
│   ├── core/
│   │   └── pvb_forge.py          # Unified .pvb compressor & bitstream router
│   ├── tokenizers/
│   │   ├── base.py               # Tokenizer abstract base class
│   │   └── jp_tokenizer.py       # Zero-dictionary Japanese tokenizer
│   └── dataset.py                # PyTorch streaming stereoscopic data pipeline
├── models/
│   └── transformer.py            # PyTorch Dual-Channel Transformer skeleton
├── benchmarks/
│   ├── run_advanced_benchmark.py # English training parity runner
│   ├── run_jp_benchmark.py       # Japanese context density runner
│   └── results/
│       ├── en_l4_dashboard.png   # English convergence baseline dashboard
│       └── jp_l4_dashboard.png   # Japanese performance dashboard
├── phonobyte-white-paper.pdf
├── LICENSE
└── README.md


🌐 The Multi-Linguistic Paradigm Shift

Traditional tokenizers treat non-Latin scripts inefficiently. To represent a single Japanese Kanji or Kana, standard BPE algorithms explode the sequence length by breaking the character into 3 to 4 raw UTF-8 bytes—imposing a heavy "Token Tax" on localized AI deployment.

Phonobyte resolves this by routing characters through bit-level prefixes directly into a single, unified bitstream:

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

📊 Empirical Benchmark: BPE vs. Phonobyte-JP

We evaluated the contextual density of the zero-dictionary Japanese tokenizer against a state-of-the-art BPE tokenizer (LLaMA-3) across a high-variance validation corpus:

Sentence 1 (Dialogue/Verbs): 12 Phonobyte Tokens vs. 15 BPE Tokens (20.0% context savings)

Sentence 3 (Technical Statement): 37 Phonobyte Tokens vs. 46 BPE Tokens (19.6% context savings)

Sentence 5 (Complex System Design): 39 Phonobyte Tokens vs. 49 BPE Tokens (20.4% context savings)

Mixed-Language Compression Efficiency: When evaluating a bilingual string ("The developer creates a lean engine. 日本語のトークン化は非常に非効率的です。"), the upgraded UnifiedPVBForge compressed the raw input from 776 bits down to 419 bits, achieving a 46.0% overall data reduction across the unified binary layout.

⚡ Quickstart Guide

1. Tokenization and Decoding

from src.tokenizers.jp_tokenizer import JapanesePhonobyteTokenizer

tokenizer = JapanesePhonobyteTokenizer()
masks, ids = tokenizer.encode("赤色の竜が歩いた。", return_tensors="list")

print("Packed 8-bit Mora Registers:", masks)


2. High-Density Storage Compression (.pvb)

from src.core.pvb_forge import UnifiedPVBForge

forge = UnifiedPVBForge()
text = "The developer creates a lean engine. 日本語のトークン化は非常に非効率的です。"
bitstream = forge.compress_arbitrary_stream(text)

ratio = forge.evaluate_efficiency(text, bitstream)
print(f"Data Reduction: {ratio:.1f}%")


Every human syllable processed by the framework is mapped to a static 8-bit register representing a structured acoustic envelope consisting of three distinct segments:

Onset Bits [7 : 5] (3 bits): Tracks the physical manner, placement, and complexity of the initial consonant cluster.

Nucleus Bit [4 : 4] (1 bit): Tracks the core vowel sound or diphthong.

Coda Bits [3 : 0] (4 bits): Tracks the terminal consonant cluster constraints.

This structure locks the baseline storage requirement of any valid spoken syllable to a constant 8 bits, yielding instant, deterministic data compression without requiring a statistical vocabulary dictionary.

We challenge the research community to verify the empirical advantages of the Phonobyte framework. Configure two identical sequence-to-sequence autoregressive models using the following parameters:

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

Empirical Baseline Results (Nvidia L4 GPU Run)

We executed this exact protocol for 50,000 steps on an enterprise Nvidia L4 GPU:

1. Input Parameter & Memory Footprint

Model A (Legacy BPE-32k): 8,192,000 parameters

Model B (Phonobyte): 1,312,000 parameters (84% reduction in input layer overhead)

2. Convergence Velocity & Stability

Model A (Legacy BPE): Commences training at a high initial entropy baseline (~10.0 loss) and plateaus early. Because BPE lacks a unified phonological structure, its loss curve exhibits high chaotic jitter, struggling to settle around the $1.0$ loss mark.

Model B (Phonobyte): Commences at a much lower initial entropy (~5.0 loss) due to its constrained 256-state output register. It exhibits a smooth, stable, continuous diagonal slide, maintaining an average loss of 0.6 (frequently dipping to 0.5).

Execution Command

Download TinyStories-valid.txt from Hugging Face.

Save it to your root directory as tinystories_sample.txt.

Run the optimized benchmarking suite:

python benchmarks/run_advanced_benchmark.py


@preprint{phonobyte2026,
  author    = {Michael D. Flowers},
  title     = {The Phonobyte: A Deterministic, Phonotactically Aligned Alternative to Sub-Word Tokenization},
  year      = {2026},
  publisher = {ResearchGate},
  doi       = {DOI: 10.13140/RG.2.2.30807.43687}
}


License

This project is licensed under the MIT License.
