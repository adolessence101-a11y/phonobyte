import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from src.tokenizers.jp_tokenizer import JapanesePhonobyteTokenizer

def run_benchmark():
    # 1. Initialize Tokenizers
    print("Initializing tokenizers...")
    jp_phonobyte = JapanesePhonobyteTokenizer()
    
    # We compare against the standard LLaMA-3 tokenizer (a heavy, state-of-the-art BPE)
    # If internet is slow or restricted, fallback to standard GPT2
    try:
        bpe_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    except Exception:
        print("Could not load LLaMA-3 tokenizer online, falling back to GPT-2...")
        bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Dataset: High-variance Japanese sentences representing dialogue, formal text, and verbs
    test_corpus = [
        "赤色の竜が歩いた。",  # The red dragon walked (Verbs + Kanji + Kana)
        "私の名前は誰もいない花です。システムを構築しています。", # Introduction (Pronouns, Names, Technical)
        "日本語のトークン化は非常に非効率的で、開発コストが高くなります。", # Complex technical statement about tokenization
        "遠い国から来た手紙が机の上に置かれていた。", # Literary descriptive sentence
        "ゲーム開発において、プロシージャル生成は無限の可能性を提供します。" # Creative technology / Game design sentence
    ]

    phonobyte_token_counts = []
    bpe_token_counts = []
    characters_counts = []

    print("\n--- Running Token Density Battle ---")
    for idx, sentence in enumerate(test_corpus):
        # Phonobyte execution
        pb_masks, _ = jp_phonobyte.encode(sentence)
        pb_count = len(pb_masks)
        phonobyte_token_counts.append(pb_count)
        
        # BPE execution
        bpe_tokens = bpe_tokenizer.encode(sentence)
        bpe_count = len(bpe_tokens)
        bpe_token_counts.append(bpe_count)
        
        characters_counts.append(len(sentence))
        
        # Print metrics
        saving = ((bpe_count - pb_count) / bpe_count) * 100
        print(f"\nSentence {idx + 1}: '{sentence[:15]}...' (Length: {len(sentence)} chars)")
        print(f"  * BPE Token Count:       {bpe_count}")
        print(f"  * Phonobyte Token Count: {pb_count}")
        print(f"  * Context Window Saving:  {saving:.1f}% fewer tokens required!")

    # 3. Generate Comparative Metrics Dashboard
    print("\nGenerating Diagnostic Dashboard...")
    os.makedirs("benchmarks/results", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_indices = range(1, len(test_corpus) + 1)
    
    # Draw bars
    bar_width = 0.35
    ax.bar([x - bar_width/2 for x in x_indices], bpe_token_counts, bar_width, label='Standard BPE (LLaMA-3/GPT)', color='#E06666')
    ax.bar([x + bar_width/2 for x in x_indices], phonobyte_token_counts, bar_width, label='Phonobyte-JP (8-bit Mora)', color='#6AA84F')
    
    ax.set_xlabel('Test Sentence ID')
    ax.set_ylabel('Required Tokens (Context Window Space)')
    ax.set_title('Japanese Token Tax: BPE vs. Phonobyte-JP\n(Lower is Better / More Dense Context)')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"S{i}" for i in x_indices])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Text callout displaying aggregate savings
    total_bpe = sum(bpe_token_counts)
    total_pb = sum(phonobyte_token_counts)
    avg_saving = ((total_bpe - total_pb) / total_bpe) * 100
    
    plt.text(0.5, 0.02, f"Total BPE Tokens: {total_bpe} | Total Phonobyte Tokens: {total_pb}\nWeighted Context Window Savings: {avg_saving:.1f}% fewer resources",
             horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    output_path = "benchmarks/results/jp_l4_dashboard.png"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Comparative chart saved to: {output_path}")

if __name__ == "__main__":
    run_benchmark()