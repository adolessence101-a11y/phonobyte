import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.tokenizers.en_tokenizer import PhonobyteTokenizer
from src.training.dataset import get_dataloader
from models.transformer import RawPhonobyteTransformer, PhonobyteConfig

class LegacyBPEConfig:
    def __init__(self):
        self.vocab_size = 32000
        self.max_seq_len = 128
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 4
        self.d_ff = 512
        self.dropout = 0.1

class SingleChannelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer('pe', self._build_positional_encoding(config.max_seq_len, config.d_model))

    def _build_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return self.embeddings(x) + self.pe[:, :seq_len]

class StandardBPETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = SingleChannelEmbedding(config)
        
        from models.transformer import PhonobyteAttentionBlock
        self.layers = nn.ModuleList([PhonobyteAttentionBlock(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _generate_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        h = self.embedding(x)
        causal_mask = self._generate_causal_mask(seq_len, device)
        for layer in self.layers:
            h = layer(h, causal_mask=causal_mask)
        h = self.ln_final(h)
        return self.lm_head(h)

def run_benchmark(dataset_path, device, max_steps=5000):
    print("Initializing Multi-Metric Verification Run...")
    
    config_a = LegacyBPEConfig()
    config_b = PhonobyteConfig()
    config_b.max_seq_len = 128
    
    bpe_emb_params = config_a.vocab_size * config_a.d_model
    phono_emb_params = (config_b.vocab_size * (config_b.d_model // 2)) + (10000 * (config_b.d_model // 2))
    
    print(f"Model A Input Embedding Parameters: {bpe_emb_params:,}")
    print(f"Model B Input Embedding Parameters: {phono_emb_params:,}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        sample_text = f.read()[:50000]
        
    tokenizer_b = PhonobyteTokenizer()
    masks, ids = tokenizer_b.encode(sample_text, return_tensors="list")
    
    bpe_chars_per_window = 128 * 4.1 
    phono_chars_per_window = (len(sample_text) / len(masks)) * 128
    
    print(f"Model A Attention Window Density: {bpe_chars_per_window:.1f} characters")
    print(f"Model B Attention Window Density: {phono_chars_per_window:.1f} characters")

    print("\nStarting Model A (BPE Control) Training Loop...")
    model_a = StandardBPETransformer(config_a).to(device)
    optimizer_a = optim.AdamW(model_a.parameters(), lr=0.0001)
    criterion_a = nn.CrossEntropyLoss()
    
    char_to_id = {char: (i % config_a.vocab_size) for i, char in enumerate(set(sample_text))}
    encoded_a = [char_to_id[c] for c in sample_text if c in char_to_id]
    
    x_a = []
    y_a = []
    for i in range(len(encoded_a) - config_a.max_seq_len - 1):
        x_a.append(encoded_a[i : i + config_a.max_seq_len])
        y_a.append(encoded_a[i + 1 : i + config_a.max_seq_len + 1])
        if len(x_a) >= max_steps:
            break
    x_tensor_a = torch.tensor(x_a, dtype=torch.long)
    y_tensor_a = torch.tensor(y_a, dtype=torch.long)
    
    loss_history_a = []
    model_a.train()
    for step in range(len(x_tensor_a)):
        if step >= max_steps:
            break
        bx = x_tensor_a[step:step+16].to(device)
        by = y_tensor_a[step:step+16].to(device)
        if bx.size(0) < 16:
            break
            
        optimizer_a.zero_grad()
        logits = model_a(bx)
        loss = criterion_a(logits.view(-1, config_a.vocab_size), by.view(-1))
        loss.backward()
        optimizer_a.step()
        loss_history_a.append(loss.item())
        
        if step % 500 == 0:
            print(f"Model A | Step {step}/{max_steps} | Loss: {loss.item():.4f}")

    print("\nStarting Model B (Phonobyte) Training Loop...")
    dataloader_b = get_dataloader(dataset_path, tokenizer_b, batch_size=16, max_seq_len=config_b.max_seq_len)
    model_b = RawPhonobyteTransformer(config_b).to(device)
    optimizer_b = optim.AdamW(model_b.parameters(), lr=0.0001)
    criterion_b = nn.CrossEntropyLoss()
    
    loss_history_b = []
    model_b.train()
    step = 0
    break_outer = False
    while step < max_steps and not break_outer:
        for x_masks, x_ids, y_masks in dataloader_b:
            if step >= max_steps:
                break_outer = True
                break
            x_masks, x_ids, y_masks = x_masks.to(device), x_ids.to(device), y_masks.to(device)
            
            optimizer_b.zero_grad()
            logits = model_b(x_masks, x_ids)
            loss = criterion_b(logits.view(-1, config_b.vocab_size), y_masks.view(-1))
            loss.backward()
            optimizer_b.step()
            loss_history_b.append(loss.item())
            
            if step % 500 == 0:
                print(f"Model B | Step {step}/{max_steps} | Loss: {loss.item():.4f}")
            step += 1

    torch.save(model_b.state_dict(), "phonobyte_advanced_weights.pt")

    print("\nCompiling Advanced Verification Dashboard...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    
    categories = ['Model A\n(Legacy BPE-32k)', 'Model B\n(Phonobyte)']
    params = [bpe_emb_params, phono_emb_params]
    axs[0].bar(categories, params, color=['#c0392b', '#1a365d'], width=0.5, edgecolor='black', alpha=0.85)
    axs[0].set_title("Input Layer Parameter Footprint\n(Lower is Better)", fontsize=11, fontweight="bold", pad=10)
    axs[0].set_ylabel("Parameter Count (Millions)", fontsize=10)
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.3)
    for i, v in enumerate(params):
        axs[0].text(i, v + (v * 0.02), f"{v:,}", ha='center', fontweight='bold', fontsize=9)
        
    axs[1].plot(loss_history_a, color="#c0392b", alpha=0.7, linewidth=2, label="Model A (BPE)")
    axs[1].plot(loss_history_b, color="#1a365d", alpha=0.9, linewidth=2, label="Model B (Phonobyte)")
    axs[1].set_title("Optimization Convergence Velocity\n(Cross-Entropy Trajectories)", fontsize=11, fontweight="bold", pad=10)
    axs[1].set_xlabel("Gradient Step", fontsize=10)
    axs[1].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.3)
    axs[1].legend(frameon=True, facecolor="#f4f6f9")
    
    densities = [bpe_chars_per_window, phono_chars_per_window]
    axs[2].bar(categories, densities, color=['#c0392b', '#1a365d'], width=0.5, edgecolor='black', alpha=0.85)
    axs[2].set_title("Attention Window Information Density\n(Higher is Better)", fontsize=11, fontweight="bold", pad=10)
    axs[2].set_ylabel("Characters Processed per 128-Token Window", fontsize=10)
    axs[2].grid(True, axis='y', linestyle='--', alpha=0.3)
    for i, v in enumerate(densities):
        axs[2].text(i, v + (v * 0.02), f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)
        
    plt.suptitle("Phonobyte Verification Gauntlet: Architectural Evaluation Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_png = "verification_gauntlet_report.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Verification dashboard successfully written to: {output_png}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Harness Device: {device}")
    
    dataset_file = "tinystories_sample.txt"
    if not os.path.exists(dataset_file):
        print("Dataset not found. Generating mock file for quick validation...")
        dummy = "The Red Dragon walked. It was running. " * 1000
        with open(dataset_file, "w", encoding="utf-8") as f:
            f.write(dummy)
            
    run_benchmark(dataset_file, device, max_steps=50000)