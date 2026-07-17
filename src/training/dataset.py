import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizers.en_tokenizer import PhonobyteTokenizer

class TinyStoriesPhonobyteDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print("Tokenizing dataset into Phonobyte streams...")
        self.masks, self.ids = self.tokenizer.encode(text, return_tensors="list")
        print(f"Dataset compiled: {len(self.masks)} phonobytes found.")

    def __len__(self):
        return max(0, (len(self.masks) - self.max_seq_len - 1))

    def __getitem__(self, idx):
        x_masks = self.masks[idx : idx + self.max_seq_len]
        x_ids = self.ids[idx : idx + self.max_seq_len]

        y_masks = self.masks[idx + 1 : idx + self.max_seq_len + 1]
        
        return (
            torch.tensor(x_masks, dtype=torch.long),
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y_masks, dtype=torch.long)
        )

def get_dataloader(file_path, tokenizer, batch_size=32, max_seq_len=128, shuffle=True):
    dataset = TinyStoriesPhonobyteDataset(file_path, tokenizer, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)