import pykakasi
from typing import List, Tuple, Union
import torch
import re
from src.tokenizers.base import BasePhonobyteTokenizer

class JapanesePhonobyteTokenizer(BasePhonobyteTokenizer):
    """
    Mora-aligned Japanese Phonobyte Tokenizer.
    Maps Japanese text to static 8-bit phonotactic registers.
    """
    
    def __init__(self):
        super().__init__()
        self.kks = pykakasi.kakasi()
        
        # 4 bits (0-15) for Onset
        self.onset_map = {
            "": 0, "k": 1, "s": 2, "t": 3, "n": 4, "h": 5, "m": 6, "y": 7, 
            "r": 8, "w": 9, "g": 10, "z": 11, "d": 12, "b": 13, "p": 14, "special": 15
        }
        self.rev_onset_map = {v: k for k, v in self.onset_map.items()}
        
        # 3 bits (0-7) for Nucleus
        self.nucleus_map = {
            "a": 0, "i": 1, "u": 2, "e": 3, "o": 4, "n": 5, "sokuon": 6, "special": 7
        }
        self.rev_nucleus_map = {v: k for k, v in self.nucleus_map.items()}

    def _syllabify_romaji(self, romaji: str) -> List[Tuple[str, str, str]]:
        """
        Decomposes romaji text into individual structured morae (Onset, Glide, Nucleus).
        Handles Japanese phonotactics cleanly.
        """
        # Clean double vowels (aa, ii, uu, ee, oo) to simple long vowels or separate morae
        romaji = romaji.lower()
        
        # RegEx to capture Consonants (including blends like sh, ch, ts, ry, ky, gy, etc.)
        # Groups: 1=Consonant, 2=Optional 'y' glide, 3=Vowel, 4=Nasal Coda, 5=Double Consonant (Sokuon)
        pattern = r"([kshntmyrwgzdbp]{1,2})(y?)([aiueo])|([aiueo])|([n](?![aiueo]))|((.)\7)"
        matches = re.finditer(pattern, romaji)
        
        morae = []
        for m in matches:
            if m.group(1) is not None and m.group(3) is not None:
                onset = m.group(1)
                glide = m.group(2)
                vowel = m.group(3)
                
                # Standardize onset to base consonant for mapping
                # e.g., 'sh' -> 's', 'ch' -> 't', 'ts' -> 't', 'ry' -> 'r'
                base_onset = onset[0]
                if onset in ["sh", "ch", "ts"]:
                    base_onset = "s" if onset == "sh" else "t"
                if len(onset) > 1 and onset[1] == 'y':
                    base_onset = onset[0]
                    glide = "y"
                    
                morae.append((base_onset, glide, vowel))
                
            elif m.group(4) is not None:  # Pure vowel mora (a, i, u, e, o)
                morae.append(("", "", m.group(4)))
                
            elif m.group(5) is not None:  # Nasal Coda 'n'
                morae.append(("", "", "n"))
                
            elif m.group(6) is not None:  # Double Consonant (Sokuon)
                morae.append(("", "", "sokuon"))
                
        return morae

    def encode(self, text: str, return_tensors: str = "list") -> Tuple[Union[List[int], torch.Tensor], Union[List[int], torch.Tensor]]:
        result = self.kks.convert(text)
        romaji_chunks = []
        for item in result:
            # Reconstruct with spaces representing word boundaries
            romaji_chunks.append(item['hepburn'])
        
        masks = []
        ids = []
        
        for chunk in romaji_chunks:
            morae = self._syllabify_romaji(chunk)
            for onset, glide, nucleus in morae:
                onset_val = self.onset_map.get(onset, 15)
                glide_val = 1 if glide == "y" else 0
                nucleus_val = self.nucleus_map.get(nucleus, 7)
                
                # Register Layout: Onset (4 bits) | Glide (1 bit) | Nucleus (3 bits)
                packed_byte = (onset_val << 4) | (glide_val << 3) | nucleus_val
                
                masks.append(packed_byte)
                ids.append(packed_byte)
            
            # Add a boundary/space marker between words if it isn't the last chunk
            if len(morae) > 0 and chunk != romaji_chunks[-1]:
                # Special separator byte: [Onset=15, Glide=0, Nucleus=7] -> 0xF7 (247)
                space_byte = (15 << 4) | (0 << 3) | 7
                masks.append(space_byte)
                ids.append(space_byte)
                
        return self._to_tensor(masks, return_tensors), self._to_tensor(ids, return_tensors)

    def decode(self, ids: List[int]) -> str:
        reconstructed = []
        
        for packed_byte in ids:
            onset_val = (packed_byte >> 4) & 0x0F
            glide_val = (packed_byte >> 3) & 0x01
            nucleus_val = packed_byte & 0x07
            
            # If it is our word boundary marker (247), skip adding a space 
            # to preserve standard, non-spaced Japanese reading flow!
            if onset_val == 15 and nucleus_val == 7:
                continue
                
            onset = self.rev_onset_map.get(onset_val, "")
            glide = "y" if glide_val == 1 else ""
            nucleus = self.rev_nucleus_map.get(nucleus_val, "")
            
            if onset == "special":
                continue
                
            if nucleus in ["n", "sokuon"]:
                char = "ん" if nucleus == "n" else "っ"
                reconstructed.append(char)
            elif nucleus == "special":
                continue
            else:
                reconstructed.append(f"{onset}{glide}{nucleus}")
                
        return "".join(reconstructed)