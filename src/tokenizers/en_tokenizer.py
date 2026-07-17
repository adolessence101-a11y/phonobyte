import re
import torch
from src.tokenizers.base import BasePhonobyteTokenizer

VOWELS = set('aeiouy')
DIGRAPHS = {'th', 'sh', 'ch', 'ph', 'wh', 'ng', 'ck', 'qu'}

SPACE_BEACON = 0x00

PUNCTUATION_MATRIX = {
    '.': 0x01, ',': 0x02, '?': 0x03, '!': 0x04,
    ';': 0x05, ':': 0x06, '\n': 0x07
}

FORMATTING_MATRIX = {
    '<': 0x08, '>': 0x09, '/': 0x0A, '=': 0x0B,
    '"': 0x0C, '[': 0x0D, ']': 0x0E, '{': 0x0F,
    '}': 0x11, '#': 0x12, '-': 0x13, '_': 0x14,
    '(': 0x15, ')': 0x16, "'": 0x17
}

TACTICAL_RESERVE_MATRIX = {
    '[CAP_NEXT]': 0x42,
    '[CAP_LOCK]': 0x43,
    '[SUFFIX_S]': 0x44,
    '[SUFFIX_ED]': 0x45,
    '[SUFFIX_ING]': 0x46,
    '[BOLD_NEXT]': 0x47,
    '[ITALIC_NEXT]': 0x48,
    '[ASCII_LITERAL]': 0x39
}

ALL_MATRICES = [
    PUNCTUATION_MATRIX, FORMATTING_MATRIX, TACTICAL_RESERVE_MATRIX
]

class PhonobyteTokenizer(BasePhonobyteTokenizer):
    def __init__(self):
        super().__init__()
        self.syllable_to_id = {}
        self.id_to_syllable = {}
        self.next_syllable_id = 256

    def is_vowel(self, phoneme):
        return all(char.lower() in VOWELS for char in phoneme)

    def chunk_phonemes(self, word):
        phonemes = []
        i = 0
        while i < len(word):
            if i < len(word) - 1:
                pair = word[i:i+2]
                if pair.lower() in DIGRAPHS or (pair[0].lower() in VOWELS and pair[1].lower() in VOWELS):
                    phonemes.append(pair)
                    i += 2
                    continue
            phonemes.append(word[i])
            i += 1
        return phonemes

    def syllabify(self, phonemes):
        syllables = []
        current_syllable = []
        i = 0
        while i < len(phonemes):
            current_syllable.append(phonemes[i])
            if self.is_vowel(phonemes[i]) and i + 1 < len(phonemes):
                if i + 2 < len(phonemes) and not self.is_vowel(phonemes[i+1]) and self.is_vowel(phonemes[i+2]):
                    syllables.append(current_syllable)
                    current_syllable = []
                elif i + 3 < len(phonemes) and not self.is_vowel(phonemes[i+1]) and not self.is_vowel(phonemes[i+2]) and self.is_vowel(phonemes[i+3]):
                    current_syllable.append(phonemes[i+1])
                    syllables.append(current_syllable)
                    current_syllable = []
                    i += 1
            i += 1
        if current_syllable:
            syllables.append(current_syllable)
        return syllables

    def build_8bit_envelope(self, syllable_phonemes):
        nucleus_idx = -1
        for i, p in enumerate(syllable_phonemes):
            if self.is_vowel(p):
                nucleus_idx = i
                break
        if nucleus_idx == -1: return 0x00
            
        onset_count = min(nucleus_idx, 3)
        coda_count = min(len(syllable_phonemes) - nucleus_idx - 1, 4)
        
        byte_val = 0x10 
        if onset_count >= 1: byte_val |= 0x20
        if onset_count >= 2: byte_val |= 0x40
        if onset_count >= 3: byte_val |= 0x80
        if coda_count >= 1: byte_val |= 0x08
        if coda_count >= 2: byte_val |= 0x04
        if coda_count >= 3: byte_val |= 0x02
        if coda_count >= 4: byte_val |= 0x01
        return byte_val

    def detect_suffixes(self, word):
        lower_word = word.lower()
        if lower_word.endswith('ing') and len(lower_word) > 4:
            return word[:-3], TACTICAL_RESERVE_MATRIX['[SUFFIX_ING]']
        elif lower_word.endswith('ed') and len(lower_word) > 3:
            return word[:-2], TACTICAL_RESERVE_MATRIX['[SUFFIX_ED]']
        elif lower_word.endswith('s') and len(lower_word) > 2 and not lower_word.endswith('ss'):
            return word[:-1], TACTICAL_RESERVE_MATRIX['[SUFFIX_S]']
        return word, None

    def push_zero_core_state(self, val, stream_masks, stream_ids):
        stream_masks.append(val)
        stream_ids.append(val)

    def encode(self, text, return_tensors="pt", device="cpu"):
        tokens = re.findall(r"\*\*|\*|[\w']+|[.,!?;:\n<>/=\"\[\]{}#\-_()+]| ", text)
        stream_masks = []
        stream_ids = []
        
        for token in tokens:
            if token == ' ':
                self.push_zero_core_state(SPACE_BEACON, stream_masks, stream_ids)
                continue
                
            if token == '**':
                self.push_zero_core_state(TACTICAL_RESERVE_MATRIX['[BOLD_NEXT]'], stream_masks, stream_ids)
                continue
            if token == '*':
                self.push_zero_core_state(TACTICAL_RESERVE_MATRIX['[ITALIC_NEXT]'], stream_masks, stream_ids)
                continue

            is_operator = False
            for matrix in ALL_MATRICES:
                if token in matrix:
                    self.push_zero_core_state(matrix[token], stream_masks, stream_ids)
                    is_operator = True
                    break
            if is_operator: continue

            if not token.isalnum():
                self.push_zero_core_state(TACTICAL_RESERVE_MATRIX['[ASCII_LITERAL]'], stream_masks, stream_ids)
                stream_masks.append(0x00)
                stream_ids.append(ord(token[0]))
                continue

            if token.istitle():
                self.push_zero_core_state(TACTICAL_RESERVE_MATRIX['[CAP_NEXT]'], stream_masks, stream_ids)
            elif token.isupper() and len(token) > 1:
                self.push_zero_core_state(TACTICAL_RESERVE_MATRIX['[CAP_LOCK]'], stream_masks, stream_ids)

            base_word, suffix_val = self.detect_suffixes(token)
            
            phonemes = self.chunk_phonemes(base_word.lower())
            syllables = self.syllabify(phonemes)
            
            for syl in syllables:
                mask = self.build_8bit_envelope(syl)
                syl_text = "".join(syl)
                
                if syl_text not in self.syllable_to_id:
                    self.syllable_to_id[syl_text] = self.next_syllable_id
                    self.id_to_syllable[self.next_syllable_id] = syl_text
                    self.next_syllable_id += 1
                    
                stream_masks.append(mask)
                stream_ids.append(self.syllable_to_id[syl_text])

            if suffix_val:
                self.push_zero_core_state(suffix_val, stream_masks, stream_ids)
                
        if return_tensors == "pt":
            return (torch.tensor(stream_masks, dtype=torch.long, device=device),
                    torch.tensor(stream_ids, dtype=torch.long, device=device))
        return stream_masks, stream_ids

    def decode(self, stream_ids):
        if torch.is_tensor(stream_ids):
            stream_ids = stream_ids.tolist()
            
        reconstructed_text = ""
        cap_next = False
        cap_lock = False
        bold_active = False
        italic_active = False
        ascii_literal_next = False
        
        for lex_id in stream_ids:
            if ascii_literal_next:
                reconstructed_text += chr(lex_id)
                ascii_literal_next = False
                continue

            if lex_id == SPACE_BEACON:
                reconstructed_text += " "
                continue
                
            if lex_id == TACTICAL_RESERVE_MATRIX['[ASCII_LITERAL]']:
                ascii_literal_next = True
                continue
            if lex_id == TACTICAL_RESERVE_MATRIX['[CAP_NEXT]']:
                cap_next = True
                continue
            if lex_id == TACTICAL_RESERVE_MATRIX['[CAP_LOCK]']:
                cap_lock = not cap_lock
                continue
            if lex_id == TACTICAL_RESERVE_MATRIX['[BOLD_NEXT]']:
                bold_active = not bold_active
                reconstructed_text += "**"
                continue
            if lex_id == TACTICAL_RESERVE_MATRIX['[ITALIC_NEXT]']:
                italic_active = not italic_active
                reconstructed_text += "*"
                continue
                
            if lex_id == TACTICAL_RESERVE_MATRIX['[SUFFIX_S]']:
                reconstructed_text += "s"
                continue
            if lex_id == TACTICAL_RESERVE_MATRIX['[SUFFIX_ED]']:
                reconstructed_text += "ed"
                continue
            if lex_id == TACTICAL_RESERVE_MATRIX['[SUFFIX_ING]']:
                reconstructed_text += "ing"
                continue

            if lex_id in self.id_to_syllable:
                text_chunk = self.id_to_syllable[lex_id]
                
                if cap_lock: text_chunk = text_chunk.upper()
                elif cap_next:
                    text_chunk = text_chunk.capitalize()
                    cap_next = False
                    
                reconstructed_text += text_chunk
                continue
            
            found = False
            for matrix in ALL_MATRICES:
                for char, val in matrix.items():
                    if val == lex_id:
                        if matrix != TACTICAL_RESERVE_MATRIX:
                            reconstructed_text += char
                        found = True
                        break
                if found: break

        return reconstructed_text
