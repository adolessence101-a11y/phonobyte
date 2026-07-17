import re
from src.tokenizers.jp_tokenizer import JapanesePhonobyteTokenizer

class UnifiedPVBForge:
    def __init__(self):
        self.vowels = set('aeiouyAEIOUY')
        self.syllable_to_id = {}
        self.next_id = 256
        self.jp_tokenizer = JapanesePhonobyteTokenizer()  # Inject our zero-vocab engine
        
        self.ESCAPE_MATRIX = {
            '.': 0x01, ',': 0x02, '?': 0x03, '!': 0x04, ';': 0x05, ':': 0x06, '\n': 0x07,
            '<': 0x08, '>': 0x09, '/': 0x0A, '=': 0x0B, '"': 0x0C, '[': 0x0D, ']': 0x0E,
            '{': 0x0F, '}': 0x11, '#': 0x12, '-': 0x13, '_': 0x14, '(': 0x15, ')': 0x16,
            "'": 0x17, ' ': 0x00,
            '[CAP_NEXT]': 0x42, '[CAP_LOCK]': 0x43,
            '[SUFFIX_S]': 0x44, '[SUFFIX_ED]': 0x45, '[SUFFIX_ING]': 0x46,
            '[BOLD_NEXT]': 0x47, '[ITALIC_NEXT]': 0x48, '[ASCII_LITERAL]': 0x39
        }

    def is_japanese(self, text: str) -> bool:
        # Regex to detect Japanese Kanji, Hiragana, and Katakana blocks
        return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))

    def detect_suffixes(self, word):
        lower_word = word.lower()
        if lower_word.endswith('ing') and len(lower_word) > 4:
            return lower_word[:-3], '[SUFFIX_ING]'
        elif lower_word.endswith('ed') and len(lower_word) > 3:
            return lower_word[:-2], '[SUFFIX_ED]'
        elif lower_word.endswith('s') and len(lower_word) > 2 and not lower_word.endswith('ss'):
            return lower_word[:-1], '[SUFFIX_S]'
        return lower_word, None

    def encode_escape_state(self, state_key):
        hex_val = self.ESCAPE_MATRIX[state_key]
        return "111" + format(hex_val, '07b')

    def compress_arbitrary_stream(self, text):
        bitstream = ""
        
        tokens = re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+|\*\*|\*|[\w']+|[.,!?;:\n<>/=\"\[\]{}#\-_()*]| ", text)
        
        for idx, token in enumerate(tokens):

            if token == ' ':
                prev_is_jp = idx > 0 and self.is_japanese(tokens[idx - 1])
                next_is_jp = idx < len(tokens) - 1 and self.is_japanese(tokens[idx + 1])
                if prev_is_jp or next_is_jp:
                    continue  # Safely discard formatting spaces within native Japanese text segments
            
            # 1. Route Native Japanese Text Route
            if self.is_japanese(token):
                packed_bytes, _ = self.jp_tokenizer.encode(token)
                for byte in packed_bytes:
                    bitstream += "110" + format(byte, '08b')
                continue

            # 2. Standard English / Markdown Route
            if token == '**':
                bitstream += self.encode_escape_state('[BOLD_NEXT]')
                continue
            if token == '*':
                bitstream += self.encode_escape_state('[ITALIC_NEXT]')
                continue

            if token in self.ESCAPE_MATRIX:
                bitstream += self.encode_escape_state(token)
                continue
                
            if not token.isalnum() and token not in self.ESCAPE_MATRIX:
                bitstream += self.encode_escape_state('[ASCII_LITERAL]')
                bitstream += format(ord(token[0]), '08b')
                continue

            if token.istitle():
                bitstream += self.encode_escape_state('[CAP_NEXT]')
            elif token.isupper() and len(token) > 1:
                bitstream += self.encode_escape_state('[CAP_LOCK]')

            base_word, suffix_flag = self.detect_suffixes(token)
            
            if base_word not in self.syllable_to_id:
                self.syllable_to_id[base_word] = self.next_id
                self.next_id += 1
                
            vowel_count = sum(1 for char in base_word if char in self.vowels)
            consonant_count = len(base_word) - vowel_count
            
            if consonant_count <= 2: header, payload_size = "0", 8
            elif consonant_count <= 3: header, payload_size = "10", 12
            else: header, payload_size = "11", 16
                
            bitstream += header + format(self.syllable_to_id[base_word], f'0{payload_size}b')
            
            if suffix_flag:
                bitstream += self.encode_escape_state(suffix_flag)
                
        return bitstream

    def evaluate_efficiency(self, raw_text, bitstream):
        # Japanese/Unicode text uses 24 bits minimum per character in raw standard text space
        # We calculate the real byte array footprint length to find accurate savings
        raw_bytes_bits = len(raw_text.encode('utf-8')) * 8
        pvb_bits = len(bitstream)
        ratio = (1 - (pvb_bits / raw_bytes_bits)) * 100
        return ratio
