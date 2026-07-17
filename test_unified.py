from src.core.pvb_forge import UnifiedPVBForge

def run_unified_test():
    forge = UnifiedPVBForge()
    
    # A mixed paragraph containing markdown, English structure, and Japanese text
    mixed_text = "The developer creates a lean engine. 日本語のトークン化は非常に非効率的です。"
    
    print("--- Running Unified Cross-Lingual Compression ---")
    print(f"Raw Input: '{mixed_text}'")
    
    # Execute the bitstream compression
    bitstream = forge.compress_arbitrary_stream(mixed_text)
    
    # Evaluate efficiency
    raw_bits = len(mixed_text.encode('utf-8')) * 8
    pvb_bits = len(bitstream)
    savings = forge.evaluate_efficiency(mixed_text, bitstream)
    
    print("\n--- Bitstream Metrics ---")
    print(f"  * Raw UTF-8 Footprint: {raw_bits} bits ({raw_bits // 8} bytes)")
    print(f"  * Unified .pvb Footprint: {pvb_bits} bits ({pvb_bits // 8} bytes)")
    print(f"  * Data Conservation Ratio: {savings:.1f}% data reduction!")
    
    # Sanity check bit signatures
    print("\n--- Bitstream Peek (First 60 bits) ---")
    print(f"  {bitstream[:60]}...")

if __name__ == "__main__":
    run_unified_test()