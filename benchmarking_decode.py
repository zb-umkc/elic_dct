import numpy as np
import time
from collections import Counter
import h5py

# Import encoding functions from the original files
from benchmarking import (
    rle_encode_original, exp_golomb_encode_original, exp_golomb_encode_lookup,
    exp_golomb_encode_unsigned_lookup, create_sample_sar_data, rle_encode_v1
)

# ========== ORIGINAL DECODING FUNCTION ==========

def exp_golomb_decode_original(bitstream):
    """Original implementation from rle.py"""
    arr_decoded = []
    i = 0
    while i < len(bitstream):
        # Count leading zeros
        m = 0
        while i < len(bitstream) and bitstream[i] == "0":
            m += 1
            i += 1
        # Read next m bits
        val_bits = bitstream[i:i+m+1]
        i += m + 1
        try:
            xp = int(val_bits, 2) - 1
        except:
            print(f"Error at position {i}, val_bits: {val_bits}")
            break
        if (xp % 2) == 0:
            x = int(-0.5*xp)
        else:
            x = int((xp + 1) / 2)

        arr_decoded.append(x)

    return arr_decoded


# ========== OPTIMIZED DECODING IMPLEMENTATIONS ==========

def exp_golomb_decode_optimized_v1(bitstream):
    """
    Optimized version 1: Reduce string operations and use list pre-allocation.
    """
    if not bitstream:
        return []
    
    arr_decoded = []
    i = 0
    bitstream_len = len(bitstream)
    
    while i < bitstream_len:
        # Count leading zeros - optimized loop
        m = 0
        while i + m < bitstream_len and bitstream[i + m] == '0':
            m += 1
        
        # Skip the leading zeros and the '1' bit
        start_pos = i + m
        end_pos = start_pos + m + 1
        
        if end_pos > bitstream_len:
            break
            
        # Read the binary value directly
        val_bits = bitstream[start_pos:end_pos]
        
        if val_bits and val_bits[0] == '1':  # Valid code must start with 1
            xp = int(val_bits, 2) - 1
            
            # Optimized signed/unsigned conversion
            if xp & 1:  # Odd numbers (faster than modulo)
                x = (xp + 1) >> 1  # Faster division by 2
            else:
                x = -(xp >> 1)  # Faster division by 2
            
            arr_decoded.append(x)
        
        i = end_pos

    return arr_decoded


def exp_golomb_decode_optimized_v2(bitstream):
    """
    Optimized version 2: Use bit manipulation and minimize string slicing.
    """
    if not bitstream:
        return []
    
    arr_decoded = []
    i = 0
    bitstream_len = len(bitstream)
    
    # Pre-compute powers of 2 for common cases (up to 16 bits)
    powers_of_2 = [1 << k for k in range(17)]  # 2^0 to 2^16
    
    while i < bitstream_len:
        # Count leading zeros
        m = 0
        while i + m < bitstream_len and bitstream[i + m] == '0':
            m += 1
        
        # Calculate positions
        start_pos = i + m
        end_pos = start_pos + m + 1
        
        if end_pos > bitstream_len or start_pos >= bitstream_len:
            break
            
        # For small codes, use lookup table approach
        if m < len(powers_of_2):
            # Build integer value bit by bit (faster for small values)
            val = 0
            for bit_pos in range(start_pos, end_pos):
                if bit_pos < bitstream_len and bitstream[bit_pos] == '1':
                    val += powers_of_2[end_pos - 1 - bit_pos]
            
            if val > 0:
                xp = val - 1
                # Fast signed conversion
                x = (xp + 1) >> 1 if xp & 1 else -(xp >> 1)
                arr_decoded.append(x)
        else:
            # Fallback to string method for very large codes
            val_bits = bitstream[start_pos:end_pos]
            if val_bits and val_bits[0] == '1':
                xp = int(val_bits, 2) - 1
                x = (xp + 1) >> 1 if xp & 1 else -(xp >> 1)
                arr_decoded.append(x)
        
        i = end_pos

    return arr_decoded


def exp_golomb_decode_lookup_table(bitstream):
    """
    Optimized version 3: Use lookup table for common codes.
    """
    if not bitstream:
        return []
    
    # Pre-computed lookup table for common Exp-Golomb codes
    # Maps bit patterns to decoded values
    lookup_table = {
        '1': 0,           # 1 bit
        '010': 1,         # 3 bits
        '011': -1,        # 3 bits
        '00100': 2,       # 5 bits
        '00101': -2,      # 5 bits
        '00110': 3,       # 5 bits
        '00111': -3,      # 5 bits
        '0001000': 4,     # 7 bits
        '0001001': -4,    # 7 bits
        '0001010': 5,     # 7 bits
        '0001011': -5,    # 7 bits
        '0001100': 6,     # 7 bits
        '0001101': -6,    # 7 bits
        '0001110': 7,     # 7 bits
        '0001111': -7,    # 7 bits
        '000010000': 8,   # 9 bits
        '000010001': -8,  # 9 bits
        '000010010': 9,   # 9 bits
        '000010011': -9,  # 9 bits
        '000010100': 10,  # 9 bits
        '000010101': -10, # 9 bits
    }
    
    # Sort lookup keys by length for greedy matching
    sorted_keys = sorted(lookup_table.keys(), key=len, reverse=True)
    
    arr_decoded = []
    i = 0
    bitstream_len = len(bitstream)
    
    while i < bitstream_len:
        found = False
        
        # Try lookup table first (for common values)
        for pattern in sorted_keys:
            pattern_len = len(pattern)
            if i + pattern_len <= bitstream_len:
                if bitstream[i:i+pattern_len] == pattern:
                    arr_decoded.append(lookup_table[pattern])
                    i += pattern_len
                    found = True
                    break
        
        if not found:
            # Fall back to standard algorithm for uncommon values
            m = 0
            while i + m < bitstream_len and bitstream[i + m] == '0':
                m += 1
            
            start_pos = i + m
            end_pos = start_pos + m + 1
            
            if end_pos > bitstream_len:
                break
                
            val_bits = bitstream[start_pos:end_pos]
            if val_bits and val_bits[0] == '1':
                xp = int(val_bits, 2) - 1
                x = (xp + 1) >> 1 if xp & 1 else -(xp >> 1)
                arr_decoded.append(x)
            
            i = end_pos

    return arr_decoded


def exp_golomb_decode_unsigned(bitstream):
    """
    Optimized decoder for unsigned Exp-Golomb codes.
    This is for decoding lengths arrays that were encoded with unsigned encoding.
    """
    if not bitstream:
        return []
    
    arr_decoded = []
    i = 0
    bitstream_len = len(bitstream)
    
    while i < bitstream_len:
        # Count leading zeros
        m = 0
        while i + m < bitstream_len and bitstream[i + m] == '0':
            m += 1
        
        # Calculate positions
        start_pos = i + m
        end_pos = start_pos + m + 1
        
        if end_pos > bitstream_len:
            break
            
        # Read the binary value
        val_bits = bitstream[start_pos:end_pos]
        
        if val_bits and val_bits[0] == '1':
            # For unsigned: the decoded value IS the binary value
            # No signed-to-unsigned conversion needed!
            x = int(val_bits, 2)
            arr_decoded.append(x)
        
        i = end_pos

    return arr_decoded


def exp_golomb_decode_unsigned_lookup(bitstream):
    """
    Unsigned decoder with lookup table for common small positive values.
    """
    if not bitstream:
        return []
    
    # Lookup table for unsigned values (lengths are typically small)
    unsigned_lookup = {
        '1': 1,           # 1
        '010': 2,         # 2
        '011': 3,         # 3
        '00100': 4,       # 4
        '00101': 5,       # 5
        '00110': 6,       # 6
        '00111': 7,       # 7
        '0001000': 8,     # 8
        '0001001': 9,     # 9
        '0001010': 10,    # 10
        '0001011': 11,    # 11
        '0001100': 12,    # 12
        '0001101': 13,    # 13
        '0001110': 14,    # 14
        '0001111': 15,    # 15
        '000010000': 16,  # 16
    }
    
    sorted_keys = sorted(unsigned_lookup.keys(), key=len, reverse=True)
    
    arr_decoded = []
    i = 0
    bitstream_len = len(bitstream)
    
    while i < bitstream_len:
        found = False
        
        # Try lookup table first
        for pattern in sorted_keys:
            pattern_len = len(pattern)
            if i + pattern_len <= bitstream_len:
                if bitstream[i:i+pattern_len] == pattern:
                    arr_decoded.append(unsigned_lookup[pattern])
                    i += pattern_len
                    found = True
                    break
        
        if not found:
            # Standard unsigned decoding for larger values
            m = 0
            while i + m < bitstream_len and bitstream[i + m] == '0':
                m += 1
            
            start_pos = i + m
            end_pos = start_pos + m + 1
            
            if end_pos > bitstream_len:
                break
                
            val_bits = bitstream[start_pos:end_pos]
            if val_bits and val_bits[0] == '1':
                x = int(val_bits, 2)  # No conversion needed for unsigned
                arr_decoded.append(x)
            
            i = end_pos

    return arr_decoded


def exp_golomb_decode_vectorized_attempt(bitstream):
    """
    Attempt at vectorizing some operations (limited success expected with strings).
    """
    if not bitstream:
        return []
    
    # This is challenging because bitstreams don't have fixed-length codes
    # But we can optimize the inner loops
    
    arr_decoded = []
    i = 0
    bitstream_len = len(bitstream)
    
    # Convert string to numpy array for potential speedup
    bit_array = np.array([int(b) for b in bitstream], dtype=np.uint8)
    
    while i < bitstream_len:
        # Count leading zeros using numpy
        start_idx = i
        while start_idx < bitstream_len and bit_array[start_idx] == 0:
            start_idx += 1
        
        m = start_idx - i
        
        # Calculate positions
        end_pos = start_idx + m + 1
        
        if end_pos > bitstream_len:
            break
        
        # Extract bits and convert
        if start_idx < bitstream_len and bit_array[start_idx] == 1:
            val_bits = bit_array[start_idx:end_pos]
            
            # Convert binary array to integer
            x_val = 0
            for bit in val_bits:
                x_val = (x_val << 1) + bit
            
            xp = x_val - 1
            x = (xp + 1) >> 1 if xp & 1 else -(xp >> 1)
            arr_decoded.append(x)
        
        i = end_pos

    return arr_decoded


# ========== BENCHMARKING FRAMEWORK ==========

def benchmark_decode_implementations(encoded_vals, encoded_lens, original_vals, original_lens, iterations=20):
    """Benchmark all decoding implementations."""
    print(f"\n=== EXP-GOLOMB DECODE BENCHMARK ({iterations} iterations) ===")
    print(f"Encoded values: {len(encoded_vals)} characters")
    print(f"Encoded lengths: {len(encoded_lens)} characters")
    print(f"Total characters to decode: {len(encoded_vals) + len(encoded_lens)}")
    
    implementations = [
        ("Original", exp_golomb_decode_original),
        ("Optimized v1", exp_golomb_decode_optimized_v1),
        ("Optimized v2", exp_golomb_decode_optimized_v2),
        ("Lookup Table", exp_golomb_decode_lookup_table),
        ("Vectorized Attempt", exp_golomb_decode_vectorized_attempt),
    ]
    
    results = {}
    
    for name, func in implementations:
        try:
            # Warmup and verify correctness
            vals_decoded = func(encoded_vals)
            lens_decoded = func(encoded_lens)
            
            # Check correctness
            vals_correct = np.array_equal(vals_decoded, original_vals)
            lens_correct = np.array_equal(lens_decoded, original_lens)
            
            if not (vals_correct and lens_correct):
                print(f"{name:20}: ❌ INCORRECT RESULTS")
                if not vals_correct:
                    print(f"   Values mismatch: expected {len(original_vals)}, got {len(vals_decoded)}")
                if not lens_correct:
                    print(f"   Lengths mismatch: expected {len(original_lens)}, got {len(lens_decoded)}")
                continue
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                vals_decoded = func(encoded_vals)
                lens_decoded = func(encoded_lens)
            elapsed = (time.time() - start_time) / iterations
            
            results[name] = elapsed
            print(f"{name:20}: {elapsed*1000:.3f} ms (both vals + lens)")
            
        except Exception as e:
            print(f"{name:20}: FAILED ({str(e)})")
    
    # Find best
    if results:
        best_name = min(results.keys(), key=lambda k: results[k])
        best_time = results[best_name]
        print(f"\n🏆 Best decoder: {best_name} ({best_time*1000:.3f} ms)")
        
        for name, time_val in results.items():
            if name != best_name:
                speedup = time_val / best_time
                print(f"   {name} is {speedup:.1f}x slower")
        
        return best_name
    
    return None


def benchmark_unsigned_decode(encoded_lens_unsigned, original_lens, iterations=50):
    """Benchmark unsigned decoding specifically for lengths."""
    print(f"\n=== UNSIGNED DECODE BENCHMARK ===")
    print(f"Encoded lengths (unsigned): {len(encoded_lens_unsigned)} characters")
    
    implementations = [
        ("Signed Original", exp_golomb_decode_original),
        ("Unsigned Basic", exp_golomb_decode_unsigned),
        ("Unsigned Lookup", exp_golomb_decode_unsigned_lookup),
    ]
    
    results = {}
    
    for name, func in implementations:
        try:
            # Test correctness
            decoded = func(encoded_lens_unsigned)
            correct = np.array_equal(decoded, original_lens)
            
            if not correct:
                # For unsigned vs signed, the values might be encoded differently
                # so we only test unsigned decoders on unsigned-encoded data
                if "Signed" in name:
                    print(f"{name:20}: ⚠️  SKIPPED (signed decoder on unsigned data)")
                    continue
                else:
                    print(f"{name:20}: ❌ INCORRECT (expected {len(original_lens)}, got {len(decoded)})")
                    continue
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                decoded = func(encoded_lens_unsigned)
            elapsed = (time.time() - start_time) / iterations
            
            results[name] = elapsed
            print(f"{name:20}: {elapsed*1000:.3f} ms")
            
        except Exception as e:
            print(f"{name:20}: FAILED ({str(e)})")
    
    # Show improvements
    if "Signed Original" in results and "Unsigned Lookup" in results:
        speedup = results["Signed Original"] / results["Unsigned Lookup"]
        print(f"\n💡 Unsigned lookup is {speedup:.1f}x faster than signed original for lengths!")
    
    return results


def benchmark_complete_pipeline_with_decode(arr, iterations=10):
    """Benchmark complete encode + decode pipeline."""
    print(f"\n=== COMPLETE ENCODE + DECODE PIPELINE ===")
    
    flat_arr = arr.flatten()
    
    # Test different encoding/decoding combinations
    pipelines = [
        ("Original encode/decode", 
         exp_golomb_encode_original, exp_golomb_decode_original,
         exp_golomb_encode_original, exp_golomb_decode_original),
        ("Lookup encode, optimized decode",
         exp_golomb_encode_lookup, exp_golomb_decode_lookup_table,
         exp_golomb_encode_lookup, exp_golomb_decode_lookup_table),
        ("Hybrid encode, mixed decode",
         exp_golomb_encode_lookup, exp_golomb_decode_lookup_table,
         exp_golomb_encode_unsigned_lookup, exp_golomb_decode_unsigned_lookup),
    ]
    
    for name, vals_enc, vals_dec, lens_enc, lens_dec in pipelines:
        try:
            # Warmup and verify
            vals, lens = rle_encode_original(flat_arr)
            vals_encoded = vals_enc(vals)
            lens_encoded = lens_enc(lens)
            vals_decoded = vals_dec(vals_encoded)
            lens_decoded = lens_dec(lens_encoded)
            reconstructed = np.repeat(vals_decoded, lens_decoded)
            
            if not np.array_equal(reconstructed, flat_arr):
                print(f"{name}: ❌ PIPELINE FAILED")
                continue
            
            # Benchmark complete pipeline
            start_time = time.time()
            for _ in range(iterations):
                vals, lens = rle_encode_original(flat_arr)
                vals_encoded = vals_enc(vals)
                lens_encoded = lens_enc(lens)
                vals_decoded = vals_dec(vals_encoded)
                lens_decoded = lens_dec(lens_encoded)
                reconstructed = np.repeat(vals_decoded, lens_decoded)
            elapsed = (time.time() - start_time) / iterations
            
            print(f"{name:30}: {elapsed*1000:.3f} ms")
            
        except Exception as e:
            print(f"{name:30}: FAILED ({str(e)})")


def analyze_decode_characteristics(encoded_vals, encoded_lens):
    """Analyze characteristics that affect decoding performance."""
    print(f"\n=== DECODE CHARACTERISTICS ANALYSIS ===")
    
    print(f"Encoded values string: {len(encoded_vals)} characters")
    print(f"Encoded lengths string: {len(encoded_lens)} characters")
    
    # Analyze bit patterns
    vals_zeros = encoded_vals.count('0')
    vals_ones = encoded_vals.count('1')
    lens_zeros = encoded_lens.count('0')
    lens_ones = encoded_lens.count('1')
    
    print(f"\nBit distribution:")
    print(f"Values - Zeros: {vals_zeros}, Ones: {vals_ones} ({vals_ones/(vals_zeros+vals_ones)*100:.1f}% ones)")
    print(f"Lengths - Zeros: {lens_zeros}, Ones: {lens_ones} ({lens_ones/(lens_zeros+lens_ones)*100:.1f}% ones)")
    
    # Estimate average code length
    vals_decoded = exp_golomb_decode_original(encoded_vals)
    lens_decoded = exp_golomb_decode_original(encoded_lens)
    
    avg_vals_code_len = len(encoded_vals) / len(vals_decoded) if vals_decoded else 0
    avg_lens_code_len = len(encoded_lens) / len(lens_decoded) if lens_decoded else 0
    
    print(f"\nAverage code lengths:")
    print(f"Values: {avg_vals_code_len:.1f} bits/value")
    print(f"Lengths: {avg_lens_code_len:.1f} bits/value")
    
    return vals_decoded, lens_decoded


def main():
    """Main benchmarking function."""
    print("SAR Latent Features Compression Benchmark")
    print("=" * 50)
    
    # Ask user for data or use synthetic
    print("Options:")
    print("1. Load your SAR latent features data")
    print("2. Use synthetic SAR-like data for testing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        filepath="/scratch/zb7df/data/latents/latent_tensors.h5"
        with h5py.File(filepath, 'r') as f:
            arr = f['y_hat'][:1].flatten()
    else:
        print("\nGenerating synthetic SAR-like data...")
        arr = create_sample_sar_data((500, 500))
    
    # Encode using different methods
    vals, lens = rle_encode_v1(arr.flatten())
    
    print(f"RLE results: {len(vals)} values, {len(lens)} lengths")
    
    # Generate encoded data for testing
    encoded_vals_orig = exp_golomb_encode_original(vals)
    encoded_lens_orig = exp_golomb_encode_original(lens)
    encoded_vals_lookup = exp_golomb_encode_lookup(vals)
    encoded_lens_unsigned = exp_golomb_encode_unsigned_lookup(lens)
    
    # Analyze decode characteristics
    analyze_decode_characteristics(encoded_vals_orig, encoded_lens_orig)
    
    # Benchmark standard decoding
    benchmark_decode_implementations(
        encoded_vals_lookup, encoded_lens_orig, 
        vals, lens, iterations=20
    )
    
    # Benchmark unsigned decoding
    benchmark_unsigned_decode(encoded_lens_unsigned, lens, iterations=50)
    
    # Benchmark complete pipelines
    benchmark_complete_pipeline_with_decode(arr, iterations=10)
    
    print(f"\n🎯 DECODE OPTIMIZATION SUMMARY:")
    print(f"   Test the best performing decoder in your ELIC pipeline")
    print(f"   Consider using unsigned decoding for lengths arrays")
    print(f"   Lookup tables help significantly for common values")


if __name__ == "__main__":
    main()
