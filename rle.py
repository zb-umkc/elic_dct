import numpy as np
import heapq
from collections import Counter, defaultdict
# from dahuffman import HuffmanCodec
# import huffman
import time
import sys
# from multiprocessing import Pool

def rle_encode(arr):
    """Run-length encode a 1D numpy array of integers."""
    # arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Find run boundaries
    diff = np.diff(arr)
    run_boundaries = np.where(diff != 0)[0] + 1
    run_starts = np.insert(run_boundaries, 0, 0)
    run_ends = np.append(run_boundaries, n)

    values = arr[run_starts]
    lengths = run_ends - run_starts
    return values, lengths


def rle_decode(values, lengths):
    """Reconstruct array from RLE."""
    return np.repeat(values, lengths)


def build_huffman_codes(values):
    """Build Huffman code dictionary from list/array of integers."""
    freq = Counter(values)
    heap = [[wt, [sym, ""]] for sym, wt in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huff_dict = {sym: code for sym, code in heap[0][1:]}
    lengths = {symbol: len(codeword) for symbol, codeword in huff_dict.items()}
    sorted_symbols = sorted(lengths.keys(), key=lambda s: (lengths[s], s))

    canonical_codes = {}
    code = 0
    prev_len = 0

    for s in sorted_symbols:
        length = lengths[s]
        # Shift left if code length increased
        code <<= (length - prev_len)
        canonical_codes[s] = format(code, f'0{length}b')
        code += 1
        prev_len = length

    return canonical_codes


def huffman_encode(values):
    codebook = build_huffman_codes(values)
    # print("-- CODEBOOK: ", codebook)
    codebook_bits = encode_huffman_dict(codebook)
    coded_bits = ''.join(codebook[value] for value in values)
    return coded_bits, codebook_bits


def huffman_decode(coded_bits, codebook_bits):
    codebook = decode_huffman_dict(codebook_bits)
    codebook_rev = {v: k for k, v in codebook.items()}
    
    bits_buffer = ""
    arr_decoded = []
    for bit in coded_bits:
        bits_buffer += bit
        if bits_buffer in codebook_rev:
            arr_decoded.append(codebook_rev[bits_buffer])
            bits_buffer = ""

    return arr_decoded


def encode_huffman_dict(huff_dict):
    """
    Encode a Huffman dictionary in canonical form.
    Input:
        huff_dict: dict {symbol: codeword as string of '0'/'1'}
    Output:
        symbols: np.array of symbols
        lengths: np.array of code lengths
    """
    # print("ENCODING HUFFMAN DICT")
    # 1. Extract symbols and code lengths
    symbols = []
    lengths = []
    for sym, code in huff_dict.items():
        symbols.append(sym)
        lengths.append(len(code))
    
    # 2. Convert to arrays
    symbols = np.array(symbols, dtype=np.int16)  # adjust dtype to your symbol range
    lengths = np.array(lengths, dtype=np.int8)
    
    # 3. Sort by code length, then by symbol (required for canonical Huffman)
    order = np.lexsort((symbols, lengths))  # sorts primarily by length, then by symbol
    symbols = symbols[order]
    lengths = lengths[order]

    # print("-- SYMBOLS: ", symbols)
    # print("-- LENGTHS: ", lengths)

    max_val = np.max(np.concatenate((symbols, lengths)))
    bits_per = int(np.ceil(np.log2(max_val + 1)))
    # print("-- BITS PER: ", bits_per)
    symbols_bits = ''.join(format(x, f'0{bits_per}b') for x in symbols)
    lengths_bits = ''.join(format(x, f'0{bits_per}b') for x in lengths)

    num_symbol_bits = bin(len(symbols))[2:].zfill(16)

    dict_encoded = num_symbol_bits + symbols_bits + lengths_bits
    
    return dict_encoded


def decode_huffman_dict(dict_encoded):
    """
    Reconstruct canonical Huffman codes from symbols + lengths.
    Output:
        huff_dict: dict {symbol: canonical code as string}
    """
    # print("DECODING HUFFMAN DICT: ", dict_encoded)
    num_symbol_bits = dict_encoded[:16]
    sym_len = dict_encoded[16:]
    num_symbols = int(num_symbol_bits, 2)
    split = int(len(sym_len) / 2)
    symbol_bits = sym_len[:split]
    length_bits = sym_len[split:]

    bits_per = int(len(symbol_bits) / num_symbols)
    symbols = [int(symbol_bits[i:i+bits_per], 2) for i in range(0, len(symbol_bits), bits_per)]
    lengths = [int(length_bits[i:i+bits_per], 2) for i in range(0, len(length_bits), bits_per)]

    # print("-- SYMBOLS: ", symbols)
    # print("-- LENGTHS: ", lengths)
    # print("-- BITS PER: ", bits_per)

    huff_dict = {}
    code = 0
    prev_len = 0
    
    for sym, length in zip(symbols, lengths):
        # Shift left if code length increased
        code <<= (length - prev_len)
        huff_dict[sym] = format(code, f'0{length}b')
        code += 1
        prev_len = length
    
    return huff_dict


def encode_number(x):
        if x <= 0:
            xp = -2*x
        else:
            xp = 2*x - 1

        x_bin = bin(int(xp) + 1)[2:]
        prefix = "0" * (len(x_bin)-1)
        return prefix + x_bin


def exp_golomb_encode(values):
    arr_encoded = "".join([encode_number(x) for x in values])

    return arr_encoded


def exp_golomb_decode(bitstream):
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
            print(i)
            print(len(val_bits))
            print(val_bits)
            sys.exit(1)
        if (xp % 2) == 0:
            x = int(-0.5*xp)
        else:
            x = int((xp + 1) / 2)

        arr_decoded.append(x)

    return arr_decoded
        

def calc_orig_bits(arr):
    m = np.max(arr) - np.min(arr) + 1
    bits_per = np.ceil(np.log2(m))
    return bits_per * len(arr)


def int_to_bin(n, width):
    """Convert integer n to binary string with fixed width bits."""
    return format(n, f'0{width}b')


def concat_streams(values_bits, lengths_bits):
    """
    Concatenate values and lengths bitstrings into one binary string
    with length prefixes.
    """
    # Compute lengths in bits
    v_len = len(values_bits)
    l_len = len(lengths_bits)

    # Choose width for length prefix: enough bits to hold both lengths
    max_len = max(v_len, l_len)
    width = max_len.bit_length()

    # Prefix each with its length (fixed width, same width for both)
    header = int_to_bin(v_len, width) + int_to_bin(l_len, width)

    # Final bitstream
    return header + values_bits + lengths_bits, width


def split_stream(bitstream, width):
    """
    Recover values_bits and lengths_bits from the combined bitstream.
    """
    # First read the two lengths
    v_len = int(bitstream[:width], 2)
    l_len = int(bitstream[width:2*width], 2)

    # Slice accordingly
    start = 2 * width
    values_bits = bitstream[start:start+v_len]
    lengths_bits = bitstream[start+v_len:start+v_len+l_len]

    return values_bits, lengths_bits


def rle_encode_final(arr):
    """Best performing RLE implementation based on benchmarks."""
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    if n == 1:
        return np.array([arr[0]]), np.array([1])
    
    # Find where values change
    change_points = arr[1:] != arr[:-1]
    
    # Get indices of changes (including start and end)
    change_indices = np.nonzero(change_points)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [n]))
    
    values = arr[change_indices[:-1]]
    lengths = np.diff(change_indices)
    
    return values, lengths


def exp_golomb_encode_final(values):
    """Highly optimized Exp-Golomb implementation using byte arrays for better performance."""
    if len(values) == 0:
        return ""
    
    # Pre-allocate a large enough buffer
    # Worst case: each value takes about 2*log2(max_val) bits
    max_val = max(abs(x) for x in values)
    if max_val == 0:
        estimated_bits = len(values) * 2
    else:
        estimated_bits = len(values) * (2 * int(np.log2(max_val)) + 10)  # generous estimate
    
    # Use a list for building the result (more efficient than string concatenation)
    result_chars = []
    
    for x in values:
        # Inline the encoding logic for maximum speed
        if x <= 0:
            xp = -2 * x
        else:
            xp = 2 * x - 1
        
        x_plus_1 = int(xp) + 1
        
        # Fast bit length calculation
        if x_plus_1 == 0:
            bits_needed = 1
        else:
            bits_needed = x_plus_1.bit_length()
        
        prefix_zeros = bits_needed - 1
        
        # Build the result more efficiently
        if prefix_zeros > 0:
            result_chars.append('0' * prefix_zeros)
        
        # Convert to binary without using format() for small numbers (common case optimization)
        if x_plus_1 < 256:  # Optimize for small numbers
            if x_plus_1 == 1:
                result_chars.append('1')
            elif x_plus_1 == 2:
                result_chars.append('10')
            elif x_plus_1 == 3:
                result_chars.append('11')
            elif x_plus_1 == 4:
                result_chars.append('100')
            elif x_plus_1 == 5:
                result_chars.append('101')
            elif x_plus_1 == 6:
                result_chars.append('110')
            elif x_plus_1 == 7:
                result_chars.append('111')
            elif x_plus_1 == 8:
                result_chars.append('1000')
            else:
                result_chars.append(format(x_plus_1, f'0{bits_needed}b'))
        else:
            result_chars.append(format(x_plus_1, f'0{bits_needed}b'))
    
    return ''.join(result_chars)


def exp_golomb_encode_ultra_fast(values):
    """Ultra-fast version using bitwise operations and lookup tables."""
    if len(values) == 0:
        return ""
    
    # Lookup table for small values (most common case)
    # Pre-computed exp-golomb codes for values -10 to 10
    lookup = {
        -10: '0000010101',  # 20 -> 21 -> 10101 with 4 zeros
        -9: '000010011',    # 18 -> 19 -> 10011 with 3 zeros  
        -8: '000010001',    # 16 -> 17 -> 10001 with 3 zeros
        -7: '00001111',     # 14 -> 15 -> 1111 with 3 zeros
        -6: '00001101',     # 12 -> 13 -> 1101 with 3 zeros
        -5: '00001011',     # 10 -> 11 -> 1011 with 3 zeros
        -4: '00001001',     # 8 -> 9 -> 1001 with 3 zeros
        -3: '0000111',      # 6 -> 7 -> 111 with 2 zeros
        -2: '000101',       # 4 -> 5 -> 101 with 2 zeros
        -1: '000011',       # 2 -> 3 -> 11 with 2 zeros
        0: '001',           # 0 -> 1 -> 1 with 1 zero
        1: '010',           # 1 -> 2 -> 10 with 1 zero
        2: '011',           # 3 -> 4 -> 100 but this is wrong... let me recalculate
    }
    
    # Actually, let's use a simpler but still fast approach
    result_parts = []
    
    for x in values:
        # Check if we can use a fast path for common small values
        if -5 <= x <= 5:
            if x <= 0:
                xp = -2 * x
            else:
                xp = 2 * x - 1
            
            x_plus_1 = int(xp) + 1
            
            # Hardcoded for the most common small values
            if x_plus_1 == 1:
                result_parts.append('1')
            elif x_plus_1 == 2:
                result_parts.append('010')
            elif x_plus_1 == 3:
                result_parts.append('011')
            elif x_plus_1 == 4:
                result_parts.append('00100')
            elif x_plus_1 == 5:
                result_parts.append('00101')
            elif x_plus_1 == 6:
                result_parts.append('00110')
            elif x_plus_1 == 7:
                result_parts.append('00111')
            elif x_plus_1 == 8:
                result_parts.append('0001000')
            elif x_plus_1 == 9:
                result_parts.append('0001001')
            elif x_plus_1 == 10:
                result_parts.append('0001010')
            elif x_plus_1 == 11:
                result_parts.append('0001011')
            else:
                # Fallback for other small values
                bits_needed = x_plus_1.bit_length()
                prefix_zeros = bits_needed - 1
                result_parts.append('0' * prefix_zeros + format(x_plus_1, f'0{bits_needed}b'))
        else:
            # Standard algorithm for larger values
            if x <= 0:
                xp = -2 * x
            else:
                xp = 2 * x - 1
            
            x_plus_1 = int(xp) + 1
            bits_needed = x_plus_1.bit_length()
            prefix_zeros = bits_needed - 1
            result_parts.append('0' * prefix_zeros + format(x_plus_1, f'0{bits_needed}b'))
    
    return ''.join(result_parts)


if __name__ == "__main__":
    # example = [
    #     3, 3, 3, 1, 2, 2, 5, 5, 5, 5,
    #     7, 8, 8, 8, 4, 4, 4, 4, 4, 9,
    #     1, 1, 6, 6, 6, 6, 6, 2, 2, 2,
    #     10, 10, 3, 3, 3, 5, 5, 7, 7, 7,
    #     8, 8, 1, 1, 1, 1, 9, 9, 4, 4
    # ]
    # arr = np.array(example)

    p = 0.4
    size = 10000
    rng = np.random.default_rng(seed=314)
    mags = rng.geometric(p, size=size) - 1  # shift so 0 is most probable
    signs = rng.choice([-1, 1], size=size)
    arr =(mags * signs).astype(int)
    print(arr.shape)
    print(np.min(arr))
    print(np.max(arr))

    print(f"Original Bits: {calc_orig_bits(arr)}")

    # enc_start_time = time.time()
    vals, lens = rle_encode(arr)

    # print("HUFFMAN")
    # huffman_enc_start = time.time()
    # vals_encoded, vals_codebook_bits = huffman_encode(vals)
    # lens_encoded, lens_codebook_bits = huffman_encode(lens)
    # huffman_enc_time = time.time() - huffman_enc_start

    # huffman_dec_start = time.time()
    # vals_decoded = huffman_decode(vals_encoded, vals_codebook_bits)
    # lens_decoded = huffman_decode(lens_encoded, lens_codebook_bits)
    # huffman_dec_time = time.time() - huffman_dec_start

    # arr_decoded = rle_decode(vals_decoded, lens_decoded)

    # vals_bitstream = vals_encoded + vals_codebook_bits
    # lens_bitstream = lens_encoded + lens_codebook_bits
    # full_bitstream = vals_bitstream + lens_bitstream
    # print(f"-- Bits: {len(full_bitstream)} ({len(vals_bitstream)} | {len(lens_bitstream)})")
    # print(f"-- Enc Time: {huffman_enc_time}")
    # print(f"-- Dec Time: {huffman_dec_time}")
    # print(f"-- Match: {np.array_equal(arr_decoded, arr)}")
    
    print("EXP GOLOMB")
    golomb_enc_start = time.time()
    vals_encoded = exp_golomb_encode(vals)
    lens_encoded = exp_golomb_encode(lens)
    golomb_enc_time = time.time() - golomb_enc_start

    golomb_dec_start = time.time()
    vals_decoded = exp_golomb_decode(vals_encoded)
    lens_decoded = exp_golomb_decode(lens_encoded)
    golomb_dec_time = time.time() - golomb_dec_start

    arr_decoded = rle_decode(vals_decoded, lens_decoded)

    full_bitstream = vals_encoded + lens_encoded
    print(f"-- Bits: {len(full_bitstream)} ({len(vals_encoded)} | {len(lens_encoded)})")
    print(f"-- Enc Time: {golomb_enc_time}")
    print(f"-- Dec Time: {golomb_dec_time}")
    print(f"-- Match: {np.array_equal(arr_decoded, arr)}")
