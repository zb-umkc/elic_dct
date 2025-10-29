import numpy as np
import heapq
from collections import Counter
from bitarray import bitarray


def rle_encode(arr):
    """Optimized with concatenation approach."""
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    changes = np.concatenate(([True], arr[1:] != arr[:-1], [True]))
    indices = np.where(changes)[0]
    
    values = arr[indices[:-1]]
    lengths = np.diff(indices)
    
    return values, lengths


def rle_decode(values, lengths):
    """Reconstruct array from RLE."""
    return np.repeat(values, lengths)


def exp_golomb_encode_signed(values):
    """Version with lookup table for common values."""
    if len(values) == 0:
        return ""
    
    # Build lookup table for values that appear frequently
    value_counts = Counter(values)
    common_values = set(val for val, count in value_counts.most_common(20) if -50 <= val <= 50)
    
    # Pre-compute codes for common values
    lookup = {}
    for val in common_values:
        if val <= 0:
            xp = -2 * val
        else:
            xp = 2 * val - 1
        
        x_plus_1 = int(xp) + 1
        bits_needed = x_plus_1.bit_length()
        prefix_zeros = bits_needed - 1
        lookup[val] = '0' * prefix_zeros + format(x_plus_1, f'0{bits_needed}b')
    
    result_parts = []
    
    for x in values:
        if x in lookup:
            result_parts.append(lookup[x])
        else:
            if x <= 0:
                xp = -2 * x
            else:
                xp = 2 * x - 1
            
            x_plus_1 = int(xp) + 1
            bits_needed = x_plus_1.bit_length()
            prefix_zeros = bits_needed - 1
            result_parts.append('0' * prefix_zeros + format(x_plus_1, f'0{bits_needed}b'))
    
    return ''.join(result_parts)


def exp_golomb_encode_unsigned(values):
    """
    Unsigned Exp-Golomb with lookup table for common values.
    
    This is optimized for RLE lengths which are typically small positive integers.
    """
    if len(values) == 0:
        return ""
    
    # Verify all values are positive
    if np.any(np.array(values) <= 0):
        raise ValueError("exp_golomb_encode_unsigned_lookup requires all values to be >= 1")
    
    # Build lookup table for the most frequent values
    value_counts = Counter(values)
    lookup_threshold = max(1, len(values) // 100)  # At least 1% frequency
    common_values = {val for val, count in value_counts.items() if count >= lookup_threshold}
    
    # Pre-compute codes for common values
    lookup = {}
    for val in common_values:
        bits_needed = int(val).bit_length()
        prefix_zeros = bits_needed - 1
        lookup[val] = '0' * prefix_zeros + format(val, f'0{bits_needed}b')
    
    # Static lookup for very common small values (1-16) that appear in most RLE data
    static_lookup = {
        1: '1',           # 1 bit
        2: '010',         # 1 zero + 2 bits  
        3: '011',         # 1 zero + 2 bits
        4: '00100',       # 2 zeros + 3 bits
        5: '00101',       # 2 zeros + 3 bits
        6: '00110',       # 2 zeros + 3 bits
        7: '00111',       # 2 zeros + 3 bits
        8: '0001000',     # 3 zeros + 4 bits
        9: '0001001',     # 3 zeros + 4 bits
        10: '0001010',    # 3 zeros + 4 bits
        11: '0001011',    # 3 zeros + 4 bits
        12: '0001100',    # 3 zeros + 4 bits
        13: '0001101',    # 3 zeros + 4 bits
        14: '0001110',    # 3 zeros + 4 bits
        15: '0001111',    # 3 zeros + 4 bits
        16: '000010000',  # 4 zeros + 5 bits
    }
    
    # Merge static and dynamic lookup tables
    combined_lookup = {**static_lookup, **lookup}
    
    result_parts = []
    
    for x in values:
        if x in combined_lookup:
            result_parts.append(combined_lookup[x])
        else:
            # Standard encoding for less common values
            bits_needed = int(x).bit_length()
            prefix_zeros = bits_needed - 1
            result_parts.append('0' * prefix_zeros + format(x, f'0{bits_needed}b'))
    
    return ''.join(result_parts)


def exp_golomb_decode_signed(bitstream):
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


##########################
### Try Huffman Coding ###
##########################

def build_huffman_codebook(arr):
    """
    Builds a Huffman codebook for a numpy array of non-negative integers.
    Returns a list `codebook` where index i gives the bitstring for symbol i.
    """
    freq = Counter(arr.tolist())
    if len(freq) == 1:
        # Edge case: only one symbol
        symbol = next(iter(freq))
        max_sym = symbol
        codebook = ["0"] * (max_sym + 1)
        codebook[symbol] = "0"
        return codebook

    # Build Huffman tree using a min-heap
    heap = [[wt, [sym, ""]] for sym, wt in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Extract codes
    huff_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: p[0])

    # Create list-based codebook
    max_sym = max(freq.keys())
    codebook = [""] * (max_sym + 1)
    for sym, code in huff_codes:
        codebook[sym] = code

    return codebook


def huffman_encode(arr, codebook):
    """
    Encodes a numpy array of non-negative integers using a precomputed list-based codebook.
    Returns a bitarray object containing the encoded bits.
    """
    encoded = bitarray(endian='big')
    for val in arr:
        encoded.extend(codebook[val])
    return encoded.to01()


def huffman_decode(bitstream, codebook):
    """
    Decodes a Huffman-encoded bitstream using a list-based codebook.

    Parameters:
        bitstream (bitarray or str): Encoded bits.
        codebook (list[str]): List of Huffman codes (index = symbol).

    Returns:
        list[int]: Decoded sequence of integers.
    """
    # Build reverse lookup: {code: symbol}
    decode_map = {code: i for i, code in enumerate(codebook) if code}

    # Allow bitarray or str input
    if isinstance(bitstream, bitarray):
        bits = bitstream.to01()
    else:
        bits = bitstream

    decoded = []
    current = ""

    # Sequentially match prefixes
    for bit in bits:
        current += bit
        if current in decode_map:
            decoded.append(decode_map[current])
            current = ""

    return decoded