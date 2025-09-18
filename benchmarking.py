import numpy as np
import time
from collections import Counter
import h5py

# ========== ALL IMPLEMENTATIONS TO TEST ==========

def rle_encode_original(arr):
    """Your original implementation."""
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    diff = np.diff(arr)
    run_boundaries = np.where(diff != 0)[0] + 1
    run_starts = np.insert(run_boundaries, 0, 0)
    run_ends = np.append(run_boundaries, n)

    values = arr[run_starts]
    lengths = run_ends - run_starts
    return values, lengths


def rle_encode_v1(arr):
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


def rle_encode_v2(arr):
    """Optimized with nonzero approach."""
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    if n == 1:
        return np.array([arr[0]]), np.array([1])
    
    change_points = arr[1:] != arr[:-1]
    change_indices = np.nonzero(change_points)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [n]))
    
    values = arr[change_indices[:-1]]
    lengths = np.diff(change_indices)
    
    return values, lengths


def rle_encode_simple_loop(arr):
    """Simple Python loop - sometimes faster for certain data patterns."""
    arr = np.asarray(arr)
    if len(arr) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    values = []
    lengths = []
    current_val = arr[0]
    current_len = 1
    
    for i in range(1, len(arr)):
        if arr[i] == current_val:
            current_len += 1
        else:
            values.append(current_val)
            lengths.append(current_len)
            current_val = arr[i]
            current_len = 1
    
    values.append(current_val)
    lengths.append(current_len)
    
    return np.array(values), np.array(lengths)


def encode_number_original(x):
    """Your original encode_number function."""
    if x <= 0:
        xp = -2*x
    else:
        xp = 2*x - 1

    x_bin = bin(int(xp) + 1)[2:]
    prefix = "0" * (len(x_bin)-1)
    return prefix + x_bin


def exp_golomb_encode_original(values):
    """Your original implementation."""
    arr_encoded = "".join([encode_number_original(x) for x in values])
    return arr_encoded


def exp_golomb_encode_list_comprehension(values):
    """Using list comprehension."""
    if len(values) == 0:
        return ""
    
    def encode_single(x):
        xp = -2 * x if x <= 0 else 2 * x - 1
        x_plus_1 = int(xp) + 1
        bits_needed = x_plus_1.bit_length()
        return '0' * (bits_needed - 1) + format(x_plus_1, f'0{bits_needed}b')
    
    return ''.join(encode_single(x) for x in values)


def exp_golomb_encode_prealloc(values):
    """Pre-allocate list for better memory performance."""
    if len(values) == 0:
        return ""
    
    result_parts = []
    
    for x in values:
        if x <= 0:
            xp = -2 * x
        else:
            xp = 2 * x - 1
        
        x_plus_1 = int(xp) + 1
        bits_needed = x_plus_1.bit_length()
        prefix_zeros = bits_needed - 1
        
        result_parts.append('0' * prefix_zeros + format(x_plus_1, f'0{bits_needed}b'))
    
    return ''.join(result_parts)


def exp_golomb_encode_lookup(values):
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


def exp_golomb_encode_unsigned_lookup(values):
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


def exp_golomb_encode_numpy_attempt(values):
    """Attempt at numpy vectorization."""
    if len(values) == 0:
        return ""
    
    values = np.asarray(values)
    xp = np.where(values <= 0, -2 * values, 2 * values - 1)
    x_plus_1 = xp + 1
    
    # For bit length calculation, we need to handle each value individually
    result_parts = []
    for i in range(len(values)):
        val = x_plus_1[i]
        bits_needed = int(val).bit_length()
        prefix_zeros = bits_needed - 1
        result_parts.append('0' * prefix_zeros + format(val, f'0{bits_needed}b'))
    
    return ''.join(result_parts)


# ========== BENCHMARKING FRAMEWORK ==========

def analyze_data_characteristics(arr):
    """Analyze the statistical properties of your data."""
    print("=== DATA CHARACTERISTICS ===")
    print(f"Array shape: {arr.shape}")
    print(f"Data type: {arr.dtype}")
    print(f"Min value: {np.min(arr)}")
    print(f"Max value: {np.max(arr)}")
    print(f"Mean: {np.mean(arr):.2f}")
    print(f"Std: {np.std(arr):.2f}")
    print(f"Unique values: {len(np.unique(arr))}")
    
    # Analyze run-length characteristics
    vals, lens = rle_encode_original(arr.flatten())
    compression_ratio = len(arr.flatten()) / len(vals)
    print(f"RLE compression ratio: {compression_ratio:.2f}x")
    print(f"Average run length: {np.mean(lens):.2f}")
    print(f"Max run length: {np.max(lens)}")
    print(f"Min run length: {np.min(lens)}")
    
    # Value distribution
    value_counts = Counter(arr.flatten())
    most_common = value_counts.most_common(10)
    print(f"Most common values: {most_common}")
    
    # Analyze vals vs lens characteristics for Exp-Golomb
    analyze_vals_vs_lens_for_exp_golomb(vals, lens)
    
    return vals, lens


def analyze_vals_vs_lens_for_exp_golomb(vals, lens):
    """Analyze why vals might be slower than lens for Exp-Golomb encoding."""
    print(f"\n=== VALS vs LENS ANALYSIS FOR EXP-GOLOMB ===")
    
    # Basic statistics
    print(f"Vals array: {len(vals)} elements")
    print(f"  Min: {np.min(vals)}, Max: {np.max(vals)}")
    print(f"  Mean: {np.mean(vals):.2f}, Std: {np.std(vals):.2f}")
    print(f"  Unique values: {len(np.unique(vals))}")
    
    print(f"Lens array: {len(lens)} elements")
    print(f"  Min: {np.min(lens)}, Max: {np.max(lens)}")
    print(f"  Mean: {np.mean(lens):.2f}, Std: {np.std(lens):.2f}")
    print(f"  Unique values: {len(np.unique(lens))}")
    
    # Exp-Golomb complexity analysis
    def estimate_exp_golomb_cost(values):
        """Estimate the computational cost of Exp-Golomb encoding."""
        total_bits = 0
        for val in values:
            if val <= 0:
                xp = -2 * val
            else:
                xp = 2 * val - 1
            
            x_plus_1 = int(xp) + 1
            bits_needed = x_plus_1.bit_length()
            total_bits += (bits_needed - 1) + bits_needed  # prefix zeros + binary
        
        return total_bits
    
    vals_estimated_bits = estimate_exp_golomb_cost(vals)
    lens_estimated_bits = estimate_exp_golomb_cost(lens)
    
    print(f"\nEstimated Exp-Golomb output:")
    print(f"  Vals: {vals_estimated_bits} bits ({vals_estimated_bits/len(vals):.1f} bits/value)")
    print(f"  Lens: {lens_estimated_bits} bits ({lens_estimated_bits/len(lens):.1f} bits/value)")
    
    # Distribution analysis
    print(f"\nValue distribution analysis:")
    vals_counter = Counter(vals)
    lens_counter = Counter(lens)
    
    print(f"Vals - most common: {vals_counter.most_common(5)}")
    print(f"Lens - most common: {lens_counter.most_common(5)}")
    
    # Negative values analysis (these are more expensive in Exp-Golomb)
    vals_negative = np.sum(vals < 0)
    lens_negative = np.sum(lens < 0)
    
    print(f"\nNegative values (expensive in Exp-Golomb):")
    print(f"  Vals: {vals_negative}/{len(vals)} ({100*vals_negative/len(vals):.1f}%)")
    print(f"  Lens: {lens_negative}/{len(lens)} ({100*lens_negative/len(lens):.1f}%)")
    
    # Large values analysis (also expensive)
    vals_large = np.sum(np.abs(vals) > 10)
    lens_large = np.sum(lens > 10)
    
    print(f"Large values (|val| > 10, expensive in Exp-Golomb):")
    print(f"  Vals: {vals_large}/{len(vals)} ({100*vals_large/len(vals):.1f}%)")
    print(f"  Lens: {lens_large}/{len(lens)} ({100*lens_large/len(lens):.1f}%)")


def benchmark_vals_vs_lens_separately(vals, lens, iterations=20):
    """Benchmark encoding vals vs lens separately to identify the bottleneck."""
    print(f"\n=== SEPARATE VALS vs LENS BENCHMARK ===")
    
    # Test original implementation
    func = exp_golomb_encode_original
    
    # Benchmark vals only
    start_time = time.time()
    for _ in range(iterations):
        vals_encoded = func(vals)
    vals_time = (time.time() - start_time) / iterations
    
    # Benchmark lens only  
    start_time = time.time()
    for _ in range(iterations):
        lens_encoded = func(lens)
    lens_time = (time.time() - start_time) / iterations
    
    print(f"Encoding VALS only:   {vals_time*1000:.3f} ms")
    print(f"Encoding LENS only:   {lens_time*1000:.3f} ms")
    print(f"Total (vals + lens):  {(vals_time + lens_time)*1000:.3f} ms")
    
    ratio = vals_time / lens_time if lens_time > 0 else float('inf')
    print(f"Vals is {ratio:.1f}x slower than lens")
    
    # Analyze why vals is slower
    if ratio > 2:
        print(f"\n🔍 VALS IS SIGNIFICANTLY SLOWER - Likely reasons:")
        
        # Check value ranges
        vals_range = np.max(vals) - np.min(vals)
        lens_range = np.max(lens) - np.min(lens)
        print(f"  Value range - Vals: {vals_range}, Lens: {lens_range}")
        
        # Check negative values
        vals_neg_pct = 100 * np.sum(vals < 0) / len(vals)
        lens_neg_pct = 100 * np.sum(lens < 0) / len(lens)
        print(f"  Negative values - Vals: {vals_neg_pct:.1f}%, Lens: {lens_neg_pct:.1f}%")
        
        # Check large values
        vals_large_pct = 100 * np.sum(np.abs(vals) > 10) / len(vals)
        lens_large_pct = 100 * np.sum(lens > 10) / len(lens)
        print(f"  Large values (>10) - Vals: {vals_large_pct:.1f}%, Lens: {lens_large_pct:.1f}%")
        
        # Check distribution entropy
        vals_unique_pct = 100 * len(np.unique(vals)) / len(vals)
        lens_unique_pct = 100 * len(np.unique(lens)) / len(lens)
        print(f"  Unique values - Vals: {vals_unique_pct:.1f}%, Lens: {lens_unique_pct:.1f}%")
    
    return vals_time, lens_time


def benchmark_rle_implementations(arr, iterations=100):
    """Benchmark all RLE implementations."""
    print(f"\n=== RLE BENCHMARK ({iterations} iterations) ===")
    
    flat_arr = arr.flatten()
    
    implementations = [
        ("Original", rle_encode_original),
        ("Concatenation (v1)", rle_encode_v1),
        ("Nonzero (v2)", rle_encode_v2),
        ("Simple Loop", rle_encode_simple_loop),
    ]
    
    results = {}
    
    for name, func in implementations:
        # Warmup
        func(flat_arr)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            vals, lens = func(flat_arr)
        elapsed = (time.time() - start_time) / iterations
        
        results[name] = elapsed
        print(f"{name:20}: {elapsed*1000:.3f} ms")
        
        # Verify correctness
        vals_orig, lens_orig = rle_encode_original(flat_arr)
        if not (np.array_equal(vals, vals_orig) and np.array_equal(lens, lens_orig)):
            print(f"  ⚠️  ERROR: {name} produces different results!")
    
    # Find best
    best_name = min(results.keys(), key=lambda k: results[k])
    best_time = results[best_name]
    print(f"\n🏆 Best RLE: {best_name} ({best_time*1000:.3f} ms)")
    
    for name, time_val in results.items():
        if name != best_name:
            speedup = time_val / best_time
            print(f"   {name} is {speedup:.1f}x slower")
    
    return best_name


def benchmark_exp_golomb_implementations(vals, lens, iterations=20):
    """Benchmark all Exp-Golomb implementations using real RLE data."""
    print(f"\n=== EXP-GOLOMB BENCHMARK ({iterations} iterations) ===")
    print(f"Values array: {len(vals)} elements")
    print(f"Lengths array: {len(lens)} elements")
    print(f"Total elements to encode: {len(vals) + len(lens)}")
    
    implementations = [
        ("Original", exp_golomb_encode_original),
        ("List Comprehension", exp_golomb_encode_list_comprehension),
        ("Pre-allocated", exp_golomb_encode_prealloc),
        ("Lookup Table", exp_golomb_encode_lookup),
        ("Numpy Attempt", exp_golomb_encode_numpy_attempt),
    ]
    
    results = {}
    
    for name, func in implementations:
        # Warmup - encode both vals and lens like the real pipeline
        try:
            vals_result = func(vals)
            lens_result = func(lens)
            
            # Benchmark - time both calls together like the real pipeline
            start_time = time.time()
            for _ in range(iterations):
                vals_encoded = func(vals)
                lens_encoded = func(lens)
            elapsed = (time.time() - start_time) / iterations
            
            results[name] = elapsed
            print(f"{name:20}: {elapsed*1000:.3f} ms (both vals + lens)")
            
            # Verify correctness
            vals_orig = exp_golomb_encode_original(vals)
            lens_orig = exp_golomb_encode_original(lens)
            if vals_result != vals_orig or lens_result != lens_orig:
                print(f"  ⚠️  ERROR: {name} produces different results!")
                
        except Exception as e:
            print(f"{name:20}: FAILED ({str(e)})")
    
    # Find best
    if results:
        best_name = min(results.keys(), key=lambda k: results[k])
        best_time = results[best_name]
        print(f"\n🏆 Best Exp-Golomb: {best_name} ({best_time*1000:.3f} ms)")
        
        for name, time_val in results.items():
            if name != best_name:
                speedup = time_val / best_time
                print(f"   {name} is {speedup:.1f}x slower")
        
        return best_name
    
    return None


def benchmark_complete_pipeline(arr, rle_func, exp_golomb_func, iterations=50):
    """Benchmark the complete compression pipeline."""
    print(f"\n=== COMPLETE PIPELINE BENCHMARK ({iterations} iterations) ===")
    
    flat_arr = arr.flatten()
    
    # Warmup
    vals, lens = rle_func(flat_arr)
    vals_encoded = exp_golomb_func(vals)
    lens_encoded = exp_golomb_func(lens)
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        vals, lens = rle_func(flat_arr)
        vals_encoded = exp_golomb_func(vals)
        lens_encoded = exp_golomb_func(lens)
    elapsed = (time.time() - start_time) / iterations
    
    print(f"Complete pipeline: {elapsed*1000:.3f} ms")
    
    # Calculate compression stats
    original_bits = len(flat_arr) * 32  # Assuming 32-bit integers
    compressed_bits = len(vals_encoded) + len(lens_encoded)
    compression_ratio = original_bits / compressed_bits
    
    print(f"Original size: {original_bits} bits")
    print(f"Compressed size: {compressed_bits} bits")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return elapsed


def create_sample_sar_data(shape=(100, 100), seed=42):
    """Create synthetic data that mimics SAR latent features."""
    np.random.seed(seed)
    
    # SAR latent features often have:
    # - Many zeros or near-zero values
    # - Some high-magnitude outliers
    # - Spatial correlation (similar values near each other)
    
    # Base layer with mostly small values
    data = np.random.laplace(0, 2, shape).astype(int)
    
    # Add some spatial correlation
    from scipy import ndimage
    data = ndimage.gaussian_filter(data.astype(float), sigma=1.0).astype(int)
    
    # Add sparse high-magnitude values
    outlier_mask = np.random.random(shape) < 0.05  # 5% outliers
    data[outlier_mask] = np.random.choice([-50, -20, 20, 50], size=np.sum(outlier_mask))
    
    # Quantize to simulate latent features
    data = np.clip(data, -100, 100)
    
    return data


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
    
    # Analyze data characteristics
    vals, lens = analyze_data_characteristics(arr)
    
    # Benchmark RLE implementations
    best_rle = benchmark_rle_implementations(arr, iterations=100)
    
    # Get the best RLE function
    rle_functions = {
        "Original": rle_encode_original,
        "Concatenation (v1)": rle_encode_v1,
        "Nonzero (v2)": rle_encode_v2,
        "Simple Loop": rle_encode_simple_loop,
    }
    best_rle_func = rle_functions[best_rle]
    
    # First, benchmark vals vs lens separately to understand the bottleneck
    benchmark_vals_vs_lens_separately(vals, lens, iterations=20)
    
    # Benchmark Exp-Golomb implementations using actual RLE vals and lens
    best_exp_golomb = benchmark_exp_golomb_implementations(vals, lens, iterations=20)
    
    if best_exp_golomb:
        exp_golomb_functions = {
            "Original": exp_golomb_encode_original,
            "List Comprehension": exp_golomb_encode_list_comprehension,
            "Pre-allocated": exp_golomb_encode_prealloc,
            "Lookup Table": exp_golomb_encode_lookup,
            "Numpy Attempt": exp_golomb_encode_numpy_attempt,
        }
        best_exp_golomb_func = exp_golomb_functions[best_exp_golomb]
        
        # Benchmark complete pipeline
        benchmark_complete_pipeline(arr, best_rle_func, best_exp_golomb_func, iterations=20)
    
    print(f"\n🎯 RECOMMENDATION FOR YOUR DATA:")
    print(f"   Best RLE: {best_rle}")
    if best_exp_golomb:
        print(f"   Best Exp-Golomb: {best_exp_golomb}")


if __name__ == "__main__":
    main()
