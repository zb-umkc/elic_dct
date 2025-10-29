import random
import numpy as np
import time
from fractions import Fraction
from decimal import Decimal, getcontext
from math import factorial, ceil, log2
from tqdm import tqdm

# Set high precision for Decimal arithmetic
# getcontext().prec = 3000  # Adjust precision as needed

# -----------------------------
# Fenwick Tree (for Factoradic)
# -----------------------------
class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0]*(n+1)
    def add(self, i, v=1):
        i += 1
        while i <= self.n:
            self.tree[i] += v
            i += i & -i
    def sum(self, i):
        # prefix sum [0..i]
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s
    def find_index(self, k):
        """Find 0-based index of the (k)-th unused element."""
        i = 0
        bit = 1 << (self.n.bit_length())
        while bit:
            nxt = i + bit
            if nxt <= self.n and self.tree[nxt] <= k:
                k -= self.tree[nxt]
                i = nxt
            bit >>= 1
        return i

# -----------------------------
# Factoradic encode/decode
# -----------------------------
def encode_factoradic(perm):
    n = len(perm)
    fw = Fenwick(n)
    for i in range(n):
        fw.add(i, 1)

    rank = 0
    for k, x in enumerate(perm):
        idx = fw.sum(x-1) if x > 0 else 0
        rank += idx * factorial(n-1-k)
        fw.add(x, -1)

    bitlen = ceil(log2(factorial(n)))
    return format(rank, f'0{bitlen}b')

def decode_factoradic(bitstring, n):
    rank = int(bitstring, 2)
    fw = Fenwick(n)
    for i in range(n):
        fw.add(i, 1)

    perm = []
    for k in range(n):
        f = factorial(n-1-k)
        idx = rank // f
        rank = rank % f
        x = fw.find_index(idx)
        perm.append(x)
        fw.add(x, -1)
    return perm


if __name__ == "__main__":
    n = 320
    for _ in range(10):
        perm = list(range(n))
        random.shuffle(perm)
        # print("Original perm:", perm[:10])

        # Factoradic (baseline)
        # print("\n--- Factoradic ---")
        f_enc_start = time.time()
        bits_f = encode_factoradic(perm)
        f_enc_time = time.time() - f_enc_start

        f_dec_start = time.time()
        decoded_f = decode_factoradic(bits_f, n)
        f_dec_time = time.time() - f_dec_start
        assert np.array_equal(perm, decoded_f)
        # print(f"Length: {len(bits_f)} bits")
        print(f"Encode: {f_enc_time:.6f}s | Decode: {f_dec_time:.6f}s")
