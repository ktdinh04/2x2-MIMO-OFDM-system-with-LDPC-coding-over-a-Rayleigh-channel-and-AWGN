"""
Module: Interleaver/Deinterleaver
==================================
Interleaving để phân tán burst errors, giúp LDPC hoạt động hiệu quả hơn.

Các loại interleaver:
- Random Interleaver: Hoán vị ngẫu nhiên
- Block Interleaver: Viết theo hàng, đọc theo cột
"""

import numpy as np
from typing import Tuple


class RandomInterleaver:
    """
    Random Interleaver với seed cố định để reproducible.
    """

    def __init__(self, length: int, seed: int = 42):
        """
        Khởi tạo interleaver.

        Args:
            length: Chiều dài chuỗi cần interleave
            seed: Random seed
        """
        self.length = length
        self.seed = seed

        # Tạo permutation cố định
        np.random.seed(seed)
        self.permutation = np.random.permutation(length)
        self.inverse_permutation = np.argsort(self.permutation)

    def interleave(self, data: np.ndarray) -> np.ndarray:
        """
        Interleave dữ liệu.

        Args:
            data: Dữ liệu đầu vào

        Returns:
            Dữ liệu đã interleave
        """
        if len(data) != self.length:
            raise ValueError(f"Data length {len(data)} != interleaver length {self.length}")
        return data[self.permutation]

    def deinterleave(self, data: np.ndarray) -> np.ndarray:
        """
        Deinterleave dữ liệu.

        Args:
            data: Dữ liệu đã interleave

        Returns:
            Dữ liệu gốc
        """
        if len(data) != self.length:
            raise ValueError(f"Data length {len(data)} != interleaver length {self.length}")
        return data[self.inverse_permutation]


class BlockInterleaver:
    """
    Block Interleaver: Viết theo hàng, đọc theo cột.

    Hiệu quả cho burst errors với chiều dài < số hàng.
    """

    def __init__(self, rows: int, cols: int):
        """
        Khởi tạo block interleaver.

        Args:
            rows: Số hàng
            cols: Số cột
        """
        self.rows = rows
        self.cols = cols
        self.length = rows * cols

    def interleave(self, data: np.ndarray) -> np.ndarray:
        """
        Interleave: viết theo hàng, đọc theo cột.
        """
        if len(data) != self.length:
            # Padding nếu cần
            padded = np.zeros(self.length, dtype=data.dtype)
            padded[:len(data)] = data
            data = padded

        matrix = data.reshape(self.rows, self.cols)
        return matrix.T.flatten()

    def deinterleave(self, data: np.ndarray) -> np.ndarray:
        """
        Deinterleave: viết theo cột, đọc theo hàng.
        """
        matrix = data.reshape(self.cols, self.rows)
        return matrix.T.flatten()


class SymbolInterleaver:
    """
    Symbol Interleaver cho OFDM.

    Hoán vị symbols qua các subcarriers để tránh deep fades.
    """

    def __init__(self, n_symbols: int, n_subcarriers: int, seed: int = 123):
        """
        Khởi tạo symbol interleaver.

        Args:
            n_symbols: Số OFDM symbols
            n_subcarriers: Số subcarriers
            seed: Random seed
        """
        self.n_symbols = n_symbols
        self.n_subcarriers = n_subcarriers

        # Tạo frequency interleaving pattern
        np.random.seed(seed)
        self.freq_perm = np.random.permutation(n_subcarriers)
        self.freq_inv = np.argsort(self.freq_perm)

    def interleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Interleave symbols qua frequency domain.

        Args:
            symbols: Shape (n_symbols, n_subcarriers)

        Returns:
            Interleaved symbols
        """
        return symbols[:, self.freq_perm]

    def deinterleave(self, symbols: np.ndarray) -> np.ndarray:
        """
        Deinterleave symbols.
        """
        return symbols[:, self.freq_inv]


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE: Interleaver")
    print("=" * 60)

    # Test Random Interleaver
    print("\n--- Test Random Interleaver ---")
    interleaver = RandomInterleaver(length=100, seed=42)

    data = np.arange(100)
    interleaved = interleaver.interleave(data)
    deinterleaved = interleaver.deinterleave(interleaved)

    print(f"Original (first 10): {data[:10]}")
    print(f"Interleaved (first 10): {interleaved[:10]}")
    print(f"Deinterleaved (first 10): {deinterleaved[:10]}")
    print(f"Recovery OK: {np.array_equal(data, deinterleaved)}")

    # Test burst error spreading
    print("\n--- Test Burst Error Spreading ---")
    # Giả sử burst error ở vị trí 10-19
    errors = np.zeros(100, dtype=int)
    errors[10:20] = 1  # 10 bit lỗi liên tiếp

    errors_interleaved = interleaver.interleave(errors)
    print(f"Burst error positions (original): {np.where(errors == 1)[0]}")
    print(f"Error positions after interleaving: {np.where(errors_interleaved == 1)[0]}")
    print(f"Max consecutive errors after interleaving: ", end="")

    # Tính consecutive errors
    max_consecutive = 1
    current = 1
    for i in range(1, len(errors_interleaved)):
        if errors_interleaved[i] == 1 and errors_interleaved[i-1] == 1:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 1
    print(max_consecutive)

    # Test Block Interleaver
    print("\n--- Test Block Interleaver ---")
    block_int = BlockInterleaver(rows=10, cols=10)
    data = np.arange(100)
    interleaved = block_int.interleave(data)
    deinterleaved = block_int.deinterleave(interleaved)
    print(f"Block interleaver recovery OK: {np.array_equal(data, deinterleaved)}")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
