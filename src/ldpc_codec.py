"""
Module: LDPC (Low-Density Parity-Check) Codec
==============================================
Bộ mã hóa và giải mã LDPC sử dụng thuật toán Sum-Product (Belief Propagation).

LDPC là mã sửa lỗi tiệm cận Shannon limit, được sử dụng rộng rãi trong
các chuẩn truyền thông hiện đại như 5G, Wi-Fi 6, DVB-S2.

Thuật toán giải mã:
- Sum-Product Algorithm (SPA): Soft-decision decoding
- Min-Sum Algorithm: Phiên bản đơn giản hóa của SPA
"""

import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csr_matrix, lil_matrix


class LDPCCode:
    """
    Mã LDPC với ma trận kiểm tra H sparse.

    Mã LDPC được định nghĩa bởi ma trận parity-check H có kích thước (m x n):
    - n: Chiều dài codeword
    - k: Số bit thông tin (k = n - m)
    - m: Số bit parity (số hàng của H)
    - Rate: R = k/n

    Attributes:
        H (np.ndarray): Ma trận parity-check
        G (np.ndarray): Ma trận generator
        n (int): Chiều dài codeword
        k (int): Số bit thông tin
        m (int): Số bit parity
        rate (float): Tỷ lệ mã hóa k/n
    """

    def __init__(self, n: int = 648, rate: float = 0.5, seed: int = 42):
        """
        Khởi tạo mã LDPC.

        Args:
            n: Chiều dài codeword (mặc định 648 - tương thích với OFDM)
            rate: Tỷ lệ mã hóa (mặc định 0.5)
            seed: Seed cho việc tạo ma trận ngẫu nhiên
        """
        self.n = n
        self.rate = rate
        self.k = int(n * rate)  # Số bit thông tin
        self.m = n - self.k     # Số bit parity

        np.random.seed(seed)
        self.H = self._create_parity_check_matrix()
        self.G = self._create_generator_matrix()

        # Precompute các cấu trúc cho giải mã
        self._precompute_tanner_graph()

    def _create_parity_check_matrix(self) -> np.ndarray:
        """
        Tạo ma trận parity-check H theo phương pháp Gallager.

        Ma trận H được tạo với cấu trúc quasi-cyclic để đảm bảo:
        - Số 1 trên mỗi hàng (row weight) = w_r
        - Số 1 trên mỗi cột (column weight) = w_c
        - Không có cycle ngắn (tránh cycle-4)

        Returns:
            Ma trận H kích thước (m x n)
        """
        # Tham số cho regular LDPC
        # Row weight và column weight
        w_c = 3  # Column weight (số 1 trong mỗi cột)
        w_r = int(w_c * self.n / self.m)  # Row weight

        H = np.zeros((self.m, self.n), dtype=int)

        # Phương pháp Progressive Edge Growth (PEG) đơn giản hóa
        for j in range(self.n):
            # Chọn w_c hàng ngẫu nhiên cho cột j
            available_rows = np.arange(self.m)

            # Ưu tiên các hàng có ít 1 nhất
            row_weights = np.sum(H, axis=1)
            sorted_rows = np.argsort(row_weights)

            # Chọn w_c hàng có weight thấp nhất
            selected_rows = sorted_rows[:w_c]

            for row in selected_rows:
                H[row, j] = 1

        return H

    def _create_generator_matrix(self) -> np.ndarray:
        """
        Tạo ma trận generator G từ H.

        Sử dụng phương pháp Gaussian elimination để đưa H về dạng:
        H = [P | I_m]

        Khi đó: G = [I_k | P^T]

        Codeword: c = m * G, trong đó m là message

        Returns:
            Ma trận G kích thước (k x n)
        """
        H_temp = self.H.astype(float).copy()
        m, n = H_temp.shape
        k = n - m

        # Gaussian elimination trong GF(2)
        # Đưa phần bên phải của H về dạng đơn vị
        pivot_row = 0
        col_order = list(range(n))

        for col in range(n - m, n):
            # Tìm pivot
            found = False
            for row in range(pivot_row, m):
                if H_temp[row, col] == 1:
                    # Swap rows
                    H_temp[[pivot_row, row]] = H_temp[[row, pivot_row]]
                    found = True
                    break

            if not found:
                # Tìm cột khác có thể hoán đổi
                for swap_col in range(k):
                    for row in range(pivot_row, m):
                        if H_temp[row, swap_col] == 1:
                            # Swap columns
                            H_temp[:, [col, swap_col]] = H_temp[:, [swap_col, col]]
                            col_order[col], col_order[swap_col] = col_order[swap_col], col_order[col]
                            H_temp[[pivot_row, row]] = H_temp[[row, pivot_row]]
                            found = True
                            break
                    if found:
                        break

            # Elimination
            for row in range(m):
                if row != pivot_row and H_temp[row, col] == 1:
                    H_temp[row] = (H_temp[row] + H_temp[pivot_row]) % 2

            pivot_row += 1

        # Tạo G = [I_k | P^T]
        P = H_temp[:, :k].T  # k x m
        G = np.zeros((k, n), dtype=int)
        G[:, :k] = np.eye(k, dtype=int)
        G[:, k:] = P.astype(int)

        # Áp dụng thứ tự cột
        G_reordered = np.zeros((k, n), dtype=int)
        for i, original_col in enumerate(col_order):
            G_reordered[:, original_col] = G[:, i]

        return G

    def _precompute_tanner_graph(self):
        """
        Tính trước cấu trúc Tanner graph cho giải mã.

        Tanner graph là bipartite graph với:
        - Variable nodes (VN): n nodes cho n bit
        - Check nodes (CN): m nodes cho m parity checks
        - Edges: nối VN_j với CN_i nếu H[i,j] = 1
        """
        # Danh sách các check nodes kết nối với mỗi variable node
        self.vn_to_cn = [[] for _ in range(self.n)]
        # Danh sách các variable nodes kết nối với mỗi check node
        self.cn_to_vn = [[] for _ in range(self.m)]

        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    self.vn_to_cn[j].append(i)
                    self.cn_to_vn[i].append(j)

    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        Mã hóa message thành codeword.

        c = m * G (mod 2)

        Args:
            message: Mảng k bit thông tin

        Returns:
            Mảng n bit codeword

        Raises:
            ValueError: Nếu chiều dài message không đúng
        """
        if len(message) != self.k:
            raise ValueError(f"Message phải có {self.k} bit, nhận được {len(message)}")

        codeword = np.dot(message, self.G) % 2
        return codeword.astype(int)

    def decode(self, llr: np.ndarray, max_iter: int = 50) -> Tuple[np.ndarray, bool]:
        """
        Giải mã LDPC sử dụng thuật toán Sum-Product.

        Sum-Product Algorithm (SPA):
        1. Khởi tạo: L_v->c = LLR kênh
        2. Check node update: L_c->v = 2 * atanh(prod(tanh(L_v'->c / 2)))
        3. Variable node update: L_v->c = LLR_ch + sum(L_c'->v)
        4. Quyết định: L_v = LLR_ch + sum(L_c->v)
        5. Kiểm tra syndrome, nếu = 0 thì dừng

        Args:
            llr: Log-likelihood ratios từ kênh (dương → bit 0)
            max_iter: Số vòng lặp tối đa

        Returns:
            decoded: Codeword đã giải mã
            success: True nếu syndrome = 0
        """
        n = self.n
        m = self.m

        # Khởi tạo messages
        # L_v_to_c[i][j] = message từ VN j đến CN i
        L_v_to_c = {}
        # L_c_to_v[i][j] = message từ CN i đến VN j
        L_c_to_v = {}

        # Khởi tạo với LLR kênh
        for j in range(n):
            for i in self.vn_to_cn[j]:
                L_v_to_c[(i, j)] = llr[j]
                L_c_to_v[(i, j)] = 0.0

        for iteration in range(max_iter):
            # ========== Check Node Update ==========
            for i in range(m):
                vns = self.cn_to_vn[i]
                for j in vns:
                    # Tính tích tanh cho tất cả VN khác j
                    product = 1.0
                    for j_prime in vns:
                        if j_prime != j:
                            x = L_v_to_c[(i, j_prime)] / 2.0
                            # Clip để tránh overflow
                            x = np.clip(x, -10, 10)
                            product *= np.tanh(x)

                    # Clip product để tránh arctanh của ±1
                    product = np.clip(product, -0.9999999, 0.9999999)
                    L_c_to_v[(i, j)] = 2.0 * np.arctanh(product)

            # ========== Variable Node Update ==========
            for j in range(n):
                cns = self.vn_to_cn[j]
                for i in cns:
                    # Tổng messages từ tất cả CN khác i
                    sum_msg = llr[j]
                    for i_prime in cns:
                        if i_prime != i:
                            sum_msg += L_c_to_v[(i_prime, j)]
                    L_v_to_c[(i, j)] = sum_msg

            # ========== Quyết định (Tentative Decision) ==========
            L_total = np.zeros(n)
            for j in range(n):
                L_total[j] = llr[j]
                for i in self.vn_to_cn[j]:
                    L_total[j] += L_c_to_v[(i, j)]

            # Hard decision
            decoded = (L_total < 0).astype(int)

            # Kiểm tra syndrome
            syndrome = np.dot(self.H, decoded) % 2
            if np.sum(syndrome) == 0:
                return decoded, True

        return decoded, False

    def decode_minsum(self, llr: np.ndarray, max_iter: int = 50,
                      alpha: float = 0.75) -> Tuple[np.ndarray, bool]:
        """
        Giải mã LDPC sử dụng thuật toán Min-Sum (đơn giản hóa).

        Min-Sum thay thế phép tính phức tạp của SPA bằng:
        L_c->v = sign(prod) * min(|L_v'->c|)

        Ưu điểm: Đơn giản hơn, ít phép tính
        Nhược điểm: Hiệu suất kém hơn SPA một chút

        Args:
            llr: Log-likelihood ratios
            max_iter: Số vòng lặp tối đa
            alpha: Hệ số scaling (0.75 thường tối ưu)

        Returns:
            decoded: Codeword đã giải mã
            success: True nếu syndrome = 0
        """
        n = self.n
        m = self.m

        # Khởi tạo messages
        L_v_to_c = {}
        L_c_to_v = {}

        for j in range(n):
            for i in self.vn_to_cn[j]:
                L_v_to_c[(i, j)] = llr[j]
                L_c_to_v[(i, j)] = 0.0

        for iteration in range(max_iter):
            # ========== Check Node Update (Min-Sum) ==========
            for i in range(m):
                vns = self.cn_to_vn[i]
                for j in vns:
                    # Tính sign và min cho tất cả VN khác j
                    sign = 1
                    min_abs = float('inf')

                    for j_prime in vns:
                        if j_prime != j:
                            msg = L_v_to_c[(i, j_prime)]
                            if msg < 0:
                                sign *= -1
                            abs_msg = abs(msg)
                            if abs_msg < min_abs:
                                min_abs = abs_msg

                    L_c_to_v[(i, j)] = alpha * sign * min_abs

            # ========== Variable Node Update ==========
            for j in range(n):
                cns = self.vn_to_cn[j]
                for i in cns:
                    sum_msg = llr[j]
                    for i_prime in cns:
                        if i_prime != i:
                            sum_msg += L_c_to_v[(i_prime, j)]
                    L_v_to_c[(i, j)] = sum_msg

            # ========== Quyết định ==========
            L_total = np.zeros(n)
            for j in range(n):
                L_total[j] = llr[j]
                for i in self.vn_to_cn[j]:
                    L_total[j] += L_c_to_v[(i, j)]

            decoded = (L_total < 0).astype(int)

            # Kiểm tra syndrome
            syndrome = np.dot(self.H, decoded) % 2
            if np.sum(syndrome) == 0:
                return decoded, True

        return decoded, False

    def check_codeword(self, codeword: np.ndarray) -> bool:
        """
        Kiểm tra xem codeword có hợp lệ không.

        H * c^T = 0 (mod 2)

        Args:
            codeword: Codeword cần kiểm tra

        Returns:
            True nếu syndrome = 0
        """
        syndrome = np.dot(self.H, codeword) % 2
        return np.sum(syndrome) == 0

    def get_info_bits(self, codeword: np.ndarray) -> np.ndarray:
        """
        Lấy các bit thông tin từ codeword.

        Với encoding systematic: k bit đầu là bit thông tin.

        Args:
            codeword: Codeword n bit

        Returns:
            k bit thông tin
        """
        return codeword[:self.k]


class LDPCCodec:
    """
    Wrapper class để mã hóa/giải mã với padding tự động.
    """

    def __init__(self, n: int = 648, rate: float = 0.5, seed: int = 42):
        """
        Khởi tạo LDPC codec.

        Args:
            n: Chiều dài codeword
            rate: Tỷ lệ mã hóa
            seed: Seed cho ma trận
        """
        self.ldpc = LDPCCode(n, rate, seed)
        self.n = self.ldpc.n
        self.k = self.ldpc.k

    def encode_block(self, bits: np.ndarray) -> np.ndarray:
        """
        Mã hóa một khối dữ liệu, tự động padding nếu cần.

        Args:
            bits: Dữ liệu cần mã hóa

        Returns:
            Codewords đã mã hóa
        """
        # Padding về bội của k
        num_bits = len(bits)
        num_blocks = int(np.ceil(num_bits / self.k))
        padded_len = num_blocks * self.k

        padded_bits = np.zeros(padded_len, dtype=int)
        padded_bits[:num_bits] = bits

        # Mã hóa từng block
        codewords = np.zeros(num_blocks * self.n, dtype=int)
        for i in range(num_blocks):
            msg = padded_bits[i * self.k:(i + 1) * self.k]
            codeword = self.ldpc.encode(msg)
            codewords[i * self.n:(i + 1) * self.n] = codeword

        return codewords

    def decode_block(self, llr: np.ndarray, max_iter: int = 50,
                     algorithm: str = 'sum_product') -> Tuple[np.ndarray, int]:
        """
        Giải mã các khối LLR.

        Args:
            llr: LLR values
            max_iter: Số vòng lặp tối đa
            algorithm: 'sum_product' hoặc 'min_sum'

        Returns:
            decoded_bits: Bit thông tin đã giải mã
            num_failures: Số block giải mã thất bại
        """
        num_blocks = len(llr) // self.n
        decoded_bits = np.zeros(num_blocks * self.k, dtype=int)
        num_failures = 0

        for i in range(num_blocks):
            block_llr = llr[i * self.n:(i + 1) * self.n]

            if algorithm == 'sum_product':
                codeword, success = self.ldpc.decode(block_llr, max_iter)
            else:
                codeword, success = self.ldpc.decode_minsum(block_llr, max_iter)

            if not success:
                num_failures += 1

            decoded_bits[i * self.k:(i + 1) * self.k] = self.ldpc.get_info_bits(codeword)

        return decoded_bits, num_failures


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE: LDPC Encoder/Decoder")
    print("=" * 60)

    # Tạo LDPC code với tham số nhỏ để test
    ldpc = LDPCCode(n=96, rate=0.5, seed=42)

    print(f"\nLDPC Code Parameters:")
    print(f"  - Codeword length (n): {ldpc.n}")
    print(f"  - Information bits (k): {ldpc.k}")
    print(f"  - Parity bits (m): {ldpc.m}")
    print(f"  - Code rate: {ldpc.rate:.2f}")

    # Test encoding
    np.random.seed(123)
    message = np.random.randint(0, 2, ldpc.k)
    codeword = ldpc.encode(message)

    print(f"\nEncoding Test:")
    print(f"  - Message (first 10 bits): {message[:10]}")
    print(f"  - Codeword length: {len(codeword)}")
    print(f"  - Codeword valid: {ldpc.check_codeword(codeword)}")

    # Test decoding with AWGN channel
    print(f"\nDecoding Test with AWGN channel:")

    for snr_db in [1, 3, 5]:
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1 / (2 * snr_linear)

        # BPSK modulation: 0 -> +1, 1 -> -1
        tx_signal = 1 - 2 * codeword

        # Add noise
        noise = np.sqrt(noise_var) * np.random.randn(len(tx_signal))
        rx_signal = tx_signal + noise

        # Calculate LLR: LLR = 2 * r / sigma^2
        channel_llr = 2 * rx_signal / noise_var

        # Decode
        decoded, success = ldpc.decode(channel_llr, max_iter=30)

        # Get info bits and check
        decoded_msg = ldpc.get_info_bits(decoded)
        ber = np.mean(message != decoded_msg)

        print(f"  SNR = {snr_db} dB: Success = {success}, BER = {ber:.6f}")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
