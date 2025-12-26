# 2×2 MIMO-OFDM System with LDPC Coding

Mô phỏng hệ thống truyền thông không dây 2×2 MIMO-OFDM sử dụng mã hóa LDPC trên kênh Rayleigh fading với nhiễu trắng Gaussian (AWGN).

## Tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRANSMITTER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Info Bits → LDPC Encoder → 64-QAM Mapper → OFDM Modulator → TX Ant 1  │
│                                                              → TX Ant 2  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MIMO CHANNEL                                     │
│                   Rayleigh Fading + AWGN                                 │
│                                                                          │
│                      ┌───────────────┐                                   │
│         TX1 ───────→ │  h11    h12  │ ───────→ RX1                      │
│         TX2 ───────→ │  h21    h22  │ ───────→ RX2                      │
│                      └───────────────┘                                   │
│                           + AWGN                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            RECEIVER                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  RX Ant 1 → OFDM Demod → MIMO Detect → 64-QAM Demap → LDPC Decode → Bits│
│  RX Ant 2 →                (ZF/MMSE)                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Thành phần hệ thống

### 1. Điều chế 64-QAM (`src/qam_modulation.py`)
- Constellation 8×8 với Gray mapping
- Hỗ trợ hard và soft demodulation
- Tính LLR cho LDPC decoding

### 2. Mã hóa LDPC (`src/ldpc_codec.py`)
- Mã LDPC regular với rate 1/2
- Giải mã Sum-Product và Min-Sum
- Hỗ trợ kích thước codeword linh hoạt

### 3. OFDM (`src/ofdm_system.py`)
- IFFT/FFT modulation
- Cyclic Prefix để chống ISI
- Pilot insertion cho channel estimation

### 4. Kênh MIMO Rayleigh (`src/mimo_channel.py`)
- 2×2 MIMO channel
- Rayleigh fading (flat và frequency-selective)
- AWGN noise

### 5. MIMO Detection (`src/mimo_detector.py`)
- Zero-Forcing (ZF)
- MMSE
- ZF-SIC và MMSE-SIC
- Maximum Likelihood (cho constellation nhỏ)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy mô phỏng

```bash
python simulation.py
```

## Kết quả

Chương trình sẽ:
1. Chạy mô phỏng BER/SER qua dải SNR 0-25 dB
2. So sánh ZF và MMSE detection
3. Đánh giá coding gain của LDPC
4. Xuất đồ thị BER và SER

## Tham số mặc định

| Tham số | Giá trị |
|---------|---------|
| MIMO | 2×2 |
| FFT Size | 64 |
| Cyclic Prefix | 16 |
| Modulation | 64-QAM |
| LDPC Rate | 1/2 |
| Channel | Rayleigh + AWGN |

## Cấu trúc thư mục

```
.
├── src/
│   ├── __init__.py
│   ├── qam_modulation.py    # 64-QAM modulator/demodulator
│   ├── ldpc_codec.py        # LDPC encoder/decoder
│   ├── ofdm_system.py       # OFDM system
│   ├── mimo_channel.py      # MIMO Rayleigh channel
│   └── mimo_detector.py     # MIMO detectors (ZF, MMSE)
├── simulation.py             # Main simulation script
├── requirements.txt
└── README.md
```

## Lý thuyết

### MIMO Channel Model
```
y = H·x + n
```
- `y`: Vector tín hiệu nhận (Nr × 1)
- `H`: Ma trận kênh (Nr × Nt), h_ij ~ CN(0,1)
- `x`: Vector tín hiệu phát (Nt × 1)
- `n`: Nhiễu AWGN, n ~ CN(0, σ²I)

### LDPC Decoding
- Sum-Product Algorithm (SPA)
- Min-Sum với scaling factor
- Soft-decision based on LLR

### 64-QAM
- 6 bits per symbol
- Gray mapping để giảm BER
- Normalized average power = 1
