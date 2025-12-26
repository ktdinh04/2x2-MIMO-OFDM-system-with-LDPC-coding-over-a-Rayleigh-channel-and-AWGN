"""
2x2 MIMO-OFDM System with LDPC Coding
=====================================

Hệ thống mô phỏng truyền thông không dây bao gồm:
- 2×2 MIMO (Multiple Input Multiple Output)
- OFDM (Orthogonal Frequency Division Multiplexing)
- LDPC (Low-Density Parity-Check) coding
- 64-QAM modulation
- Rayleigh fading channel with AWGN
"""

from .qam_modulation import QAM64
from .ldpc_codec import LDPCCode, LDPCCodec
from .ofdm_system import OFDMSystem, MIMOOFDMSystem
from .mimo_channel import RayleighMIMOChannel, FrequencyFlatMIMOChannel
from .mimo_detector import MIMODetector, MIMOOFDMDetector

__all__ = [
    'QAM64',
    'LDPCCode',
    'LDPCCodec',
    'OFDMSystem',
    'MIMOOFDMSystem',
    'RayleighMIMOChannel',
    'FrequencyFlatMIMOChannel',
    'MIMODetector',
    'MIMOOFDMDetector',
]
