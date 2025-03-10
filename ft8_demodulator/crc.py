"""
CRC计算模块，用于FT8/FT4消息的CRC校验和计算和验证。
基于 https://barrgroup.com/Embedded-Systems/How-To/CRC-Calculation-C-Code 的实现。
"""

# FT8/FT4使用的CRC常量
CRC_WIDTH = 14
CRC_POLYNOMIAL = 0x2757  # CRC-14多项式，不包括最高位的1
TOPBIT = 1 << (CRC_WIDTH - 1)

def compute_crc(message: bytearray, num_bits: int) -> int:
    """
    计算FT8/FT4消息的CRC校验和
    
    Args:
        message: 包含消息位的字节数组（MSB优先）
        num_bits: 要计算CRC的位数
    
    Returns:
        14位CRC校验和
    """
    remainder = 0
    idx_byte = 0
    
    # 一次处理一位，完全匹配C代码的实现
    for idx_bit in range(num_bits):
        if idx_bit % 8 == 0:
            # 将下一个字节移入余数的高位
            remainder ^= (message[idx_byte] << (CRC_WIDTH - 8))
            idx_byte += 1
        
        # 处理当前位
        if remainder & TOPBIT:
            remainder = (remainder << 1) ^ CRC_POLYNOMIAL
        else:
            remainder = (remainder << 1)
    
    # 最后应用掩码，与C代码完全一致
    return remainder & ((TOPBIT << 1) - 1)

def extract_crc(a91: bytearray) -> int:
    """
    从91位FT8/FT4消息中提取CRC校验和
    
    Args:
        a91: 包含91位消息的字节数组
    
    Returns:
        14位CRC校验和
    """
    # CRC位于77-91位之间
    # 字节9的后3位 + 字节10的全部8位 + 字节11的前3位
    chksum = ((a91[9] & 0x07) << 11) | (a91[10] << 3) | (a91[11] >> 5)
    return chksum

def add_crc(payload: bytearray, a91: bytearray) -> None:
    """
    将CRC添加到FT8/FT4消息中
    
    Args:
        payload: 输入的77位消息
        a91: 输出的91位消息（包含CRC）
    """
    # 复制77位载荷数据
    for i in range(10):
        a91[i] = payload[i]
        
    # 清除载荷后的位，准备进行CRC计算
    a91[9] &= 0xF8  # 保留前5位（77-72=5位）
    a91[10] = 0
    a91[11] = 0
    
    # 计算CRC
    checksum = compute_crc(a91, 82)  # 计算前82位的CRC
    
    # 将CRC存储在77位消息的末尾
    a91[9] |= (checksum >> 11)         # 存储最高3位
    a91[10] = (checksum >> 3) & 0xFF   # 存储中间8位
    a91[11] = (checksum << 5) & 0xE0   # 存储最低3位，左移5位