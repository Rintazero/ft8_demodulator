"""
CRC计算模块，用于FT8/FT4消息的CRC校验和计算和验证。
"""

# FT8/FT4使用的CRC多项式: x^14 + x^13 + x^11 + x^10 + x^9 + x^8 + x^6 + x^5 + x^4 + x^1 + 1
# 表示为二进制: 0b11011001110011 (0x36E3)
CRC_POLYNOMIAL = 0x36E3
CRC_WIDTH = 14

def compute_crc(data: bytearray, num_bits: int) -> int:
    """
    计算FT8/FT4消息的CRC校验和
    
    Args:
        data: 包含消息位的字节数组
        num_bits: 要计算CRC的位数
    
    Returns:
        14位CRC校验和
    """
    # 初始化CRC寄存器为全1
    reg = (1 << CRC_WIDTH) - 1
    
    # 处理每一位
    for i in range(num_bits):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)  # MSB优先
        
        # 获取当前位
        bit = (data[byte_idx] >> bit_idx) & 1
        
        # 计算反馈位
        feedback_bit = ((reg >> (CRC_WIDTH - 1)) ^ bit) & 1
        
        # 更新CRC寄存器
        reg = ((reg << 1) & ((1 << CRC_WIDTH) - 1)) | feedback_bit
        
        # 如果反馈位为1，则与多项式异或
        if feedback_bit:
            reg ^= CRC_POLYNOMIAL
    
    # 返回最终的CRC值
    return reg & ((1 << CRC_WIDTH) - 1)

def extract_crc(data: bytearray) -> int:
    """
    从FT8/FT4消息中提取CRC校验和
    
    Args:
        data: 包含消息和CRC的字节数组
    
    Returns:
        提取的14位CRC校验和
    """
    # CRC位于77-91位之间
    crc = 0
    
    # 提取CRC位 (77-90)
    for i in range(77, 77 + CRC_WIDTH):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)  # MSB优先
        
        bit = (data[byte_idx] >> bit_idx) & 1
        crc = (crc << 1) | bit
    
    return crc

def check_crc(data: bytearray) -> bool:
    """
    检查FT8/FT4消息的CRC是否有效
    
    Args:
        data: 包含消息和CRC的字节数组
    
    Returns:
        如果CRC有效则为True，否则为False
    """
    extracted_crc = extract_crc(data)
    
    # 计算前77位的CRC
    calculated_crc = compute_crc(data, 77)
    
    return extracted_crc == calculated_crc 