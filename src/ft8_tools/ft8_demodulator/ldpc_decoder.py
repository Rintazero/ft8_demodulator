import numpy as np
from typing import Tuple, List
from .constants import (
    FTX_LDPC_M,
    FTX_LDPC_N,
    kFTX_LDPC_Mn,
    kFTX_LDPC_Nm,
    kFTX_LDPC_Num_rows
)

def fast_tanh(x: np.ndarray) -> np.ndarray:
    """
    快速tanh近似实现
    使用有理分式近似tanh函数
    """
    # 限制输入范围
    x = np.clip(x, -4.97, 4.97)
    x2 = x * x
    a = x * (945.0 + x2 * (105.0 + x2))
    b = 945.0 + x2 * (420.0 + x2 * 15.0)
    return a / b

def fast_atanh(x: np.ndarray) -> np.ndarray:
    """
    快速atanh近似实现
    使用有理分式近似atanh函数
    """
    x2 = x * x
    a = x * (945.0 + x2 * (-735.0 + x2 * 64.0))
    b = (945.0 + x2 * (-1050.0 + x2 * 225.0))
    return a / b

def ldpc_check(codeword: np.ndarray) -> int:
    """
    检查174位码字是否通过FT8的LDPC奇偶校验
    
    Args:
        codeword: 174位码字
    
    Returns:
        int: 奇偶校验错误的数量，0表示完全正确
    """
    errors = 0
    
    for m in range(FTX_LDPC_M):
        x = 0
        for i in range(kFTX_LDPC_Num_rows[m]):
            x ^= codeword[kFTX_LDPC_Nm[m][i] - 1]
        if x != 0:
            errors += 1
            
    return errors

def bp_decode(codeword: np.ndarray, max_iterations: int) -> Tuple[np.ndarray, int]:
    """
    使用信念传播(Belief Propagation)算法进行LDPC解码
    
    Args:
        codeword: 174个对数似然比值
        max_iterations: 最大迭代次数
        
    Returns:
        Tuple[np.ndarray, int]: (解码后的码字, 错误数)
    """
    # 初始化消息数组
    tov = np.zeros((FTX_LDPC_N, 3))  # 变量节点到校验节点的消息
    toc = np.zeros((FTX_LDPC_M, 7))  # 校验节点到变量节点的消息
    plain = np.zeros(FTX_LDPC_N, dtype=np.uint8)
    min_errors = FTX_LDPC_M

    for iter in range(max_iterations):
        # 硬判决猜测
        messages = codeword + np.sum(tov, axis=1)
        plain = (messages > 0).astype(np.uint8)
        
        if np.sum(plain) == 0:
            # 消息收敛到全零，这是被禁止的
            break
            
        # 检查是否得到有效码字
        errors = ldpc_check(plain)
        
        if errors < min_errors:
            min_errors = errors
            if errors == 0:
                break  # 找到完美解
                
        # 从变量节点发送消息到校验节点
        for m in range(FTX_LDPC_M):
            for n_idx in range(kFTX_LDPC_Num_rows[m]):
                n = kFTX_LDPC_Nm[m][n_idx] - 1
                # 对每个(n,m)
                Tnm = codeword[n]
                for m_idx in range(3):
                    if (kFTX_LDPC_Mn[n][m_idx] - 1) != m:
                        Tnm += tov[n][m_idx]
                toc[m][n_idx] = fast_tanh(-Tnm / 2)
                
        # 从校验节点发送消息到变量节点
        for n in range(FTX_LDPC_N):
            for m_idx in range(3):
                m = kFTX_LDPC_Mn[n][m_idx] - 1
                # 对每个(n,m)
                Tmn = 1.0
                for n_idx in range(kFTX_LDPC_Num_rows[m]):
                    if (kFTX_LDPC_Nm[m][n_idx] - 1) != n:
                        Tmn *= toc[m][n_idx]
                tov[n][m_idx] = -2 * fast_atanh(Tmn)
    

    # print(f"解码后的码字: {plain}")
    # print(f"错误数: {min_errors}")
    return plain, min_errors 