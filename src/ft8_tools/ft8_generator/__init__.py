from .crc import crc_generator, get_crc_from_a91
from .encoder import symbolIdSequence_generator, itones_generator, ft8_encode
from .modulator import gfsk_modulation_waveform_generator, ft8_modulation_waveform_generator, ft8_baseband_generator, ft8_generator

# Optionally, you can define what should be accessible when the module is imported
__all__ = ['crc_generator', 'get_crc_from_a91', 'symbolIdSequence_generator', 'itones_generator', 'ft8_encode', 'gfsk_modulation_waveform_generator', 'ft8_modulation_waveform_generator', 'ft8_baseband_generator', 'ft8_generator']

# You can also add any initialization code or helper functions here if needed

def generate_crc(data):
    """
    A helper function to generate CRC for the given data using crc_generator.
    
    :param data: The input data for which CRC needs to be generated.
    :return: The generated CRC value.
    """
    return crc_generator(data)

def get_crc_from_a91(a91_12bytes):
    """
    A helper function to get CRC from a91_12bytes using get_crc_from_a91.
    
    :param a91_12bytes: The input data for which CRC needs to be generated.
    :return: The generated CRC value.
    """
    return get_crc_from_a91(a91_12bytes)

# Example of how you might initialize or configure something when the module is imported
def initialize_module():
    """
    Perform any necessary initialization for the ft8_generator module.
    """
    print("FT8 Generator module initialized.")

# Call the initialization function
initialize_module()


