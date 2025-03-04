import numpy as np

from .abstract import BenchmarkStateVector


class BalancedHammingWeight(BenchmarkStateVector):
    # ref : https://mathoverflow.net/questions/301733/how-to-create-a-quantum-algorithm-that-produces-2-n-bit-sequences-with-equal-num
    # authored by : seungmin.jeon@kaist.ac.kr
    """
    Construct state vector for bit measure state (number of ones on the left half and number of ones on the right half are equal)

    Example
    -------
    n: 2 -> 00, 11
    n: 4 -> 0000, 0101, 0110, 1001, 1010, 1111
    n: 6 -> 000000, 001001, 001010, 001100, 010001, 010010, 010100, 100001, 100010, 100100, 011011, 011101, 011110, 101011, 101101, 101110, 110011, 110101, 110110, 111111
    """

    def __call__(self, n: int):
        sv = np.zeros(shape=(2**n,))

        valid_indices = []
        for i in range(2**n):
            bit_string = bin(i)[2:].zfill(n)  # Convert to binary and pad with zeros
            left_half = bit_string[: n // 2]
            right_half = bit_string[n // 2 :]

            if left_half.count("1") == right_half.count("1"):
                valid_indices.append(i)

        norm_factor = 1 / np.sqrt(len(valid_indices))
        for idx in valid_indices:
            sv[idx] = norm_factor

        return sv
