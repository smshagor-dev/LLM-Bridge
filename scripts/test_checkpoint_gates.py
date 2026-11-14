import unittest
from check_streamed_parity import CHECKPOINTS, validate_checkpoint, validate_proof_cap


class CheckpointGateTests(unittest.TestCase):
    def test_q4_requires_its_exact_explicit_identity(self):
        validate_checkpoint('stories15M-derived-q4_0', CHECKPOINTS['stories15M-derived-q4_0'])
        with self.assertRaises(ValueError):
            validate_checkpoint('stories15M-q8_0', CHECKPOINTS['stories15M-derived-q4_0'])
        with self.assertRaises(ValueError):
            validate_checkpoint('stories15M-derived-q4_0', CHECKPOINTS['stories15M-q8_0'])

    def test_unknown_identity_or_hash_never_reaches_golden_execution(self):
        for name, digest in [('unknown', CHECKPOINTS['stories15M-q8_0']),
                             ('stories15M-derived-q4_0', '0' * 64)]:
            with self.assertRaises(ValueError):
                validate_checkpoint(name, digest)

    def test_smaller_q4_file_cannot_be_claimed_as_the_24_mib_proof(self):
        validate_proof_cap(24, 26671328, 'linux')
        validate_proof_cap(0, 14469344, 'linux')
        for size, system in [(14469344, 'linux'), (24 * 1024 * 1024, 'linux'),
                             (26671328, 'darwin')]:
            with self.assertRaises(ValueError):
                validate_proof_cap(24, size, system)
        with self.assertRaises(ValueError):
            validate_proof_cap(-1, 26671328, 'linux')


if __name__ == '__main__':
    unittest.main()
