"""Regression fixtures for numerical transparency versus source-read policy."""
import copy
import tempfile
import unittest
from pathlib import Path
from compare_cache_results import compare_reports


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cold = Path(self.directory.name) / 'cold'; self.cold.mkdir()
        self.warm = Path(self.directory.name) / 'warm'; self.warm.mkdir()
        for directory in (self.cold, self.warm):
            (directory / 'streamed-0.f32').write_bytes(b'\x00\x00\x80\x3f')
        metrics = {'source_read_calls': 10, 'source_bytes_read': 1000, 'weight_read_calls': 10,
                   'weight_bytes_requested': 500, 'reader_buffer_bytes': 64,
                   'cache_allocated_bytes': 0, 'cache_block_hits': 0, 'cache_block_misses': 0,
                   'cache_evictions': 0, 'cache_release_events': 0}
        self.a = {'pass': True, 'model_sha256': 'trusted', 'binary_sha256': 'binary', 'git_sha': 'source',
                  'address_space_limit_bytes': None, 'cache_budget_bytes': 0,
                  'verified_session_requested': False,
                  'trials': [{'pass': True, 'prompt': 'fixture', 'streamed_text': 'fixture output',
                              'reader_metrics': metrics, 'memory': {'peak_rss_bytes': 100},
                              'integrity_pass': True,
                              'integrity_metrics': {'verified': '1', 'sha256': 'trusted', 'blocks_verified': '10'}}]}
        self.b = copy.deepcopy(self.a); self.b['cache_budget_bytes'] = 128
        self.b['trials'][0]['reader_metrics']['cache_allocated_bytes'] = 100

    def result(self, bypass=False):
        return compare_reports(self.a, self.b, self.cold, self.warm, bypass)['pass']

    def verified(self):
        self.a['verified_session_requested'] = self.b['verified_session_requested'] = True

    def test_unverified_mode_keeps_strict_read_savings(self):
        self.assertFalse(self.result())
        self.b['trials'][0]['reader_metrics']['source_read_calls'] = 5
        self.assertTrue(self.result())

    def test_authenticated_bypass_requires_explicit_policy(self):
        self.verified()
        self.assertFalse(self.result())
        self.assertTrue(self.result(True))

    def test_bypass_rejects_cache_hits_or_missing_fresh_weight_reads(self):
        self.verified()
        for field, value in [('cache_block_hits', 1), ('cache_block_misses', 1),
                             ('weight_read_calls', 9), ('source_read_calls', 9), ('cache_release_events', 1)]:
            old = self.b['trials'][0]['reader_metrics'][field]
            self.b['trials'][0]['reader_metrics'][field] = value
            self.assertFalse(self.result(True), field)
            self.b['trials'][0]['reader_metrics'][field] = old

    def test_bypass_rejects_wrong_missing_or_insufficient_integrity(self):
        self.verified()
        for field, value in [('sha256', 'wrong'), ('verified', '0'), ('blocks_verified', '0')]:
            old = self.b['trials'][0]['integrity_metrics'][field]
            self.b['trials'][0]['integrity_metrics'][field] = value
            self.assertFalse(self.result(True), field)
            self.b['trials'][0]['integrity_metrics'][field] = old
        self.b['verified_session_requested'] = False
        self.assertFalse(self.result(True))

    def test_output_logits_and_reference_configuration_stay_strict(self):
        self.verified()
        self.b['trials'][0]['streamed_text'] = 'different'
        self.assertFalse(self.result(True))
        self.b['trials'][0]['streamed_text'] = self.a['trials'][0]['streamed_text']
        (self.warm / 'streamed-0.f32').write_bytes(b'changed')
        self.assertFalse(self.result(True))
        (self.warm / 'streamed-0.f32').write_bytes((self.cold / 'streamed-0.f32').read_bytes())
        self.b['reference_weight_repacking'] = 'different'
        self.assertFalse(self.result(True))

    def test_empty_logit_files_cannot_pass(self):
        self.verified()
        for directory in (self.cold, self.warm):
            (directory / 'streamed-0.f32').write_bytes(b'')
        self.assertFalse(self.result(True))


if __name__ == '__main__':
    unittest.main()
