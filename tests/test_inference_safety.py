"""CPU regressions for causal inference; fixtures are not music experiments."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from infer import CachedDecoder, generate_sequence
from model import ModelConfig, MusicTransformer
from music_codec import Vocabulary


class InferenceSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, context=128):
        torch.manual_seed(14942154)
        return MusicTransformer(ModelConfig(Vocabulary().size, width=32, layers=2,
                                            heads=4, context=context, dropout=0.0)).eval()

    def test_kv_cache_matches_full_forward_for_single_tokens_and_chunks(self):
        model = self.make_model()
        ids = torch.tensor([[1, 3, 7, 11, 5, 19, 22, 28, 35, 41, 44, 50, 2]])
        with torch.inference_mode():
            expected, _, _ = model(ids)
            for chunk_sizes in ([1] * ids.shape[1], [5, 1, 7], [3, 4, 2, 4]):
                past, pieces, position = None, [], 0
                for length in chunk_sizes:
                    actual, _, past = model(ids[:, position:position + length], past=past, use_cache=True)
                    pieces.append(actual)
                    position += length
                torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-5, atol=1e-6)

    def test_future_tokens_cannot_change_prefix_logits(self):
        model = self.make_model()
        a = torch.tensor([[1, 3, 7, 11, 5, 19, 22, 28, 35, 41]])
        b = a.clone()
        b[:, 6:] = torch.tensor([[81, 82, 83, 84]])
        with torch.inference_mode():
            scores_a, _, _ = model(a)
            scores_b, _, _ = model(b)
        torch.testing.assert_close(scores_a[:, :6], scores_b[:, :6], rtol=0, atol=0)
        self.assertFalse(torch.equal(scores_a[:, 6:], scores_b[:, 6:]))

    def test_context_overflow_is_explicit(self):
        model = self.make_model(context=8)
        with torch.inference_mode():
            _, _, past = model(torch.tensor([[1, 3, 7, 11, 5, 19, 22]]), use_cache=True)
            with self.assertRaisesRegex(ValueError, "Context overflow"):
                model(torch.tensor([[28, 35]]), past=past, use_cache=True)

    def test_cached_decoder_rollover_starts_at_a_complete_bar(self):
        model = self.make_model(context=12)
        decoder = CachedDecoder(model, "cpu")
        decoder.feed([1, 3, 7, 8, 9, 3, 10, 11, 12, 13])
        actual = decoder.feed([14, 15, 16])
        # The retained eleven-token window begins inside bar one. Re-prefill
        # must skip that incomplete bar and begin with the next Bar token.
        retained = [1, 3, 10, 11, 12, 13, 14, 15, 16]
        with torch.inference_mode():
            expected, _, _ = model(torch.tensor([retained]))
        torch.testing.assert_close(torch.from_numpy(actual), expected[0, -1], rtol=1e-5, atol=1e-6)
        self.assertEqual(decoder.reprefills, 1)

    def test_prompt_notes_do_not_hide_an_empty_continuation(self):
        vocab = Vocabulary()
        prefix = [vocab.id(name) for name in
                  ("<BOS>", "Bar", "Meter_4", "Tempo_120", "Pos_0", "Pitch_60", "Vel_16", "Dur_4", "<EOS>")]
        with patch("infer.sample_token", return_value=vocab.bar_id):
            with self.assertRaisesRegex(ValueError, "continuation contains no new notes"):
                generate_sequence(self.make_model(), {}, vocab, seed=0, bars=2, prefix=prefix)

    def test_unconditional_empty_generation_is_rejected(self):
        vocab = Vocabulary()
        with patch("infer.sample_token", return_value=vocab.bar_id):
            with self.assertRaisesRegex(ValueError, "empty piece"):
                generate_sequence(self.make_model(), {}, vocab, seed=0, bars=2)

    def test_generated_note_count_excludes_the_prompt(self):
        vocab = Vocabulary()
        prefix = [vocab.id(name) for name in
                  ("<BOS>", "Bar", "Meter_4", "Tempo_120", "Pos_0", "Pitch_60", "Vel_16", "Dur_4", "<EOS>")]
        generated = [vocab.id(name) for name in ("Pos_0", "Pitch_64", "Vel_16", "Dur_4", "Bar")]
        with patch("infer.sample_token", side_effect=generated):
            _, diagnostics = generate_sequence(self.make_model(), {}, vocab, seed=0, bars=2, prefix=prefix)
        self.assertEqual(diagnostics["n_notes"], 2)
        self.assertEqual(diagnostics["generated_note_count"], 1)


if __name__ == "__main__":
    unittest.main()
