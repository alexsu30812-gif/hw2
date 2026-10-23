"""Tiny random models test the pipeline; their outputs are not music results."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_bonus import generate_accompaniment, export_with_original_melody
from model import ModelConfig, MusicTransformer
from music_codec import Vocabulary, parse_tokens, tokenize_midi

ROOT = Path(__file__).resolve().parents[1]


class BonusGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.vocab = Vocabulary()
        cls.model = MusicTransformer(ModelConfig(cls.vocab.size, 32, 1, 2, 256, 0.)).eval()
        cls.metadata = {"inference_context": 128}
        cls.prompt = ROOT / "POP909/prompts/bonus/001"
        cls.encoded = tokenize_midi(cls.prompt / "melody.mid", cls.prompt / "beat_midi.txt", mode="roles", max_bars=16)

    def generate(self, seed=42, bars=16):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_accompaniment(self.model, self.metadata, self.encoded, self.vocab,
                                           seed, bars=bars, max_notes_per_bar=4)

    def test_seed_reproducibility_and_nonempty_samples(self):
        a, _ = self.generate(bars=2)
        b, _ = self.generate(bars=2)
        c, _ = self.generate(seed=43, bars=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_cached_sampling_roles_and_same_pitch_overlap(self):
        tokens, diagnostics = self.generate()
        parsed = parse_tokens(tokens)
        self.assertEqual(len(parsed["bars"]), 16)
        self.assertEqual(parsed["invalid_events"], 0)
        self.assertGreater(diagnostics["context_reprefills"], 0)
        active = {}
        for note in parsed["notes"]:
            if note["track"] != "Piano":
                continue
            start = self.encoded.beat_map.bar_indices[note["bar"]] + note["pos"] / 8
            self.assertGreaterEqual(start + 1e-9, active.get(note["pitch"], -1))
            active[note["pitch"]] = start + note["duration"] / 8
            self.assertLessEqual(active[note["pitch"]], self.encoded.beat_map.bar_indices[16])

    def test_full_export_keeps_exact_original_melody_and_two_tracks(self):
        tokens, _ = self.generate()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "001.mid"
            _, audit = export_with_original_melody(tokens, self.prompt / "melody.mid", self.encoded.beat_map,
                                                   target, self.vocab)
            readback = pretty_midi.PrettyMIDI(str(target))
            self.assertEqual([inst.name for inst in readback.instruments], ["MELODY", "PIANO"])
            self.assertEqual(audit["notes"], 75)
            self.assertTrue(audit["verified"])
            self.assertLess(audit["max_timing_error_seconds"], .002)


if __name__ == "__main__":
    unittest.main()
