"""Tiny random-model fixtures test ablation behavior, never musical quality."""
from argparse import Namespace
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pretty_midi
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_bonus import (melody_condition_bars, generate_accompaniment,
                            export_with_original_melody, generate_bonus)
from model import ModelConfig, MusicTransformer, save_checkpoint
from music_codec import Vocabulary, parse_tokens, tokenize_midi

ROOT = Path(__file__).resolve().parents[1]


class BonusAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.vocab = Vocabulary()
        cls.model = MusicTransformer(ModelConfig(cls.vocab.size, 16, 1, 2, 256, 0.)).eval()
        cls.metadata = {"inference_context": 128}
        cls.prompt_dir = ROOT / "POP909/prompts/bonus"
        cls.prompt = cls.prompt_dir / "001"
        cls.encoded = tokenize_midi(cls.prompt / "melody.mid", cls.prompt / "beat_midi.txt",
                                    mode="roles", max_bars=16)

    def test_empty_condition_removes_only_melody_notes(self):
        normal = melody_condition_bars(self.encoded, self.vocab)
        default = melody_condition_bars(self.encoded, self.vocab, conditioning="normal")
        empty = melody_condition_bars(self.encoded, self.vocab, conditioning="empty-melody")
        self.assertEqual(normal, default)
        self.assertTrue(any(len(bar) > 5 for bar in normal))
        for original, ablated in zip(normal, empty):
            self.assertEqual(ablated, original[:4] + [self.vocab.id("Track_Piano")])
            self.assertEqual(len(ablated), 5)
        self.assertGreater(len(parse_tokens([self.vocab.bos_id] + sum(normal, []))["notes"]), 0)
        self.assertEqual(parse_tokens([self.vocab.bos_id] + sum(empty, []))["notes"], [])

    def test_empty_condition_export_keeps_original_melody(self):
        with contextlib.redirect_stdout(io.StringIO()):
            tokens, diagnostic = generate_accompaniment(
                self.model, self.metadata, self.encoded, self.vocab, 42,
                max_notes_per_bar=1, min_notes_per_bar=1, conditioning="empty-melody")
        parsed = parse_tokens(tokens)
        self.assertFalse(any(n["track"] == "Melody" for n in parsed["notes"]))
        self.assertEqual(diagnostic["conditioning"], "empty-melody")
        self.assertEqual(diagnostic["generated_piano_notes"], 16)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "empty.mid"
            _, audit = export_with_original_melody(tokens, self.prompt / "melody.mid",
                                                   self.encoded.beat_map, target, self.vocab)
            output = pretty_midi.PrettyMIDI(str(target))
            self.assertEqual([inst.name for inst in output.instruments], ["MELODY", "PIANO"])
            self.assertTrue(audit["verified"])
            self.assertEqual(audit["notes"], 75)
            self.assertLess(audit["max_timing_error_seconds"], .002)

    def test_full_ablation_manifest_is_ineligible_and_default_remains_normal(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            # Synthetic provenance tests the eligibility gate. These tiny random
            # fixtures and generated files remain inside this temporary folder.
            metadata = dict(self.metadata, mode="roles", vocabulary_sha256=self.vocab.sha256,
                            all_official_training_songs_used=True,
                            seen_song_ids=[f"{i:03d}" for i in range(1, 897)])
            checkpoint = folder / "fixture.pt"
            save_checkpoint(checkpoint, self.model, metadata)
            common = dict(checkpoint=checkpoint, prompt_dir=self.prompt_dir, temperature=1.,
                          top_k=0, top_p=.95, seed=42, device="cpu", render=False,
                          max_notes_per_bar=1, min_notes_per_bar=1)
            default_args = Namespace(**common, output_dir=folder / "normal")
            empty_args = Namespace(**common, output_dir=folder / "empty", conditioning="empty-melody")
            with contextlib.redirect_stdout(io.StringIO()):
                normal = generate_bonus(default_args)
                empty = generate_bonus(empty_args)
            self.assertEqual(normal["conditioning"], "normal")
            self.assertTrue(normal["submission_eligible"])
            self.assertTrue(empty["complete"])
            self.assertEqual(len(empty["samples"]), 10)
            self.assertEqual(empty["conditioning"], "empty-melody")
            self.assertFalse(empty["submission_eligible"])
            self.assertIn("out-of-distribution", empty["interpretation_limitations"])
            self.assertTrue(all(s["melody_verification"]["verified"] for s in empty["samples"]))


if __name__ == "__main__":
    unittest.main()
