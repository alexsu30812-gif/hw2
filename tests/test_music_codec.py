"""Musically meaningful clock, role, overlap and prompt-preservation checks."""
import sys
from pathlib import Path
import tempfile
import unittest

import mido
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_codec import BeatMap, Vocabulary, decode_tokens, parse_tokens, preserve_prompt_notes, tokenize_midi

ROOT = Path(__file__).resolve().parents[1]


class MusicCodecTests(unittest.TestCase):
    def test_vocabulary_is_fixed_and_complete(self):
        a, b = Vocabulary(), Vocabulary()
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(len(set(a.tokens)), a.size)
        self.assertEqual(a.size, 683)

    def test_annotation_clock_and_roles_on_real_prompts(self):
        for folder in sorted((ROOT / "POP909/prompts/basic-b").iterdir()):
            source = next(folder.glob("*.mid"))
            song = tokenize_midi(source, folder / "beat_midi.txt", max_bars=8)
            self.assertEqual(song.metadata["n_bars"], 8)
            self.assertEqual(len(song.bar_offsets), 9)
            self.assertEqual(parse_tokens(song.tokens)["invalid_events"], 0)
            for seconds in song.beat_map.beat_times:
                self.assertAlmostEqual(song.beat_map.beat_to_time(song.beat_map.time_to_beat(seconds)), seconds)
        for folder in sorted((ROOT / "POP909/prompts/bonus").iterdir()):
            song = tokenize_midi(folder / "melody.mid", folder / "beat_midi.txt", mode="roles", max_bars=16)
            self.assertEqual(song.metadata["n_bars"], 16)
            self.assertTrue(all(n["track"] == "Melody" for n in parse_tokens(song.tokens)["notes"]))

    def test_variable_meter_tempo_and_overlapping_same_pitch(self):
        vocab = Vocabulary()
        names = ["<BOS>", "Bar", "Meter_3", "Tempo_60",
                 "Pos_0", "Pitch_60", "Vel_16", "Dur_16",
                 "Pos_4", "Pitch_60", "Vel_20", "Dur_4",
                 "Bar", "Meter_6", "Tempo_120", "Pos_0", "Pitch_72", "Vel_25", "Dur_8", "<EOS>"]
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "synthetic.mid"
            decoded = decode_tokens([vocab.id(name) for name in names], target)
            notes = [n for inst in pretty_midi.PrettyMIDI(str(target)).instruments for n in inst.notes]
            self.assertEqual(len(notes), 3)
            self.assertAlmostEqual(decoded["duration_sec"], 6.)
            self.assertAlmostEqual(mido.MidiFile(str(target)).length, 6., places=5)
            high = next(note for note in notes if note.pitch == 72)
            self.assertAlmostEqual(high.start, 3., places=5)
            self.assertAlmostEqual(high.end, 3.5, places=5)

    def test_external_annotation_map_preserves_rubato_seconds(self):
        vocab = Vocabulary()
        names = ["<BOS>", "Bar", "Meter_2", "Tempo_120", "Pos_8", "Pitch_60", "Vel_16", "Dur_8", "<EOS>"]
        beat_map = BeatMap([0.1, 0.6, 1.6], [0, 2])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "rubato.mid"
            decode_tokens([vocab.id(name) for name in names], target, beat_map=beat_map)
            note = pretty_midi.PrettyMIDI(str(target)).instruments[0].notes[0]
            self.assertAlmostEqual(note.start, .6, places=5)
            self.assertAlmostEqual(note.end, 1.6, places=5)
            self.assertAlmostEqual(mido.MidiFile(str(target)).length, 1.6, places=5)

    def test_original_prompt_notes_and_silent_endpoint_survive(self):
        folder = ROOT / "POP909/prompts/basic-b/001"
        original = folder / "001.mid"
        encoded = tokenize_midi(original, folder / "beat_midi.txt", max_bars=8)
        vocab = Vocabulary()
        # Add 24 empty bars to exercise preservation of a long final rest.
        tokens = encoded.tokens[:-1]
        for _ in range(24):
            tokens += [vocab.bar_id, vocab.id("Meter_4"), vocab.id("Tempo_64")]
        tokens.append(vocab.eos_id)
        with tempfile.TemporaryDirectory() as temp:
            generated, merged = Path(temp) / "generated.mid", Path(temp) / "merged.mid"
            decode_tokens(tokens, generated)
            cutoff = encoded.beat_map.bar_times[8]
            info = preserve_prompt_notes(original, generated, merged, cutoff)
            src_notes = sorted((n.pitch, n.velocity, n.start, n.end) for inst in pretty_midi.PrettyMIDI(str(original)).instruments
                               for n in inst.notes if n.start < cutoff - 1e-7)
            out_notes = sorted((n.pitch, n.velocity, n.start, n.end) for inst in pretty_midi.PrettyMIDI(str(merged)).instruments
                               for n in inst.notes if n.start < cutoff - 1e-7)
            self.assertEqual(len(src_notes), info["prefix_notes"])
            self.assertEqual(len(src_notes), len(out_notes))
            for source, output in zip(src_notes, out_notes):
                self.assertEqual(source[:2], output[:2])
                self.assertLessEqual(abs(source[2] - output[2]), .001)
                self.assertLessEqual(abs(source[3] - output[3]), .001)
            self.assertAlmostEqual(mido.MidiFile(str(generated)).length, mido.MidiFile(str(merged)).length, places=4)
            # The production continuation caller replaces its file in place.
            before = mido.MidiFile(str(generated)).length
            preserve_prompt_notes(original, generated, generated, cutoff)
            self.assertAlmostEqual(mido.MidiFile(str(generated)).length, before, places=4)


if __name__ == "__main__":
    unittest.main()
