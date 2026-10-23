"""Prevent a resumed run from silently mixing models or modified MIDI files."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_pipeline import valid_generation


class ResumeIntegrityTests(unittest.TestCase):
    def test_completed_manifest_rejects_replaced_model_and_modified_music(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, midi, manifest = (root / name for name in ('own.pt', 'piece.mid', 'manifest.json'))
            checkpoint.write_bytes(b'own checkpoint version one')
            midi.write_bytes(b'generated midi bytes')
            configuration = {'temperature': .8, 'seed': 10, 'device': 'cpu'}
            payload = {'complete': True, 'task': 'unconditional', 'configuration': configuration,
                       'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       'samples': [{'song_name': 'sample_001', 'bars': 32, 'midi_path': str(midi),
                                    'midi_sha256': hashlib.sha256(midi.read_bytes()).hexdigest()}]}
            manifest.write_text(json.dumps(payload))
            def valid(settings=configuration):
                return valid_generation(manifest, checkpoint, 'unconditional', 1, settings, 32)
            self.assertTrue(valid())
            self.assertFalse(valid({**configuration, 'temperature': 1.2}))
            checkpoint.write_bytes(b'own checkpoint version two')
            self.assertFalse(valid())
            checkpoint.write_bytes(b'own checkpoint version one')
            midi.write_bytes(b'changed after evaluation')
            self.assertFalse(valid())

    def test_duplicate_piece_or_wrong_condition_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, midi, manifest = (root / name for name in ('own.pt', 'piece.mid', 'manifest.json'))
            checkpoint.write_bytes(b'own roles model')
            midi.write_bytes(b'midi')
            sample = {'song_name': '001', 'bars': 16, 'midi_path': str(midi),
                      'midi_sha256': hashlib.sha256(midi.read_bytes()).hexdigest()}
            payload = {'complete': True, 'task': 'accompaniment', 'configuration': {},
                       'conditioning': 'normal', 'samples': [sample, sample],
                       'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            manifest.write_text(json.dumps(payload))
            self.assertFalse(valid_generation(manifest, checkpoint, 'accompaniment', 2, {}, 16, 'normal'))
            self.assertFalse(valid_generation(manifest, checkpoint, 'accompaniment', 2, {}, 16, 'empty-melody'))


if __name__ == '__main__':
    unittest.main()
