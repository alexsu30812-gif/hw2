"""Contract tests use explicit synthetic fixtures, never experiment results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
import struct
import hashlib
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_report import format_value, hosting_warnings, validate_payload
from gather_report_data import csv_rows
from package_submission import owned_file, submission_samples, wav_info, verify_archive


def fixture():
    """Small values below are synthetic tests of validation, not measured music."""
    a = []
    for config in range(3):
        a.append({"id": f"TEST-{config}", "parameters": {"temperature": 0.8 + config*0.2},
                  "n": 20, "metrics": {"H4": 2.0, "GS": 0.5},
                  "samples": [{"bars": 32, "output_midi": f"TEST/a-{config}-{i}.mid", "output_wav": f"TEST/a-{config}-{i}.wav"} for i in range(20)]})
    b = [{"prompt_id": f"{p:03d}", "configuration_id": f"TEST-{c}",
          "bars": 32, "prompt_bars": 8, "new_bars": 24, "is_best": c == 0,
          "output_midi": f"TEST/b-{p}-{c}.mid", "output_wav": f"TEST/b-{p}-{c}.wav"}
         for p in range(1, 4) for c in range(3)]
    return {"_synthetic_test_fixture": True, "status": "complete", "student": {"student_id": "r14942154"},
            "cloud_drive_url": "https://drive.google.com/drive/folders/SYNTHETIC_TEST_ONLY",
            "data": {"training_songs": 896, "final_all_896_songs_verified": True}, "model": {"random_initialization": True},
            "basic_a": {"configurations": a, "selected_configuration": "TEST-0", "real_baseline": {"H4": 2.0, "GS": 0.5}},
            "basic_b": {"results": b, "same_checkpoint_as_basic_a": True, "prompt_preservation_verified": True},
            "bonus": {"attempted": False}, "references": ["Synthetic validation fixture only"]}


class ContractTests(unittest.TestCase):
    def test_complete_shape(self):
        self.assertEqual(validate_payload(fixture()), [])

    def test_no_imputed_missing_score(self):
        d = fixture()
        d["basic_a"]["configurations"][0]["metrics"]["H4"] = None
        self.assertTrue(any("fewer than three" in r for r in validate_payload(d)))

    def test_metric_range_and_nan(self):
        for value in (-0.1, 1.1, float("nan")):
            d = fixture()
            d["basic_a"]["real_baseline"]["GS"] = value
            self.assertTrue(any("baseline" in r for r in validate_payload(d)))

    def test_duplicate_decoding_settings_are_not_three_experiments(self):
        d = fixture()
        for row in d["basic_a"]["configurations"]:
            row["parameters"] = {"temperature": 1.0}
        self.assertTrue(any("distinct basic-a" in r for r in validate_payload(d)))

    def test_count_is_actual_manifest_length(self):
        d = fixture()
        d["basic_a"]["configurations"][0]["samples"].pop()
        self.assertTrue(any("fewer than three" in r for r in validate_payload(d)))

    def test_final_checkpoint_must_expose_every_song(self):
        d = fixture()
        d["data"]["final_all_896_songs_verified"] = False
        self.assertTrue(any("exposure to all 896" in reason for reason in validate_payload(d)))

    def test_skipped_or_incomplete_generation_cannot_pass(self):
        for field, bad_value in (("manifest_complete", False), ("skipped", 1), ("evaluated", 19)):
            d = fixture()
            checks = {"manifest_complete": True, "skipped": 0, "evaluated": 20, "valid_bar_counts": True}
            checks[field] = bad_value
            d["basic_a"]["configurations"][0]["checks"] = checks
            self.assertTrue(any("fewer than three" in reason for reason in validate_payload(d)))

    def test_exactly_one_best_per_prompt(self):
        d = fixture()
        d["basic_b"]["results"][1]["is_best"] = True
        self.assertTrue(any("exactly one best" in r for r in validate_payload(d)))

    def test_github_does_not_replace_cloud_drive(self):
        d = fixture()
        d["cloud_drive_url"] = "https://github.com/alexsu30812-gif/hw1"
        self.assertTrue(any("cloud-drive" in r for r in validate_payload(d)))

    def test_artifact_url_and_unverified_flag_remain_separate(self):
        d = fixture()
        d["artifact_url"] = "https://github.com/alexsu30812-gif/hw2"
        d["cloud_drive_url"] = ""
        d["cloud_drive_requirement_unverified"] = True
        reasons = validate_payload(d)
        self.assertTrue(any("cloud-drive folder URL is pending" in r for r in reasons))
        self.assertTrue(any("unverified" in r for r in reasons))

    def test_explicit_repository_choice_is_complete_with_hosting_warning(self):
        d = fixture()
        d.update(artifact_url="https://github.com/alexsu30812-gif/hw2", cloud_drive_url="",
                 user_selected_repository_hosting=True, cloud_drive_requirement_unverified=True)
        self.assertEqual(validate_payload(d), [])
        self.assertTrue(any("unverified" in warning for warning in hosting_warnings(d)))
        self.assertTrue(d["cloud_drive_requirement_unverified"])

    def test_repository_choice_does_not_claim_ta_acceptance(self):
        d = fixture()
        d.update(artifact_url="https://github.com/alexsu30812-gif/hw2", cloud_drive_url="",
                 user_selected_repository_hosting=True, cloud_drive_requirement_unverified=False)
        self.assertTrue(any("acceptance as unverified" in r for r in validate_payload(d)))

    def test_repository_choice_cannot_hide_missing_experiments(self):
        d = fixture()
        d.update(artifact_url="https://github.com/alexsu30812-gif/hw2", cloud_drive_url="",
                 user_selected_repository_hosting=True, cloud_drive_requirement_unverified=True)
        d["basic_a"]["configurations"][0]["samples"].pop()
        self.assertTrue(any("fewer than three" in r for r in validate_payload(d)))

    def test_official_bonus_column_names_are_not_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_bonus.csv"
            path.write_text("song_name,CA,DP (Pitch-class OA),DV (Velocity OA),DD (Duration OA),DIOI (Inter-Onset OA)\n001,0,0.1,0.2,0.3,0.4\n")
            row = csv_rows(path)["001"]
            self.assertEqual([row[name] for name in ("CA", "OA_P", "OA_V", "OA_D", "OA_IOI")], ["0", "0.1", "0.2", "0.3", "0.4"])

    def test_bonus_missing_metric_or_prompt_fails(self):
        d = fixture()
        d["bonus"] = {"attempted": True, "results": [], "melody_preservation_verified": True}
        self.assertTrue(any("ten prompt" in r for r in validate_payload(d)))

    def test_submission_selects_twenty_not_all_sixty(self):
        samples, counts = submission_samples(fixture())
        self.assertEqual(counts, {"basic_a": 20, "basic_b": 9, "bonus": 0})
        self.assertEqual(len(samples), 29)

    def test_parameters_are_rendered_not_replaced_by_pending(self):
        self.assertIn("temperature", format_value({"temperature": 1.0}))
        self.assertEqual(format_value(None), "PENDING")

    def test_dataset_path_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "POP909").mkdir()
            (root / "POP909/song.mid").write_bytes(b"test")
            with self.assertRaises(ValueError):
                owned_file(root, "POP909/song.mid")
            with self.assertRaises(ValueError):
                owned_file(root, "../outside.mid")

    def test_real_wav_header_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "synthetic.wav"
            with wave.open(str(p), "wb") as out:
                out.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                out.writeframes(struct.pack("<h", 1000) * 24000)
            self.assertAlmostEqual(wav_info(p)["seconds"], 1.0)
            p.write_bytes(p.read_bytes()[:-10])
            with self.assertRaises(ValueError):
                wav_info(p)

    def test_silent_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "silent.wav"
            with wave.open(str(p), "wb") as out:
                out.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                out.writeframes(b"\0\0" * 22050)
            with self.assertRaisesRegex(ValueError, "Silent"):
                wav_info(p)

    def test_archive_readback_detects_changed_and_unlisted_files(self):
        # These bytes test only archive integrity, not MIDI or audio validity.
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "synthetic.zip"
            members = {"TEST/a.mid": b"synthetic midi bytes", "TEST/a.wav": b"synthetic wav bytes"}
            hashes = {name: hashlib.sha256(value).hexdigest() for name, value in members.items()}
            checks = {"midi_wav_pairs": [{"midi": "TEST/a.mid", "wav": "TEST/a.wav"}]}

            def write(changed=False, extra=False):
                with zipfile.ZipFile(archive, "w") as handle:
                    for name, value in members.items():
                        handle.writestr("SYNTHETIC/" + name, b"changed" if changed else value)
                    handle.writestr("SYNTHETIC/FILE_SHA256.json", json.dumps(hashes))
                    handle.writestr("SYNTHETIC/SUBMISSION_CHECKS.json", json.dumps(checks))
                    if extra:
                        handle.writestr("SYNTHETIC/unlisted.txt", "unlisted")

            write()
            self.assertTrue(verify_archive(archive)["all_file_hashes_match"])
            write(changed=True)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_archive(archive)
            write(extra=True)
            with self.assertRaisesRegex(ValueError, "unlisted or missing"):
                verify_archive(archive)


if __name__ == "__main__":
    unittest.main()
