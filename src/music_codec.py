"""Beat-annotation-aware REMI representation for POP909.

The annotation beat clock, rather than the unreliable MIDI meter metadata,
defines the bars. Eight grid subdivisions represent one annotation beat.
Basic notes merge every source track; roles notes retain MELODY and PIANO only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import bisect
import copy
import hashlib
import json
import math
import statistics

STEPS_PER_BEAT = 8
MAX_POSITIONS = 128
MAX_DURATION = 256


class Vocabulary:
    """A fixed vocabulary independent of validation or prompt contents."""

    def __init__(self):
        self.tokens = ["<PAD>", "<BOS>", "<EOS>", "Bar"]
        self.ranges = {}
        groups = {
            "Meter": [f"Meter_{i}" for i in range(1, 17)],
            "Tempo": [f"Tempo_{i}" for i in range(8, 241, 2)],
            "Pos": [f"Pos_{i}" for i in range(MAX_POSITIONS)],
            "Pitch": [f"Pitch_{i}" for i in range(128)],
            "Vel": [f"Vel_{i}" for i in range(32)],
            "Dur": [f"Dur_{i}" for i in range(1, MAX_DURATION + 1)],
            "Track": ["Track_Melody", "Track_Piano"],
        }
        for key, values in groups.items():
            start = len(self.tokens)
            self.tokens.extend(values)
            self.ranges[key] = (start, len(self.tokens))
        self.id_to_token = self.tokens
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.size = len(self.tokens)
        self.pad_id, self.bos_id, self.eos_id, self.bar_id = range(4)
        self.sha256 = hashlib.sha256(json.dumps(self.tokens, separators=(",", ":")).encode()).hexdigest()

    def id(self, name):
        return self.token_to_id[name]

    def group_ids(self, name):
        return list(range(*self.ranges[name]))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"tokens": self.tokens, "token_to_id": self.token_to_id,
                                         "sha256": self.sha256, "steps_per_beat": STEPS_PER_BEAT,
                                         "ranges": self.ranges}, indent=2) + "\n")

    def __len__(self):
        return self.size


@dataclass
class BeatMap:
    beat_times: list[float]
    bar_indices: list[int]

    @classmethod
    def from_file(cls, path, end_time=None):
        rows = [line.split() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        times = [float(row[0]) for row in rows]
        if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"Invalid increasing beat annotation: {path}")
        down = [i for i, row in enumerate(rows) if len(row) > 2 and abs(float(row[2]) - 1) < 1e-6]
        if not down:
            raise ValueError(f"No downbeats in {path}")
        if down[0] != 0:
            down.insert(0, 0)  # Retain a pickup as its own short opening bar.
        lengths = [b - a for a, b in zip(down, down[1:])]
        meter = lengths[-1] if lengths else 4
        # A final annotated downbeat is an endpoint, not another empty bar.
        if down[-1] < len(times) - 1:
            down.append(down[-1] + meter)
        dt = statistics.median([b - a for a, b in zip(times[-9:], times[-8:])]) if len(times) > 9 else times[-1] - times[-2]
        while len(times) <= down[-1]:
            times.append(times[-1] + dt)
        if end_time is not None:
            while float(end_time) >= times[down[-1]] - 1e-7:
                down.append(down[-1] + meter)
                while len(times) <= down[-1]:
                    times.append(times[-1] + dt)
        if len(down) < 2:
            down.append(down[0] + meter)
            while len(times) <= down[-1]:
                times.append(times[-1] + dt)
        result = cls(times, down)
        if any(n < 1 or n > 16 for n in result.bar_beats):
            raise ValueError(f"Bar length outside vocabulary in {path}: {result.bar_beats}")
        return result

    @property
    def n_bars(self):
        return len(self.bar_indices) - 1

    @property
    def bar_times(self):
        return [self.beat_times[i] for i in self.bar_indices]

    @property
    def bar_beats(self):
        return [b - a for a, b in zip(self.bar_indices, self.bar_indices[1:])]

    def time_to_beat(self, seconds):
        i = max(0, min(len(self.beat_times) - 2, bisect.bisect_right(self.beat_times, seconds) - 1))
        return i + (float(seconds) - self.beat_times[i]) / (self.beat_times[i + 1] - self.beat_times[i])

    def beat_to_time(self, beat):
        i = max(0, min(len(self.beat_times) - 2, int(math.floor(beat))))
        return self.beat_times[i] + (float(beat) - i) * (self.beat_times[i + 1] - self.beat_times[i])

    def bar_at_time(self, seconds):
        return max(0, min(self.n_bars - 1, bisect.bisect_right(self.bar_times, float(seconds) + 1e-7) - 1))

    def as_dict(self):
        return {"beat_times": self.beat_times, "bar_indices": self.bar_indices,
                "bar_times": self.bar_times, "bar_beats": self.bar_beats}


@dataclass
class EncodedSong:
    tokens: list[int]
    bar_offsets: list[int]
    beat_map: BeatMap
    metadata: dict

    def token_strings(self, vocab=None):
        vocab = vocab or Vocabulary()
        return [vocab.tokens[int(i)] for i in self.tokens]


def _tempo_token(bpm):
    return f"Tempo_{min(240, max(8, int(round(float(bpm) / 2) * 2)))}"


def tokenize_midi(midi_path, beat_path, mode="basic", max_bars=None, vocab=None):
    import pretty_midi

    if mode not in ("basic", "roles"):
        raise ValueError("mode must be basic or roles")
    vocab = vocab or Vocabulary()
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    selected = [(inst, note) for inst in midi.instruments for note in inst.notes
                if mode == "basic" or inst.name.strip().upper() in ("MELODY", "PIANO")]
    last_onset = max((note.start for _, note in selected), default=0)
    beat_map = BeatMap.from_file(beat_path, end_time=last_onset if max_bars is None else None)
    n_bars = beat_map.n_bars if max_bars is None else min(max_bars, beat_map.n_bars)
    if n_bars < 1:
        raise ValueError(f"No bars in {midi_path}")
    notes = [[] for _ in range(n_bars)]
    stats = {"source_notes": len(selected), "encoded_notes": 0, "duration_clipped": 0,
             "onsets_before_first_beat": 0, "onsets_after_requested_range": 0,
             "onset_error_sec_sum": 0., "onset_error_sec_max": 0.,
             "offset_error_sec_sum": 0., "offset_error_sec_max": 0.}
    for inst, note in selected:
        if note.start >= beat_map.bar_times[n_bars] - 1e-7:
            stats["onsets_after_requested_range"] += 1
            continue
        b = beat_map.bar_at_time(note.start)
        start_beat = beat_map.time_to_beat(note.start)
        if start_beat < 0:
            stats["onsets_before_first_beat"] += 1
        pos = max(0, min(beat_map.bar_beats[b] * STEPS_PER_BEAT - 1,
                         int(round((start_beat - beat_map.bar_indices[b]) * STEPS_PER_BEAT))))
        q_start_beat = beat_map.bar_indices[b] + pos / STEPS_PER_BEAT
        dur_raw = max(1, int(round((beat_map.time_to_beat(note.end) - start_beat) * STEPS_PER_BEAT)))
        duration = min(MAX_DURATION, dur_raw)
        stats["duration_clipped"] += int(dur_raw > MAX_DURATION)
        velocity_bin = min(31, max(0, int(note.velocity) // 4))
        role = "Melody" if inst.name.strip().upper() == "MELODY" else "Piano"
        notes[b].append({"bar": b, "pos": pos, "pitch": int(note.pitch), "velocity": velocity_bin * 4 + 2,
                         "duration": duration, "track": role})
        onset_error = abs(beat_map.beat_to_time(q_start_beat) - note.start)
        offset_error = abs(beat_map.beat_to_time(q_start_beat + duration / STEPS_PER_BEAT) - note.end)
        stats["onset_error_sec_sum"] += onset_error
        stats["offset_error_sec_sum"] += offset_error
        stats["onset_error_sec_max"] = max(stats["onset_error_sec_max"], onset_error)
        stats["offset_error_sec_max"] = max(stats["offset_error_sec_max"], offset_error)
        stats["encoded_notes"] += 1
    tokens = [vocab.bos_id]
    bar_offsets = []
    for b in range(n_bars):
        bar_offsets.append(len(tokens))
        duration_sec = beat_map.bar_times[b + 1] - beat_map.bar_times[b]
        tokens.extend([vocab.bar_id, vocab.id(f"Meter_{beat_map.bar_beats[b]}"),
                       vocab.id(_tempo_token(60 * beat_map.bar_beats[b] / duration_sec))])
        groups = ("Melody", "Piano") if mode == "roles" else (None,)
        for role in groups:
            if role:
                tokens.append(vocab.id(f"Track_{role}"))
            chosen = [n for n in notes[b] if role is None or n["track"] == role]
            for note in sorted(chosen, key=lambda n: (n["pos"], n["pitch"], n["duration"], n["velocity"])):
                tokens.extend([vocab.id(f"Pos_{note['pos']}"), vocab.id(f"Pitch_{note['pitch']}"),
                               vocab.id(f"Vel_{note['velocity'] // 4}"), vocab.id(f"Dur_{note['duration']}")])
    tokens.append(vocab.eos_id)
    bar_offsets.append(len(tokens) - 1)
    metadata = {"song_name": Path(midi_path).parent.name, "source_midi": str(midi_path), "mode": mode,
                "n_bars": n_bars, "bar_times": beat_map.bar_times[:n_bars + 1],
                "bar_beats": beat_map.bar_beats[:n_bars], "steps_per_beat": STEPS_PER_BEAT,
                "vocabulary_sha256": vocab.sha256, "stats": stats}
    return EncodedSong(tokens, bar_offsets, beat_map, metadata)


def parse_tokens(tokens, vocab=None):
    """Return bars, quantized notes and ignored malformed token count."""
    vocab = vocab or Vocabulary()
    strings = [vocab.tokens[int(x)] if not isinstance(x, str) else x for x in tokens]
    bars, notes = [], []
    b, track, invalid, i = -1, "Piano", 0, 0
    while i < len(strings):
        token = strings[i]
        if token == "Bar":
            b += 1
            bars.append({"meter": 4, "tempo": 120.})
            track = "Piano"
        elif token.startswith("Meter_") and b >= 0:
            bars[b]["meter"] = int(token.split("_")[1])
        elif token.startswith("Tempo_") and b >= 0:
            bars[b]["tempo"] = float(token.split("_")[1])
        elif token.startswith("Track_"):
            track = token.split("_", 1)[1]
        elif token.startswith("Pos_"):
            following = strings[i + 1:i + 4]
            if b >= 0 and len(following) == 3 and all(t.startswith(prefix) for t, prefix in zip(following, ("Pitch_", "Vel_", "Dur_"))):
                pos, pitch, velocity, duration = [int(t.split("_")[1]) for t in [token] + following]
                if pos < bars[b]["meter"] * STEPS_PER_BEAT:
                    notes.append({"bar": b, "pos": pos, "pitch": pitch, "velocity": velocity * 4 + 2,
                                  "duration": duration, "track": track})
                else:
                    invalid += 1
                i += 3
            else:
                invalid += 1
        elif token not in ("<BOS>", "<EOS>", "<PAD>"):
            invalid += 1
        i += 1
    return {"bars": bars, "notes": notes, "invalid_events": invalid}


def decode_notes(tokens, vocab=None):
    return parse_tokens(tokens, vocab)["notes"]


def decode_tokens(tokens, path, vocab=None, beat_map=None, bpm=None, clip_at_end=False):
    """Write MIDI. An external annotation map preserves original absolute time.

    Without an external map, generated meter/tempo tokens define the clock.
    Meter n is written as n/4; each annotation beat is represented as one
    quarter note, including source compound meters, preserving bar boundaries.
    """
    import mido

    parsed = parse_tokens(tokens, vocab)
    bars = parsed["bars"]
    if not bars:
        raise ValueError("Cannot decode a sequence with no bars")
    if beat_map is not None and len(bars) > beat_map.n_bars:
        raise ValueError("External beat map has fewer bars than token sequence")
    starts = [0.]
    beats = [0]
    for bar in bars:
        starts.append(starts[-1] + bar["meter"] * 60 / (bpm or bar["tempo"]))
        beats.append(beats[-1] + bar["meter"])

    # Encode the actual clock into MIDI tempo changes. Using a single export
    # tempo would retain seconds but corrupt MIDI bar locations after rubato.
    ppq = 960
    midi = mido.MidiFile(type=1, ticks_per_beat=ppq)
    meta_events = []
    if beat_map is not None:
        first_dt = beat_map.beat_times[1] - beat_map.beat_times[0]
        tick_offset = beat_map.beat_times[0] / first_dt * ppq
        def to_tick(bar, pos=0, duration=0):
            return max(0, round(tick_offset + (beat_map.bar_indices[bar] + (pos + duration) / STEPS_PER_BEAT) * ppq))
        first_tempo = round(first_dt * 1e6)
        meta_events.append((0, mido.MetaMessage("set_tempo", tempo=first_tempo)))
        previous = first_tempo
        for index, (a, z) in enumerate(zip(beat_map.beat_times, beat_map.beat_times[1:])):
            value = max(1, min(16777215, round((z - a) * 1e6)))
            if value != previous:
                meta_events.append((max(0, round(tick_offset + index * ppq)), mido.MetaMessage("set_tempo", tempo=value)))
                previous = value
        endpoint = max(0, round(tick_offset + beat_map.bar_indices[len(bars)] * ppq))
        end_sec = beat_map.bar_times[len(bars)]
    else:
        def to_tick(bar, pos=0, duration=0):
            return max(0, round((beats[bar] + (pos + duration) / STEPS_PER_BEAT) * ppq))
        previous = None
        for index, bar in enumerate(bars):
            tempo = mido.bpm2tempo(float(bpm or bar["tempo"]))
            if tempo != previous:
                meta_events.append((beats[index] * ppq, mido.MetaMessage("set_tempo", tempo=tempo)))
                previous = tempo
        endpoint = beats[-1] * ppq
        end_sec = starts[-1]
    for b, bar in enumerate(bars):
        if b == 0 or bars[b - 1]["meter"] != bar["meter"]:
            meta_events.append((to_tick(b), mido.MetaMessage("time_signature", numerator=bar["meter"], denominator=4)))

    def make_track(events, name):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        last = 0
        # note_off precedes note_on at a shared tick for a repeated pitch.
        for tick, message in sorted(events, key=lambda item: (item[0], item[1].type == "note_on")):
            track.append(message.copy(time=max(0, int(tick) - last)))
            last = int(tick)
        track.append(mido.MetaMessage("end_of_track", time=max(0, endpoint - last)))
        return track

    midi.tracks.append(make_track(meta_events, "ANNOTATION_CLOCK"))
    tracks = {}
    clipped_final_notes = 0
    active_until = {}
    channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    channel_use = {}
    for note in sorted(parsed["notes"], key=lambda n: (n["bar"], n["pos"], n["pitch"], n["track"])):
        name = note["track"].upper()
        if name not in tracks:
            tracks[name] = []
        start = to_tick(note["bar"], note["pos"])
        end = max(start + 1, to_tick(note["bar"], note["pos"], note["duration"]))
        if clip_at_end and end > endpoint:
            end = max(start + 1, endpoint)
            clipped_final_notes += 1
        preferred = 1 if name == "MELODY" else 0
        available = [ch for ch in [preferred] + [ch for ch in channels if ch != preferred]
                     if active_until.get((ch, note["pitch"]), -1) <= start]
        if not available:
            raise ValueError("More than 15 simultaneous overlapping voices on one pitch")
        channel = available[0]
        active_until[(channel, note["pitch"])] = end
        channel_use.setdefault(name, set()).add(channel)
        tracks[name].extend([(start, mido.Message("note_on", note=note["pitch"], velocity=note["velocity"], channel=channel)),
                             (end, mido.Message("note_off", note=note["pitch"], velocity=0, channel=channel))])
    for name, events in tracks.items():
        for channel in sorted(channel_use[name]):
            events.append((0, mido.Message("program_change", program=0, channel=channel)))
        midi.tracks.append(make_track(events, name))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))
    return {"path": str(path), "n_bars": len(bars), "bar_times": beat_map.bar_times[:len(bars) + 1] if beat_map is not None else starts,
            "bar_beats": [bar["meter"] for bar in bars], "notes": parsed["notes"],
            "invalid_events": parsed["invalid_events"], "duration_sec": end_sec,
            "clipped_final_notes": clipped_final_notes}


def restore_midi_endpoint(path, endpoint_seconds):
    """Restore a silent endpoint after pretty_midi rewrites a file."""
    import mido
    original_end = float(endpoint_seconds)
    exported = mido.MidiFile(str(path))
    tempo_events = [(0, 500000)]
    for track in exported.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempo_events.append((tick, message.tempo))
    clock = sorted(dict(tempo_events).items())
    elapsed, last_tick, tempo = 0., 0, clock[0][1]
    endpoint_tick = None
    for tick, next_tempo in clock[1:]:
        span = mido.tick2second(tick - last_tick, exported.ticks_per_beat, tempo)
        if elapsed + span >= original_end:
            endpoint_tick = last_tick + round(mido.second2tick(original_end - elapsed, exported.ticks_per_beat, tempo))
            break
        elapsed += span
        last_tick, tempo = tick, next_tempo
    if endpoint_tick is None:
        endpoint_tick = last_tick + round(mido.second2tick(original_end - elapsed, exported.ticks_per_beat, tempo))
    for track in exported.tracks:
        total_tick = sum(message.time for message in track)
        if track and track[-1].type == "end_of_track":
            track[-1].time += max(0, endpoint_tick - total_tick)
    exported.save(str(path))


def preserve_prompt_notes(prompt_path, generated_path, output_path, cutoff_sec):
    """Replace generated prefix with exact source notes, including seam sustain."""
    import pretty_midi
    import mido

    prompt = pretty_midi.PrettyMIDI(str(prompt_path))
    generated = pretty_midi.PrettyMIDI(str(generated_path))
    original_end = mido.MidiFile(str(generated_path)).length
    output = copy.deepcopy(generated)
    for instrument in output.instruments:
        instrument.notes = [n for n in instrument.notes if n.start >= cutoff_sec - 1e-7]
    prefix_count = 0
    for instrument in prompt.instruments:
        prefix = copy.deepcopy(instrument)
        prefix.notes = [copy.deepcopy(n) for n in instrument.notes if n.start < cutoff_sec - 1e-7]
        prefix.control_changes = [copy.deepcopy(event) for event in instrument.control_changes if event.time < cutoff_sec]
        prefix.pitch_bends = [copy.deepcopy(event) for event in instrument.pitch_bends if event.time < cutoff_sec]
        prefix_count += len(prefix.notes)
        output.instruments.append(prefix)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.write(str(output_path))
    restore_midi_endpoint(output_path, original_end)
    return {"prefix_notes": prefix_count, "cutoff_sec": cutoff_sec, "duration_sec": original_end}
