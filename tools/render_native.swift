// Offline piano rendering with macOS's system sound bank. No sound bank is bundled.
import Foundation
import AVFoundation
import AudioToolbox

struct Event: Decodable { let frame: Int64; let pitch: UInt8; let velocity: UInt8; let channel: UInt8; let on: Bool }
struct Request: Decodable { let output: String; let sample_rate: Double; let duration_frames: Int64; let events: [Event] }
func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(1) }
if CommandLine.arguments.count != 2 { fail("usage: render_native request.json") }
do {
    let req = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
    let engine = AVAudioEngine()
    let sampler = AVAudioUnitSampler()
    engine.attach(sampler)
    let format = AVAudioFormat(standardFormatWithSampleRate: req.sample_rate, channels: 2)!
    engine.connect(sampler, to: engine.mainMixerNode, format: format)
    engine.mainMixerNode.outputVolume = 0.60
    let bank = URL(fileURLWithPath: "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls")
    try sampler.loadSoundBankInstrument(at: bank, program: 0, bankMSB: UInt8(kAUSampler_DefaultMelodicBankMSB), bankLSB: 0)
    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
    try engine.start()
    let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: req.sample_rate, AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
    let file = try AVAudioFile(forWriting: URL(fileURLWithPath: req.output), settings: settings)
    let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4096)!
    var idx = 0
    var failed = 0
    while engine.manualRenderingSampleTime < req.duration_frames {
        let now = engine.manualRenderingSampleTime
        while idx < req.events.count && req.events[idx].frame <= now {
            let event = req.events[idx]
            if event.on { sampler.startNote(event.pitch, withVelocity: event.velocity, onChannel: event.channel) }
            else { sampler.stopNote(event.pitch, onChannel: event.channel) }
            idx += 1
        }
        let next = idx < req.events.count ? req.events[idx].frame : req.duration_frames
        let count = AVAudioFrameCount(max(1, min(4096, min(next - now, req.duration_frames - now))))
        let status = try engine.renderOffline(count, to: buffer)
        switch status {
        case .success: try file.write(from: buffer); failed = 0
        case .cannotDoInCurrentContext: failed += 1; if failed > 100 { fail("renderer made no progress") }
        case .insufficientDataFromInputNode: fail("offline sampler has insufficient input")
        case .error: fail("offline render failed")
        @unknown default: fail("unknown rendering status")
        }
    }
    engine.stop()
} catch { fail("\(error)") }
