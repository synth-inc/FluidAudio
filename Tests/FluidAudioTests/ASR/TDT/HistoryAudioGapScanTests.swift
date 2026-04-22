import Foundation
import XCTest

@testable import FluidAudio

/// Scans all history audio files from ~/Library/Application Support/Onit/transcription_history_audio/
/// and reports what percentage would trigger the ≥12f blank-gap window inference.
///
/// Run with:
///   cd FluidAudio && swift test --filter HistoryAudioGapScanTests
final class HistoryAudioGapScanTests: XCTestCase {

    private static let audioDir =
        FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Onit/transcription_history_audio")
    private static let outputPath = "/tmp/history_gap_scan.json"
    private static let recoveryOutputPath = "/tmp/history_gap_recovery.json"

    private static let gapThresholdFrames = 12  // 960ms — the production threshold
    private static let frameSeconds = 0.08
    private static let samplesPerFrame = 1280  // 80ms @ 16kHz

    func testScanHistoryAudioGaps() async throws {
        guard FileManager.default.fileExists(atPath: Self.audioDir.path) else {
            throw XCTSkip("History audio dir not found: \(Self.audioDir.path)")
        }

        let allFiles = try FileManager.default.contentsOfDirectory(
            at: Self.audioDir, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }

        print("Found \(allFiles.count) history audio files")

        let tdtModels = try await AsrModels.loadFromCache()
        let manager = AsrManager()
        try await manager.initialize(models: tdtModels)

        let vad = try await VadManager()
        let chunkDuration = Double(VadManager.chunkSize) / 16000.0

        let converter = AudioConverter()

        var totalFiles = 0
        var errors = 0
        var filesWithGap = 0          // ≥12f gap exists at all
        var filesWithSpeechGap = 0    // ≥12f gap + VAD confirms speech
        var totalSpeechGaps = 0       // total qualifying gaps across all files
        var gapFrameDistribution: [Int: Int] = [:]
        var results: [[String: Any]] = []

        for (i, url) in allFiles.enumerated() {
            if (i + 1) % 100 == 0 || i == 0 {
                print("[\(i+1)/\(allFiles.count)] processed=\(totalFiles) "
                    + "triggered=\(filesWithSpeechGap) errors=\(errors)")
            }

            do {
                let samples = try converter.resampleAudioFile(path: url.path)
                totalFiles += 1

                // ── TDT inference ─────────────────────────────────────────
                let padded = manager.padAudioIfNeeded(samples, targetLength: ASRConstants.maxModelSamples)
                var state = TdtDecoderState.make()
                let (hypothesis, _) = try await manager.executeMLInferenceWithTimings(
                    padded,
                    originalLength: samples.count,
                    actualAudioFrames: nil,
                    decoderState: &state,
                    contextFrameAdjustment: 0,
                    isLastChunk: true
                )

                let ts = hypothesis.timestamps
                guard ts.count >= 2 else { continue }

                // ── Find gaps ≥ threshold ─────────────────────────────────
                struct Gap {
                    let afterFrame: Int
                    let beforeFrame: Int
                    let gapFrames: Int
                }
                var gaps: [Gap] = []
                for gi in 0..<(ts.count - 1) {
                    let gf = ts[gi + 1] - ts[gi]
                    if gf >= Self.gapThresholdFrames {
                        gaps.append(Gap(afterFrame: ts[gi], beforeFrame: ts[gi + 1], gapFrames: gf))
                    }
                }

                guard !gaps.isEmpty else { continue }
                filesWithGap += 1

                // ── VAD on full audio ─────────────────────────────────────
                let fullVad = try await vad.process(samples)
                func hasSpeechInRange(t1: Double, t2: Double) -> (Bool, Float) {
                    var maxP: Float = 0
                    var active = false
                    for (ci, r) in fullVad.enumerated() {
                        let cs = Double(ci) * chunkDuration
                        let ce = cs + chunkDuration
                        guard cs <= t2 && ce >= t1 else { continue }
                        if r.probability > maxP { maxP = r.probability }
                        if r.isVoiceActive { active = true }
                    }
                    return (active, maxP)
                }

                var speechGaps: [[String: Any]] = []
                for gap in gaps {
                    let t1 = Double(gap.afterFrame + 1) * Self.frameSeconds
                    let t2 = Double(gap.beforeFrame) * Self.frameSeconds
                    let (hasSpeech, vadProb) = hasSpeechInRange(t1: t1, t2: t2)
                    gapFrameDistribution[gap.gapFrames, default: 0] += 1
                    if hasSpeech {
                        totalSpeechGaps += 1
                        speechGaps.append([
                            "gapFrames": gap.gapFrames,
                            "gapSeconds": Double(gap.gapFrames) * Self.frameSeconds,
                            "vadMaxProb": Double(vadProb),
                        ])
                    }
                }

                if !speechGaps.isEmpty {
                    filesWithSpeechGap += 1
                    results.append([
                        "file": url.lastPathComponent,
                        "durationSamples": samples.count,
                        "durationSeconds": Double(samples.count) / 16000.0,
                        "gaps": speechGaps,
                    ])
                }

            } catch {
                errors += 1
            }
        }

        // ── Summary ───────────────────────────────────────────────────────
        let pctTriggered = totalFiles > 0 ? 100.0 * Double(filesWithSpeechGap) / Double(totalFiles) : 0
        let summary: [String: Any] = [
            "totalFiles": totalFiles,
            "errors": errors,
            "filesWithGap_noVad": filesWithGap,
            "filesWithSpeechGap": filesWithSpeechGap,
            "pctTriggered": pctTriggered,
            "totalSpeechGaps": totalSpeechGaps,
            "gapThresholdFrames": Self.gapThresholdFrames,
        ]

        let output: [String: Any] = ["summary": summary, "triggered": results]
        let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys, .prettyPrinted])
        try data.write(to: URL(fileURLWithPath: Self.outputPath))

        print("\n=== GAP SCAN SUMMARY (≥\(Self.gapThresholdFrames)f / \(Int(Double(Self.gapThresholdFrames) * Self.frameSeconds * 1000))ms) ===")
        print("Total files processed : \(totalFiles)")
        print("Errors                : \(errors)")
        print("Files with gap (raw)  : \(filesWithGap)")
        print("Files with speech gap : \(filesWithSpeechGap) (\(String(format: "%.1f", pctTriggered))%)")
        print("Total speech gaps     : \(totalSpeechGaps)")
        print("Written to            : \(Self.outputPath)")

        print("\nGap size distribution (≥\(Self.gapThresholdFrames)f):")
        for size in gapFrameDistribution.keys.sorted() {
            let count = gapFrameDistribution[size]!
            print("  \(size)f (\(String(format: "%.2f", Double(size) * Self.frameSeconds))s): \(count)")
        }
    }

    /// Runs window TDT on only the files that triggered in testScanHistoryAudioGaps,
    /// to see what gets recovered and whether any are dictation commands.
    ///
    /// Run with:
    ///   cd FluidAudio && swift test --filter HistoryAudioGapScanTests/testRecoverTriggeredGaps
    func testRecoverTriggeredGaps() async throws {
        guard FileManager.default.fileExists(atPath: Self.outputPath) else {
            throw XCTSkip("Run testScanHistoryAudioGaps first to generate \(Self.outputPath)")
        }

        let scanData = try Data(contentsOf: URL(fileURLWithPath: Self.outputPath))
        let scan = try JSONSerialization.jsonObject(with: scanData) as! [String: Any]
        let triggered = scan["triggered"] as! [[String: Any]]

        print("Running window TDT on \(triggered.count) triggered files…")

        let tdtModels = try await AsrModels.loadFromCache()
        let manager = AsrManager()
        try await manager.initialize(models: tdtModels)
        let converter = AudioConverter()

        let dictationCommands: Set<String> = [
            "period", "comma", "semicolon", "colon",
            "exclamation point", "question mark",
            "quote", "unquote", "end quote", "open quote", "close quote",
            "new line", "newline", "new paragraph", "dash", "hyphen",
        ]

        func normalize(_ s: String) -> String {
            s.lowercased()
                .replacingOccurrences(of: "[^a-z0-9 ]", with: " ", options: .regularExpression)
                .components(separatedBy: .whitespaces).filter { !$0.isEmpty }.joined(separator: " ")
        }

        var results: [[String: Any]] = []
        var dictationHits = 0
        var windowInferenceMs: [Double] = []
        var errors = 0

        for (i, entry) in triggered.enumerated() {
            let fileName = entry["file"] as! String
            let fileURL = Self.audioDir.appendingPathComponent(fileName)
            let gaps = entry["gaps"] as! [[String: Any]]

            if (i + 1) % 20 == 0 || i == 0 {
                print("[\(i+1)/\(triggered.count)] hits=\(dictationHits)")
            }

            do {
                let samples = try converter.resampleAudioFile(path: fileURL.path)

                // Full TDT to get token timestamps
                let padded = manager.padAudioIfNeeded(samples, targetLength: ASRConstants.maxModelSamples)
                var state = TdtDecoderState.make()
                let (hypothesis, _) = try await manager.executeMLInferenceWithTimings(
                    padded,
                    originalLength: samples.count,
                    actualAudioFrames: nil,
                    decoderState: &state,
                    contextFrameAdjustment: 0,
                    isLastChunk: true
                )
                let ts = hypothesis.timestamps
                guard ts.count >= 2 else { continue }

                let baseline = manager.processTranscriptionResult(
                    tokenIds: hypothesis.ySequence,
                    timestamps: hypothesis.timestamps,
                    confidences: hypothesis.tokenConfidences,
                    encoderSequenceLength: 0,
                    audioSamples: samples,
                    processingTime: 0
                ).text

                // Re-find qualifying gaps
                var gapRecoveries: [[String: Any]] = []
                for gi in 0..<(ts.count - 1) {
                    let gf = ts[gi + 1] - ts[gi]
                    guard gf >= Self.gapThresholdFrames else { continue }

                    let afterFrame = ts[gi]
                    let beforeFrame = ts[gi + 1]

                    // Only process gaps that VAD confirmed as speech in the scan
                    guard gaps.contains(where: { ($0["gapFrames"] as? Int) == gf }) else { continue }

                    let winStart = (afterFrame + 3) * Self.samplesPerFrame
                    let winEnd = min((beforeFrame + 2) * Self.samplesPerFrame, samples.count)
                    guard winStart < samples.count && winStart < winEnd else { continue }

                    let winSamples = Array(samples[winStart..<winEnd])
                    let paddedWin = manager.padAudioIfNeeded(
                        winSamples, targetLength: ASRConstants.maxModelSamples)
                    var winState = TdtDecoderState.make()
                    let winStart_t = Date()
                    let (winHyp, _) = try await manager.executeMLInferenceWithTimings(
                        paddedWin,
                        originalLength: winSamples.count,
                        actualAudioFrames: nil,
                        decoderState: &winState,
                        contextFrameAdjustment: 0,
                        isLastChunk: true
                    )
                    windowInferenceMs.append(Date().timeIntervalSince(winStart_t) * 1000)

                    let recovered = manager.processTranscriptionResult(
                        tokenIds: winHyp.ySequence,
                        timestamps: winHyp.timestamps,
                        confidences: winHyp.tokenConfidences,
                        encoderSequenceLength: 0,
                        audioSamples: winSamples,
                        processingTime: 0
                    ).text.trimmingCharacters(in: .whitespacesAndNewlines)

                    let norm = normalize(recovered)
                    let isDictation = dictationCommands.contains(norm)
                    if isDictation { dictationHits += 1 }

                    gapRecoveries.append([
                        "gapFrames": gf,
                        "gapSeconds": Double(gf) * Self.frameSeconds,
                        "recovered": recovered,
                        "isDictation": isDictation,
                    ])
                }

                let anyDictation = gapRecoveries.contains { $0["isDictation"] as? Bool == true }
                results.append([
                    "file": fileName,
                    "baseline": baseline,
                    "gaps": gapRecoveries,
                    "anyDictation": anyDictation,
                ])

            } catch {
                errors += 1
            }
        }

        let avgWindowMs = windowInferenceMs.isEmpty ? 0.0
            : windowInferenceMs.reduce(0, +) / Double(windowInferenceMs.count)
        let medianWindowMs = windowInferenceMs.isEmpty ? 0.0
            : windowInferenceMs.sorted()[windowInferenceMs.count / 2]

        let summary: [String: Any] = [
            "triggeredFiles": triggered.count,
            "errors": errors,
            "dictationHits": dictationHits,
            "filesWithDictationHit": results.filter { $0["anyDictation"] as? Bool == true }.count,
            "windowInferences": windowInferenceMs.count,
            "avgWindowInferenceMs": avgWindowMs,
            "medianWindowInferenceMs": medianWindowMs,
        ]
        let output: [String: Any] = ["summary": summary, "results": results]
        let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys, .prettyPrinted])
        try data.write(to: URL(fileURLWithPath: Self.recoveryOutputPath))

        print("\n=== RECOVERY SUMMARY ===")
        print("Triggered files      : \(triggered.count)")
        print("Dictation hits       : \(dictationHits)")
        print("Files with hit       : \(results.filter { $0["anyDictation"] as? Bool == true }.count)")
        print("Errors               : \(errors)")
        print("Window inference     : avg=\(String(format: "%.0f", avgWindowMs))ms  median=\(String(format: "%.0f", medianWindowMs))ms  n=\(windowInferenceMs.count)")
        print("Written to           : \(Self.recoveryOutputPath)")

        print("\nAll recovered texts (non-empty):")
        for r in results {
            for g in (r["gaps"] as? [[String: Any]] ?? []) {
                let rec = g["recovered"] as? String ?? ""
                guard !rec.isEmpty else { continue }
                let isDictation = g["isDictation"] as? Bool == true
                let tag = isDictation ? " ★ DICTATION" : ""
                print("  \(r["file"] as? String ?? "")  gap=\(g["gapFrames"] as? Int ?? 0)f  recovered=\"\(rec)\"\(tag)")
            }
        }
    }
}
