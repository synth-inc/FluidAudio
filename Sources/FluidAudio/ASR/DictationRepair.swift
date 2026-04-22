import Foundation

extension AsrManager {

    internal static let dictationRepairGapThreshold = 12  // 960ms at 80ms/frame
    private static let frameSeconds: Double = 0.08
    private static let samplesPerFrame = 1280  // 80ms @ 16kHz

    private static let dictationCommands: Set<String> = [
        "period", "comma", "semicolon", "colon",
        "exclamation point", "question mark",
        "quote", "unquote", "end quote", "open quote", "close quote",
        "new line", "newline", "new paragraph", "dash", "hyphen",
    ]

    /// Detects blank gaps ≥960ms with VAD-confirmed speech, re-runs TDT on each gap window,
    /// and inserts recovered dictation commands (period, comma, end quote, etc.) into the text.
    ///
    /// Returns the repaired text, or nil if nothing was changed.
    internal func repairDictationGaps(
        hypothesis: TdtHypothesis,
        audioSamples: [Float]
    ) async -> String? {
        guard let vad = vadManager else { return nil }

        let ts = hypothesis.timestamps
        guard ts.count >= 2 else { return nil }

        // Find blank gaps ≥ threshold
        struct Gap {
            let afterTokenIdx: Int
            let afterFrame: Int
            let beforeFrame: Int
        }
        var gaps: [Gap] = []
        for i in 0..<(ts.count - 1) {
            let gf = ts[i + 1] - ts[i]
            if gf >= Self.dictationRepairGapThreshold {
                gaps.append(Gap(afterTokenIdx: i, afterFrame: ts[i], beforeFrame: ts[i + 1]))
            }
        }
        guard !gaps.isEmpty else { return nil }

        // Run VAD on full audio once
        guard let vadResults = try? await vad.process(audioSamples) else { return nil }
        let chunkDuration = Double(VadManager.chunkSize) / 16_000.0

        func hasSpeech(t1: Double, t2: Double) -> Bool {
            for (ci, r) in vadResults.enumerated() {
                let cs = Double(ci) * chunkDuration
                let ce = cs + chunkDuration
                guard cs <= t2 && ce >= t1 else { continue }
                if r.isVoiceActive { return true }
            }
            return false
        }

        // For each speech-confirmed gap, run window TDT and check for a dictation command
        var repairs: [Int: String] = [:]
        for gap in gaps {
            let t1 = Double(gap.afterFrame + 1) * Self.frameSeconds
            let t2 = Double(gap.beforeFrame) * Self.frameSeconds
            guard hasSpeech(t1: t1, t2: t2) else { continue }

            let winStart = (gap.afterFrame + 3) * Self.samplesPerFrame
            let winEnd = min((gap.beforeFrame + 2) * Self.samplesPerFrame, audioSamples.count)
            guard winStart < audioSamples.count, winStart < winEnd else { continue }

            let winSamples = Array(audioSamples[winStart..<winEnd])
            let paddedWin = padAudioIfNeeded(winSamples, targetLength: ASRConstants.maxModelSamples)
            var winState = TdtDecoderState.make()
            guard
                let (winHyp, _) = try? await executeMLInferenceWithTimings(
                    paddedWin,
                    originalLength: winSamples.count,
                    actualAudioFrames: nil,
                    decoderState: &winState,
                    contextFrameAdjustment: 0,
                    isLastChunk: true
                )
            else { continue }

            let recovered = processTranscriptionResult(
                tokenIds: winHyp.ySequence,
                timestamps: winHyp.timestamps,
                confidences: winHyp.tokenConfidences,
                tokenDurations: winHyp.tokenDurations,
                encoderSequenceLength: 0,
                audioSamples: winSamples,
                processingTime: 0
            ).text.trimmingCharacters(in: .whitespacesAndNewlines)

            let normalized =
                recovered
                .lowercased()
                .replacingOccurrences(of: "[^a-z0-9 ]", with: " ", options: .regularExpression)
                .components(separatedBy: .whitespaces)
                .filter { !$0.isEmpty }
                .joined(separator: " ")

            if Self.dictationCommands.contains(normalized) {
                repairs[gap.afterTokenIdx] = recovered
                logger.info(
                    "Dictation repair: inserted \"\(recovered)\" after token \(gap.afterTokenIdx) (gap=\(ts[gap.afterTokenIdx + 1] - ts[gap.afterTokenIdx])f)"
                )
            }
        }

        guard !repairs.isEmpty else { return nil }

        // Reconstruct text from vocabulary tokens with insertions at repair points
        var parts: [String] = []
        for (i, tokenId) in hypothesis.ySequence.enumerated() {
            let tok = vocabulary[tokenId]?.replacingOccurrences(of: "▁", with: " ") ?? ""
            parts.append(tok)
            if let inserted = repairs[i] {
                parts.append(" \(inserted)")
            }
        }

        let repaired = parts.joined()
            .replacingOccurrences(of: "  +", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)

        return repaired
    }
}
