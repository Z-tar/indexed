import Foundation

/// Byte-for-byte compatible implementation of Pillow's 8-bit BICUBIC resize.
///
/// WeMM's published Python path performs a forced Pillow resize to 448×448
/// before Qwen patchification.  Core Image's similarly named filter uses a
/// different downsampling kernel, which is enough to move image embeddings in
/// the unified text/image space.  This is a small Swift port of Pillow's
/// `libImaging/Resample.c` coefficient and two-pass integer code.
enum PillowBicubic {
    private static let precisionBits = 22
    private static let coefficientScale = 1 << precisionBits
    private static let rounding = 1 << (precisionBits - 1)

    private struct CacheKey: Hashable {
        let input: Int
        let output: Int
    }

    private struct Coefficients {
        let starts: [Int]
        let counts: [Int]
        let kernelSize: Int
        let values: [Int32]
    }

    private static let cacheLock = NSLock()
    private nonisolated(unsafe) static var coefficientCache: [CacheKey: Coefficients] = [:]

    static func resizeRGBA(
        _ source: [UInt8],
        width sourceWidth: Int,
        height sourceHeight: Int,
        to targetWidth: Int,
        _ targetHeight: Int
    ) -> [UInt8] {
        precondition(source.count == sourceWidth * sourceHeight * 4)
        let horizontal = cachedCoefficients(input: sourceWidth, output: targetWidth)
        let vertical = cachedCoefficients(input: sourceHeight, output: targetHeight)

        var intermediate = [UInt8](
            repeating: 255,
            count: targetWidth * sourceHeight * 4
        )
        source.withUnsafeBufferPointer { srcBuf in
            intermediate.withUnsafeMutableBufferPointer { interBuf in
                horizontal.starts.withUnsafeBufferPointer { startsBuf in
                    horizontal.counts.withUnsafeBufferPointer { countsBuf in
                        horizontal.values.withUnsafeBufferPointer { valuesBuf in
                            let src = srcBuf.baseAddress!
                            let inter = interBuf.baseAddress!
                            let starts = startsBuf.baseAddress!
                            let counts = countsBuf.baseAddress!
                            let values = valuesBuf.baseAddress!
                            let kSize = horizontal.kernelSize
                            for y in 0 ..< sourceHeight {
                                let rowSrcBase = y * sourceWidth * 4
                                let rowDstBase = y * targetWidth * 4
                                for x in 0 ..< targetWidth {
                                    let start = starts[x]
                                    let count = counts[x]
                                    let kernelOffset = x * kSize
                                    let destination = rowDstBase + x * 4
                                    let srcOffset = rowSrcBase + start * 4
                                    var sum0 = rounding
                                    var sum1 = rounding
                                    var sum2 = rounding
                                    for tap in 0 ..< count {
                                        let coeff = Int(values[kernelOffset + tap])
                                        let px = srcOffset + tap * 4
                                        sum0 += Int(src[px]) * coeff
                                        sum1 += Int(src[px + 1]) * coeff
                                        sum2 += Int(src[px + 2]) * coeff
                                    }
                                    inter[destination] = clipped(sum0 >> precisionBits)
                                    inter[destination + 1] = clipped(sum1 >> precisionBits)
                                    inter[destination + 2] = clipped(sum2 >> precisionBits)
                                }
                            }
                        }
                    }
                }
            }
        }

        var output = [UInt8](repeating: 255, count: targetWidth * targetHeight * 4)
        intermediate.withUnsafeBufferPointer { interBuf in
            output.withUnsafeMutableBufferPointer { outBuf in
                vertical.starts.withUnsafeBufferPointer { startsBuf in
                    vertical.counts.withUnsafeBufferPointer { countsBuf in
                        vertical.values.withUnsafeBufferPointer { valuesBuf in
                            let inter = interBuf.baseAddress!
                            let out = outBuf.baseAddress!
                            let starts = startsBuf.baseAddress!
                            let counts = countsBuf.baseAddress!
                            let values = valuesBuf.baseAddress!
                            let kSize = vertical.kernelSize
                            let rowStride = targetWidth * 4
                            for y in 0 ..< targetHeight {
                                let start = starts[y]
                                let count = counts[y]
                                let kernelOffset = y * kSize
                                let rowDstBase = y * rowStride
                                for x in 0 ..< targetWidth {
                                    let destination = rowDstBase + x * 4
                                    let colBase = start * rowStride + x * 4
                                    var sum0 = rounding
                                    var sum1 = rounding
                                    var sum2 = rounding
                                    for tap in 0 ..< count {
                                        let coeff = Int(values[kernelOffset + tap])
                                        let px = colBase + tap * rowStride
                                        sum0 += Int(inter[px]) * coeff
                                        sum1 += Int(inter[px + 1]) * coeff
                                        sum2 += Int(inter[px + 2]) * coeff
                                    }
                                    out[destination] = clipped(sum0 >> precisionBits)
                                    out[destination + 1] = clipped(sum1 >> precisionBits)
                                    out[destination + 2] = clipped(sum2 >> precisionBits)
                                }
                            }
                        }
                    }
                }
            }
        }
        return output
    }

    private static func cachedCoefficients(input: Int, output: Int) -> Coefficients {
        let key = CacheKey(input: input, output: output)
        cacheLock.lock()
        if let cached = coefficientCache[key] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        let computed = coefficients(input: input, output: output)
        cacheLock.lock()
        if coefficientCache.count >= 32 {
            coefficientCache.removeAll(keepingCapacity: true)
        }
        coefficientCache[key] = computed
        cacheLock.unlock()
        return computed
    }

    private static func coefficients(input: Int, output: Int) -> Coefficients {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        let kernelSize = Int(ceil(support)) * 2 + 1
        let inverseFilterScale = 1.0 / filterScale
        var starts = [Int](repeating: 0, count: output)
        var counts = [Int](repeating: 0, count: output)
        var values = [Int32](repeating: 0, count: output * kernelSize)

        for destination in 0 ..< output {
            let center = (Double(destination) + 0.5) * scale
            let minimum = max(0, Int(center - support + 0.5))
            let maximum = min(input, Int(center + support + 0.5))
            let count = maximum - minimum
            var weights = [Double](repeating: 0, count: count)
            var total = 0.0
            for tap in 0 ..< count {
                let distance = (Double(tap + minimum) - center + 0.5) * inverseFilterScale
                let weight = bicubic(distance)
                weights[tap] = weight
                total += weight
            }
            starts[destination] = minimum
            counts[destination] = count
            if total != 0 {
                for tap in 0 ..< count {
                    let normalized = weights[tap] / total
                    let scaled = normalized * Double(coefficientScale)
                    let rounded = normalized < 0 ? Int(scaled - 0.5) : Int(scaled + 0.5)
                    values[destination * kernelSize + tap] = Int32(rounded)
                }
            }
        }
        return Coefficients(
            starts: starts,
            counts: counts,
            kernelSize: kernelSize,
            values: values
        )
    }

    private static func bicubic(_ input: Double) -> Double {
        let x = abs(input)
        let a = -0.5
        if x < 1.0 {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        }
        if x < 2.0 {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        }
        return 0.0
    }

    private static func clipped(_ value: Int) -> UInt8 {
        UInt8(clamping: value)
    }
}
