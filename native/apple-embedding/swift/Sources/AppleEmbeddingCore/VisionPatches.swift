import Foundation

/// CPU-only vision staging; no MLX arrays or streams cross the stage boundary.
/// The 256-value normalization table is materialized once with the original MLX
/// expression during model load, preserving its exact FP32 rounding.
struct VisionPatches: Sendable {
    let values: [Float]
    let rows: Int
    static let columns = 3 * 2 * 16 * 16

    static func normalizedCHW(_ rgba: [UInt8], side: Int, table: [Float]) -> [Float] {
        precondition(table.count == 256 && rgba.count == side * side * 4)
        let pixels = side * side
        var result = [Float](repeating: 0, count: pixels * 3)
        rgba.withUnsafeBufferPointer { rgbaBuf in
            table.withUnsafeBufferPointer { tableBuf in
                result.withUnsafeMutableBufferPointer { outBuf in
                    let src = rgbaBuf.baseAddress!
                    let lut = tableBuf.baseAddress!
                    let dst = outBuf.baseAddress!
                    let c1 = pixels
                    let c2 = pixels * 2
                    for pixel in 0..<pixels {
                        let base = pixel * 4
                        dst[pixel] = lut[Int(src[base])]
                        dst[c1 + pixel] = lut[Int(src[base + 1])]
                        dst[c2 + pixel] = lut[Int(src[base + 2])]
                    }
                }
            }
        }
        return result
    }

    static func pair(_ first: [Float], _ second: [Float], side: Int) -> VisionPatches {
        precondition(side > 0 && side.isMultiple(of: 32))
        precondition(first.count == side * side * 3 && second.count == first.count)
        let grid = side / 16
        var result = [Float](repeating: 0, count: grid * grid * columns)
        // Match reshape [1,2,3,gh/2,2,16,gw/2,2,16], transpose
        // [0,3,6,4,7,2,1,5,8], then flatten [gh*gw,1536].
        first.withUnsafeBufferPointer { firstBuf in
            second.withUnsafeBufferPointer { secondBuf in
                result.withUnsafeMutableBufferPointer { outBuf in
                    let f0 = firstBuf.baseAddress!
                    let f1 = secondBuf.baseAddress!
                    let dst = outBuf.baseAddress!
                    let halfGrid = grid / 2
                    let planeStride = side * side
                    for blockY in 0..<halfGrid {
                        for blockX in 0..<halfGrid {
                            for mergeY in 0..<2 {
                                for mergeX in 0..<2 {
                                    let patch = ((blockY * halfGrid + blockX) * 2 + mergeY) * 2 + mergeX
                                    let patchBase = patch * columns
                                    let spatialOrigin = (blockY * 32 + mergeY * 16) * side + blockX * 32 + mergeX * 16
                                    for channel in 0..<3 {
                                        let origin = channel * planeStride + spatialOrigin
                                        let dest0 = patchBase + (channel * 2) * 256
                                        let dest1 = dest0 + 256
                                        for y in 0..<16 {
                                            let rowOffset = origin + y * side
                                            (dst + dest0 + y * 16).update(from: f0 + rowOffset, count: 16)
                                            (dst + dest1 + y * 16).update(from: f1 + rowOffset, count: 16)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return VisionPatches(values: result, rows: grid * grid)
    }
}

struct VisionFeatures: Sendable {
    let values: [Float]
    let shape: [Int]
}
