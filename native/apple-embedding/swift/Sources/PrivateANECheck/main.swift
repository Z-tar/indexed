import Foundation
import MLX
import MLXNN
import PrivateANEBridge
import AppleEmbeddingCore

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: message, code: 1) }
}

func errorText(_ buffer: [CChar]) -> String {
    String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

func check() throws {
    if CommandLine.arguments.contains("--pipeline-check") {
        try InferenceValidation.checkScheduling()
        try InferenceValidation.checkRouter()
        let values = try InferenceValidation.checkVisionPatches()
        print("{\"scheduling\":\"passed\",\"cpu_patches\":\"bit-exact\",\"values\":\(values)}")
        return
    }
    try PrivateGDNValidation.checkGraphAndProfile()
    if CommandLine.arguments.contains("--gdn") {
        let values = try PrivateGDNValidation.checkHardware()
        print("{\"gdn\":\"passed\",\"values\":\(values),\"oracle\":\"sequential-cpu\",\"routing\":\"passed\"}")
        return
    }
    if CommandLine.arguments.contains("--hybrid-mlp") {
        let values = try PrivateMLPValidation.check()
        print("{\"hybrid_mlp\":\"passed\",\"values\":\(values),\"dtypes\":[\"float16\",\"bfloat16\"],\"routing\":\"passed\"}")
        return
    }
    var error = [CChar](repeating: 0, count: 4096)
    let invalid = indexed_ane_create(nil, 0, nil, 0, nil, 0, nil, 0, nil, 0, &error, error.count)
    try require(invalid == nil && errorText(error).contains("Invalid private ANE graph"), "Invalid graph was accepted")
    try require(!indexed_ane_write(nil, 0, nil, 0, &error, error.count), "Null input accepted")
    try require(!indexed_ane_read(nil, 0, nil, 0, &error, error.count), "Null output accepted")
    try require(indexed_ane_lock_output(nil, 0, 0, &error, error.count) == nil, "Null output borrow accepted")
    indexed_ane_unlock_output(nil, 0)
    try require(indexed_ane_retain_output_surface(nil, 0) == nil, "Null retained output accepted")
    indexed_ane_release_surface(nil)
    try require(!indexed_ane_evaluate(nil, &error, error.count), "Null evaluation accepted")
    indexed_ane_free(nil)
    let quantized = try PrivateANELinearWeights(dense: [127, 2.5, 3.5, -2.5, -3.5, 0], inputDimension: 6, outputDimension: 1)
    try require(quantized.data == [127, 2, 4, -2, -4, 0], "Row quantization no longer rounds ties to even")
    guard CommandLine.arguments.contains("--hardware") else {
        print("{\"bounds\":\"passed\",\"hardware\":\"not-requested\"}")
        return
    }
    let graph = """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
    {
      func main<ios18>(tensor<fp16, [1, 128, 1, 128]> x) {
        tensor<fp16, [1, 128, 1, 128]> y = add(x=x, y=x)[name=string("double")];
      } -> (y);
    }
    """
    var inputSize = 128 * 128 * MemoryLayout<Float16>.stride
    var outputSize = inputSize
    let program = Array(graph.utf8).withUnsafeBufferPointer { text in
        indexed_ane_create(text.baseAddress, text.count, nil, 0, nil, 0, &inputSize, 1, &outputSize, 1, &error, error.count)
    }
    guard let handle = program else { throw NSError(domain: errorText(error), code: 1) }
    defer { indexed_ane_free(handle) }
    let input = (0..<(128 * 128)).map { Float16($0 % 32) / 32 }
    var output = [Float16](repeating: 0, count: input.count)
    try input.withUnsafeBytes { bytes in
        try require(!indexed_ane_write(handle, 1, bytes.baseAddress, bytes.count, &error, error.count), "Out-of-range surface accepted")
        try require(!indexed_ane_write(handle, 0, bytes.baseAddress, bytes.count + 1, &error, error.count), "Oversized copy accepted")
        try require(indexed_ane_write(handle, 0, bytes.baseAddress, bytes.count, &error, error.count), errorText(error))
    }
    try require(indexed_ane_evaluate(handle, &error, error.count), errorText(error))
    try output.withUnsafeMutableBytes { bytes in
        try require(!indexed_ane_read(handle, 0, bytes.baseAddress, bytes.count - 1, &error, error.count), "Truncated output accepted")
        try require(indexed_ane_read(handle, 0, bytes.baseAddress, bytes.count, &error, error.count), errorText(error))
    }
    try require(output == input.map { $0 * 2 }, "Private ANE output differs from exact FP16 reference")
    try require(indexed_ane_lock_output(handle, 1, output.count * 2, &error, error.count) == nil, "Out-of-range output borrow accepted")
    try require(indexed_ane_lock_output(handle, 0, output.count * 2 - 1, &error, error.count) == nil, "Truncated output borrow accepted")
    guard let borrowed = indexed_ane_lock_output(handle, 0, output.count * 2, &error, error.count) else {
        throw ServiceError.model(errorText(error))
    }
    let borrowedValues = Array(UnsafeBufferPointer(start: borrowed.assumingMemoryBound(to: Float16.self), count: output.count))
    indexed_ane_unlock_output(handle, 0)
    try require(borrowedValues == output, "Borrowed ANE output differs from copied output")
    let linearMIL = try PrivateANELinearWeights.mil(inputDimension: 128, outputDimension: 128, sequenceLength: 64)
    var identity = [Int8](repeating: 0, count: 128 * 128)
    for index in 0..<128 { identity[index * 128 + index] = 2 }
    let linear = try PrivateANEProgram(mil: linearMIL, weights: identity, scales: [Float16](repeating: 0.5, count: 128), inputElements: [128 * 64], outputElements: [128 * 64])
    let linearInput = Array(input.prefix(128 * 64))
    do {
        _ = try linear.evaluate(updating: [:], cancellation: CancellationToken())
        throw ServiceError.internalFailure("Uninitialized input surface was evaluated")
    } catch ServiceError.invalidRequest { }
    do {
        try linear.prepareInput(linearInput, at: 1)
        throw ServiceError.internalFailure("Invalid constant surface index was accepted")
    } catch ServiceError.invalidRequest { }
    try require(linear.evaluations == 0, "Rejected sparse input still submitted ANE work")
    let linearOutput = try linear.evaluate([linearInput], cancellation: CancellationToken())
    try require(linearOutput == [linearInput], "INT8 weight blob / FP16 scale linear kernel differs from identity")
    let triple = try PrivateANEProgram(mil: linearMIL, weights: identity,
        scales: [Float16](repeating: 1.5, count: 128), inputElements: [128 * 64],
        outputElements: [128 * 64], sharing: linear)
    let tripleOutput = try triple.evaluate([linearInput], cancellation: CancellationToken())
    try require(tripleOutput == [linearInput.map { $0 * 3 }], "Shared surfaces changed the second model's weights")
    try require(linearOutput == [linearInput], "Shared readback overwrote a retained earlier result")
    let subsequent = try linear.evaluate([linearInput], cancellation: CancellationToken())
    try require(subsequent == [linearInput], "Reusing the first model read another model's results")
    do {
        _ = try triple.evaluate(updating: [:], cancellation: CancellationToken())
        throw ServiceError.internalFailure("Shared workspace accepted stale inputs")
    } catch ServiceError.invalidRequest { }
    do {
        try triple.prepareInput(linearInput, at: 0)
        throw ServiceError.internalFailure("Shared workspace accepted mutable constant inputs")
    } catch ServiceError.invalidRequest { }
    do {
        _ = try PrivateANEProgram(mil: linearMIL, inputElements: [64], outputElements: [64], sharing: linear)
        throw ServiceError.internalFailure("Shared workspace accepted different surface sizes")
    } catch ServiceError.invalidRequest { }
    let cancelled = CancellationToken(); cancelled.cancel()
    do { _ = try linear.evaluate([linearInput], cancellation: cancelled); throw NSError(domain: "Cancelled program evaluated", code: 1) }
    catch ServiceError.cancelled { }
    try require(linear.evaluations == 2, "Cancellation still submitted ANE work")
    print("{\"bounds\":\"passed\",\"hardware\":\"passed\",\"values\":40960,\"linear\":\"passed\",\"shared_workspace\":\"passed\",\"cancel\":\"passed\"}")
}

func probeJSON(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) {
        print(String(decoding: data, as: UTF8.self))
    }
}

func probeValues(_ count: Int, seed: UInt32) -> [Float16] {
    var state = seed
    return (0..<count).map { _ in
        state = state &* 1664525 &+ 1013904223
        return Float16((Float(state >> 8) / Float(1 << 24) * 2 - 1) * 0.5)
    }
}

func probeMedian(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}

func probeFP16Bytes(_ values: [Float16]) -> [Int8] {
    values.withUnsafeBytes { Array($0.assumingMemoryBound(to: Int8.self)) }
}

/// M3-generation compatibility probe: the frozen int8 constexpr flavor versus
/// a pre-dequantized FP16 conv, then timing versus the GPU suffix kernels on
/// the full C-mode prefix shape (in=2048, out=2*4608, seq=2112).
func aneConvProbe() throws {
    let cancellation = CancellationToken()
    do {
        let mil = try PrivateANELinearWeights.mil(inputDimension: 128, outputDimension: 128, sequenceLength: 64)
        _ = try PrivateANEProgram(mil: mil, weights: [Int8](repeating: 1, count: 128 * 128),
            scales: [Float16](repeating: 0.5, count: 128), inputElements: [128 * 64], outputElements: [128 * 64])
        probeJSON(["probe": "int8_small", "compiled": true,
                   "note": "constexpr_blockwise_shift_scale accepted by this ANE compiler"])
    } catch {
        probeJSON(["probe": "int8_small", "compiled": false, "error": String("\(error)".prefix(400))])
    }
    let smallIn = 128, smallOut = 128, smallSeq = 64
    let smallWeight = probeValues(smallIn * smallOut, seed: 5)
    let smallX = probeValues(smallIn * smallSeq, seed: 7)
    do {
        let mil = try PrivateANELinearWeights.milFP16(inputDimension: smallIn, outputDimension: smallOut, sequenceLength: smallSeq)
        let program = try PrivateANEProgram(mil: mil, weights: probeFP16Bytes(smallWeight), scales: [],
            inputElements: [smallIn * smallSeq], outputElements: [smallOut * smallSeq])
        let output = try program.evaluate([smallX], cancellation: cancellation)[0]
        var difference = 0.0, norm = 0.0
        for out in 0..<smallOut { for token in 0..<smallSeq {
            var expected: Float = 0
            for channel in 0..<smallIn {
                expected += Float(smallWeight[out * smallIn + channel]) * Float(smallX[channel * smallSeq + token])
            }
            difference += pow(Double(output[out * smallSeq + token]) - Double(expected), 2)
            norm += pow(Double(expected), 2)
        } }
        probeJSON(["probe": "fp16_small", "compiled": true, "relative_l2": sqrt(difference / max(norm, 1e-20))])
    } catch {
        probeJSON(["probe": "fp16_small", "compiled": false, "error": String("\(error)".prefix(400))])
        return
    }
    Memory.cacheLimit = 512 * 1024 * 1024
    let inDim = 2048, outDim = 9216, sequence = 2112, iterations = 7
    let weight = probeValues(inDim * outDim, seed: 11)
    let x = probeValues(inDim * sequence, seed: 13)
    do {
        let mil = try PrivateANELinearWeights.milFP16(inputDimension: inDim, outputDimension: outDim, sequenceLength: sequence)
        let program = try PrivateANEProgram(mil: mil, weights: probeFP16Bytes(weight), scales: [],
            inputElements: [inDim * sequence], outputElements: [outDim * sequence])
        _ = try program.evaluate([x], cancellation: cancellation)
        var aneTimes: [Double] = []
        for _ in 0..<iterations {
            let started = ContinuousClock.now
            _ = try program.evaluate([x], cancellation: cancellation)
            let duration = started.duration(to: .now).components
            aneTimes.append(Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15)
        }
        probeJSON(["probe": "ane_fp16_bench", "in": inDim, "out": outDim, "seq": sequence,
                   "ms": aneTimes, "median_ms": probeMedian(aneTimes)])
    } catch {
        probeJSON(["probe": "ane_fp16_bench", "compiled": false, "error": String("\(error)".prefix(400))])
    }
    let dense = MLXArray((0..<(inDim * outDim)).map { Float(weight[$0]) }, [outDim, inDim]).asType(.float16)
    let linear = QuantizedLinear(weight: dense, bias: nil, groupSize: 64, bits: 8)
    let gpuX = MLXArray((0..<(inDim * sequence)).map { Float(x[$0]) }, [1, sequence, inDim]).asType(.float16)
    eval(gpuX, linear)
    func timed(_ variant: Bool) -> [Double] {
        var times: [Double] = []
        for _ in 0..<iterations {
            let started = ContinuousClock.now
            let output = variant
                ? PrivateMLPKernels.affine(gpuX, weight: linear.weight, scales: linear.scales, biases: linear.biases!)
                : linear(gpuX)
            eval(output)
            let duration = started.duration(to: .now).components
            times.append(Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15)
        }
        return times
    }
    _ = timed(false)
    _ = timed(true)
    let stock = timed(false)
    let variant = timed(true)
    probeJSON(["probe": "gpu_stock", "in": inDim, "out": outDim, "seq": sequence,
               "ms": stock, "median_ms": probeMedian(stock)])
    probeJSON(["probe": "gpu_variant8", "in": inDim, "out": outDim, "seq": sequence,
               "ms": variant, "median_ms": probeMedian(variant)])
}

do {
    if CommandLine.arguments.contains("--reference-server") { try await runReferenceServer() }
    else if CommandLine.arguments.contains("--gpu-projection-benchmark") { try PrivateMLPValidation.benchmarkProjections() }
    else if CommandLine.arguments.contains("--ane-conv-probe") { try aneConvProbe() }
    else { try check() }
}
catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
