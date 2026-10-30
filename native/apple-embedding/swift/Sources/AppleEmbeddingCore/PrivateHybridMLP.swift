import Foundation
import Metal
import MLX
import MLXNN

/// ANE program weight flavor. int8 keeps the frozen oMLX row-quantized blob
/// (H16+ compilers only); fp16 ships pre-dequantized weights so pre-H16 ANE
/// compilers (M1-M3 generation) accept the program. auto selects fp16 on M1-M3
/// and int8 on M4+.
public enum PrivateMLPWeightDtype: String, Sendable {
    case auto
    case int8
    case fp16

    public var resolved: PrivateMLPWeightDtype {
        guard self == .auto else { return self }
        guard let name = MTLCreateSystemDefaultDevice()?.name else { return .int8 }
        if name.hasPrefix("Apple M1") || name.hasPrefix("Apple M2") || name.hasPrefix("Apple M3") {
            return .fp16
        }
        return .int8
    }
}

/// First fixed contract being ported, matching the frozen Q8/G64 C reference.
/// Reject unsupported settings instead of silently claiming a different kernel.
public struct PrivateMLPConfiguration: Sendable {
    public let sequenceLength: Int
    public let shortSequenceLength: Int?
    public let fraction: Double
    public let maxLayers: Int
    public let weightDtype: PrivateMLPWeightDtype
    public var tailMinimumTokens: Int { sequenceLength / 2 }
    public var shortMinimumTokens: Int { 16 }

    public init(sequenceLength: Int = 2112, shortSequenceLength: Int? = 256, fraction: Double = 0.75, maxLayers: Int = 24,
                weightDtype: PrivateMLPWeightDtype = .auto) throws {
        guard sequenceLength >= 64, sequenceLength <= 8192, sequenceLength.isMultiple(of: 64),
              shortSequenceLength == nil || (shortSequenceLength! >= 64 && shortSequenceLength! <= sequenceLength && shortSequenceLength!.isMultiple(of: 64)),
              fraction > 0, fraction < 1, fraction.isFinite, maxLayers > 0, maxLayers <= 24 else {
            throw ServiceError.invalidRequest("Invalid private MLP configuration")
        }
        self.sequenceLength = sequenceLength; self.shortSequenceLength = shortSequenceLength
        self.fraction = fraction; self.maxLayers = maxLayers
        self.weightDtype = weightDtype.resolved
    }
}

/// Request scope plus counters are shared by all substituted layers. The model
/// remains single-writer; device overlap happens inside a layer, not by running
/// two complete language models concurrently.
public final class PrivateMLPContext: @unchecked Sendable {
    private let requestLock = NSLock()
    private let counterLock = NSLock()
    fileprivate var cancellation = CancellationToken()
    var requestCancellation: CancellationToken { cancellation }
    var recurrenceAllowed = false
    var recurrence: PrivateGDNRecurrence?
    var aneLane: InferenceGate?
    // All MLP layers execute under the same language owner. Their compiled
    // weights differ, but only one layer needs the fixed-shape I/O at a time.
    fileprivate var sharedProgram: PrivateANEProgram?
    fileprivate var sharedShortProgram: PrivateANEProgram?
    fileprivate var activeWeightDtype: PrivateMLPWeightDtype = .int8
    private var operationCount = 0
    private var projectionCount = 0
    private var stageMilliseconds: [String: Double] = [:]
    private var fallbackCounts: [String: Int] = [:]
    private var failures: [String: String] = [:]
    public var operations: Int { counterLock.withLock { operationCount } }
    public var fallbacks: [String: Int] { counterLock.withLock { fallbackCounts } }

    public init() {}

    public func withRequest<T>(_ token: CancellationToken, _ body: () throws -> T) throws -> T {
        try requestLock.withLock {
            cancellation = token
            recurrenceAllowed = false
            defer { cancellation = CancellationToken(); recurrenceAllowed = false }
            try token.check()
            let result = try body()
            try token.check()
            return result
        }
    }

    fileprivate func recordOperations(_ count: Int) { counterLock.withLock { operationCount += count } }
    func recordGPUProjection() { counterLock.withLock { projectionCount += 1 } }
    fileprivate func recordStage(_ name: String, since start: ContinuousClock.Instant) {
        let duration = start.duration(to: .now).components
        let milliseconds = Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15
        counterLock.withLock { stageMilliseconds[name, default: 0] += milliseconds }
    }
    fileprivate func recordStages(_ stages: [String: Double]) {
        counterLock.withLock { for (name, time) in stages { stageMilliseconds[name, default: 0] += time } }
    }
    func fallback(_ reason: String, layer: String, error: Error? = nil) {
        counterLock.withLock {
            fallbackCounts[reason, default: 0] += 1
            if let error { failures[layer] = String(describing: error) }
        }
    }

    public var diagnostics: JSONValue {
        counterLock.withLock {
            .object([
                "backend": .string("swift-private-ane-mlp-candidate"),
                "weight_dtype": .string(activeWeightDtype.rawValue),
                "gpu_kernel": .string("classic-q8-g64-bm64-bk32-bn64"),
                "long_projection_kernel": .string(FastQ8Projection.isSupportedDevice ? "classic-q8-g64-bm64-bk32-bn64" : "stock-mlx"),
                "native_profile": .object(["mlp": .object([
                    "operations": .number(Double(operationCount)),
                    "gpu_projection_operations": .number(Double(projectionCount)),
                    "stage_ms": .object(stageMilliseconds.mapValues { .number($0) }),
                ])]),
                "fallbacks": .object(fallbackCounts.mapValues { .number(Double($0)) }),
                "failed_layers": .object(failures.mapValues { .string($0) }),
                "recurrence": recurrence?.diagnostics ?? .null,
            ])
        }
    }
}

/// Fixed ANE gate/up prefix + asynchronous Q8 GPU suffix + Q8 down projection.
/// Original weights are retained for short/tail/failed-kernel fallback; GPU
/// suffix slices borrow original storage directly without concatenating copies.
public final class PrivateHybridMLP: Module, UnaryLayer {
    private let original: any UnaryLayer
    private let down: QuantizedLinear
    private let program: PrivateANEProgram
    private let shortProgram: PrivateANEProgram?
    private let gpuGateWeight: MLXArray
    private let gpuGateScales: MLXArray
    private let gpuGateBiases: MLXArray
    private let gpuUpWeight: MLXArray
    private let gpuUpScales: MLXArray
    private let gpuUpBiases: MLXArray
    private let configuration: PrivateMLPConfiguration
    private let context: PrivateMLPContext
    private let layer: String
    private let inputDimension: Int
    private let aneHidden: Int
    private let gpuHidden: Int
    private let dtype: DType
    private var failed = false

    public init(original module: Module, configuration: PrivateMLPConfiguration,
                context: PrivateMLPContext, layer: String) throws {
        let children = Dictionary(uniqueKeysWithValues: module.children().flattened())
        guard let original = module as? any UnaryLayer,
              let gate = children["gate_proj"] as? QuantizedLinear,
              let up = children["up_proj"] as? QuantizedLinear,
              let down = children["down_proj"] as? QuantizedLinear,
              gate.bits == 8, up.bits == 8, down.bits == 8,
              gate.groupSize == 64, up.groupSize == 64, down.groupSize == 64,
              gate.mode == .affine, up.mode == .affine, down.mode == .affine,
              gate.bias == nil, up.bias == nil, down.bias == nil,
              gate.shape == up.shape, down.shape == (gate.shape.1, gate.shape.0),
              let gateBias = gate.biases, let upBias = up.biases,
              [DType.float16, .bfloat16].contains(gate.scales.dtype),
              gate.scales.dtype == up.scales.dtype, gate.scales.dtype == down.scales.dtype else {
            throw ServiceError.model("Private MLP requires matching dense Q8/G64 affine gate/up/down layers")
        }
        let aneHidden = (Int(Double(gate.shape.0) * configuration.fraction) / 64) * 64
        let gpuHidden = gate.shape.0 - aneHidden
        guard aneHidden > 0, gpuHidden > 0, gpuHidden.isMultiple(of: 64),
              gate.shape.1.isMultiple(of: 64) else {
            throw ServiceError.model("Private MLP channel partition is unsupported")
        }
        let builtProgram: PrivateANEProgram
        let builtShortProgram: PrivateANEProgram?
        let resolvedDtype = configuration.weightDtype.resolved
        context.activeWeightDtype = resolvedDtype
        switch resolvedDtype {
        case .int8, .auto:
            let dense = concatenated([gate, up].map { linear in
                dequantized(linear.weight[0..<aneHidden], scales: linear.scales[0..<aneHidden],
                            biases: linear.biases![0..<aneHidden], groupSize: 64, bits: 8).asType(.float32)
            }, axis: 0)
            let quantized = try PrivateANELinearWeights(dense: dense.asArray(Float.self),
                inputDimension: gate.shape.1, outputDimension: 2 * aneHidden)
            builtProgram = try PrivateANEProgram(
                mil: PrivateANELinearWeights.mil(inputDimension: gate.shape.1, outputDimension: 2 * aneHidden,
                                                sequenceLength: configuration.sequenceLength),
                weights: quantized.data, scales: quantized.scales,
                inputElements: [gate.shape.1 * configuration.sequenceLength],
                outputElements: [2 * aneHidden * configuration.sequenceLength], sharing: context.sharedProgram)
            if let shortLen = configuration.shortSequenceLength {
                builtShortProgram = try PrivateANEProgram(
                    mil: PrivateANELinearWeights.mil(inputDimension: gate.shape.1, outputDimension: 2 * aneHidden,
                                                    sequenceLength: shortLen),
                    weights: quantized.data, scales: quantized.scales,
                    inputElements: [gate.shape.1 * shortLen],
                    outputElements: [2 * aneHidden * shortLen], sharing: context.sharedShortProgram)
            } else { builtShortProgram = nil }
        case .fp16:
            // Host-side pre-dequantization: pre-H16 ANE compilers reject the
            // constexpr int8 ops, so ship raw FP16 conv weights directly.
            let denseFP16 = concatenated([gate, up].map { linear in
                dequantized(linear.weight[0..<aneHidden], scales: linear.scales[0..<aneHidden],
                            biases: linear.biases![0..<aneHidden], groupSize: 64, bits: 8).asType(.float16)
            }, axis: 0)
            let packed = denseFP16.asArray(Float16.self)
            let bytes = packed.withUnsafeBytes { Array($0.assumingMemoryBound(to: Int8.self)) }
            builtProgram = try PrivateANEProgram(
                mil: PrivateANELinearWeights.milFP16(inputDimension: gate.shape.1, outputDimension: 2 * aneHidden,
                                                     sequenceLength: configuration.sequenceLength),
                weights: bytes, scales: [],
                inputElements: [gate.shape.1 * configuration.sequenceLength],
                outputElements: [2 * aneHidden * configuration.sequenceLength], sharing: context.sharedProgram)
            if let shortLen = configuration.shortSequenceLength {
                builtShortProgram = try PrivateANEProgram(
                    mil: PrivateANELinearWeights.milFP16(inputDimension: gate.shape.1, outputDimension: 2 * aneHidden,
                                                         sequenceLength: shortLen),
                    weights: bytes, scales: [],
                    inputElements: [gate.shape.1 * shortLen],
                    outputElements: [2 * aneHidden * shortLen], sharing: context.sharedShortProgram)
            } else { builtShortProgram = nil }
        }
        self.program = builtProgram
        if context.sharedProgram == nil { context.sharedProgram = builtProgram }
        self.shortProgram = builtShortProgram
        if context.sharedShortProgram == nil { context.sharedShortProgram = builtShortProgram }
        self.gpuGateWeight = gate.weight[aneHidden...]
        self.gpuGateScales = gate.scales[aneHidden...]
        self.gpuGateBiases = gateBias[aneHidden...]
        self.gpuUpWeight = up.weight[aneHidden...]
        self.gpuUpScales = up.scales[aneHidden...]
        self.gpuUpBiases = upBias[aneHidden...]
        eval(gpuGateWeight, gpuGateScales, gpuGateBiases, gpuUpWeight, gpuUpScales, gpuUpBiases)
        self.original = original
        self.down = FastQ8Projection.isSupportedDevice ? FastQ8Projection(down, context: context) : down
        self.configuration = configuration
        self.context = context; self.layer = layer; self.inputDimension = gate.shape.1
        self.aneHidden = aneHidden; self.gpuHidden = gpuHidden; self.dtype = gate.scales.dtype
        super.init()
    }

    private func exact(targetProgram: PrivateANEProgram? = nil, targetLength: Int? = nil,
                       aneInput: MLXArray, gpuInput: MLXArray, validRows: Int, project: Bool = true) throws -> MLXArray {
        let activeProgram = targetProgram ?? self.program
        let activeLength = targetLength ?? configuration.sequenceLength
        try context.cancellation.check()
        let producerStarted = ContinuousClock.now
        let packed = PrivateMLPKernels.packedANEInput(aneInput)
        // Complete the input producer before starting either consumer. ANE's
        // evaluation can overlap the already submitted GPU suffix. The packed
        // buffer is consumed in place: no intermediate [Float16] allocation.
        eval(packed)
        context.recordStage("input_producer", since: producerStarted)
        try context.cancellation.check()
        let gpuGate = PrivateMLPKernels.affine(gpuInput, weight: gpuGateWeight, scales: gpuGateScales, biases: gpuGateBiases)
        let gpuUp = PrivateMLPKernels.affine(gpuInput, weight: gpuUpWeight, scales: gpuUpScales, biases: gpuUpBiases)
        asyncEval(gpuGate, gpuUp)
        // Always join submitted GPU work, including ANE error/cancellation paths,
        // before releasing request-local arrays or reusing buffers.
        defer { eval(gpuGate, gpuUp) }
        let before = activeProgram.evaluations
        defer { context.recordOperations(activeProgram.evaluations - before) }
        let aneStarted = ContinuousClock.now
        let activation = try activeProgram.evaluateActivationPacked(
            packed, gpuGate: gpuGate, gpuUp: gpuUp,
            validRows: validRows, rows: activeLength,
            aneHidden: aneHidden, gpuHidden: gpuHidden,
            cancellation: context.cancellation, lane: context.aneLane
        )
        context.recordStage("ane_evaluate_activation", since: aneStarted)
        context.recordStages(activeProgram.lastEvaluationStages)
        try context.cancellation.check()
        return project ? down(activation) : activation
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if context.cancellation.isCancelled { return .zeros(x.shape, dtype: x.dtype) }
        guard !failed else {
            context.fallback("latched_failure", layer: layer)
            return original(x)
        }
        guard x.ndim == 3, x.dim(0) == 1, x.dim(-1) == inputDimension, x.dtype == dtype else {
            context.fallback("unsupported_shape_or_dtype", layer: layer)
            return original(x)
        }
        let rows = x.dim(1)
        let minTokens = shortProgram != nil ? configuration.shortMinimumTokens : configuration.tailMinimumTokens
        guard rows >= minTokens else {
            context.fallback("short_input", layer: layer)
            return original(x)
        }
        do {
            // Path 1: Short input bucket (rows < tailMinimumTokens, e.g. < 1056)
            if let shortProg = shortProgram, let shortLen = configuration.shortSequenceLength,
               rows < configuration.tailMinimumTokens {
                var parts: [MLXArray] = []
                for start in stride(from: 0, to: rows, by: shortLen) {
                    let count = min(shortLen, rows - start)
                    let part = x[0..., start..<(start + count), 0...]
                    if count == shortLen {
                        parts.append(try exact(targetProgram: shortProg, targetLength: shortLen,
                            aneInput: part, gpuInput: part, validRows: count))
                    } else if count >= configuration.shortMinimumTokens {
                        let padded = concatenated([part, .zeros([1, shortLen - count, inputDimension], dtype: dtype)], axis: 1)
                        parts.append(try exact(targetProgram: shortProg, targetLength: shortLen,
                            aneInput: padded, gpuInput: part, validRows: count))
                    } else {
                        context.fallback("short_tail", layer: layer)
                        parts.append(original(part))
                    }
                }
                return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
            }

            // Path 2: Long input bucket (rows >= tailMinimumTokens, e.g. >= 1056)
            var parts: [MLXArray] = []
            let length = configuration.sequenceLength
            let tail = rows % length
            // Keep the fixed ANE blocks intact. On measured long M4/M3 inputs,
            // trim their padding before one Q8 down projection so GPU work is
            // proportional to real tokens. On tail blocks (1056..<2112), only
            // pad the ANE input while keeping GPU suffix and merge on `count`
            // real rows so GPU never wastes compute on zero-padded rows.
            let joinedDown = FastQ8Projection.isSupportedDevice && rows >= 2112
                && (tail == 0 || tail >= configuration.tailMinimumTokens)
            for start in stride(from: 0, to: rows, by: length) {
                let count = min(length, rows - start)
                let part = x[0..., start..<(start + count), 0...]
                if count == length {
                    parts.append(try exact(aneInput: part, gpuInput: part, validRows: length, project: !joinedDown))
                } else if count >= configuration.tailMinimumTokens {
                    let padded = concatenated([part, .zeros([1, length - count, inputDimension], dtype: dtype)], axis: 1)
                    parts.append(try exact(aneInput: padded, gpuInput: part, validRows: count, project: !joinedDown))
                } else if let shortProg = shortProgram, let shortLen = configuration.shortSequenceLength {
                    for subStart in stride(from: 0, to: count, by: shortLen) {
                        let subCount = min(shortLen, count - subStart)
                        let subPart = part[0..., subStart..<(subStart + subCount), 0...]
                        if subCount == shortLen {
                            parts.append(try exact(targetProgram: shortProg, targetLength: shortLen,
                                aneInput: subPart, gpuInput: subPart, validRows: subCount, project: !joinedDown))
                        } else if subCount >= configuration.shortMinimumTokens {
                            let padded = concatenated([subPart, .zeros([1, shortLen - subCount, inputDimension], dtype: dtype)], axis: 1)
                            parts.append(try exact(targetProgram: shortProg, targetLength: shortLen,
                                aneInput: padded, gpuInput: subPart, validRows: subCount, project: !joinedDown))
                        } else {
                            context.fallback("short_tail", layer: layer)
                            parts.append(original(subPart))
                        }
                    }
                } else {
                    context.fallback("short_tail", layer: layer)
                    parts.append(original(part))
                }
            }
            let result = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
            return joinedDown ? down(result) : result
        } catch ServiceError.cancelled {
            return .zeros(x.shape, dtype: x.dtype)
        } catch {
            failed = true
            context.fallback("runtime_failure", layer: layer, error: error)
            return original(x)
        }
    }

    /// Compile all replacements before changing the model. No partially patched
    /// model escapes if a layer fails to compile or the expected layout differs.
    /// Populate all 24 layer entries (keeping original modules for trailing
    /// layers when maxLayers < 24) so MLX's unflattened update never hits a
    /// sparse-array crash.
    public static func install(in model: Module, configuration: PrivateMLPConfiguration,
                               context: PrivateMLPContext) throws -> Int {
        let modules = model.namedModules().filter { $0.0.hasSuffix(".mlp") }
        guard modules.count == 24 else { throw ServiceError.model("Private C candidate requires the 24-layer WeMM model") }
        var replacements: [(String, Module)] = []
        var patchedCount = 0
        for (index, (name, module)) in modules.enumerated() {
            if index < configuration.maxLayers {
                replacements.append((name, try PrivateHybridMLP(original: module,
                    configuration: configuration, context: context, layer: name)))
                patchedCount += 1
            } else {
                replacements.append((name, module))
            }
        }
        try model.update(modules: .unflattened(replacements), verify: .all)
        return patchedCount
    }
}
