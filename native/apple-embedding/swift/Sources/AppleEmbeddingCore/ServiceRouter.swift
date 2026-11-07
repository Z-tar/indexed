import Foundation

public struct ServiceConfiguration: Sendable {
    public var host: String
    public var port: UInt16
    public var defaultDimension: Int
    public var maxRequestBytes: Int
    public var maxConcurrentRequests: Int
    public var maxQueuedRequests: Int
    public var maxRequestDurationMS: Int
    public var authToken: String?

    public init(
        host: String = "127.0.0.1",
        port: UInt16 = 18_768,
        defaultDimension: Int = 2048,
        maxRequestBytes: Int = 64 * 1024 * 1024,
        maxConcurrentRequests: Int = 1,
        maxQueuedRequests: Int = 16,
        maxRequestDurationMS: Int = 300_000,
        authToken: String? = nil
    ) {
        self.host = host
        self.port = port
        self.defaultDimension = defaultDimension
        self.maxRequestBytes = maxRequestBytes
        self.maxConcurrentRequests = maxConcurrentRequests
        self.maxQueuedRequests = maxQueuedRequests
        self.authToken = authToken
        self.maxRequestDurationMS = maxRequestDurationMS
    }
}

public struct HTTPRequest: Sendable {
    public let method: String
    public let path: String
    public let headers: [String: String]
    public let body: Data
}

public struct HTTPResponse: Sendable {
    public let status: Int
    public let body: Data
    public let headers: [String: String]

    public init(status: Int, body: Data = Data(), headers: [String: String] = [:]) {
        self.status = status
        self.body = body
        self.headers = headers
    }
}

public final class ServiceRouter: @unchecked Sendable {
    private let engine: EmbeddingEngine
    private let configuration: ServiceConfiguration
    public let vectorStore: NativeVectorStore?
    private let startedAt = Date()
    private let requestSlots: DispatchSemaphore
    private let requestCapacity: DispatchSemaphore
    private let requestsLock = NSLock()
    private let loadLock = NSLock()
    private let metricsLock = NSLock()
    private var activeRequests: [String: CancellationToken] = [:]
    private var queuedRequests = 0
    private var runningRequests = 0
    private var requestsTotal = 0
    private var requestsSucceeded = 0
    private var requestsFailed = 0
    private var cumulativeRequestMilliseconds = 0.0
    private var lastRequestMilliseconds = 0.0

    public init(
        engine: EmbeddingEngine,
        configuration: ServiceConfiguration,
        vectorStore: NativeVectorStore? = nil
    ) throws {
        guard officialDimensions.contains(configuration.defaultDimension) else {
            throw ServiceError.invalidRequest("默认维度不受支持")
        }
        guard configuration.maxConcurrentRequests > 0 else {
            throw ServiceError.invalidRequest("maxConcurrentRequests 必须大于零")
        }
        guard configuration.maxConcurrentRequests <= engine.maxConcurrentRequests else {
            throw ServiceError.invalidRequest("Requested concurrency exceeds engine capacity")
        }
        guard configuration.maxQueuedRequests >= 0 else {
            throw ServiceError.invalidRequest("maxQueuedRequests 不能小于零")
        }
        guard (1...600_000).contains(configuration.maxRequestDurationMS) else {
            throw ServiceError.invalidRequest("Invalid maximum request duration")
        }
        self.engine = engine
        self.configuration = configuration
        self.vectorStore = vectorStore
        self.requestSlots = DispatchSemaphore(value: configuration.maxConcurrentRequests)
        self.requestCapacity = DispatchSemaphore(
            value: configuration.maxConcurrentRequests + configuration.maxQueuedRequests
        )
    }

    public func route(_ request: HTTPRequest, cancellation: CancellationToken = CancellationToken()) -> HTTPResponse {
        do {
            if request.method == "OPTIONS" { return try response(status: 204, value: nil) }
            try authenticate(request)
            let pathname = request.path.components(separatedBy: "?").first ?? request.path

            if request.method == "GET", pathname == "/health" { return try health() }
            if request.method == "GET", pathname == "/v1/models" { return try models() }
            if request.method == "GET", pathname == "/api/status" { return try apiStatus() }
            if request.method == "GET", pathname == "/api/embedding-backend" { return try apiEmbeddingBackend() }
            if request.method == "GET", pathname == "/api/vectors" { return try apiVectors() }
            if request.method == "GET", pathname == "/api/extension-config" { return try apiExtensionConfig() }
            if request.method == "GET", pathname == "/api/videos" { return try apiVideos(request) }
            if request.method == "GET", pathname == "/api/extension-library" { return try apiExtensionLibrary(request) }
            if request.method == "DELETE", pathname.hasPrefix("/api/videos/") { return try apiDeleteVideo(pathname) }
            if request.method == "POST", pathname == "/api/search" {
                return try apiSearch(request, cancellation: cancellation)
            }
            if request.method == "POST", pathname == "/api/search/image" {
                return try apiSearchImage(request, cancellation: cancellation)
            }
            if request.method == "POST", (pathname == "/api/ingest" || pathname.hasPrefix("/api/ingest/")) {
                return try apiIngest(request, cancellation: cancellation)
            }
            if request.method == "DELETE", (pathname.hasPrefix("/v1/requests/") || pathname.hasPrefix("/api/embedding/v1/requests/")) {
                return try cancel(pathname)
            }
            if request.method == "POST", (pathname == "/v1/embeddings" || pathname == "/api/embedding/v1/embeddings" || pathname == "/api/embedding") {
                return try embeddings(request, cancellation: cancellation)
            }
            if (request.method == "GET" || request.method == "HEAD"), let staticResponse = serveStaticAsset(pathname, method: request.method) {
                return staticResponse
            }
            throw ServiceError.notFound("Not found")
        } catch ServiceError.cancelled {
            return errorResponse(cancellation.hasExpired ? .deadlineExceeded : .cancelled)
        } catch let error as ServiceError {
            return errorResponse(error)
        } catch {
            fputs("[apple-embedding] request failed: \(error)\n", stderr)
            return errorResponse(.internalFailure(String(describing: error)))
        }
    }

    func configureDeadline(_ request: HTTPRequest, cancellation: CancellationToken) throws {
        var duration = configuration.maxRequestDurationMS
        if let header = request.headers["x-indexed-timeout-ms"] {
            guard !header.isEmpty, header.utf8.allSatisfy({ (48...57).contains($0) }),
                  let value = Int(header), (1...600_000).contains(value) else {
                throw ServiceError.invalidRequest("X-Indexed-Timeout-MS must be an integer from 1 to 600000")
            }
            duration = min(duration, value)
        }
        cancellation.limit(toMilliseconds: duration)
    }

    private func authenticate(_ request: HTTPRequest) throws {
        guard let expected = configuration.authToken, !expected.isEmpty else { return }
        let bearer = request.headers["authorization"].flatMap { value -> String? in
            let prefix = "Bearer "
            guard value.count >= prefix.count,
                  value.prefix(prefix.count).caseInsensitiveCompare(prefix) == .orderedSame else {
                return nil
            }
            return String(value.dropFirst(prefix.count))
        }
        let supplied = bearer ?? request.headers["x-indexed-token"]
        guard let supplied, constantTimeEqual(supplied, expected) else {
            throw ServiceError.unauthorized("缺少或无效的本地服务令牌")
        }
    }

    private func constantTimeEqual(_ left: String, _ right: String) -> Bool {
        let a = Array(left.utf8)
        let b = Array(right.utf8)
        var difference = a.count ^ b.count
        for index in 0 ..< max(a.count, b.count) {
            difference |= Int((index < a.count ? a[index] : 0)
                ^ (index < b.count ? b[index] : 0))
        }
        return difference == 0
    }

    private func health() throws -> HTTPResponse {
        let metadata = engine.metadata
        let load = loadLock.withLock { (queued: queuedRequests, running: runningRequests) }
        let metrics = metricsLock.withLock {
            (
                total: requestsTotal,
                succeeded: requestsSucceeded,
                failed: requestsFailed,
                cumulativeMS: cumulativeRequestMilliseconds,
                lastMS: lastRequestMilliseconds
            )
        }
        return try response(status: 200, value: JSONValue.object([
            "status": .string("ready"),
            "backend": .string(metadata.backend),
            "uptime_seconds": .number(Date().timeIntervalSince(startedAt)),
            "load_seconds": .number(metadata.loadSeconds),
            "warmup_seconds": .number(metadata.warmupSeconds),
            "allocated_bytes": .number(Double(metadata.allocatedBytes)),
            "mlx_allocated_bytes": .number(Double(metadata.allocatedBytes)),
            "mps_allocated_bytes": .number(
                Double(metadata.mlxDevice == "gpu" ? metadata.allocatedBytes : 0)
            ),
            "package_fingerprint": metadata.packageFingerprint.map(JSONValue.string) ?? .null,
            "compute_devices": .array(metadata.computeDevices.map(JSONValue.string)),
            "decoder_segment_loaded": .bool(metadata.decoderSegmentLoaded),
            "decoder_bundle_loaded": .bool(metadata.decoderBundleLoaded),
            "decoder_minimum_tokens": metadata.decoderMinimumTokens
                .map { .number(Double($0)) } ?? .null,
            "decoder_maximum_tokens": metadata.decoderMaximumTokens
                .map { .number(Double($0)) } ?? .null,
            "decoder_bucket_token_limits": .array(
                metadata.decoderBucketTokenLimits.map { .number(Double($0)) }
            ),
            "decoder_segment_fingerprint": metadata.decoderSegmentFingerprint
                .map(JSONValue.string) ?? .null,
            "decoder_bundle_fingerprint": metadata.decoderBundleFingerprint
                .map(JSONValue.string) ?? .null,
            "decoder_loading_strategy": metadata.decoderLoadingStrategy
                .map(JSONValue.string) ?? .null,
            "language_compute": .string(metadata.languageCompute),
            "mlx_decoder_loaded": .bool(metadata.mlxDecoderLoaded),
            "mlx_device": .string(metadata.mlxDevice),
            "max_model_len": .number(Double(metadata.maxModelLength)),
            "private_ane": metadata.privateANE ?? .null,
            "scheduling": metadata.scheduling ?? .null,
            "running_requests": .number(Double(load.running)),
            "queued_requests": .number(Double(load.queued)),
            "max_concurrent_requests": .number(Double(configuration.maxConcurrentRequests)),
            "max_queued_requests": .number(Double(configuration.maxQueuedRequests)),
            "requests_total": .number(Double(metrics.total)),
            "requests_succeeded": .number(Double(metrics.succeeded)),
            "requests_failed": .number(Double(metrics.failed)),
            "request_latency_ms_average": .number(
                metrics.total > 0 ? metrics.cumulativeMS / Double(metrics.total) : 0
            ),
            "request_latency_ms_last": .number(metrics.lastMS),
        ]))
    }

    private func models() throws -> HTTPResponse {
        let metadata = engine.metadata
        let ordered = [configuration.defaultDimension]
            + officialDimensions.filter { $0 != configuration.defaultDimension }
        let data = try ordered.map { dimension -> JSONValue in
            .object([
                "id": .string(try modelID(dimension: dimension)),
                "object": .string("model"),
                "owned_by": .string("indexed-apple"),
                "dimension": .number(Double(dimension)),
                "embedding_space": .string(embeddingSpace(dimension)),
                "max_model_len": .number(Double(metadata.maxModelLength)),
                "modalities": .array([.string("text"), .string("image"), .string("video")]),
                "backend": .string(metadata.backend),
                "language_compute": .string(metadata.languageCompute),
            ])
        }
        return try response(status: 200, value: .object([
            "object": .string("list"), "data": .array(data),
        ]))
    }

    private func embeddings(_ request: HTTPRequest, cancellation: CancellationToken) throws -> HTTPResponse {
        try configureDeadline(request, cancellation: cancellation)
        try cancellation.check()
        let requestStartedAt = Date()
        var succeeded = false
        defer {
            let elapsed = Date().timeIntervalSince(requestStartedAt) * 1_000
            metricsLock.withLock {
                requestsTotal += 1
                if succeeded { requestsSucceeded += 1 }
                else { requestsFailed += 1 }
                cumulativeRequestMilliseconds += elapsed
                lastRequestMilliseconds = elapsed
            }
        }
        guard !request.body.isEmpty else { throw ServiceError.invalidRequest("请求体为空") }
        guard request.body.count <= configuration.maxRequestBytes else {
            throw ServiceError.invalidRequest("请求体超过 Apple helper 限制", status: 413)
        }
        let input: EmbeddingRequest
        do { input = try JSONDecoder().decode(EmbeddingRequest.self, from: request.body) }
        catch { throw ServiceError.invalidRequest("请求体不是有效 JSON 对象") }

        let selectedDimension = try dimension(
            for: input.model,
            default: configuration.defaultDimension
        )
        let messages = try normalizedMessages(from: input)
        let requestID = input.requestID ?? UUID().uuidString.lowercased()
        guard (1...128).contains(requestID.utf8.count), requestID.utf8.allSatisfy({
            (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0)
                || [45, 46, 58, 95].contains($0)
        }) else { throw ServiceError.invalidRequest("request_id must contain 1–128 ASCII letters, digits, or -._:") }
        guard requestCapacity.wait(timeout: .now()) == .success else {
            throw ServiceError.overloaded("原生 embedding 服务队列已满，请稍后重试")
        }
        defer { requestCapacity.signal() }

        try requestsLock.withLock {
            guard activeRequests[requestID] == nil else {
                throw ServiceError.invalidRequest("request_id is already active", status: 409)
            }
            activeRequests[requestID] = cancellation
        }
        defer { requestsLock.withLock { activeRequests[requestID] = nil } }

        loadLock.withLock { queuedRequests += 1 }
        var acquiredRequestSlot = false
        defer {
            loadLock.withLock {
                if acquiredRequestSlot { runningRequests -= 1 }
                else { queuedRequests -= 1 }
            }
            if acquiredRequestSlot { requestSlots.signal() }
        }
        while requestSlots.wait(timeout: .now() + .milliseconds(50)) != .success {
            try cancellation.check()
        }
        loadLock.withLock {
            queuedRequests -= 1
            runningRequests += 1
        }
        acquiredRequestSlot = true

        try cancellation.check()
        let result = try engine.embed(
            messages: messages, dimension: selectedDimension, cancellation: cancellation
        )
        guard result.vector.count == selectedDimension else {
            throw ServiceError.internalFailure(
                "推理引擎返回 \(result.vector.count) 维，预期 \(selectedDimension) 维"
            )
        }
        let vector = result.vector.map { JSONValue.number(Double($0)) }
        let timings = result.timingsMS.mapValues(JSONValue.number)
        let output = try response(status: 200, value: .object([
            "object": .string("list"),
            "model": .string(try modelID(dimension: selectedDimension)),
            "data": .array([.object([
                "object": .string("embedding"),
                "index": .number(0),
                "embedding": .array(vector),
            ])]),
            "usage": .object([
                "prompt_tokens": .number(Double(result.promptTokens)),
                "total_tokens": .number(Double(result.promptTokens)),
            ]),
            "indexed": .object([
                "backend": .string(engine.metadata.backend),
                "modality": .string(result.modality),
                "dimension": .number(Double(selectedDimension)),
                "embedding_space": .string(embeddingSpace(selectedDimension)),
                "request_id": .string(requestID),
                "timings_ms": .object(timings),
            ]),
        ]))
        succeeded = true
        return output
    }

    private func cancel(_ path: String) throws -> HTTPResponse {
        let requestID = String(path.dropFirst("/v1/requests/".count))
        guard !requestID.isEmpty else { throw ServiceError.notFound("Not found") }
        let token = requestsLock.withLock { activeRequests[requestID] }
        guard let token else { throw ServiceError.notFound("未找到运行中的请求：\(requestID)") }
        token.cancel()
        return try response(status: 202, value: .object([
            "status": .string("cancelling"), "request_id": .string(requestID),
        ]))
    }

    private func apiEmbeddingBackend() throws -> HTTPResponse {
        let dim = configuration.defaultDimension
        let space = embeddingSpace(dim)
        let model = (try? modelID(dimension: dim)) ?? "wemm-embedding-2b-apple-2048"
        return try response(status: 200, value: .object([
            "managed": .bool(true),
            "state": .string("ready"),
            "mode": .string("fast"),
            "visionCompute": .string("ane"),
            "executionMode": .string("c"),
            "pid": .number(Double(ProcessInfo.processInfo.processIdentifier)),
            "baseUrl": .string("http://127.0.0.1:\(configuration.port)"),
            "model": .string(model),
            "dimension": .number(Double(dim)),
            "embeddingSpace": .string(space),
            "startedAt": .number(startedAt.timeIntervalSince1970 * 1000),
            "restartCount": .number(0),
            "lastExit": .null,
            "lastError": .null,
            "stderrTail": .array([]),
            "ready": .object([
                "status": .string("ready"),
                "url": .string("http://127.0.0.1:\(configuration.port)"),
                "default_model": .string(model),
                "backend": .string(engine.metadata.backend),
                "execution_mode": .string("c")
            ])
        ]))
    }

    private func apiStatus() throws -> HTTPResponse {
        var storageVal: JSONValue = .object([
            "ok": .bool(false),
            "provider": .string("none"),
            "collections": .array([])
        ])
        var visualReachable = false
        var transcriptReachable = false

        if let store = vectorStore {
            let counts = store.countByCollection()
            var collectionsArr: [JSONValue] = []
            for (col, cnt) in counts {
                collectionsArr.append(.object([
                    "name": .string(col),
                    "count": .number(Double(cnt))
                ]))
            }
            if !counts.keys.contains("video_visual") {
                collectionsArr.append(.object(["name": .string("video_visual"), "count": .number(0)]))
            }
            if !counts.keys.contains("video_transcript") {
                collectionsArr.append(.object(["name": .string("video_transcript"), "count": .number(0)]))
            }
            storageVal = .object([
                "ok": .bool(true),
                "provider": .string("native-sqlite"),
                "provider_label": .string("Apple Native SQLite + SIMD Accelerate"),
                "path": .string(store.databasePath),
                "collections": .array(collectionsArr)
            ])
            visualReachable = true
            transcriptReachable = true
        }

        let dim = configuration.defaultDimension
        let space = embeddingSpace(dim)
        let model = (try? modelID(dimension: dim)) ?? "wemm-embedding-base"

        let payload: JSONValue = .object([
            "ok": .bool(true),
            "activeProfile": .string("apple-wemm"),
            "label": .string("Apple Silicon Native (Zero-Node)"),
            "embeddingProvider": .string("apple-native"),
            "embeddingModel": .string(model),
            "embeddingDimension": .number(Double(dim)),
            "embeddingSpace": .string(space),
            "storageProvider": .string("apple-native-sqlite"),
            "embedding": .object([
                "ok": .bool(true),
                "dimension": .number(Double(dim))
            ]),
            "storage": storageVal,
            "visualReachable": .bool(visualReachable),
            "transcriptReachable": .bool(transcriptReachable),
            "reranker": .object([
                "ok": .bool(true),
                "enabled": .bool(false)
            ])
        ])
        return try response(status: 200, value: payload)
    }

    private func apiVectors() throws -> HTTPResponse {
        guard let store = vectorStore else {
            return try response(status: 200, value: .object([
                "total": .number(0),
                "collections": .object([:])
            ]))
        }
        let counts = store.countByCollection()
        var countsObj: [String: JSONValue] = [:]
        for (k, v) in counts {
            countsObj[k] = .number(Double(v))
        }
        return try response(status: 200, value: .object([
            "total": .number(Double(store.count())),
            "collections": .object(countsObj)
        ]))
    }

    private func apiExtensionConfig() throws -> HTTPResponse {
        let dim = configuration.defaultDimension
        let space = embeddingSpace(dim)
        let model = (try? modelID(dimension: dim)) ?? "wemm-embedding-base"
        let payload: JSONValue = .object([
            "profileId": .string("apple-wemm"),
            "label": .string("Apple Silicon Native (Zero-Node)"),
            "embeddingBaseUrl": .string("http://127.0.0.1:\(configuration.port)"),
            "embeddingModel": .string(model),
            "embeddingDimension": .number(Double(dim)),
            "embeddingInputStyle": .string("wemm"),
            "embeddingSpace": .string(space),
            "embeddingProvider": .string("apple-native"),
            "storageProvider": .string("native-sqlite"),
            "ossRegion": .string("local"),
            "ossAccountId": .string("local"),
            "ossBucket": .string("local"),
            "ossVisualIndex": .string("video_visual"),
            "ossTranscriptIndex": .string("video_transcript")
        ])
        return try response(status: 200, value: payload)
    }

    private func parseQueryParam(_ path: String, name: String) -> String? {
        guard let query = path.components(separatedBy: "?").dropFirst().first else { return nil }
        for pair in query.components(separatedBy: "&") {
            let parts = pair.components(separatedBy: "=")
            if parts.first == name {
                return parts.count > 1 ? parts[1].removingPercentEncoding ?? parts[1] : ""
            }
        }
        return nil
    }

    private func withInferenceSlot<T>(cancellation: CancellationToken, action: () throws -> T) throws -> T {
        guard requestCapacity.wait(timeout: .now()) == .success else {
            throw ServiceError.overloaded("原生 embedding 服务队列已满，请稍后重试")
        }
        defer { requestCapacity.signal() }

        loadLock.withLock { queuedRequests += 1 }
        var acquiredRequestSlot = false
        defer {
            loadLock.withLock {
                if acquiredRequestSlot { runningRequests -= 1 }
                else { queuedRequests -= 1 }
            }
            if acquiredRequestSlot { requestSlots.signal() }
        }
        while requestSlots.wait(timeout: .now() + .milliseconds(50)) != .success {
            try cancellation.check()
        }
        loadLock.withLock {
            queuedRequests -= 1
            runningRequests += 1
        }
        acquiredRequestSlot = true

        try cancellation.check()
        let startedAt = Date()
        var succeeded = false
        defer {
            let elapsed = Date().timeIntervalSince(startedAt) * 1_000
            metricsLock.withLock {
                requestsTotal += 1
                if succeeded { requestsSucceeded += 1 }
                else { requestsFailed += 1 }
                cumulativeRequestMilliseconds += elapsed
                lastRequestMilliseconds = elapsed
            }
        }
        let result = try action()
        succeeded = true
        return result
    }

    private func apiSearch(_ request: HTTPRequest, cancellation: CancellationToken) throws -> HTTPResponse {
        try configureDeadline(request, cancellation: cancellation)
        try cancellation.check()
        guard !request.body.isEmpty else { throw ServiceError.invalidRequest("Search body cannot be empty") }
        guard let bodyObj = try? JSONDecoder().decode([String: JSONValue].self, from: request.body) else {
            throw ServiceError.invalidRequest("Search body must be a JSON object")
        }
        let query = bodyObj["query"]?.stringValue ?? ""
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ServiceError.invalidRequest("搜索文字不能为空")
        }
        let limit = Int(bodyObj["limit"]?.numberValue ?? 30)
        var filter: [String: String]? = nil
        if let videoId = bodyObj["videoId"]?.stringValue, !videoId.isEmpty {
            filter = ["video_id": videoId]
        }

        let dim = configuration.defaultDimension
        guard let store = vectorStore else {
            throw ServiceError.internalFailure("Native vector store not configured")
        }

        let visualMsg: [JSONValue] = [
            .object([
                "role": .string("user"),
                "content": .array([
                    .object(["type": .string("text"), "text": .string("Find an image or video that best matches the following description: \(query)")])
                ])
            ])
        ]
        let transcriptMsg: [JSONValue] = [
            .object([
                "role": .string("user"),
                "content": .array([
                    .object(["type": .string("text"), "text": .string("Instruct: Given a search request, retrieve the video transcript segment whose spoken content best matches it.\nQuery: \(query)")])
                ])
            ])
        ]

        let visualEmbedding = try withInferenceSlot(cancellation: cancellation) {
            try engine.embed(messages: visualMsg, dimension: dim, cancellation: cancellation)
        }
        let transcriptEmbedding = try withInferenceSlot(cancellation: cancellation) {
            try engine.embed(messages: transcriptMsg, dimension: dim, cancellation: cancellation)
        }

        let visualHits = store.search(collection: "video_visual", queryVector: visualEmbedding.vector, limit: limit, filter: filter)
        let transcriptHits = store.search(collection: "video_transcript", queryVector: transcriptEmbedding.vector, limit: limit, filter: filter)

        let payload: JSONValue = .object([
            "query": .string(query),
            "profile": .string("apple-wemm"),
            "embeddingModel": .string((try? modelID(dimension: dim)) ?? "wemm-embedding-base"),
            "embeddingSpace": .string(embeddingSpace(dim)),
            "reranked": .bool(false),
            "warnings": .array([]),
            "visual": .array(visualHits.map { formatVectorHit($0, modality: "visual") }),
            "transcript": .array(transcriptHits.map { formatVectorHit($0, modality: "transcript") })
        ])
        return try response(status: 200, value: payload)
    }

    private func apiSearchImage(_ request: HTTPRequest, cancellation: CancellationToken) throws -> HTTPResponse {
        try configureDeadline(request, cancellation: cancellation)
        try cancellation.check()
        guard !request.body.isEmpty else { throw ServiceError.invalidRequest("Search body cannot be empty") }
        guard let bodyObj = try? JSONDecoder().decode([String: JSONValue].self, from: request.body) else {
            throw ServiceError.invalidRequest("Search body must be a JSON object")
        }
        guard let imageBase64 = bodyObj["imageBase64"]?.stringValue, !imageBase64.isEmpty else {
            throw ServiceError.invalidRequest("imageBase64 不能为空")
        }
        let mimeType = bodyObj["mimeType"]?.stringValue ?? "image/jpeg"
        let query = bodyObj["query"]?.stringValue ?? ""
        let limit = Int(bodyObj["limit"]?.numberValue ?? 30)
        var filter: [String: String]? = nil
        if let videoId = bodyObj["videoId"]?.stringValue, !videoId.isEmpty {
            filter = ["video_id": videoId]
        }

        let dim = configuration.defaultDimension
        guard let store = vectorStore else {
            throw ServiceError.internalFailure("Native vector store not configured")
        }

        let dataUri = "data:\(mimeType);base64,\(imageBase64)"
        var contentItems: [JSONValue] = [
            .object([
                "type": .string("image_url"),
                "image_url": .object(["url": .string(dataUri)])
            ])
        ]
        if !query.isEmpty {
            contentItems.append(.object(["type": .string("text"), "text": .string(query)]))
        } else {
            contentItems.append(.object(["type": .string("text"), "text": .string("Represent this image for retrieval.")]))
        }
        let msgs: [JSONValue] = [.object(["role": .string("user"), "content": .array(contentItems)])]

        let embedding = try withInferenceSlot(cancellation: cancellation) {
            try engine.embed(messages: msgs, dimension: dim, cancellation: cancellation)
        }
        let visualHits = store.search(collection: "video_visual", queryVector: embedding.vector, limit: limit, filter: filter)

        let payload: JSONValue = .object([
            "query": .string(query),
            "profile": .string("apple-wemm"),
            "embeddingModel": .string((try? modelID(dimension: dim)) ?? "wemm-embedding-base"),
            "embeddingSpace": .string(embeddingSpace(dim)),
            "reranked": .bool(false),
            "warnings": .array([]),
            "visual": .array(visualHits.map { formatVectorHit($0, modality: "visual") }),
            "transcript": .array([])
        ])
        return try response(status: 200, value: payload)
    }

    private func formatVectorHit(_ hit: VectorSearchResult, modality: String) -> JSONValue {
        let meta = hit.metadata
        return .object([
            "id": .string(hit.id),
            "modality": .string(modality),
            "score": .number(Double(hit.score)),
            "distance": .number(Double(hit.distance)),
            "videoId": .string(meta["video_id"]?.stringValue ?? ""),
            "sourceSite": .string(meta["source_site"]?.stringValue ?? "youtube"),
            "title": .string(meta["title"]?.stringValue ?? "Untitled"),
            "channel": .string(meta["channel_name"]?.stringValue ?? ""),
            "sourceUrl": .string(meta["source_url"]?.stringValue ?? ""),
            "startTime": meta["start_time"] ?? meta["timestamp"] ?? .number(0),
            "endTime": meta["end_time"] ?? meta["timestamp"] ?? .number(0),
            "transcript": .string(meta["transcript_text"]?.stringValue ?? ""),
            "preview": .string(meta["preview_data_uri"]?.stringValue ?? meta["thumbnail_url"]?.stringValue ?? ""),
            "embeddingModel": .string(meta["embedding_model"]?.stringValue ?? ""),
            "embeddingSpace": .string(meta["embedding_space"]?.stringValue ?? "")
        ])
    }

    private func apiIngest(_ request: HTTPRequest, cancellation: CancellationToken) throws -> HTTPResponse {
        try configureDeadline(request, cancellation: cancellation)
        try cancellation.check()
        guard let store = vectorStore else {
            throw ServiceError.internalFailure("Native vector store not configured")
        }
        guard !request.body.isEmpty else { throw ServiceError.invalidRequest("Ingest body cannot be empty") }
        guard let bodyObj = try? JSONDecoder().decode(JSONValue.self, from: request.body) else {
            throw ServiceError.invalidRequest("Ingest body must be valid JSON")
        }

        let dim = configuration.defaultDimension
        let pathname = request.path.components(separatedBy: "?").first ?? request.path
        let kindFromPath = pathname.hasPrefix("/api/ingest/")
            ? String(pathname.dropFirst("/api/ingest/".count))
            : ""

        let records: [[String: JSONValue]]
        if let arr = bodyObj.arrayValue {
            records = arr.compactMap { $0.objectValue }
        } else if let obj = bodyObj.objectValue {
            records = [obj]
        } else {
            throw ServiceError.invalidRequest("Expected record object or array")
        }

        var lastId = ""
        for record in records {
            try cancellation.check()
            let kind = kindFromPath.isEmpty
                ? (record["kind"]?.stringValue ?? record["record_type"]?.stringValue ?? "visual")
                : kindFromPath
            let isTranscript = kind == "transcript" || kind.contains("transcript")
            let collection = isTranscript ? "video_transcript" : "video_visual"

            let id = record["key"]?.stringValue ?? record["id"]?.stringValue ?? UUID().uuidString
            lastId = id

            var vector: [Float] = []
            if let vecArr = record["vector"]?.arrayValue ?? record["embedding"]?.arrayValue {
                vector = vecArr.compactMap { $0.numberValue.map(Float.init) }
            } else if let float32Arr = record["data"]?.objectValue?["float32"]?.arrayValue {
                vector = float32Arr.compactMap { $0.numberValue.map(Float.init) }
            }

            if vector.isEmpty {
                if isTranscript {
                    let text = record["text"]?.stringValue ?? record["transcript_text"]?.stringValue ?? ""
                    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        continue
                    }
                    let msgs: [JSONValue] = [
                        .object([
                            "role": .string("user"),
                            "content": .array([
                                .object(["type": .string("text"), "text": .string("Represent the spoken content of this video segment for semantic retrieval: \(text)")])
                            ])
                        ])
                    ]
                    let res = try withInferenceSlot(cancellation: cancellation) {
                        try engine.embed(messages: msgs, dimension: dim, cancellation: cancellation)
                    }
                    vector = res.vector
                } else {
                    if let frameArr = record["video_frames"]?.arrayValue, !frameArr.isEmpty {
                        let normalizedFrames: [JSONValue] = frameArr.map { frameVal in
                            guard let obj = frameVal.objectValue else { return frameVal }
                            var updated = obj
                            if let b64 = obj["image_base64"]?.stringValue {
                                let mime = obj["mime_type"]?.stringValue ?? "image/jpeg"
                                updated["image_url"] = .object(["url": .string("data:\(mime);base64,\(b64)")])
                            }
                            return .object(updated)
                        }
                        let msgs: [JSONValue] = [
                            .object([
                                "role": .string("user"),
                                "content": .array([
                                    .object(["type": .string("video_frames"), "frames": .array(normalizedFrames)]),
                                    .object(["type": .string("text"), "text": .string("Represent this video.")])
                                ])
                            ])
                        ]
                        let res = try withInferenceSlot(cancellation: cancellation) {
                            try engine.embed(messages: msgs, dimension: dim, cancellation: cancellation)
                        }
                        vector = res.vector
                    } else if let videoBase64 = record["video_base64"]?.stringValue, !videoBase64.isEmpty {
                        let mimeType = record["mime_type"]?.stringValue ?? "video/webm"
                        let msgs: [JSONValue] = [
                            .object([
                                "role": .string("user"),
                                "content": .array([
                                    .object(["type": .string("video_url"), "video_url": .object(["url": .string("data:\(mimeType);base64,\(videoBase64)")])]),
                                    .object(["type": .string("text"), "text": .string("Represent this video.")])
                                ])
                            ])
                        ]
                        let res = try withInferenceSlot(cancellation: cancellation) {
                            try engine.embed(messages: msgs, dimension: dim, cancellation: cancellation)
                        }
                        vector = res.vector
                    } else if let imageBase64 = record["image_base64"]?.stringValue ?? record["preview_base64"]?.stringValue,
                       !imageBase64.isEmpty {
                        let mimeType = record["mime_type"]?.stringValue ?? "image/jpeg"
                        let msgs: [JSONValue] = [
                            .object([
                                "role": .string("user"),
                                "content": .array([
                                    .object([
                                        "type": .string("image_url"),
                                        "image_url": .object(["url": .string("data:\(mimeType);base64,\(imageBase64)")])
                                    ])
                                ])
                            ])
                        ]
                        let res = try withInferenceSlot(cancellation: cancellation) {
                            try engine.embed(messages: msgs, dimension: dim, cancellation: cancellation)
                        }
                        vector = res.vector
                    } else {
                        continue
                    }
                }
            }

            guard vector.count == dim else {
                throw ServiceError.invalidRequest("Vector dimension \(vector.count) does not match expected \(dim)")
            }

            try store.upsert(collection: collection, id: id, vector: vector, metadata: record)
        }

        return try response(status: 200, value: .object([
            "ok": .bool(true),
            "cached": .bool(false),
            "record_id": .string(lastId)
        ]))
    }

    private func apiVideos(_ request: HTTPRequest) throws -> HTTPResponse {
        let limit = parseQueryParam(request.path, name: "limit").flatMap(Int.init) ?? 100
        guard let store = vectorStore else {
            return try response(status: 200, value: .object([
                "profile": .string("apple-wemm"),
                "embeddingModel": .string((try? modelID(dimension: configuration.defaultDimension)) ?? "wemm-embedding-base"),
                "count": .number(0),
                "videos": .array([])
            ]))
        }
        let videos = aggregateVideos(store: store)
        let dim = configuration.defaultDimension
        let sliced = Array(videos.prefix(limit))
        return try response(status: 200, value: .object([
            "profile": .string("apple-wemm"),
            "embeddingModel": .string((try? modelID(dimension: dim)) ?? "wemm-embedding-base"),
            "count": .number(Double(videos.count)),
            "videos": .array(sliced.map { $0.toJSON() })
        ]))
    }

    private func apiExtensionLibrary(_ request: HTTPRequest) throws -> HTTPResponse {
        let includeThumbnail = parseQueryParam(request.path, name: "includeThumbnail") == "true"
        guard let store = vectorStore else {
            return try response(status: 200, value: .object([
                "videos": .array([]),
                "video_count": .number(0),
                "visual_count": .number(0),
                "transcript_count": .number(0),
                "storage_provider": .string("native-sqlite"),
                "storage_label": .string("Apple Native SQLite + SIMD Accelerate")
            ]))
        }
        let videos = aggregateVideos(store: store)
        let totalVisual = videos.reduce(0) { $0 + $1.visualCount }
        let totalTranscript = videos.reduce(0) { $0 + $1.transcriptCount }
        return try response(status: 200, value: .object([
            "videos": .array(videos.map { $0.toJSON(includeThumbnail: includeThumbnail) }),
            "video_count": .number(Double(videos.count)),
            "visual_count": .number(Double(totalVisual)),
            "transcript_count": .number(Double(totalTranscript)),
            "storage_provider": .string("native-sqlite"),
            "storage_label": .string("Apple Native SQLite + SIMD Accelerate")
        ]))
    }

    private func apiDeleteVideo(_ path: String) throws -> HTTPResponse {
        guard let store = vectorStore else {
            throw ServiceError.internalFailure("Native vector store not configured")
        }
        let pathname = path.components(separatedBy: "?").first ?? path
        let parts = pathname.split(separator: "/").map(String.init)
        guard parts.count >= 4, parts[1] == "videos" else {
            throw ServiceError.invalidRequest("URL format must be /api/videos/:site/:videoId")
        }
        let site = parts[2]
        let videoId = parts[3]

        func idsToDelete(collection: String) -> [String] {
            var matching: [String] = []
            for (id, meta) in store.allMetadatas(collection: collection) {
                let vId = meta["video_id"]?.stringValue ?? meta["videoId"]?.stringValue ?? ""
                let sSite = meta["source_site"]?.stringValue ?? meta["sourceSite"]?.stringValue ?? ""
                if vId == videoId && (site.isEmpty || sSite.isEmpty || sSite == site) {
                    matching.append(id)
                }
            }
            return matching
        }

        let visualIds = idsToDelete(collection: "video_visual")
        let transcriptIds = idsToDelete(collection: "video_transcript")

        try store.delete(collection: "video_visual", ids: visualIds)
        try store.delete(collection: "video_transcript", ids: transcriptIds)

        return try response(status: 200, value: .object([
            "ok": .bool(true),
            "sourceSite": .string(site),
            "videoId": .string(videoId),
            "visualDeleted": .number(Double(visualIds.count)),
            "transcriptDeleted": .number(Double(transcriptIds.count))
        ]))
    }

    private struct VideoAggregate {
        var sourceSite: String
        var videoId: String
        var title: String
        var channel: String
        var sourceUrl: String
        var thumbnail: String
        var visualCount: Int
        var transcriptCount: Int
        var visualSegments: Set<Int>
        var transcriptSegments: Set<Int>
        var duration: Double
        var segmentInterval: Double
        var indexedAt: Double

        func toJSON(includeThumbnail: Bool = true) -> JSONValue {
            .object([
                "video_id": .string(videoId),
                "source_site": .string(sourceSite),
                "title": .string(title),
                "channel_name": .string(channel),
                "source_url": .string(sourceUrl),
                "thumbnail_url": .string(includeThumbnail ? thumbnail : ""),
                "visual_count": .number(Double(visualCount)),
                "transcript_count": .number(Double(transcriptCount)),
                "saved_visual_segments": .array(visualSegments.sorted().map { .number(Double($0)) }),
                "saved_transcript_segments": .array(transcriptSegments.sorted().map { .number(Double($0)) }),
                "duration": .number(duration),
                "segment_interval": .number(segmentInterval),
                "last_indexed_at": .number(indexedAt)
            ])
        }
    }

    private func aggregateVideos(store: NativeVectorStore) -> [VideoAggregate] {
        var videos: [String: VideoAggregate] = [:]

        func processRow(_ id: String, _ meta: [String: JSONValue], modality: String) {
            let site = meta["source_site"]?.stringValue ?? "youtube"
            let videoId = meta["video_id"]?.stringValue ?? ""
            guard !videoId.isEmpty else { return }
            let key = "\(site):\(videoId)"

            var current = videos[key] ?? VideoAggregate(
                sourceSite: site,
                videoId: videoId,
                title: meta["title"]?.stringValue ?? "Untitled",
                channel: meta["channel_name"]?.stringValue ?? meta["channel"]?.stringValue ?? "",
                sourceUrl: meta["source_url"]?.stringValue ?? meta["sourceUrl"]?.stringValue ?? "",
                thumbnail: meta["preview_data_uri"]?.stringValue ?? meta["thumbnail_url"]?.stringValue ?? "",
                visualCount: 0,
                transcriptCount: 0,
                visualSegments: [],
                transcriptSegments: [],
                duration: 0,
                segmentInterval: 0,
                indexedAt: 0
            )

            let segmentIndex = meta["segment_index"]?.numberValue.map(Int.init)
                ?? (meta["segment_index"]?.stringValue.flatMap(Int.init))
            if modality == "visual" {
                current.visualCount += 1
                if let seg = segmentIndex, seg >= 0 { current.visualSegments.insert(seg) }
            } else {
                current.transcriptCount += 1
                if let seg = segmentIndex, seg >= 0 { current.transcriptSegments.insert(seg) }
            }

            let dur = meta["duration"]?.numberValue ?? meta["end_time"]?.numberValue ?? 0
            current.duration = max(current.duration, dur)
            let interval = meta["segment_interval"]?.numberValue ?? 0
            current.segmentInterval = max(current.segmentInterval, interval)
            let indexedMs = meta["indexed_at_ms"]?.numberValue ?? 0
            current.indexedAt = max(current.indexedAt, indexedMs)

            if current.thumbnail.isEmpty {
                current.thumbnail = meta["preview_data_uri"]?.stringValue ?? meta["thumbnail_url"]?.stringValue ?? ""
            }
            videos[key] = current
        }

        for (id, meta) in store.allMetadatas(collection: "video_visual") {
            processRow(id, meta, modality: "visual")
        }
        for (id, meta) in store.allMetadatas(collection: "video_transcript") {
            processRow(id, meta, modality: "transcript")
        }

        return videos.values.sorted { $0.indexedAt > $1.indexedAt }
    }

    private func embeddingSpace(_ dimension: Int) -> String {
        if let template = engine.metadata.embeddingSpaceTemplate {
            return template.replacingOccurrences(of: "{dimension}", with: String(dimension))
        }
        return "\(modelPrefix)-\(dimension)-wemm-indexed-v1"
    }

    private func response(status: Int, value: JSONValue?) throws -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let body = try value.map { try encoder.encode($0) } ?? Data()
        return HTTPResponse(status: status, body: body, headers: defaultHeaders)
    }

    private func errorResponse(_ error: ServiceError) -> HTTPResponse {
        let value = JSONValue.object(["error": .object([
            "message": .string(error.localizedDescription),
            "type": .string(error.responseType),
            "code": .string(error.responseCode),
        ])])
        return (try? response(status: error.status, value: value))
            ?? HTTPResponse(status: 500, body: Data())
    }

    private var defaultHeaders: [String: String] {
        [
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Indexed-Token, X-Indexed-Timeout-MS",
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        ]
    }

    private func serveStaticAsset(_ pathname: String, method: String = "GET") -> HTTPResponse? {
        let cleanPath = pathname == "/" ? "/index.html" : pathname
        let validExtensions = [".html", ".js", ".css", ".svg", ".png", ".jpg", ".jpeg", ".ico", ".json", ".woff", ".woff2"]
        guard validExtensions.contains(where: { cleanPath.hasSuffix($0) }) else { return nil }

        let candidates = [
            ProcessInfo.processInfo.environment["INDEXED_DASHBOARD_DIR"],
            "/Users/r/Documents/antigravity/indexed/dist/dashboard",
            FileManager.default.currentDirectoryPath + "/dist/dashboard"
        ].compactMap { $0 }

        for dir in candidates {
            let fileURL = URL(fileURLWithPath: dir).appendingPathComponent(cleanPath.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
            if FileManager.default.fileExists(atPath: fileURL.path),
               let data = try? Data(contentsOf: fileURL) {
                var mimeType = "application/octet-stream"
                if cleanPath.hasSuffix(".html") { mimeType = "text/html; charset=utf-8" }
                else if cleanPath.hasSuffix(".js") { mimeType = "text/javascript; charset=utf-8" }
                else if cleanPath.hasSuffix(".css") { mimeType = "text/css; charset=utf-8" }
                else if cleanPath.hasSuffix(".svg") { mimeType = "image/svg+xml; charset=utf-8" }
                else if cleanPath.hasSuffix(".json") { mimeType = "application/json; charset=utf-8" }
                else if cleanPath.hasSuffix(".png") { mimeType = "image/png" }

                let body = method == "HEAD" ? Data() : data
                return HTTPResponse(status: 200, body: body, headers: [
                    "Content-Type": mimeType,
                    "Content-Length": String(data.count),
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*"
                ])
            }
        }
        return nil
    }
}
