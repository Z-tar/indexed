import Accelerate
import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public struct VectorSearchResult: Sendable, Codable {
    public let id: String
    public let score: Float
    public let distance: Float
    public let metadata: [String: JSONValue]

    public init(id: String, score: Float, distance: Float, metadata: [String: JSONValue]) {
        self.id = id
        self.score = score
        self.distance = distance
        self.metadata = metadata
    }
}

public final class NativeVectorStore: @unchecked Sendable {
    private var db: OpaquePointer?
    private let lock = NSRecursiveLock()
    private let databaseURL: URL

    // In-memory cache for ultra-fast SIMD vector scanning
    private final class CollectionCache {
        var ids: [String] = []
        var idToIndex: [String: Int] = [:]
        var vectors: [Float] = [] // Packed contiguously: count * dimension
        var metadatas: [[String: JSONValue]] = []
        let dimension: Int

        init(dimension: Int = 2048) {
            self.dimension = dimension
        }
    }

    private var caches: [String: CollectionCache] = [:]

    public init(databaseURL: URL) throws {
        self.databaseURL = databaseURL
        let directory = databaseURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        if sqlite3_open_v2(databaseURL.path, &handle, flags, nil) != SQLITE_OK {
            let message = String(cString: sqlite3_errmsg(handle))
            throw ServiceError.internalFailure("Failed to open SQLite vector store: \(message)")
        }
        self.db = handle
        try initSchema()
        try preloadCaches()
    }

    deinit {
        lock.lock()
        defer { lock.unlock() }
        if let db = db {
            sqlite3_close_v2(db)
            self.db = nil
        }
    }

    private func initSchema() throws {
        let sql = """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -64000;

        CREATE TABLE IF NOT EXISTS records (
            collection TEXT NOT NULL,
            id TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (collection, id)
        );
        CREATE INDEX IF NOT EXISTS idx_records_collection ON records(collection);
        """
        if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
            let message = String(cString: sqlite3_errmsg(db))
            throw ServiceError.internalFailure("Failed to init SQLite schema: \(message)")
        }
    }

    private func preloadCaches() throws {
        lock.lock()
        defer { lock.unlock() }

        let sql = "SELECT collection, id, metadata_json, embedding FROM records ORDER BY created_at ASC;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            return
        }
        defer { sqlite3_finalize(stmt) }

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let colPtr = sqlite3_column_text(stmt, 0),
                  let idPtr = sqlite3_column_text(stmt, 1),
                  let metaPtr = sqlite3_column_text(stmt, 2),
                  let blobPtr = sqlite3_column_blob(stmt, 3) else {
                continue
            }
            let collection = String(cString: colPtr)
            let id = String(cString: idPtr)
            let metaString = String(cString: metaPtr)
            let blobBytes = sqlite3_column_bytes(stmt, 3)
            let floatCount = Int(blobBytes) / MemoryLayout<Float>.size
            guard floatCount > 0 else { continue }

            let cache = caches[collection] ?? CollectionCache(dimension: floatCount)
            caches[collection] = cache

            let metaData = metaString.data(using: .utf8) ?? Data()
            let metadata = (try? JSONDecoder().decode([String: JSONValue].self, from: metaData)) ?? [:]

            let floatPtr = blobPtr.bindMemory(to: Float.self, capacity: floatCount)
            let floats = Array(UnsafeBufferPointer(start: floatPtr, count: floatCount))

            cache.idToIndex[id] = cache.ids.count
            cache.ids.append(id)
            cache.metadatas.append(metadata)
            cache.vectors.append(contentsOf: floats)
        }
    }

    public func upsert(
        collection: String,
        id: String,
        vector: [Float],
        metadata: [String: JSONValue]
    ) throws {
        lock.lock()
        defer { lock.unlock() }

        let metaData = try JSONEncoder().encode(metadata)
        let metaString = String(data: metaData, encoding: .utf8) ?? "{}"
        let now = Date().timeIntervalSince1970

        let sql = """
        INSERT INTO records (collection, id, metadata_json, embedding, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(collection, id) DO UPDATE SET
            metadata_json = excluded.metadata_json,
            embedding = excluded.embedding,
            created_at = excluded.created_at;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            let message = String(cString: sqlite3_errmsg(db))
            throw ServiceError.internalFailure("SQLite prepare failed: \(message)")
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (collection as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 3, (metaString as NSString).utf8String, -1, SQLITE_TRANSIENT)
        
        let byteCount = vector.count * MemoryLayout<Float>.size
        _ = vector.withUnsafeBytes { rawBuffer in
            sqlite3_bind_blob(stmt, 4, rawBuffer.baseAddress, Int32(byteCount), SQLITE_TRANSIENT)
        }
        sqlite3_bind_double(stmt, 5, now)

        if sqlite3_step(stmt) != SQLITE_DONE {
            let message = String(cString: sqlite3_errmsg(db))
            throw ServiceError.internalFailure("SQLite upsert failed: \(message)")
        }

        // Update in-memory cache
        let cache = caches[collection] ?? CollectionCache(dimension: vector.count)
        caches[collection] = cache

        if let existingIdx = cache.idToIndex[id] {
            cache.metadatas[existingIdx] = metadata
            let start = existingIdx * cache.dimension
            for i in 0..<min(vector.count, cache.dimension) {
                cache.vectors[start + i] = vector[i]
            }
        } else {
            let newIdx = cache.ids.count
            cache.ids.append(id)
            cache.idToIndex[id] = newIdx
            cache.metadatas.append(metadata)
            cache.vectors.append(contentsOf: vector)
        }
    }

    public func search(
        collection: String,
        queryVector: [Float],
        limit: Int = 30,
        filter: [String: String]? = nil
    ) -> [VectorSearchResult] {
        lock.lock()
        defer { lock.unlock() }

        guard let cache = caches[collection], !cache.ids.isEmpty else {
            return []
        }
        let totalCount = cache.ids.count
        let dimension = cache.dimension
        guard queryVector.count == dimension else {
            return []
        }

        var results: [(id: String, score: Float, metadata: [String: JSONValue])] = []
        results.reserveCapacity(totalCount)

        queryVector.withUnsafeBufferPointer { queryBuf in
            guard let qPtr = queryBuf.baseAddress else { return }
            cache.vectors.withUnsafeBufferPointer { vecBuf in
                guard let vPtr = vecBuf.baseAddress else { return }

                for i in 0..<totalCount {
                    let meta = cache.metadatas[i]

                    // Apply filter if requested
                    if let filter = filter {
                        var match = true
                        for (k, v) in filter {
                            let valStr = meta[k]?.stringValue ?? meta[k]?.numberValue.map {
                                $0.truncatingRemainder(dividingBy: 1) == 0 ? String(Int($0)) : String($0)
                            }
                            if valStr != v {
                                match = false
                                break
                            }
                        }
                        if !match { continue }
                    }

                    // Accelerate dot product: cos_sim = A · B (since both are unit-normalized)
                    let currentVecPtr = vPtr.advanced(by: i * dimension)
                    var dot: Float = 0.0
                    vDSP_dotpr(qPtr, 1, currentVecPtr, 1, &dot, vDSP_Length(dimension))

                    let clamped = max(-1.0, min(1.0, dot))
                    results.append((id: cache.ids[i], score: clamped, metadata: meta))
                }
            }
        }

        // Sort descending by similarity score
        results.sort { $0.score > $1.score }

        let topHits = results.prefix(limit)
        return topHits.map { hit in
            VectorSearchResult(
                id: hit.id,
                score: hit.score,
                distance: max(0.0, min(2.0, 1.0 - hit.score)),
                metadata: hit.metadata
            )
        }
    }

    public func count(collection: String? = nil) -> Int {
        lock.lock()
        defer { lock.unlock() }
        if let col = collection {
            return caches[col]?.ids.count ?? 0
        }
        return caches.values.reduce(0) { $0 + $1.ids.count }
    }

    public func countByCollection() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        var result: [String: Int] = [:]
        for (k, v) in caches {
            result[k] = v.ids.count
        }
        return result
    }

    public var databasePath: String {
        databaseURL.path
    }

    public func hasRecord(collection: String, id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let cache = caches[collection] else { return false }
        return cache.idToIndex[id] != nil
    }

    public func allMetadatas(collection: String) -> [(id: String, metadata: [String: JSONValue])] {
        lock.lock()
        defer { lock.unlock() }
        guard let cache = caches[collection] else { return [] }
        var result: [(id: String, metadata: [String: JSONValue])] = []
        result.reserveCapacity(cache.ids.count)
        for i in 0..<cache.ids.count {
            result.append((id: cache.ids[i], metadata: cache.metadatas[i]))
        }
        return result
    }

    public func delete(collection: String, ids: [String]) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !ids.isEmpty else { return }

        let placeholders = ids.map { _ in "?" }.joined(separator: ",")
        let sql = "DELETE FROM records WHERE collection = ? AND id IN (\(placeholders));"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            let message = String(cString: sqlite3_errmsg(db))
            throw ServiceError.internalFailure("SQLite delete prepare failed: \(message)")
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (collection as NSString).utf8String, -1, SQLITE_TRANSIENT)
        for (idx, id) in ids.enumerated() {
            sqlite3_bind_text(stmt, Int32(idx + 2), (id as NSString).utf8String, -1, SQLITE_TRANSIENT)
        }
        if sqlite3_step(stmt) != SQLITE_DONE {
            let message = String(cString: sqlite3_errmsg(db))
            throw ServiceError.internalFailure("SQLite delete failed: \(message)")
        }

        // Rebuild cache for collection
        if let cache = caches[collection] {
            let idSet = Set(ids)
            var newIds: [String] = []
            var newIdToIndex: [String: Int] = [:]
            var newMeta: [[String: JSONValue]] = []
            var newVectors: [Float] = []
            for (idx, existingId) in cache.ids.enumerated() {
                if !idSet.contains(existingId) {
                    newIdToIndex[existingId] = newIds.count
                    newIds.append(existingId)
                    newMeta.append(cache.metadatas[idx])
                    let start = idx * cache.dimension
                    newVectors.append(contentsOf: cache.vectors[start..<(start + cache.dimension)])
                }
            }
            cache.ids = newIds
            cache.idToIndex = newIdToIndex
            cache.metadatas = newMeta
            cache.vectors = newVectors
        }
    }
}
