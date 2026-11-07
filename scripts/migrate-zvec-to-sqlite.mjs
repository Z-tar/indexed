#!/usr/bin/env node
import fs from "fs";
import path from "path";
import os from "os";
import { DatabaseSync } from "node:sqlite";

const root = path.join(os.homedir(), ".local/share/indexed");
const zvecRoot = path.join(root, "zvec");
const targetSqlitePath = process.env.INDEXED_VECTOR_STORE_PATH || path.join(root, "data.sqlite");

console.log("=== 📦 Indexed zvec -> SQLite 零损耗数据迁移 ===");
console.log(`源目录 (zvec): ${zvecRoot}`);
console.log(`目标库 (SQLite): ${targetSqlitePath}`);

if (!fs.existsSync(zvecRoot)) {
  console.error(`❌ 未找到 zvec 源目录: ${zvecRoot}`);
  process.exit(1);
}

const entries = fs.readdirSync(zvecRoot);
const visualDirName = entries.find((e) => e.startsWith("v2_video_visual_"));
const transcriptDirName = entries.find((e) => e.startsWith("v2_video_transcript_"));

if (!visualDirName && !transcriptDirName) {
  console.error("❌ 未在 zvec 目录下找到任何视频向量集合目录。");
  process.exit(1);
}

const { ZVecOpen } = (await import("@zvec/zvec")).default;

const db = new DatabaseSync(targetSqlitePath);
db.exec(`
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
`);

const insertStmt = db.prepare(`
  INSERT OR REPLACE INTO records (collection, id, metadata_json, embedding, created_at)
  VALUES (?, ?, ?, ?, ?)
`);

const startTime = performance.now();
let totalMigrated = 0;

db.exec("BEGIN TRANSACTION;");

try {
  if (visualDirName) {
    const visualPath = path.join(zvecRoot, visualDirName);
    console.log(`\n⏳ 正在迁移视觉集合 [video_visual]: ${visualDirName}...`);
    const col = ZVecOpen(visualPath);
    let visualCount = 0;
    const it = col.iterDocsSync({
      outputFields: ["metadata_json", "vector_f32"],
      includeVector: true,
    });

    for (const doc of it) {
      const id = String(doc.id);
      const metadataJson = String(doc.fields.metadata_json || "{}");
      let createdAt = Date.now() / 1000;
      try {
        const meta = JSON.parse(metadataJson);
        if (meta.indexed_at_ms) createdAt = Number(meta.indexed_at_ms) / 1000;
      } catch {}

      let floatArray;
      if (Array.isArray(doc.vectors?.vector)) {
        floatArray = new Float32Array(doc.vectors.vector);
      } else if (doc.fields.vector_f32 && typeof doc.fields.vector_f32 === "string") {
        floatArray = new Float32Array(Buffer.from(doc.fields.vector_f32, "base64").buffer);
      } else {
        floatArray = new Float32Array(2048);
      }

      const embeddingBuffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
      insertStmt.run("video_visual", id, metadataJson, embeddingBuffer, createdAt);
      visualCount++;
    }
    col.closeSync();
    console.log(`✅ [video_visual] 迁移完成，共 ${visualCount} 条记录。`);
    totalMigrated += visualCount;
  }

  if (transcriptDirName) {
    const transcriptPath = path.join(zvecRoot, transcriptDirName);
    console.log(`\n⏳ 正在迁移字幕集合 [video_transcript]: ${transcriptDirName}...`);
    const col = ZVecOpen(transcriptPath);
    let transcriptCount = 0;
    const it = col.iterDocsSync({
      outputFields: ["metadata_json", "vector_f32"],
      includeVector: true,
    });

    for (const doc of it) {
      const id = String(doc.id);
      const metadataJson = String(doc.fields.metadata_json || "{}");
      let createdAt = Date.now() / 1000;
      try {
        const meta = JSON.parse(metadataJson);
        if (meta.indexed_at_ms) createdAt = Number(meta.indexed_at_ms) / 1000;
      } catch {}

      let floatArray;
      if (Array.isArray(doc.vectors?.vector)) {
        floatArray = new Float32Array(doc.vectors.vector);
      } else if (doc.fields.vector_f32 && typeof doc.fields.vector_f32 === "string") {
        floatArray = new Float32Array(Buffer.from(doc.fields.vector_f32, "base64").buffer);
      } else {
        floatArray = new Float32Array(2048);
      }

      const embeddingBuffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
      insertStmt.run("video_transcript", id, metadataJson, embeddingBuffer, createdAt);
      transcriptCount++;
    }
    col.closeSync();
    console.log(`✅ [video_transcript] 迁移完成，共 ${transcriptCount} 条记录。`);
    totalMigrated += transcriptCount;
  }

  db.exec("COMMIT;");
} catch (error) {
  db.exec("ROLLBACK;");
  console.error("❌ 迁移失败，已回滚事务:", error);
  process.exit(1);
}

const elapsed = (performance.now() - startTime).toFixed(2);
console.log(`\n🎉 迁移成功！共写入 ${totalMigrated} 条记录，总耗时: ${elapsed}ms`);

// 校验统计
const checkStmt = db.prepare("SELECT collection, count(*) as count FROM records GROUP BY collection;");
const rows = checkStmt.all();
console.log("\n📊 目标 SQLite (data.sqlite) 验证结果:");
for (const row of rows) {
  console.log(` - 集合 [${row.collection}]: ${row.count} 条记录`);
}
db.close();
