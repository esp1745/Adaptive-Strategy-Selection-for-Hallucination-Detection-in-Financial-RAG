import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";

type DetectorMetrics = {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  avg_latency_ms: number;
  total_time_s: number;
  cost: number;
  confusion_matrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
};

type BenchmarkJson = Record<string, DetectorMetrics>;

export async function GET() {
  try {
    const benchmarkPath = path.resolve(
      process.cwd(),
      "..",
      "data",
      "processed",
      "benchmark_results.json"
    );

    const raw = await fs.readFile(benchmarkPath, "utf8");
    const parsed = JSON.parse(raw) as BenchmarkJson;

    const detectors = Object.values(parsed);
    const sampleCount = detectors.length > 0
      ? detectors[0].confusion_matrix.tp +
        detectors[0].confusion_matrix.fp +
        detectors[0].confusion_matrix.tn +
        detectors[0].confusion_matrix.fn
      : 0;

    return NextResponse.json({
      dataset: "PHANTOM Financial Hallucination",
      sampleCount,
      generatedAt: new Date().toISOString(),
      detectors,
    });
  } catch {
    return NextResponse.json(
      { message: "benchmark_results.json not found. Run benchmark_detectors.py first." },
      { status: 500 }
    );
  }
}
