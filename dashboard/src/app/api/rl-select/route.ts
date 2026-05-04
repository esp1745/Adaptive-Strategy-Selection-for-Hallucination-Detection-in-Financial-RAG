import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";

import {
  type BenchmarkJson,
  type PolicyJson,
  selectDetectorForQuestion,
} from "@/lib/rlSelection";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { question?: string };
    const question = (body?.question ?? "").trim();

    if (!question) {
      return NextResponse.json({ message: "question is required" }, { status: 400 });
    }

    const benchmarkPath = path.resolve(
      process.cwd(),
      "..",
      "data",
      "processed",
      "benchmark_results.json"
    );
    const policyPath = path.resolve(
      process.cwd(),
      "..",
      "data",
      "processed",
      "rl_policy.json"
    );

    const benchmarkRaw = await fs.readFile(benchmarkPath, "utf8");
    const benchmarkParsed = JSON.parse(benchmarkRaw) as BenchmarkJson;

    let policyParsed: PolicyJson = {};
    try {
      const policyRaw = await fs.readFile(policyPath, "utf8");
      policyParsed = JSON.parse(policyRaw) as PolicyJson;
    } catch {
      policyParsed = {};
    }

    const selection = selectDetectorForQuestion(benchmarkParsed, policyParsed, question);
    return NextResponse.json(selection);
  } catch {
    return NextResponse.json({ message: "Could not run RL selector" }, { status: 500 });
  }
}
