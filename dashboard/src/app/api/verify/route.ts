import { NextResponse } from "next/server";
import path from "node:path";
import { spawn } from "node:child_process";

export const runtime = "nodejs";

type WorkflowPayload = {
  ok: boolean;
  error?: string;
  workflow?: unknown;
};

async function runPythonWorkflow(input: {
  question: string;
  top_k: number;
}): Promise<WorkflowPayload> {
  const rootDir = path.resolve(process.cwd(), "..");
  const pythonPath = path.resolve(rootDir, ".venv", "bin", "python");
  const scriptPath = path.resolve(process.cwd(), "scripts", "verify_query.py");

  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath, [scriptPath], {
      cwd: rootDir,
      env: {
        ...process.env,
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        OPENBLAS_NUM_THREADS: "1",
        PYTHONPYCACHEPREFIX: "/tmp/codex-dashboard-pycache",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python workflow exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout) as WorkflowPayload);
      } catch (error) {
        reject(
          new Error(
            `Could not parse workflow output: ${
              error instanceof Error ? error.message : "unknown error"
            }`
          )
        );
      }
    });

    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("Python workflow timed out after 60s"));
    }, 60_000);

    child.on("close", () => clearTimeout(timer));

    child.stdin.write(JSON.stringify(input));
    child.stdin.end();
  });
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { question?: string };
    const question = (body?.question ?? "").trim();

    if (!question) {
      return NextResponse.json(
        { message: "question is required" },
        { status: 400 }
      );
    }

    const workflow = await runPythonWorkflow({
      question,
      top_k: 3,
    });

    if (!workflow.ok) {
      return NextResponse.json(
        { message: workflow.error ?? "Workflow execution failed" },
        { status: 500 }
      );
    }

    return NextResponse.json(workflow);
  } catch {
    return NextResponse.json(
      { message: "Could not run the query workflow" },
      { status: 500 }
    );
  }
}
