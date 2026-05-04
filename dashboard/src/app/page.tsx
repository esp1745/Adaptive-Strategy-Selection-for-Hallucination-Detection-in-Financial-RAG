"use client";

import { useEffect, useState } from "react";

type DetectorMetrics = {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  avg_latency_ms: number;
  total_time_s: number;
  cost: number;
  confusion_matrix: { tp: number; fp: number; tn: number; fn: number };
};

type BenchmarkResponse = {
  dataset: string;
  sampleCount: number;
  generatedAt: string;
  detectors: DetectorMetrics[];
};

type EvidenceChunk = {
  company?: string;
  filingType?: string;
  filingDate?: string;
  chunkId?: number;
  score?: number;
  rawScore?: number;
  text: string;
};

type WorkflowDetails = {
  question: string;
  queryProcessor: {
    wordCount: number;
    entityCount: number;
    complexity: number;
    complexityLabel: string;
    features: Record<string, number>;
  };
  rag: {
    retrieval: {
      method: string;
      topK: number;
      confidence: number;
      topChunkScore: number;
      scoreGap: number;
      latencyMs: number;
      chunks: EvidenceChunk[];
    };
    generation: {
      method: string;
      engine: string;
      confidence: number;
      answer: string;
      sourceSentences: string[];
    };
  };
  policy: {
    algorithm: "ppo" | "bandit" | "fallback";
    stateDim: number;
    stateFeatureNames: string[];
    stateVector: number[];
    normalizedStateVector: number[];
    selectedDetector: string;
    selectedProbability: number;
    probabilities: Record<string, number>;
    explorationMode: boolean;
    actorHiddenSize: number;
    criticHiddenSize: number;
    stateValue: number;
    reasons: string[];
    policyFile: string;
  };
  detectorExecution: {
    requestedDetector: string;
    actualDetector: string;
    verdict: "grounded" | "hallucinated" | string;
    verdictLabel: string;
    confidence: number;
    hallucinationScore: number;
    explanation: string;
    detailLines: string[];
    latencyMs: number;
    cost: number;
    warning?: string | null;
  };
  resultAssembler: {
    answer: string;
    status: string;
    detector: string;
    cost: number;
    latencyMs: number;
    savingsVsBaseline: number;
  };
  trainingUpdate: {
    bufferSize: number;
    reward: number;
    updated: boolean;
    formula: string;
    message: string;
    recentActionCounts: Record<string, number>;
    averageBufferReward: number;
    policyFile: string;
    replayPreview: Array<{
      timestamp: string;
      question: string;
      action_name: string;
      reward: number;
    }>;
  };
  totalLatencyMs: number;
};

type WorkflowResponse = { ok: boolean; workflow: WorkflowDetails };

type RunHistoryEntry = {
  id: number;
  timestamp: string;
  question: string;
  detectorName: string;
  status: string;
  reward: number;
  savings: number;
  totalLatencyMs: number;
};

type DashboardTab = "answer" | "sources" | "decided" | "learning" | "grades";

const exampleQueries = [
  "What was Apple's total revenue last year?",
  "How did Tesla describe its key business risks?",
  "Did Microsoft's operating income go up or down?",
  "What are Amazon's main business areas?",
];

const loadingSteps = [
  "Reading your question",
  "Searching company filings",
  "Writing a grounded answer",
  "Choosing the best accuracy checker",
  "Checking the answer for errors",
  "Saving what was learned",
];


const tabConfig: { id: DashboardTab; label: string }[] = [
  { id: "answer", label: "Answer" },
  { id: "sources", label: "Sources" },
  { id: "decided", label: "How It Decided" },
  { id: "learning", label: "Learning" },
  { id: "grades", label: "Checker Grades" },
];

function detectorLabel(name: string): string {
  if (name === "token_overlap") return "Token Overlap";
  if (name === "semantic_similarity") return "Semantic Similarity";
  if (name === "bert_nli") return "BERT NLI";
  if (name === "llm_judge") return "LLM Judge";
  return name;
}

function detectorDescription(name: string): string {
  if (name === "token_overlap") return "Fast word-matcher — best for number-heavy questions";
  if (name === "semantic_similarity") return "Meaning comparer — good for paraphrased facts";
  if (name === "bert_nli") return "AI reasoner — best for comparisons and trends";
  if (name === "llm_judge") return "Full AI review — most thorough, uses Qwen";
  return "";
}

function detectorColorClass(name: string): string {
  if (name === "token_overlap") return "text-emerald-700";
  if (name === "semantic_similarity") return "text-sky-700";
  if (name === "bert_nli") return "text-amber-700";
  if (name === "llm_judge") return "text-rose-700";
  return "text-slate-700";
}

function detectorBgClass(name: string): string {
  if (name === "token_overlap") return "bg-emerald-500";
  if (name === "semantic_similarity") return "bg-sky-500";
  if (name === "bert_nli") return "bg-amber-500";
  if (name === "llm_judge") return "bg-rose-500";
  return "bg-slate-500";
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms.toFixed(0)}ms`;
}

function policyLabel(algorithm: WorkflowDetails["policy"]["algorithm"]): string {
  if (algorithm === "ppo") return "Adaptive";
  if (algorithm === "bandit") return "Exploratory";
  return "Standard";
}

function rewardToLabel(reward: number): { label: string; color: string } {
  if (reward > 0.3) return { label: "Excellent ✓", color: "text-emerald-700" };
  if (reward > 0) return { label: "Good", color: "text-sky-700" };
  if (reward > -0.3) return { label: "Acceptable", color: "text-amber-700" };
  return { label: "Could improve", color: "text-rose-700" };
}

function gradeLetter(f1: number): { grade: string; className: string } {
  if (f1 >= 0.75) return { grade: "A", className: "grade-a" };
  if (f1 >= 0.60) return { grade: "B", className: "grade-b" };
  if (f1 >= 0.45) return { grade: "C", className: "grade-c" };
  return { grade: "D", className: "grade-d" };
}

function StepIcon({ step }: { step: number }) {
  const p = { className: "h-[17px] w-[17px]", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.75", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (step === 0) return <svg {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>;
  if (step === 1) return <svg {...p}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>;
  if (step === 2) return <svg {...p}><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>;
  if (step === 3) return <svg {...p}><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/><circle cx="8" cy="6" r="2" fill="currentColor"/><circle cx="16" cy="12" r="2" fill="currentColor"/><circle cx="10" cy="18" r="2" fill="currentColor"/></svg>;
  if (step === 4) return <svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>;
  return <svg {...p}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>;
}

function EmptySvg({ type }: { type: "query" | "document" | "chart" | "data" | "warning" }) {
  const p = { className: "h-10 w-10", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.4", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (type === "query") return <svg {...p}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>;
  if (type === "document") return <svg {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>;
  if (type === "chart") return <svg {...p}><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>;
  if (type === "warning") return <svg {...p}><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;
  return <svg {...p}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>;
}

function featureSignals(features: Record<string, number>): string[] {
  const signals: string[] = [];
  if ((features.numeric_density ?? 0) > 0.05) signals.push("Contains numbers — factual accuracy prioritised");
  if ((features.has_percentage ?? 0) > 0) signals.push("Has percentages — precise matching needed");
  if ((features.has_comparison ?? 0) > 0) signals.push("Comparison language — AI reasoning preferred");
  if ((features.retrieval_confidence ?? 0) > 0.7) signals.push("Strong source match — lighter check is reliable");
  if ((features.complexity_score ?? 0) > 0.5) signals.push("Complex question — thorough checker selected");
  if (signals.length === 0) signals.push("Straightforward question — efficient checker selected");
  return signals;
}

export default function Home() {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState(exampleQueries[0]);
  const [isLoading, setIsLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<WorkflowDetails | null>(null);
  const [runHistory, setRunHistory] = useState<RunHistoryEntry[]>([]);
  const [activeTab, setActiveTab] = useState<DashboardTab>("answer");

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/benchmark", { cache: "no-store" });
        if (!res.ok) throw new Error("Could not load benchmark results.");
        const payload: BenchmarkResponse = await res.json();
        setData(payload);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unknown error");
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (!isLoading) return;
    const interval = window.setInterval(() => {
      setStepIndex((prev) => (prev >= loadingSteps.length - 1 ? prev : prev + 1));
    }, 650);
    return () => window.clearInterval(interval);
  }, [isLoading]);

  const runCount = runHistory.length;
  const totalSavings = runHistory.reduce((sum, r) => sum + r.savings, 0);
  const averageLatency = runCount > 0 ? runHistory.reduce((sum, r) => sum + r.totalLatencyMs, 0) / runCount : 0;
  const currentWorkflow = analysis;
  const currentDetectorName = currentWorkflow?.detectorExecution.actualDetector ?? currentWorkflow?.policy.selectedDetector ?? "";
  const sortedProbabilities = currentWorkflow
    ? Object.entries(currentWorkflow.policy.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  async function onRunWorkflow() {
    if (!query.trim()) return;
    setIsLoading(true);
    setStepIndex(0);
    setRequestError(null);
    try {
      const response = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });
      if (!response.ok) throw new Error("Could not run the workflow for this query.");
      const payload: WorkflowResponse = await response.json();
      setAnalysis(payload.workflow);
      setActiveTab("answer");
      setRunHistory((prev) => [
        {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          question: payload.workflow.question,
          detectorName: payload.workflow.detectorExecution.actualDetector,
          status: payload.workflow.resultAssembler.status,
          reward: payload.workflow.trainingUpdate.reward,
          savings: payload.workflow.resultAssembler.savingsVsBaseline,
          totalLatencyMs: payload.workflow.totalLatencyMs,
        },
        ...prev,
      ]);
    } catch (err) {
      setRequestError(err instanceof Error ? err.message : "Could not run the workflow.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background px-4 py-8 text-foreground md:px-8">
      {/* Loading progress bar */}
      {isLoading && (
        <div className="fixed left-0 top-0 z-50 h-1 w-full overflow-hidden bg-slate-200">
          <div
            className="h-full bg-[var(--color-primary)] transition-all duration-700"
            style={{ width: `${Math.min(10 + (stepIndex / (loadingSteps.length - 1)) * 85, 95)}%` }}
          />
        </div>
      )}

      <div className="mx-auto flex max-w-[1240px] flex-col gap-6">

        {/* ===== HERO ===== */}
        <header className="hero-panel card overflow-hidden p-6 md:p-8" style={{ background: "linear-gradient(135deg, #0c1d33 0%, #152d4a 100%)", borderColor: "#1e3555" }}>
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-blue-300">SEC Filing Intelligence Platform</p>
              <h1 className="mt-3 text-4xl font-bold text-white md:text-5xl">FinGuard</h1>
              <p className="mt-4 max-w-xl text-base leading-7 text-slate-300">
                Ask any question about company financial reports. Answers are sourced from official SEC filings
                and verified for accuracy using an adaptive detection model.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="hero-metric text-center">
                <p className="text-xs uppercase tracking-wide text-slate-400">Verifications</p>
                <p className="mt-1 text-2xl font-bold tabular-nums text-white">{runCount}</p>
              </div>
              <div className="hero-metric text-center">
                <p className="text-xs uppercase tracking-wide text-slate-400">Strategy</p>
                <p className="mt-1 text-lg font-semibold text-white">
                  {currentWorkflow ? policyLabel(currentWorkflow.policy.algorithm) : "Ready"}
                </p>
              </div>
              <div className="hero-metric text-center">
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  {runCount > 0 ? "Cost Saved" : "Detectors"}
                </p>
                <p className="mt-1 text-lg font-semibold tabular-nums text-emerald-400">
                  {runCount > 0 ? `$${totalSavings.toFixed(4)}` : data ? `${data.detectors.length} active` : "–"}
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* ===== QUERY SECTION ===== */}
        <section className="card p-6 md:p-7">
          <h2 className="text-2xl font-semibold">What would you like to know?</h2>
          <p className="mt-1 text-sm text-slate-500">
            Ask about revenue, profits, risks, business segments, or any financial detail from SEC filings.
          </p>

          <div className="mt-5 flex flex-wrap gap-2">
            {exampleQueries.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuery(example)}
                className="focus-ring rounded border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-800"
              >
                {example}
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-4 xl:grid-cols-[1.3fr_0.7fr]">
            <label className="flex flex-col gap-2">
              <span className="text-sm font-medium text-slate-700">Your question</span>
              <textarea
                className="input-query focus-ring min-h-[136px]"
                aria-label="Enter financial query"
                placeholder="e.g. What was Apple's total revenue last year?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </label>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
              <p className="text-sm font-semibold text-slate-900">What happens when you click Check</p>
              <ol className="mt-3 space-y-2">
                {loadingSteps.map((step, index) => (
                  <li
                    key={step}
                    className={`flex items-center gap-3 rounded px-3 py-2 text-sm transition-all ${
                      isLoading && index <= stepIndex
                        ? "border border-blue-300 bg-white font-medium text-blue-900"
                        : "border border-slate-200 bg-white text-slate-500"
                    }`}
                  >
                    <span className={isLoading && index <= stepIndex ? "text-blue-600" : "text-slate-400"}>
                      <StepIcon step={index} />
                    </span>
                    {step}
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-slate-500">
              {currentWorkflow ? (
                <>
                  Last check used{" "}
                  <span className={`font-semibold ${detectorColorClass(currentDetectorName)}`}>
                    {detectorLabel(currentDetectorName)}
                  </span>
                </>
              ) : (
                "Run your first check to get started"
              )}
            </p>
            <button
              type="button"
              onClick={onRunWorkflow}
              disabled={isLoading || !query.trim()}
              className="btn-primary focus-ring h-fit disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Checking…" : "Check for Accuracy ✓"}
            </button>
          </div>

          {requestError && (
            <div className="mt-4 flex items-start gap-2 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              {requestError}
            </div>
          )}
        </section>

        {/* ===== SUMMARY METRICS ===== */}
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <article className="card p-5 text-center">
            <p className="text-4xl font-bold text-[var(--color-navy)]">{runCount}</p>
            <p className="mt-2 text-sm text-[var(--color-slate)]">Questions Checked</p>
          </article>
          <article className="card p-5 text-center">
            <p className="text-4xl font-bold text-[var(--color-navy)]">
              {currentWorkflow ? formatPct(currentWorkflow.rag.retrieval.confidence) : "–"}
            </p>
            <p className="mt-2 text-sm text-[var(--color-slate)]">Source Confidence</p>
          </article>
          <article className="card p-5 text-center">
            <p className="text-4xl font-bold text-emerald-700">
              ${totalSavings.toFixed(2)}
            </p>
            <p className="mt-2 text-sm text-[var(--color-slate)]">Total Saved vs Expensive</p>
          </article>
          <article className="card p-5 text-center">
            <p className="text-4xl font-bold text-sky-700">
              {runCount > 0 ? formatLatency(averageLatency) : "–"}
            </p>
            <p className="mt-2 text-sm text-[var(--color-slate)]">Average Check Time</p>
          </article>
        </section>

        {/* ===== TABS ===== */}
        <nav className="tab-bar card p-3">
          <div className="flex flex-wrap gap-2">
            {tabConfig.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`tab-button focus-ring ${activeTab === tab.id ? "tab-button-active" : ""}`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </nav>

        {/* ===== ANSWER TAB ===== */}
        {activeTab === "answer" && (
          currentWorkflow ? (
            <div className="flex flex-col gap-4">

              {/* Verdict banner */}
              <div className={currentWorkflow.detectorExecution.verdict === "grounded" ? "verdict-success-banner" : "verdict-warning-banner"}>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="flex items-center gap-2 text-2xl font-bold">
                      {currentWorkflow.detectorExecution.verdict === "grounded" ? (
                        <>
                          <svg className="h-5 w-5 shrink-0 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
                          Answer Verified
                        </>
                      ) : (
                        <>
                          <svg className="h-5 w-5 shrink-0 text-amber-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                          Answer Needs Review
                        </>
                      )}
                    </p>
                    <p className="mt-1 text-sm text-slate-600">
                      Checked by{" "}
                      <strong className={detectorColorClass(currentWorkflow.detectorExecution.actualDetector)}>
                        {detectorLabel(currentWorkflow.detectorExecution.actualDetector)}
                      </strong>{" "}
                      with <strong>{formatPct(currentWorkflow.detectorExecution.confidence)}</strong> confidence
                    </p>
                  </div>
                  <div className="flex gap-4 sm:text-right">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">This check cost</p>
                      <p className="text-xl font-bold text-slate-800">${currentWorkflow.resultAssembler.cost.toFixed(2)}</p>
                    </div>
                    {currentWorkflow.resultAssembler.savingsVsBaseline > 0 && (
                      <div>
                        <p className="text-xs uppercase tracking-wide text-slate-500">You saved</p>
                        <p className="text-xl font-bold text-emerald-700">
                          ${currentWorkflow.resultAssembler.savingsVsBaseline.toFixed(2)}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* The answer */}
              <section className="card p-6">
                <p className="text-xs uppercase tracking-wide text-slate-400">Answer</p>
                <p className="answer-text mt-3">{currentWorkflow.resultAssembler.answer}</p>
                {currentWorkflow.detectorExecution.explanation && (
                  <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Why the AI said this</p>
                    <p className="mt-1 text-sm text-slate-700">{currentWorkflow.detectorExecution.explanation}</p>
                  </div>
                )}
                {currentWorkflow.detectorExecution.warning && (
                  <p className="mt-3 flex items-start gap-2 rounded border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                    <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    {currentWorkflow.detectorExecution.warning}
                  </p>
                )}
              </section>

              {/* Pipeline steps */}
              <section className="card p-6">
                <h3 className="text-lg font-semibold text-slate-900">How this answer was made</h3>
                <p className="mt-1 text-sm text-slate-500">Step by step — here's exactly what happened behind the scenes.</p>

                <div className="mt-5 flex flex-col gap-3">

                  <div className="pipeline-step">
                    <div className="step-icon"><StepIcon step={0} /></div>
                    <div>
                      <p className="font-semibold text-slate-900">Read your question</p>
                      <p className="mt-0.5 text-sm text-slate-500">
                        {currentWorkflow.queryProcessor.wordCount} words •{" "}
                        {currentWorkflow.queryProcessor.entityCount}{" "}
                        {currentWorkflow.queryProcessor.entityCount === 1 ? "company" : "companies"} mentioned •{" "}
                        Complexity: <strong className="text-slate-700">{currentWorkflow.queryProcessor.complexityLabel}</strong>
                      </p>
                    </div>
                  </div>

                  <div className="pipeline-step">
                    <div className="step-icon"><StepIcon step={1} /></div>
                    <div>
                      <p className="font-semibold text-slate-900">Searched company filings</p>
                      <p className="mt-0.5 text-sm text-slate-500">
                        Found <strong className="text-slate-700">{currentWorkflow.rag.retrieval.topK} relevant sections</strong> from SEC documents •{" "}
                        Confidence: <strong className="text-slate-700">{formatPct(currentWorkflow.rag.retrieval.confidence)}</strong> •{" "}
                        Took {formatLatency(currentWorkflow.rag.retrieval.latencyMs)}
                      </p>
                    </div>
                  </div>

                  <div className="pipeline-step">
                    <div className="step-icon"><StepIcon step={2} /></div>
                    <div className="flex-1">
                      <p className="font-semibold text-slate-900">Wrote a grounded answer</p>
                      <p className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                        {currentWorkflow.rag.generation.answer}
                      </p>
                    </div>
                  </div>

                  <div className="pipeline-step">
                    <div className="step-icon"><StepIcon step={3} /></div>
                    <div>
                      <p className="font-semibold text-slate-900">
                        Chose{" "}
                        <span className={detectorColorClass(currentWorkflow.policy.selectedDetector)}>
                          {detectorLabel(currentWorkflow.policy.selectedDetector)}
                        </span>{" "}
                        to verify the answer
                      </p>
                      <p className="mt-0.5 text-sm text-slate-500">
                        {formatPct(currentWorkflow.policy.selectedProbability)} confident this was the right checker for your question
                      </p>
                      {currentWorkflow.policy.reasons[0] && (
                        <p className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                          {currentWorkflow.policy.reasons[0]}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="pipeline-step">
                    <div className="step-icon"><StepIcon step={4} /></div>
                    <div>
                      <p className="font-semibold text-slate-900">
                        Verdict:{" "}
                        <span className={currentWorkflow.detectorExecution.verdict === "grounded" ? "text-emerald-700" : "text-rose-700"}>
                          {currentWorkflow.detectorExecution.verdict === "grounded" ? "Verified ✓" : "Needs Review ⚠️"}
                        </span>
                      </p>
                      <p className="mt-0.5 text-sm text-slate-500">
                        {formatPct(currentWorkflow.detectorExecution.confidence)} confidence •{" "}
                        Took {formatLatency(currentWorkflow.detectorExecution.latencyMs)}
                      </p>
                    </div>
                  </div>

                  <div className="pipeline-step">
                    <div className="step-icon"><StepIcon step={5} /></div>
                    <div>
                      <p className="font-semibold text-slate-900">
                        {currentWorkflow.trainingUpdate.updated ? "AI model improved from this check" : "Experience saved for future learning"}
                      </p>
                      <p className="mt-0.5 text-sm text-slate-500">
                        {currentWorkflow.trainingUpdate.bufferSize} examples in memory •{" "}
                        Quality:{" "}
                        <span className={rewardToLabel(currentWorkflow.trainingUpdate.reward).color}>
                          {rewardToLabel(currentWorkflow.trainingUpdate.reward).label}
                        </span>
                      </p>
                    </div>
                  </div>

                </div>
              </section>
            </div>
          ) : !isLoading ? (
            <section className="card">
              <div className="empty-state">
                <div className="empty-state-icon"><EmptySvg type="query" /></div>
                <p className="text-xl font-semibold text-slate-900">Ask your first question to get started</p>
                <p className="max-w-md text-sm text-slate-500">
                  Type a question about any company&apos;s financial filings above, then click
                  &ldquo;Check for Accuracy&rdquo;. The AI will find the answer and verify it for you.
                </p>
              </div>
            </section>
          ) : null
        )}

        {/* ===== SOURCES TAB ===== */}
        {activeTab === "sources" && (
          currentWorkflow ? (
            <section className="card p-6">
              <h2 className="text-2xl font-semibold text-slate-900">Where the answer came from</h2>
              <p className="mt-1 text-sm text-slate-500">
                These sections from company SEC filings were used to write the answer.
                The AI read these documents and extracted the most relevant information.
              </p>
              <div className="mt-2 flex gap-4 text-xs text-slate-400">
                <span>Top relevance: {formatPct(currentWorkflow.rag.retrieval.topChunkScore)}</span>
                <span>Confidence gap: {currentWorkflow.rag.retrieval.scoreGap.toFixed(3)}</span>
              </div>

              <div className="mt-5 grid gap-4 lg:grid-cols-3">
                {currentWorkflow.rag.retrieval.chunks.map((chunk, index) => (
                  <article key={`${chunk.company}-${chunk.chunkId}-${index}`} className="evidence-card">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-semibold text-slate-900">{chunk.company ?? "Unknown company"}</p>
                      {typeof chunk.score === "number" && (
                        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
                          {formatPct(chunk.score)} match
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs uppercase tracking-wide text-slate-400">
                      {chunk.filingType ?? "Filing"}{chunk.filingDate ? ` • ${chunk.filingDate}` : ""}
                    </p>
                    <p className="mt-3 text-sm leading-7 text-slate-700">{chunk.text}</p>
                  </article>
                ))}
              </div>
            </section>
          ) : (
            <section className="card">
              <div className="empty-state">
                <div className="empty-state-icon"><EmptySvg type="document" /></div>
                <p className="text-lg font-semibold text-slate-900">No sources yet</p>
                <p className="text-sm text-slate-500">Run a question to see which filing sections were used.</p>
              </div>
            </section>
          )
        )}

        {/* ===== HOW IT DECIDED TAB ===== */}
        {activeTab === "decided" && (
          currentWorkflow ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <section className="card p-6">
                <h2 className="text-2xl font-semibold text-slate-900">
                  Why{" "}
                  <span className={detectorColorClass(currentWorkflow.policy.selectedDetector)}>
                    {detectorLabel(currentWorkflow.policy.selectedDetector)}
                  </span>{" "}
                  was chosen
                </h2>
                <p className="mt-2 text-sm text-slate-500">
                  The AI looked at your question and picked the best accuracy checker based on these signals:
                </p>
                <div className="mt-5 flex flex-wrap gap-2">
                  {featureSignals(currentWorkflow.queryProcessor.features).map((signal) => (
                    <span key={signal} className="signal-tag">{signal}</span>
                  ))}
                </div>
                <div className="mt-5 space-y-2">
                  {currentWorkflow.policy.reasons.map((reason) => (
                    <div key={reason} className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                      {reason}
                    </div>
                  ))}
                </div>
              </section>

              <section className="card p-6">
                <h2 className="text-2xl font-semibold text-slate-900">How likely each checker was</h2>
                <p className="mt-2 text-sm text-slate-500">
                  The AI assigned each checker a probability of being the best choice for your question:
                </p>
                <div className="mt-5 space-y-5">
                  {sortedProbabilities.map(([name, probability]) => (
                    <div key={name}>
                      <div className="mb-1.5 flex items-start justify-between gap-3">
                        <div>
                          <p className={`font-semibold text-sm ${name === currentWorkflow.policy.selectedDetector ? detectorColorClass(name) : "text-slate-600"}`}>
                            {detectorLabel(name)}
                            {name === currentWorkflow.policy.selectedDetector && (
                              <span className="ml-2 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">chosen</span>
                            )}
                          </p>
                          <p className="text-xs text-slate-400">{detectorDescription(name)}</p>
                        </div>
                        <span className="text-sm font-semibold text-slate-700 shrink-0">{formatPct(probability)}</span>
                      </div>
                      <div className="probability-track">
                        <div
                          className={`h-full rounded-full ${detectorBgClass(name)}`}
                          style={{ width: `${Math.max(probability * 100, 2)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          ) : (
            <section className="card">
              <div className="empty-state">
                <div className="empty-state-icon"><EmptySvg type="chart" /></div>
                <p className="text-lg font-semibold text-slate-900">No decision made yet</p>
                <p className="text-sm text-slate-500">Run a question to see how the AI chose a checker.</p>
              </div>
            </section>
          )
        )}

        {/* ===== LEARNING TAB ===== */}
        {activeTab === "learning" && (
          currentWorkflow ? (
            <div className="flex flex-col gap-4">
              <section className="grid gap-4 lg:grid-cols-2">
                <article className="card p-6">
                  <h2 className="text-2xl font-semibold">FinGuard learns as you use it</h2>
                  <p className="mt-2 text-sm leading-6 text-slate-600">
                    After each question, the AI records what happened and uses it to make smarter decisions
                    next time. The more you use it, the better it gets at choosing the right checker.
                  </p>
                  <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">How quality is scored</p>
                    <p className="mt-1 text-sm font-medium text-slate-800">
                      Correct answer + low cost + fast speed = high score
                    </p>
                    <p className="mt-1 text-xs text-slate-400">{currentWorkflow.trainingUpdate.formula}</p>
                  </div>
                  <div className="mt-5 grid gap-3 sm:grid-cols-3">
                    <div className="metric-surface">
                      <p className="text-xs uppercase tracking-wide text-slate-500">Last Score</p>
                      <p className={`mt-1 text-base font-bold ${rewardToLabel(currentWorkflow.trainingUpdate.reward).color}`}>
                        {rewardToLabel(currentWorkflow.trainingUpdate.reward).label}
                      </p>
                    </div>
                    <div className="metric-surface">
                      <p className="text-xs uppercase tracking-wide text-slate-500">Memory Size</p>
                      <p className="mt-1 text-lg font-semibold text-slate-900">
                        {currentWorkflow.trainingUpdate.bufferSize} checks
                      </p>
                    </div>
                    <div className="metric-surface">
                      <p className="text-xs uppercase tracking-wide text-slate-500">Model Updated</p>
                      <p className={`mt-1 text-base font-semibold ${currentWorkflow.trainingUpdate.updated ? "text-emerald-700" : "text-slate-500"}`}>
                        {currentWorkflow.trainingUpdate.updated ? "Yes ✓" : "Buffered"}
                      </p>
                    </div>
                  </div>
                </article>

                <article className="card p-6">
                  <h2 className="text-2xl font-semibold">Checker usage so far</h2>
                  <p className="mt-2 text-sm text-slate-500">How often the AI has picked each checker:</p>
                  <div className="mt-5 space-y-4">
                    {Object.entries(currentWorkflow.trainingUpdate.recentActionCounts).map(([name, count]) => (
                      <div key={name}>
                        <div className="mb-1 flex items-center justify-between text-sm">
                          <div>
                            <span className={`font-medium ${detectorColorClass(name)}`}>{detectorLabel(name)}</span>
                            <span className="ml-2 text-xs text-slate-400">{detectorDescription(name)}</span>
                          </div>
                          <span className="font-semibold text-slate-900">{count}×</span>
                        </div>
                        <div className="probability-track">
                          <div
                            className={`h-full rounded-full ${detectorBgClass(name)}`}
                            style={{ width: `${Math.max((count / Math.max(runCount, 1)) * 100, 2)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </article>
              </section>

              <section className="card p-6">
                <h2 className="text-2xl font-semibold">Recent memory</h2>
                <p className="mt-1 text-sm text-slate-500">The last few checks stored in the AI&apos;s learning memory:</p>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[600px] border-collapse text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-400">
                        <th className="py-3">Time</th>
                        <th className="py-3">Question</th>
                        <th className="py-3">Checker Used</th>
                        <th className="py-3">Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentWorkflow.trainingUpdate.replayPreview.map((exp) => (
                        <tr key={`${exp.timestamp}-${exp.question}`} className="border-b border-slate-100">
                          <td className="py-3 text-slate-400 text-xs">{new Date(exp.timestamp).toLocaleTimeString()}</td>
                          <td className="max-w-[280px] py-3 font-medium text-slate-900">{exp.question}</td>
                          <td className={`py-3 font-medium ${detectorColorClass(exp.action_name)}`}>
                            {detectorLabel(exp.action_name)}
                          </td>
                          <td className={`py-3 font-semibold ${rewardToLabel(exp.reward).color}`}>
                            {rewardToLabel(exp.reward).label}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {runHistory.length > 0 && (
                <section className="card p-6">
                  <div className="flex items-center justify-between gap-4">
                    <h2 className="text-2xl font-semibold">Your full check history</h2>
                    {error && (
                      <span className="rounded-md border border-red-200 bg-red-50 px-3 py-1 text-xs text-red-700">{error}</span>
                    )}
                  </div>
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full min-w-[700px] border-collapse text-left text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-400">
                          <th className="py-3">Time</th>
                          <th className="py-3">Question</th>
                          <th className="py-3">Checker</th>
                          <th className="py-3">Status</th>
                          <th className="py-3">Score</th>
                          <th className="py-3">Duration</th>
                          <th className="py-3">Saved</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runHistory.map((run) => (
                          <tr key={run.id} className="border-b border-slate-100">
                            <td className="py-3 text-xs text-slate-400">{new Date(run.timestamp).toLocaleTimeString()}</td>
                            <td className="max-w-[240px] py-3 font-medium text-slate-900">{run.question}</td>
                            <td className={`py-3 font-medium ${detectorColorClass(run.detectorName)}`}>
                              {detectorLabel(run.detectorName)}
                            </td>
                            <td className="py-3 text-slate-600">{run.status}</td>
                            <td className={`py-3 font-semibold ${rewardToLabel(run.reward).color}`}>
                              {rewardToLabel(run.reward).label}
                            </td>
                            <td className="py-3 text-slate-500">{formatLatency(run.totalLatencyMs)}</td>
                            <td className="py-3 font-semibold text-emerald-700">${run.savings.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </div>
          ) : (
            <section className="card">
              <div className="empty-state">
                <div className="empty-state-icon"><EmptySvg type="data" /></div>
                <p className="text-lg font-semibold text-slate-900">No learning data yet</p>
                <p className="text-sm text-slate-500">Run a few questions to start building the AI&apos;s learning history.</p>
              </div>
            </section>
          )
        )}

        {/* ===== CHECKER GRADES TAB ===== */}
        {activeTab === "grades" && (
          data ? (
            <div className="flex flex-col gap-4">
              <section className="card p-6">
                <h2 className="text-2xl font-semibold text-slate-900">Accuracy Checker Report Card</h2>
                <p className="mt-1 text-sm text-slate-500">
                  We tested all four checkers on {data.sampleCount} real financial questions. Here&apos;s how they did.
                </p>

                <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  {[...data.detectors].sort((a, b) => b.f1 - a.f1).map((detector) => {
                    const { grade, className } = gradeLetter(detector.f1);
                    return (
                      <article key={detector.name} className="card p-5 flex flex-col gap-3">
                        <div className="flex items-start justify-between">
                          <div>
                            <p className={`font-bold ${detectorColorClass(detector.name)}`}>
                              {detectorLabel(detector.name)}
                            </p>
                            <p className="mt-0.5 text-xs text-slate-400">{detectorDescription(detector.name)}</p>
                          </div>
                          <span className={`grade-letter ${className}`}>{grade}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="metric-surface">
                            <p className="text-xs text-slate-500">F1 Score</p>
                            <p className="mt-0.5 text-lg font-bold text-slate-900">{formatPct(detector.f1)}</p>
                          </div>
                          <div className="metric-surface">
                            <p className="text-xs text-slate-500">Accuracy</p>
                            <p className="mt-0.5 text-lg font-bold text-slate-900">{formatPct(detector.accuracy)}</p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="metric-surface">
                            <p className="text-xs text-slate-500">Speed</p>
                            <p className="mt-0.5 text-base font-semibold text-slate-900">{formatLatency(detector.avg_latency_ms)}</p>
                          </div>
                          <div className="metric-surface">
                            <p className="text-xs text-slate-500">Cost per check</p>
                            <p className="mt-0.5 text-base font-semibold text-slate-900">${detector.cost.toFixed(2)}</p>
                          </div>
                        </div>
                        <div>
                          <p className="mb-1 text-xs text-slate-400">F1 score</p>
                          <div className="probability-track">
                            <div
                              className={`h-full rounded-full ${detectorBgClass(detector.name)}`}
                              style={{ width: `${Math.max(detector.f1 * 100, 2)}%` }}
                            />
                          </div>
                        </div>
                        <p className="text-xs text-slate-400">
                          Correct: {detector.confusion_matrix.tp + detector.confusion_matrix.tn} ·
                          Missed: {detector.confusion_matrix.fp + detector.confusion_matrix.fn}
                        </p>
                      </article>
                    );
                  })}
                </div>
              </section>

              <section className="card p-6">
                <h2 className="text-2xl font-semibold text-slate-900">Full comparison table</h2>
                <p className="mt-1 text-sm text-slate-500">All metrics side by side — sorted by F1 score (best first).</p>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[680px] border-collapse text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-400">
                        <th className="py-3">Checker</th>
                        <th className="py-3">Grade</th>
                        <th className="py-3">F1</th>
                        <th className="py-3">Accuracy</th>
                        <th className="py-3">Precision</th>
                        <th className="py-3">Recall</th>
                        <th className="py-3">Speed</th>
                        <th className="py-3">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...data.detectors].sort((a, b) => b.f1 - a.f1).map((detector) => {
                        const { grade, className } = gradeLetter(detector.f1);
                        return (
                          <tr key={detector.name} className="border-b border-slate-100 hover:bg-slate-50">
                            <td className={`py-3 font-semibold ${detectorColorClass(detector.name)}`}>
                              {detectorLabel(detector.name)}
                            </td>
                            <td className={`py-3 font-bold text-lg ${className}`}>{grade}</td>
                            <td className="py-3 font-medium text-slate-900">{formatPct(detector.f1)}</td>
                            <td className="py-3 text-slate-600">{formatPct(detector.accuracy)}</td>
                            <td className="py-3 text-slate-600">{formatPct(detector.precision)}</td>
                            <td className="py-3 text-slate-600">{formatPct(detector.recall)}</td>
                            <td className="py-3 text-slate-600">{formatLatency(detector.avg_latency_ms)}</td>
                            <td className="py-3 text-slate-600">${detector.cost.toFixed(2)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          ) : error ? (
            <section className="card">
              <div className="empty-state">
                <div className="empty-state-icon"><EmptySvg type="warning" /></div>
                <p className="text-lg font-semibold text-red-700">Could not load benchmark data</p>
                <p className="text-sm text-slate-500">
                  Run <code className="rounded bg-slate-100 px-1 py-0.5">python benchmark_detectors.py</code> to generate results.
                </p>
              </div>
            </section>
          ) : (
            <section className="card">
              <div className="empty-state">
                <p className="text-sm text-slate-500">Loading checker grades…</p>
              </div>
            </section>
          )
        )}

      </div>
    </main>
  );
}
