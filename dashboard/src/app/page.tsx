"use client";

import { useEffect, useState } from "react";

type EvidenceChunk = {
  company?: string;
  filingType?: string;
  filingDate?: string;
  chunkId?: number;
  score?: number;
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

type DashboardTab = "answer" | "decided";

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
  { id: "decided", label: "Policy Decision" },
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
  if (algorithm === "ppo") return "PPO (Adaptive)";
  if (algorithm === "bandit") return "Bandit (Exploratory)";
  return "Standard";
}

function rewardToLabel(reward: number): { label: string; color: string } {
  if (reward > 0.3) return { label: "Excellent", color: "text-emerald-700" };
  if (reward > 0) return { label: "Good", color: "text-sky-700" };
  if (reward > -0.3) return { label: "Acceptable", color: "text-amber-700" };
  return { label: "Low", color: "text-rose-700" };
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

function StepIcon({ step }: { step: number }) {
  const p = { className: "h-[17px] w-[17px]", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.75", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (step === 0) return <svg {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>;
  if (step === 1) return <svg {...p}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>;
  if (step === 2) return <svg {...p}><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>;
  if (step === 3) return <svg {...p}><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/><circle cx="8" cy="6" r="2" fill="currentColor"/><circle cx="16" cy="12" r="2" fill="currentColor"/><circle cx="10" cy="18" r="2" fill="currentColor"/></svg>;
  if (step === 4) return <svg {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>;
  return <svg {...p}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>;
}

export default function Home() {
  const [query, setQuery] = useState(exampleQueries[0]);
  const [isLoading, setIsLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<WorkflowDetails | null>(null);
  const [activeTab, setActiveTab] = useState<DashboardTab>("answer");

  useEffect(() => {
    if (!isLoading) return;
    const interval = window.setInterval(() => {
      setStepIndex((prev) => (prev >= loadingSteps.length - 1 ? prev : prev + 1));
    }, 650);
    return () => window.clearInterval(interval);
  }, [isLoading]);

  const currentDetectorName = analysis?.detectorExecution.actualDetector ?? analysis?.policy.selectedDetector ?? "";
  const sortedProbabilities = analysis
    ? Object.entries(analysis.policy.probabilities).sort((a, b) => b[1] - a[1])
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
    } catch (err) {
      setRequestError(err instanceof Error ? err.message : "Could not run the workflow.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background px-4 py-8 text-foreground md:px-8">
      {isLoading && (
        <div className="fixed left-0 top-0 z-50 h-1 w-full overflow-hidden bg-slate-200">
          <div
            className="h-full bg-[var(--color-primary)] transition-all duration-700"
            style={{ width: `${Math.min(10 + (stepIndex / (loadingSteps.length - 1)) * 85, 95)}%` }}
          />
        </div>
      )}

      <div className="mx-auto flex max-w-[1100px] flex-col gap-6">

        {/* HERO */}
        <header className="card overflow-hidden" style={{ background: "linear-gradient(135deg, #0c1d33 0%, #152d4a 100%)", borderColor: "#1e3555" }}>
          {/* Top section */}
          <div className="p-6 md:p-8">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-blue-300">Adaptive Hallucination Detection · Financial RAG</p>
            <h1 className="mt-3 text-4xl font-bold text-white md:text-5xl">FinGuard</h1>
            <p className="mt-3 max-w-2xl text-base leading-7 text-slate-300">
              Every query is different — a number-heavy earnings question needs a different accuracy check than a
              narrative risk summary. FinGuard uses a{" "}
              <span className="font-semibold text-blue-300">PPO reinforcement learning policy</span> trained on
              real query features to automatically route each question to the most cost-efficient detector,
              eliminating the need for manual tuning or a one-size-fits-all approach.
            </p>

            {/* Key stat pills */}
            <div className="mt-5 flex flex-wrap gap-3 text-sm text-slate-400">
              <span className="rounded-full border border-slate-600 px-3 py-1">PPO Neural Policy</span>
              <span className="rounded-full border border-slate-600 px-3 py-1">4 Detectors</span>
              <span className="rounded-full border border-slate-600 px-3 py-1">975 SEC Filing Chunks</span>
              {analysis && (
                <span className="rounded-full border border-blue-500 px-3 py-1 text-blue-300">
                  Last: {policyLabel(analysis.policy.algorithm)} selected {detectorLabel(currentDetectorName)}
                </span>
              )}
            </div>
          </div>

          {/* How it works — pipeline flow */}
          <div className="border-t border-[#1e3555] px-6 py-5 md:px-8">
            <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">How it works</p>
            <div className="flex flex-wrap items-center gap-2">
              {/* Step 1 */}
              <div className="flex flex-col gap-1 rounded-xl border border-[#1e3555] bg-[#0e2240] px-4 py-3 text-center" style={{ minWidth: "130px" }}>
                <span className="text-xs font-semibold text-white">Financial Query</span>
                <span className="text-[11px] text-slate-400">SEC filing question</span>
              </div>
              <svg className="text-slate-500 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              {/* Step 2 */}
              <div className="flex flex-col gap-1 rounded-xl border border-[#1e3555] bg-[#0e2240] px-4 py-3 text-center" style={{ minWidth: "130px" }}>
                <span className="text-xs font-semibold text-white">Feature Extraction</span>
                <span className="text-[11px] text-slate-400">12-dim state vector</span>
              </div>
              <svg className="text-slate-500 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              {/* Step 3 — highlighted */}
              <div className="flex flex-col gap-1 rounded-xl border border-blue-500 bg-blue-900/40 px-4 py-3 text-center" style={{ minWidth: "140px" }}>
                <span className="text-xs font-semibold text-blue-200">RL Policy (PPO)</span>
                <span className="text-[11px] text-blue-300">Picks best detector</span>
              </div>
              <svg className="text-slate-500 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              {/* Step 4 */}
              <div className="flex flex-col gap-1 rounded-xl border border-[#1e3555] bg-[#0e2240] px-4 py-3 text-center" style={{ minWidth: "130px" }}>
                <span className="text-xs font-semibold text-white">Hallucination Check</span>
                <span className="text-[11px] text-slate-400">Token / Semantic / BERT / LLM</span>
              </div>
              <svg className="text-slate-500 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              {/* Step 5 */}
              <div className="flex flex-col gap-1 rounded-xl border border-[#1e3555] bg-[#0e2240] px-4 py-3 text-center" style={{ minWidth: "130px" }}>
                <span className="text-xs font-semibold text-white">Policy Update</span>
                <span className="text-[11px] text-slate-400">Reward feeds PPO</span>
              </div>
            </div>
          </div>

          {/* Key differentiators */}
          <div className="border-t border-[#1e3555] px-6 py-5 md:px-8">
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="flex items-start gap-3">
                <span className="mt-0.5 text-blue-400">
                  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>
                  </svg>
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">Adaptive, not fixed</p>
                  <p className="mt-0.5 text-xs text-slate-400">The policy learns which detector works best per query type — not a static rule.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="mt-0.5 text-emerald-400">
                  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                  </svg>
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">Cost-efficient</p>
                  <p className="mt-0.5 text-xs text-slate-400">Avoids expensive LLM calls when a lightweight checker is sufficient.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="mt-0.5 text-amber-400">
                  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                  </svg>
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">Continuously improving</p>
                  <p className="mt-0.5 text-xs text-slate-400">Every query updates the PPO policy — the system gets smarter over time.</p>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* QUERY */}
        <section className="card p-6 md:p-7">
          <h2 className="text-xl font-semibold">Ask a financial question</h2>
          <p className="mt-1 text-sm text-slate-500">
            The RL policy will decide which detector to use based on query features — no manual selection needed.
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            {exampleQueries.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuery(example)}
                className="focus-ring rounded border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-800"
              >
                {example}
              </button>
            ))}
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-[1.4fr_0.6fr]">
            <label className="flex flex-col gap-2">
              <span className="text-sm font-medium text-slate-700">Your question</span>
              <textarea
                className="input-query focus-ring min-h-[120px]"
                aria-label="Enter financial query"
                placeholder="e.g. What was Apple's total revenue last year?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </label>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">Pipeline steps</p>
              <ol className="space-y-1.5">
                {loadingSteps.map((step, index) => (
                  <li
                    key={step}
                    className={`flex items-center gap-2 rounded px-2.5 py-1.5 text-xs transition-all ${
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

          <div className="mt-4 flex items-center justify-end">
            <button
              type="button"
              onClick={onRunWorkflow}
              disabled={isLoading || !query.trim()}
              className="btn-primary focus-ring disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Running..." : "Run Pipeline"}
            </button>
          </div>

          {requestError && (
            <div className="mt-4 flex items-start gap-2 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              {requestError}
            </div>
          )}
        </section>

        {/* TABS — only shown after first run */}
        {analysis && (
          <>
            <nav className="tab-bar card p-3">
              <div className="flex gap-2">
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

            {/* ANSWER TAB */}
            {activeTab === "answer" && (
              <div className="flex flex-col gap-4">

                {/* Verdict */}
                <div className={analysis.detectorExecution.verdict === "grounded" ? "verdict-success-banner" : "verdict-warning-banner"}>
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="flex items-center gap-2 text-xl font-bold">
                        {analysis.detectorExecution.verdict === "grounded" ? (
                          <>
                            <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
                            Answer Verified
                          </>
                        ) : (
                          <>
                            <svg className="h-5 w-5 text-amber-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            Needs Review
                          </>
                        )}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        Checked by{" "}
                        <strong className={detectorColorClass(analysis.detectorExecution.actualDetector)}>
                          {detectorLabel(analysis.detectorExecution.actualDetector)}
                        </strong>{" "}
                        · {formatPct(analysis.detectorExecution.confidence)} confidence
                        · {formatLatency(analysis.totalLatencyMs)} total
                      </p>
                    </div>
                    <div className="flex gap-6 sm:text-right text-sm">
                      <div>
                        <p className="text-xs uppercase tracking-wide text-slate-500">Cost</p>
                        <p className="text-lg font-bold text-slate-800">${analysis.resultAssembler.cost.toFixed(2)}</p>
                      </div>
                      {analysis.resultAssembler.savingsVsBaseline > 0 && (
                        <div>
                          <p className="text-xs uppercase tracking-wide text-slate-500">Saved vs LLM Judge</p>
                          <p className="text-lg font-bold text-emerald-700">
                            ${analysis.resultAssembler.savingsVsBaseline.toFixed(2)}
                          </p>
                        </div>
                      )}
                      <div>
                        <p className="text-xs uppercase tracking-wide text-slate-500">Reward</p>
                        <p className={`text-lg font-bold ${rewardToLabel(analysis.trainingUpdate.reward).color}`}>
                          {rewardToLabel(analysis.trainingUpdate.reward).label}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Answer + explanation */}
                <section className="card p-6">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Answer</p>
                  <p className="answer-text mt-3">{analysis.resultAssembler.answer}</p>
                  {analysis.detectorExecution.explanation && (
                    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Detector explanation</p>
                      <p className="mt-1 text-sm text-slate-700">{analysis.detectorExecution.explanation}</p>
                    </div>
                  )}
                  {analysis.detectorExecution.warning && (
                    <p className="mt-3 flex items-start gap-2 rounded border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                      <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                      {analysis.detectorExecution.warning}
                    </p>
                  )}
                </section>

                {/* Pipeline trace */}
                <section className="card p-6">
                  <h3 className="text-base font-semibold text-slate-900">Pipeline trace</h3>
                  <div className="mt-4 flex flex-col gap-3">
                    <div className="pipeline-step">
                      <div className="step-icon"><StepIcon step={0} /></div>
                      <div>
                        <p className="font-semibold text-slate-900">Query processed</p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          {analysis.queryProcessor.wordCount} words · {analysis.queryProcessor.entityCount}{" "}
                          {analysis.queryProcessor.entityCount === 1 ? "entity" : "entities"} · complexity{" "}
                          <strong className="text-slate-700">{analysis.queryProcessor.complexityLabel}</strong>
                        </p>
                      </div>
                    </div>

                    <div className="pipeline-step">
                      <div className="step-icon"><StepIcon step={1} /></div>
                      <div>
                        <p className="font-semibold text-slate-900">RAG retrieval</p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          {analysis.rag.retrieval.topK} chunks from SEC filings ·{" "}
                          {formatPct(analysis.rag.retrieval.confidence)} confidence ·{" "}
                          {formatLatency(analysis.rag.retrieval.latencyMs)}
                        </p>
                      </div>
                    </div>

                    <div className="pipeline-step">
                      <div className="step-icon"><StepIcon step={2} /></div>
                      <div className="flex-1">
                        <p className="font-semibold text-slate-900">Answer generated</p>
                        <p className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                          {analysis.rag.generation.answer}
                        </p>
                      </div>
                    </div>

                    <div className="pipeline-step">
                      <div className="step-icon"><StepIcon step={3} /></div>
                      <div>
                        <p className="font-semibold text-slate-900">
                          RL policy selected{" "}
                          <span className={detectorColorClass(analysis.policy.selectedDetector)}>
                            {detectorLabel(analysis.policy.selectedDetector)}
                          </span>
                        </p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          {formatPct(analysis.policy.selectedProbability)} selection probability ·{" "}
                          {policyLabel(analysis.policy.algorithm)}
                          {analysis.policy.explorationMode && " · exploration mode"}
                        </p>
                        {analysis.policy.reasons[0] && (
                          <p className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                            {analysis.policy.reasons[0]}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="pipeline-step">
                      <div className="step-icon"><StepIcon step={4} /></div>
                      <div>
                        <p className="font-semibold text-slate-900">
                          Verdict:{" "}
                          <span className={analysis.detectorExecution.verdict === "grounded" ? "text-emerald-700" : "text-rose-700"}>
                            {analysis.detectorExecution.verdict === "grounded" ? "Grounded" : "Hallucinated"}
                          </span>
                        </p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          {formatPct(analysis.detectorExecution.confidence)} confidence ·{" "}
                          {formatLatency(analysis.detectorExecution.latencyMs)}
                        </p>
                      </div>
                    </div>

                    <div className="pipeline-step">
                      <div className="step-icon"><StepIcon step={5} /></div>
                      <div>
                        <p className="font-semibold text-slate-900">
                          {analysis.trainingUpdate.updated ? "Policy updated" : "Experience buffered"}
                        </p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          {analysis.trainingUpdate.bufferSize} examples in replay buffer
                        </p>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            )}

            {/* POLICY DECISION TAB */}
            {activeTab === "decided" && (
              <div className="grid gap-4 lg:grid-cols-2">
                <section className="card p-6">
                  <h2 className="text-xl font-semibold text-slate-900">
                    Why{" "}
                    <span className={detectorColorClass(analysis.policy.selectedDetector)}>
                      {detectorLabel(analysis.policy.selectedDetector)}
                    </span>{" "}
                    was selected
                  </h2>
                  <p className="mt-2 text-sm text-slate-500">
                    The PPO policy encoded your query as a {analysis.policy.stateDim}-dimensional state vector
                    and chose the detector with the highest value estimate.
                  </p>

                  <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm">
                    <p className="font-semibold text-slate-700 mb-1">Query signals detected</p>
                    <ul className="space-y-1">
                      {featureSignals(analysis.queryProcessor.features).map((signal) => (
                        <li key={signal} className="flex items-start gap-2 text-slate-600">
                          <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
                          {signal}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {analysis.policy.reasons.length > 0 && (
                    <div className="mt-4 space-y-2">
                      {analysis.policy.reasons.map((reason) => (
                        <div key={reason} className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
                          {reason}
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="mt-5 grid grid-cols-3 gap-3 text-center text-sm">
                    <div className="metric-surface">
                      <p className="text-xs text-slate-500">Algorithm</p>
                      <p className="mt-0.5 font-semibold text-slate-900">{analysis.policy.algorithm.toUpperCase()}</p>
                    </div>
                    <div className="metric-surface">
                      <p className="text-xs text-slate-500">State dim</p>
                      <p className="mt-0.5 font-semibold text-slate-900">{analysis.policy.stateDim}</p>
                    </div>
                    <div className="metric-surface">
                      <p className="text-xs text-slate-500">State value</p>
                      <p className="mt-0.5 font-semibold text-slate-900">{analysis.policy.stateValue.toFixed(3)}</p>
                    </div>
                  </div>
                </section>

                <section className="card p-6">
                  <h2 className="text-xl font-semibold text-slate-900">Detector selection probabilities</h2>
                  <p className="mt-2 text-sm text-slate-500">
                    Softmax output of the actor network — probability assigned to each detector:
                  </p>
                  <div className="mt-5 space-y-5">
                    {sortedProbabilities.map(([name, probability]) => (
                      <div key={name}>
                        <div className="mb-1.5 flex items-start justify-between gap-3">
                          <div>
                            <p className={`font-semibold text-sm ${name === analysis.policy.selectedDetector ? detectorColorClass(name) : "text-slate-600"}`}>
                              {detectorLabel(name)}
                              {name === analysis.policy.selectedDetector && (
                                <span className="ml-2 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">selected</span>
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

                  <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm">
                    <p className="font-semibold text-slate-700 mb-2">Reward signal</p>
                    <p className="text-xs text-slate-500 mb-1 font-mono">{analysis.trainingUpdate.formula}</p>
                    <div className="flex gap-4">
                      <div>
                        <p className="text-xs text-slate-500">This run</p>
                        <p className={`font-semibold ${rewardToLabel(analysis.trainingUpdate.reward).color}`}>
                          {analysis.trainingUpdate.reward.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500">Buffer avg</p>
                        <p className="font-semibold text-slate-700">
                          {analysis.trainingUpdate.averageBufferReward.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500">Updated</p>
                        <p className={`font-semibold ${analysis.trainingUpdate.updated ? "text-emerald-700" : "text-slate-500"}`}>
                          {analysis.trainingUpdate.updated ? "Yes" : "No"}
                        </p>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </>
        )}

        {/* Empty state before first run */}
        {!analysis && !isLoading && (
          <section className="card">
            <div className="empty-state">
              <svg className="h-10 w-10 text-slate-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                <line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/>
                <circle cx="8" cy="6" r="2" fill="currentColor"/><circle cx="16" cy="12" r="2" fill="currentColor"/><circle cx="10" cy="18" r="2" fill="currentColor"/>
              </svg>
              <p className="text-lg font-semibold text-slate-900">Run the pipeline to see results</p>
              <p className="max-w-md text-sm text-slate-500">
                Select a question above and click <strong>Run Pipeline</strong>. The RL policy will adaptively
                choose the best hallucination detector for your query.
              </p>
            </div>
          </section>
        )}

      </div>
    </main>
  );
}
