export type DetectorMetrics = {
  name: string;
  accuracy: number;
  precision?: number;
  recall?: number;
  f1?: number;
  avg_latency_ms: number;
  total_time_s?: number;
  cost: number;
  confusion_matrix?: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
};

export type BenchmarkJson = Record<string, DetectorMetrics>;

export type PolicyEntry = {
  stats: { pulls: number; reward_sum: number; correct_sum: number };
  mean_reward: number;
  empirical_accuracy: number;
  prior: {
    name: string;
    accuracy: number;
    avg_latency_ms: number;
    cost: number;
  };
};

export type PPOPolicyModel = {
  type: "mlp_tanh";
  feature_names: string[];
  detectors: string[];
  normalization: {
    mean: number[];
    std: number[];
  };
  actor: {
    hidden_weight: number[][];
    hidden_bias: number[];
    output_weight: number[][];
    output_bias: number[];
  };
};

export type PolicyJson = {
  algorithm?: "bandit" | "ppo";
  config?: { epsilon?: number };
  policy?: Record<string, PolicyEntry>;
  policy_model?: PPOPolicyModel;
};

export type ExtractedFeatures = {
  questionLen: number;
  entityCount: number;
  numericDensity: number;
  hasPercentage: number;
  hasComparison: number;
  complexityScore: number;
  contextChunks: number;
  retrievalConfidence: number;
  topChunkScore: number;
  scoreGap: number;
  answerLen: number;
  answerNumericDensity: number;
};

export type SelectionResult = {
  algorithm: "bandit" | "ppo" | "fallback";
  selectedDetector: string;
  selectedProbability: number;
  explorationMode: boolean;
  features: ExtractedFeatures;
  probabilities: Record<string, number>;
  reasons: string[];
};

export function extractFeatures(
  question: string,
  options?: {
    answer?: string;
    contextChunks?: number;
    retrievalConfidence?: number;
    topChunkScore?: number;
    scoreGap?: number;
  }
): ExtractedFeatures {
  const q = question || "";
  const answer = options?.answer ?? "";
  const tokens = (q.toLowerCase().match(/\b\w+\b/g) ?? []).length;
  const entityMatches = new Set(
    q.match(/\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b/g) ?? []
  );
  const numericMatches = q.match(/\d+(?:\.\d+)?/g) ?? [];
  const numericDensity = numericMatches.length / Math.max(1, tokens);
  const hasPercentage = q.includes("%") ? 1 : 0;
  const hasComparison = /compare|increase|decrease|growth|change|versus|difference/i.test(q) ? 1 : 0;
  const answerTokens = (answer.toLowerCase().match(/\b\w+\b/g) ?? []).length;
  const answerNumericMatches = answer.match(/\d+(?:\.\d+)?/g) ?? [];
  const complexityScore = Math.min(
    1,
    0.35 * Math.min(tokens / 20, 1) +
      0.2 * Math.min(entityMatches.size / 3, 1) +
      0.2 * numericDensity +
      0.15 * hasPercentage +
      0.1 * hasComparison
  );

  return {
    questionLen: tokens,
    entityCount: entityMatches.size,
    numericDensity,
    hasPercentage,
    hasComparison,
    complexityScore,
    contextChunks: options?.contextChunks ?? 0,
    retrievalConfidence: options?.retrievalConfidence ?? 0,
    topChunkScore: options?.topChunkScore ?? 0,
    scoreGap: options?.scoreGap ?? 0,
    answerLen: answerTokens,
    answerNumericDensity: answerNumericMatches.length / Math.max(1, answerTokens),
  };
}

function contextBonus(detector: string, features: ExtractedFeatures) {
  const {
    numericDensity,
    questionLen,
    hasComparison,
    hasPercentage,
    complexityScore,
    retrievalConfidence,
    scoreGap,
    answerLen,
  } = features;

  if (detector === "token_overlap") {
    return 0.1 * numericDensity + 0.05 * hasPercentage + 0.08 * retrievalConfidence - 0.05 * complexityScore;
  }
  if (detector === "semantic_similarity") {
    return 0.06 * Math.max(0, 1 - numericDensity) + 0.04 * retrievalConfidence + 0.03 * Math.min(answerLen / 25, 1);
  }
  if (detector === "bert_nli") {
    return 0.08 * hasComparison + 0.03 * Math.min(questionLen / 30, 1) + 0.06 * complexityScore + 0.04 * scoreGap;
  }
  if (detector === "llm_judge") {
    return (
      0.1 * hasComparison +
      0.05 * Math.min(questionLen / 40, 1) +
      0.06 * numericDensity +
      0.08 * complexityScore +
      0.05 * retrievalConfidence
    );
  }
  return 0;
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / Math.max(total, 1e-8));
}

function linear(input: number[], weight: number[][], bias: number[]): number[] {
  return weight.map((row, rowIndex) =>
    row.reduce((sum, value, columnIndex) => sum + value * (input[columnIndex] ?? 0), bias[rowIndex] ?? 0)
  );
}

function toFeatureVector(features: ExtractedFeatures, featureNames: string[]): number[] {
  const lookup: Record<string, number> = {
    question_len: features.questionLen,
    entity_count: features.entityCount,
    numeric_density: features.numericDensity,
    has_percentage: features.hasPercentage,
    has_comparison: features.hasComparison,
    complexity_score: features.complexityScore,
    context_chunks: features.contextChunks,
    retrieval_confidence: features.retrievalConfidence,
    top_chunk_score: features.topChunkScore,
    score_gap: features.scoreGap,
    answer_len: features.answerLen,
    answer_numeric_density: features.answerNumericDensity,
  };

  return featureNames.map((name) => lookup[name] ?? 0);
}

function normalize(vector: number[], mean: number[], std: number[]): number[] {
  return vector.map((value, index) => {
    const scale = std[index] && std[index] > 1e-8 ? std[index] : 1;
    return (value - (mean[index] ?? 0)) / scale;
  });
}

function runPPOPolicy(model: PPOPolicyModel, features: ExtractedFeatures) {
  if (model.type !== "mlp_tanh") {
    return null;
  }

  const featureVector = toFeatureVector(features, model.feature_names);
  const normalized = normalize(
    featureVector,
    model.normalization.mean,
    model.normalization.std
  );
  const hidden = linear(normalized, model.actor.hidden_weight, model.actor.hidden_bias).map((value) =>
    Math.tanh(value)
  );
  const logits = linear(hidden, model.actor.output_weight, model.actor.output_bias);
  const probabilities = softmax(logits);

  let bestIndex = 0;
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] > probabilities[bestIndex]) {
      bestIndex = index;
    }
  }

  return {
    selectedDetector: model.detectors[bestIndex],
    selectedProbability: probabilities[bestIndex],
    probabilities: Object.fromEntries(
      model.detectors.map((detector, index) => [detector, probabilities[index] ?? 0])
    ) as Record<string, number>,
  };
}

export function detectorReasons(
  detector: string,
  features: ExtractedFeatures,
  metric: DetectorMetrics,
  leadReason?: string
): string[] {
  const reasons: string[] = [];

  if (leadReason) {
    reasons.push(leadReason);
  }
  if (features.numericDensity > 0.05) {
    reasons.push("Question includes numeric signals, so factual grounding is prioritized.");
  }
  if (features.hasComparison > 0) {
    reasons.push("Question implies comparison or trend reasoning, so consistency checks are emphasized.");
  }
  if (features.retrievalConfidence > 0.7) {
    reasons.push("Retrieval confidence is high, making grounded low-cost detectors more viable.");
  }

  reasons.push(`Expected quality: ${((metric.f1 ?? metric.accuracy) * 100).toFixed(1)}% on benchmark.`);
  reasons.push(`Speed/cost profile: ${metric.avg_latency_ms.toFixed(1)}ms at $${metric.cost.toFixed(2)} per run.`);

  if (reasons.length < 3) {
    reasons.push(`Selected ${detector} as the best tradeoff for this question pattern.`);
  }

  return reasons.slice(0, 3);
}

export function selectDetectorForQuestion(
  benchmarkParsed: BenchmarkJson,
  policyParsed: PolicyJson,
  question: string
): SelectionResult {
  const detectors = Object.values(benchmarkParsed);
  if (detectors.length === 0) {
    throw new Error("No detector benchmark data found");
  }

  const detectorByName = new Map(detectors.map((detector) => [detector.name, detector]));
  const features = extractFeatures(question);

  if (policyParsed.algorithm === "ppo" && policyParsed.policy_model) {
    const inference = runPPOPolicy(policyParsed.policy_model, features);
    if (inference?.selectedDetector) {
      const selected =
        detectorByName.get(inference.selectedDetector) ?? detectors[0];
      return {
        algorithm: "ppo",
        selectedDetector: selected.name,
        selectedProbability: inference.selectedProbability,
        explorationMode: false,
        features,
        probabilities: inference.probabilities,
        reasons: detectorReasons(
          selected.name,
          features,
          selected,
          `PPO policy confidence: ${(inference.selectedProbability * 100).toFixed(1)}%.`
        ),
      };
    }
  }

  const epsilon = policyParsed.config?.epsilon ?? 0.12;
  const scores = detectors.map((detector) => {
    const policyEntry = policyParsed.policy?.[detector.name];
    const baseScore = policyEntry?.mean_reward ?? detector.accuracy;
    const pulls = Math.max(1, policyEntry?.stats.pulls ?? 1);
    const explorationBonus = 0.15 / Math.sqrt(pulls);
    return baseScore + explorationBonus + contextBonus(detector.name, features);
  });
  const scoreProbabilities = softmax(scores);
  const probabilityMap = Object.fromEntries(
    detectors.map((detector, index) => [detector.name, scoreProbabilities[index] ?? 0])
  ) as Record<string, number>;

  const explore = Math.random() < epsilon;
  let selected = detectors[0];
  let selectedProbability = probabilityMap[selected.name] ?? 0;

  if (explore) {
    selected = detectors[Math.floor(Math.random() * detectors.length)];
    selectedProbability = 1 / detectors.length;
  } else {
    let bestScore = Number.NEGATIVE_INFINITY;

    for (let index = 0; index < detectors.length; index += 1) {
      const detector = detectors[index];
      if (scores[index] > bestScore) {
        bestScore = scores[index];
        selected = detector;
        selectedProbability = probabilityMap[detector.name] ?? 0;
      }
    }
  }

  return {
    algorithm: policyParsed.algorithm === "bandit" ? "bandit" : "fallback",
    selectedDetector: selected.name,
    selectedProbability,
    explorationMode: explore,
    features,
    probabilities: explore
      ? Object.fromEntries(detectors.map((detector) => [detector.name, 1 / detectors.length]))
      : probabilityMap,
    reasons: detectorReasons(
      selected.name,
      features,
      selected,
      explore ? "Bandit exploration mode sampled an alternative detector." : undefined
    ),
  };
}
