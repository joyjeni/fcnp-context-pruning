export interface MethodAggregate {
  method: string;
  n: number;
  recall_mean: number; recall_ci_lo: number; recall_ci_hi: number;
  precision_mean: number; precision_ci_lo: number; precision_ci_hi: number;
  f1_mean: number; f1_ci_lo: number; f1_ci_hi: number;
  ndcg_mean: number; ndcg_ci_lo: number; ndcg_ci_hi: number;
  compression_ratio_mean: number;
  reduction_pct_mean: number;
  latency_ms_mean: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
}

export interface WilcoxonResult {
  p_value: number | null;
  statistic: number | null;
  n: number;
}

export interface MetricsPayload {
  run_id: string;
  timestamp: number;
  dataset: string;
  primary_method: string;
  config: Record<string, any>;
  n_examples: number;
  methods: MethodAggregate[];
  significance_vs_primary: {
    metric: string;
    tests: Record<string, WilcoxonResult>;
  };
  per_split_summary?: Record<string, Record<string, Record<string, number>>>;
}
