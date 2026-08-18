'use client';

import {
  ResponsiveContainer, ScatterChart, Scatter,
  XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, LabelList,
} from 'recharts';
import { MetricsPayload } from '../lib/types';

const PRIMARY_COLOR = '#58a6ff';
const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f78166', '#a371f7', '#e3b341', '#fb7185', '#56d364'];

function fmt(n: number | null | undefined, d = 3): string {
  if (n == null || !isFinite(n)) return '—';
  return n.toFixed(d);
}

export default function Dashboard({ metrics }: { metrics: MetricsPayload | null }) {
  if (!metrics) {
    return (
      <div className="container">
        <h1>FCNP Dashboard</h1>
        <p className="sub">No metrics yet — run the Kaggle notebook and POST to /api/metrics.</p>
        <div className="card">
          <p>Expected payload shape: <span className="code-pill">{`{ run_id, dataset, methods[], significance_vs_primary }`}</span></p>
          <p>Send with:</p>
          <pre style={{ background: '#0d1117', padding: 12, borderRadius: 6, overflow: 'auto' }}>
{`curl -X POST $DASHBOARD_URL/api/metrics \\
  -H "Authorization: Bearer $DASHBOARD_TOKEN" \\
  -H "Content-Type: application/json" \\
  --data @results/metrics.json`}
          </pre>
        </div>
      </div>
    );
  }

  const primary = metrics.methods.find(m => m.method === metrics.primary_method);
  const noComp  = metrics.methods.find(m => m.method === 'NoCompression');

  const compressionRatio = primary?.compression_ratio_mean ?? 0;
  const reductionPct = primary?.reduction_pct_mean ?? 0;
  const f1Improvement =
    primary && noComp ? primary.f1_mean - noComp.f1_mean : 0;
  const latencyP50 = primary?.latency_ms_p50 ?? 0;

  // Pareto data
  const paretoBase = metrics.methods.map((m, i) => ({
    method: m.method,
    x: m.compression_ratio_mean,
    y: m.f1_mean,
    z: 50,
    color: m.method === metrics.primary_method ? PRIMARY_COLOR : COLORS[i % COLORS.length],
  }));

  // Group points that are visually close (normalized by axis range) so their
  // labels can be fanned out instead of stacking on top of each other.
  const xValsAll = paretoBase.map(p => p.x);
  const xSpan = (Math.max(...xValsAll) - Math.min(...xValsAll)) || 1;
  const CLUSTER_THRESH = 0.07; // normalized distance (x/xSpan, y/1) to treat as "same cluster"
  const clusterOf: number[] = new Array(paretoBase.length).fill(-1);
  let nextClusterId = 0;
  for (let i = 0; i < paretoBase.length; i++) {
    if (clusterOf[i] !== -1) continue;
    clusterOf[i] = nextClusterId;
    for (let j = i + 1; j < paretoBase.length; j++) {
      if (clusterOf[j] !== -1) continue;
      const ndx = Math.abs(paretoBase[i].x - paretoBase[j].x) / xSpan;
      const ndy = Math.abs(paretoBase[i].y - paretoBase[j].y) / 1;
      if (ndx < CLUSTER_THRESH && ndy < CLUSTER_THRESH) clusterOf[j] = nextClusterId;
    }
    nextClusterId++;
  }
  const clusterSizes: Record<number, number> = {};
  clusterOf.forEach(c => { clusterSizes[c] = (clusterSizes[c] ?? 0) + 1; });
  const FAN_OFFSETS: Record<number, { dx: number; dy: number }[]> = {
    1: [{ dx: 0, dy: -10 }],
    2: [{ dx: -6, dy: -12 }, { dx: 6, dy: 18 }],
    3: [{ dx: -46, dy: -10 }, { dx: 0, dy: 18 }, { dx: 46, dy: -10 }],
    4: [{ dx: -60, dy: -14 }, { dx: -18, dy: 20 }, { dx: 18, dy: -30 }, { dx: 60, dy: 20 }],
  };
  const seenInCluster: Record<number, number> = {};
  const pareto = paretoBase.map((p, i) => {
    const cid = clusterOf[i];
    const size = Math.min(clusterSizes[cid] ?? 1, 4);
    const memberIdx = seenInCluster[cid] ?? 0;
    seenInCluster[cid] = memberIdx + 1;
    const offsets = FAN_OFFSETS[size] ?? FAN_OFFSETS[4];
    const off = offsets[memberIdx % offsets.length];
    return { ...p, labelDx: off.dx, labelDy: off.dy };
  });

  // Latency
  const latency = [...metrics.methods]
    .sort((a, b) => a.latency_ms_p50 - b.latency_ms_p50)
    .map(m => ({ method: m.method, p50: m.latency_ms_p50, p95: m.latency_ms_p95 }));

  // Recall/F1 bars
  const f1Bars = [...metrics.methods]
    .sort((a, b) => b.f1_mean - a.f1_mean)
    .map(m => ({
      method: m.method,
      f1: m.f1_mean,
      recall: m.recall_mean,
      ndcg: m.ndcg_mean,
    }));

  return (
    <div className="container">
      <h1>FCNP — Flow-Based Context Network Pruning</h1>
      <p className="sub">
        Dataset: <strong>{metrics.dataset}</strong> · Run: <span className="code-pill">{metrics.run_id}</span> · n = {metrics.n_examples} examples · primary: <strong>{metrics.primary_method}</strong>
      </p>

      <div className="grid">
        <div className="card">
          <div className="label">Compression ratio</div>
          <div className="value">{fmt(compressionRatio, 2)}×</div>
          <div className="sublabel">input tokens / output tokens</div>
        </div>
        <div className="card">
          <div className="label">Token reduction</div>
          <div className="value">{fmt(reductionPct, 1)}%</div>
          <div className="sublabel">of original context</div>
        </div>
        <div className="card">
          <div className="label">F1 vs full context</div>
          <div className="value">{f1Improvement >= 0 ? '+' : ''}{fmt(f1Improvement, 3)}</div>
          <div className="sublabel">absolute F1 delta</div>
        </div>
        <div className="card">
          <div className="label">Latency p50</div>
          <div className="value">{fmt(latencyP50, 1)} ms</div>
          <div className="sublabel">per query</div>
        </div>
        <div className="card">
          <div className="label">F1 (oracle k)</div>
          <div className="value">{fmt(primary?.f1_mean, 3)}</div>
          <div className="sublabel">95% CI [{fmt(primary?.f1_ci_lo, 3)}, {fmt(primary?.f1_ci_hi, 3)}]</div>
        </div>
        <div className="card">
          <div className="label">Recall@k</div>
          <div className="value">{fmt(primary?.recall_mean, 3)}</div>
          <div className="sublabel">95% CI [{fmt(primary?.recall_ci_lo, 3)}, {fmt(primary?.recall_ci_hi, 3)}]</div>
        </div>
        <div className="card">
          <div className="label">nDCG@k</div>
          <div className="value">{fmt(primary?.ndcg_mean, 3)}</div>
          <div className="sublabel">ranking quality</div>
        </div>
        <div className="card">
          <div className="label">Iterations to converge</div>
          <div className="value">{metrics.config?.max_iterations ?? '—'}</div>
          <div className="sublabel">ε = {metrics.config?.epsilon ?? '—'}</div>
        </div>
      </div>

      <h2>New in this version</h2>
      <div className="grid">
        <div className="card">
          <div className="label">#1 Dynamic re-pruning trigger</div>
          <div className="sublabel">Flow-entropy signal fires re-prune on drift, not a fixed clock</div>
        </div>
        <div className="card">
          <div className="label">#2 Hybrid tiering</div>
          <div className="value">{metrics.config?.enable_hybrid_tiering ? 'ON' : 'OFF'}</div>
          <div className="sublabel">summarize top {fmt((metrics.config?.summarize_top_k_fraction ?? 0) * 100, 0)}% instead of dropping · max {metrics.config?.summary_max_tokens ?? '—'} tok/summary</div>
        </div>
        <div className="card">
          <div className="label">#3 Persistent memory tier</div>
          <div className="sublabel">Flow-earned promotion/demotion, no unbounded growth</div>
        </div>
        <div className="card">
          <div className="label">#5 Compute vs token cost</div>
          <div className="sublabel">Deterministic linear solve vs metered LLM-compression calls</div>
        </div>
        <div className="card">
          <div className="label">Real Agri data (data.gov.in)</div>
          <div className="sublabel">Karnataka-first mandi prices, national fallback when empty</div>
        </div>
      </div>

      <h2>Pareto frontier — accuracy vs compression</h2>
      <p className="sublabel" style={{ marginTop: -6, marginBottom: 10 }}>
        Methods with identical (compression, F1) values plot as overlapping markers — labels are fanned out around them so every method stays readable.
      </p>
      <div className="chart-card">
        <ResponsiveContainer width="100%" height={340}>
          <ScatterChart margin={{ top: 34, right: 48, bottom: 36, left: 8 }}>
            <CartesianGrid stroke="#21262d" />
            <XAxis type="number" dataKey="x" name="Compression ratio (×)"
                   stroke="#8b949e" tick={{ fill: '#8b949e', fontSize: 11 }}
                   label={{ value: 'Compression ratio (input / output tokens)', position: 'bottom', offset: 10, fill: '#8b949e', fontSize: 12 }} />
            <YAxis type="number" dataKey="y" name="F1@k"
                   domain={[0, 1]}
                   stroke="#8b949e" tick={{ fill: '#8b949e', fontSize: 11 }}
                   label={{ value: 'F1@k (oracle budget)', angle: -90, position: 'left', fill: '#8b949e', fontSize: 12 }} />
            <ZAxis dataKey="z" range={[80, 80]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }}
                     contentStyle={{ background: '#161b22', border: '1px solid #30363d', color: '#e6edf3' }}
                     formatter={(v: any, n: any) => [typeof v === 'number' ? v.toFixed(3) : v, n]}
                     labelFormatter={() => ''}
                     itemSorter={() => -1} />
            <Scatter name="Methods" data={pareto} fill="#58a6ff" shape="circle">
              {pareto.map((p, i) => (
                <Scatter key={i} dataKey="y" fill={p.color} />
              ))}
              <LabelList
                dataKey="method"
                content={(props: any) => {
                  const { x, y, value, index } = props;
                  const off = pareto[index] ?? { labelDx: 0, labelDy: -10 };
                  const dx = off.labelDx ?? 0;
                  const dy = off.labelDy ?? -10;
                  return (
                    <text x={x + dx} y={y + dy} fill="#c9d1d9" fontSize={11} textAnchor="middle">
                      {value}
                    </text>
                  );
                }}
              />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <h2>F1 / Recall / nDCG by method</h2>
      <div className="chart-card">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={f1Bars} margin={{ top: 8, right: 24, bottom: 36, left: 8 }}>
            <CartesianGrid stroke="#21262d" />
            <XAxis dataKey="method" stroke="#8b949e" tick={{ fill: '#8b949e', fontSize: 11 }} angle={-15} textAnchor="end" />
            <YAxis stroke="#8b949e" tick={{ fill: '#8b949e', fontSize: 11 }} domain={[0, 1]} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', color: '#e6edf3' }}
                     formatter={(v: any) => (typeof v === 'number' ? v.toFixed(3) : v)} />
            <Legend wrapperStyle={{ color: '#c9d1d9' }} />
            <Bar dataKey="recall" name="Recall" fill="#3fb950" />
            <Bar dataKey="f1" name="F1" fill="#58a6ff" />
            <Bar dataKey="ndcg" name="nDCG" fill="#d29922" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2>Per-query latency</h2>
      <div className="chart-card">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={latency} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 80 }}>
            <CartesianGrid stroke="#21262d" />
            <XAxis type="number" stroke="#8b949e" tick={{ fill: '#8b949e', fontSize: 11 }}
                   label={{ value: 'Latency (ms)', position: 'bottom', fill: '#8b949e', fontSize: 12 }} />
            <YAxis type="category" dataKey="method" stroke="#8b949e" tick={{ fill: '#8b949e', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d', color: '#e6edf3' }}
                     formatter={(v: any) => (typeof v === 'number' ? v.toFixed(2) + ' ms' : v)} />
            <Legend wrapperStyle={{ color: '#c9d1d9' }} />
            <Bar dataKey="p50" name="p50" fill="#58a6ff" />
            <Bar dataKey="p95" name="p95" fill="#a371f7" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2>Detailed results</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th><th>n</th>
            <th>F1 (95% CI)</th><th>Recall</th><th>Precision</th><th>nDCG</th>
            <th>Comp.×</th><th>Reduction %</th>
            <th>Latency p50 (ms)</th><th>Latency p95 (ms)</th>
          </tr>
        </thead>
        <tbody>
          {metrics.methods.map(m => (
            <tr key={m.method} className={m.method === metrics.primary_method ? 'primary' : ''}>
              <td>{m.method}</td>
              <td>{m.n}</td>
              <td>{fmt(m.f1_mean)} <span style={{ color: '#6e7681' }}>[{fmt(m.f1_ci_lo)}, {fmt(m.f1_ci_hi)}]</span></td>
              <td>{fmt(m.recall_mean)}</td>
              <td>{fmt(m.precision_mean)}</td>
              <td>{fmt(m.ndcg_mean)}</td>
              <td>{fmt(m.compression_ratio_mean, 2)}</td>
              <td>{fmt(m.reduction_pct_mean, 1)}</td>
              <td>{fmt(m.latency_ms_p50, 2)}</td>
              <td>{fmt(m.latency_ms_p95, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Pairwise significance vs {metrics.primary_method} (Wilcoxon, F1)</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th><th>n pairs</th><th>statistic</th><th>p-value</th><th>p &lt; 0.05</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics.significance_vs_primary?.tests ?? {}).map(([m, t]: any) => (
            <tr key={m}>
              <td>{m}</td>
              <td>{t.n ?? '—'}</td>
              <td>{t.statistic != null ? Number(t.statistic).toFixed(2) : '—'}</td>
              <td>{t.p_value != null ? Number(t.p_value).toExponential(2) : '—'}</td>
              <td>
                {t.p_value != null
                  ? (t.p_value < 0.05 ? <span className="sig yes">YES</span> : <span className="sig no">no</span>)
                  : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <footer>
        Last update: {new Date(metrics.timestamp * 1000).toISOString()} ·
        {' '}<a href="/api/metrics">raw JSON</a> ·
        {' '}<a href="https://github.com/joyjeni/fcnp-context-pruning">source</a>
      </footer>
    </div>
  );
}
