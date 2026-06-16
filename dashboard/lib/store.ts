/**
 * Metrics store.
 *
 * - If Vercel KV env vars are set, persists to KV (durable across deploys).
 * - Otherwise uses an in-memory store + bundled seed file.
 */
import fs from 'fs';
import path from 'path';
import { MetricsPayload } from './types';

const KV_URL = process.env.KV_REST_API_URL;
const KV_TOKEN = process.env.KV_REST_API_TOKEN;
const KV_KEY = 'fcnp:latest_metrics';

let mem: MetricsPayload | null = null;

function loadSeed(): MetricsPayload | null {
  try {
    const p = path.join(process.cwd(), 'public', 'metrics.json');
    if (fs.existsSync(p)) {
      return JSON.parse(fs.readFileSync(p, 'utf8')) as MetricsPayload;
    }
  } catch {
    /* ignore */
  }
  return null;
}

export async function getMetrics(): Promise<MetricsPayload | null> {
  if (KV_URL && KV_TOKEN) {
    try {
      const r = await fetch(`${KV_URL}/get/${KV_KEY}`, {
        headers: { Authorization: `Bearer ${KV_TOKEN}` },
        cache: 'no-store',
      });
      if (r.ok) {
        const data = (await r.json()) as { result: string | null };
        if (data.result) return JSON.parse(data.result) as MetricsPayload;
      }
    } catch {
      /* fall through */
    }
  }
  if (mem) return mem;
  mem = loadSeed();
  return mem;
}

export async function setMetrics(payload: MetricsPayload): Promise<void> {
  mem = payload;
  if (KV_URL && KV_TOKEN) {
    try {
      await fetch(`${KV_URL}/set/${KV_KEY}`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${KV_TOKEN}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
    } catch {
      /* keep in-memory copy */
    }
  }
}
