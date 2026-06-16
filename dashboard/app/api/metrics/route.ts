import { NextRequest, NextResponse } from 'next/server';
import { getMetrics, setMetrics } from '../../../lib/store';
import { MetricsPayload } from '../../../lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  const m = await getMetrics();
  if (!m) return NextResponse.json({ error: 'no metrics yet' }, { status: 404 });
  return NextResponse.json(m);
}

export async function POST(req: NextRequest) {
  const expected = process.env.DASHBOARD_TOKEN;
  if (!expected) {
    return NextResponse.json(
      { error: 'server misconfigured: DASHBOARD_TOKEN not set' },
      { status: 500 },
    );
  }
  const auth = req.headers.get('authorization') ?? '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : '';
  if (token !== expected) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  let payload: MetricsPayload;
  try {
    payload = (await req.json()) as MetricsPayload;
  } catch {
    return NextResponse.json({ error: 'invalid json' }, { status: 400 });
  }
  if (!payload?.methods || !Array.isArray(payload.methods)) {
    return NextResponse.json({ error: 'missing methods[]' }, { status: 400 });
  }

  await setMetrics(payload);
  return NextResponse.json({ ok: true, run_id: payload.run_id });
}
