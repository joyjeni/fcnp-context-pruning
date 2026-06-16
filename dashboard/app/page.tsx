import { getMetrics } from '../lib/store';
import Dashboard from './Dashboard';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function Page() {
  const metrics = await getMetrics();
  return <Dashboard metrics={metrics} />;
}
