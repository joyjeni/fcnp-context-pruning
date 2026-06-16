import './globals.css';

export const metadata = {
  title: 'FCNP — Flow-Based Context Network Pruning',
  description: 'Benchmark dashboard for FCNP on ToolBench',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
