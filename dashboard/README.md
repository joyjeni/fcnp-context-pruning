# FCNP Dashboard

Next.js app that visualizes FCNP benchmark metrics produced by the
Kaggle notebook.

## Environment variables (Vercel)

| Variable | Purpose |
|---|---|
| `DASHBOARD_TOKEN` | Bearer token required to POST new metrics to `/api/metrics` |
| `KV_REST_API_URL` *(optional)* | Vercel KV REST endpoint for cross-region persistence |
| `KV_REST_API_TOKEN` *(optional)* | Vercel KV bearer token |

If `KV_*` variables are absent, the app falls back to in-memory storage
(metrics survive only while the function instance is warm) plus the
bundled `public/metrics.json` as the seed value.

## Local development

```bash
npm install
npm run dev
```

## Deploy

```bash
vercel --prod
```

## API

### `POST /api/metrics`

Auth: `Authorization: Bearer $DASHBOARD_TOKEN`.
Body: the JSON payload written by `fcnp.report.write_report` →
`results/metrics.json`.

### `GET /api/metrics`

Returns the most recently posted payload (no auth required).
