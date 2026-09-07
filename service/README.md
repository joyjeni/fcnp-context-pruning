# FCNP HTTP microservice

```bash
python service/app.py 43134
curl -X POST http://127.0.0.1:43134/prune -H 'Content-Type: application/json' \
  -d '{"query":"tomato Karnataka","elements":[{"id":"a","text":"Tomato mandi Bengaluru"}]}'
```

`POST /agri/snapshot` requires `DATA_GOV_API_KEY` and never invents prices.
Karnataka-first, then explicit national fallback of **live** records.
