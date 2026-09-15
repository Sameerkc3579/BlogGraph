# Deploying BlogGraph on Render

Render's free disk is wiped on every deploy and restart, so accounts, blogs and images are
stored in PostgreSQL. The app refuses to start on Render without `DATABASE_URL`.

## 1. Create a PostgreSQL database

Use any hosted Postgres. [Neon](https://neon.tech) has a free tier that doesn't expire
(Render's own free Postgres is deleted after a trial period — check their current terms).

Copy the connection string, e.g.
`postgresql://user:password@host/dbname?sslmode=require`

Tables are created automatically on first start.

## 2. Push the code to GitHub

`render.yaml` (service settings, including the Python version) and the pinned `requirements.txt` are already in the repo.
Make sure `.env` and `bloggraph.db` are **not** committed (both are in `.gitignore`).

## 3. Create the service on Render

1. Render dashboard → **New → Blueprint** → select this repository.
2. Render reads `render.yaml` and asks for the secret values:
   - `DATABASE_URL` — the connection string from step 1

   **No AI keys go on the server.** Each user enters their own Google Gemini API key
   (and optionally a Hugging Face token for images) in the app's Configure panel. Keys are sent
   only with that user's requests and are never stored. Keep `ALLOW_SERVER_KEYS=0` and don't set
   `GOOGLE_API_KEY` / `HF_TOKEN` on Render.
3. Deploy. The health check is `/health`.

Optional settings (Environment tab):

| Variable | Default | Purpose |
|---|---|---|
| `DAILY_GENERATION_LIMIT` | `0` | Successful blogs per user per 24 h (`0` = unlimited) |
| `ENABLE_IMAGES` | `1` | `0` skips images to save Hugging Face credits |
| `HF_IMAGE_MODEL` | `black-forest-labs/FLUX.1-schnell` | Image model |
| `ALLOWED_ORIGINS` | *(none)* | Only needed if another site calls the API |

## Notes

- **Free plan sleeps** after ~15 minutes idle; the first visit afterwards takes up to a minute.
- **Generation takes 1–3 minutes** per blog and uses your Hugging Face credits.
  Every visitor must sign up, and each account is limited by `DAILY_GENERATION_LIMIT`.
- **Local development** still uses `bloggraph.db` (SQLite) when `DATABASE_URL` is empty:
  `python api.py`
- **Blogs saved before accounts existed** can be moved into your account locally:
  sign up in the app, then run `python database.py claim-blogs you@example.com`
