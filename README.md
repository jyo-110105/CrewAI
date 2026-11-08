# CrewAI Projects API

FastAPI service exposing 10 CrewAI projects through REST endpoints.

## Local setup

1. Use Python 3.13.5.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set the `GEMINI_API_KEY` environment variable.
4. Start the API:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. Open `http://localhost:8000/docs`.

## Render deployment

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variable: `GEMINI_API_KEY` (add the secret in Render; do not commit it to GitHub).

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /agent1` | Blog Post Crew |
| `POST /agent2` | Competitor Analysis Crew (send exactly 3 URLs, comma-separated) |
| `POST /agent3` | Meeting Notes Crew (send transcript text) |
| `POST /agent4` | Data Report Crew (send CSV text) |
| `POST /agent5` | Customer Support Flow |
| `POST /agent6` | Resume Screening Flow |
| `POST /agent7` | Bug Triage Flow |
| `POST /agent8` | Content Repurposing Flow |
| `POST /agent9` | Lead Qualification Flow (send a website URL) |
| `POST /agent10` | Approval Report Flow (send `topic | audience`) |
| `POST /agent10/review` | Review an Agent 10 report |

All endpoints are documented at `/docs`.

