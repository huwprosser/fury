---
name: web_search
description: Search the web. Returns a list of search results to then be explored. Use when you need current information or to verify facts.
---

# Web Search Skill

Searches the web using the Brave Search API.

## Usage

1. Get the user's query.
2. Run the Brave Search API query (API key loaded from `.env` file):

```bash
source /PATH_TO_ENV_FILE/.env
curl -H "x-subscription-token: $BRAVE_API_KEY" \
  "https://api.search.brave.com/res/v1/web/search?q=QUERY"
```

3. Summarize the top results in a single sentence. Cite sources if relevant.

## Notes

- API key is loaded from `PATH_TO_ENV_FILE/.env`
- Response is JSON. Key fields: `web.results[*].title`, `web.results[*].url`, `web.results[*].description`
- Keep summaries concise. No markdown.
