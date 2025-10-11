# Internship Finder AI — (team2f25 / port 5002)

Single-container starter for the Week 1–2 milestone.
Tech: **Python**, **Streamlit**, **Playwright**, **pandas**, **LangChain (optional)**, **Docker**.

Target path: `https://sec.cse.csusb.edu/team2f25` (reverse-proxy to container **port 5002**).

---

## Run with Docker (recommended)

```bash
# stop anything already using 5002 (safe no-op if none)
docker rm -f team2f25 2>/dev/null || true
for id in $(docker ps -q --filter "publish=5002"); do docker rm -f "$id"; done

# build
docker build -t internship-finder:team2f25 .

# run (mount only /data so code inside the image can't be overridden)
docker run --rm -p 5002:5002 \
  -v "$(pwd)/data:/app/data" \
  --name team2f25 \
  internship-finder:team2f25


