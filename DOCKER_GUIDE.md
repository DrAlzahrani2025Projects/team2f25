#  Docker Guide — CSUSB Internship Finder (team2f25)

This guide provides full Docker instructions for building, running, testing, and maintaining the **CSUSB Internship Finder** web app.

---

## Overview

This project includes:

- **Dockerfile** — Main Streamlit app (production)
- **Dockerfile.test** — Test container (unit/integration tests)
- **Helper scripts** — Easy setup and cleanup (`scripts/startup.sh`, `scripts/cleanup.sh`)

---

## Production Container (Dockerfile)

### Build

```bash
docker build -f docker/Dockerfile -t team2f25-streamlit .
```

### Run

```bash
# Default configuration
docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit
```

### Access

Open your browser:  
👉 **http://localhost:5002/team2f25**

---

## Test Container (Dockerfile.test)

### Build

```bash
docker build -f docker/Dockerfile.test -t team2f25-tests .
```

### Run Tests

```bash
# Run unit tests (default)
docker run --rm team2f25-tests

# Run integration tests
docker run --rm -e API_KEY="your-key" team2f25-tests pytest tests/integration

# Run end-to-end (E2E) tests
docker run --rm -e API_KEY="your-key" team2f25-tests pytest tests/e2e

# Run all tests
docker run --rm -e API_KEY="your-key" team2f25-tests pytest tests
```

### Verbose and Debug Modes

```bash
# Verbose test output
docker run --rm team2f25-tests pytest -vv

# Show print() statements
docker run --rm team2f25-tests pytest -s

# Stop on first failure
docker run --rm team2f25-tests pytest -x
```

### Generate Coverage Reports

```bash
# Terminal summary
docker run --rm team2f25-tests pytest --cov=app --cov-report=term

# Detailed missing coverage
docker run --rm team2f25-tests pytest --cov=app --cov-report=term-missing

# Save HTML coverage (requires volume mount)
docker run --rm -v $(pwd)/htmlcov:/app/htmlcov   team2f25-tests pytest --cov=app --cov-report=html
```

---

## Scripted Setup (Recommended)

### Start the App

```bash
./scripts/startup.sh
```

Accessible at → **http://localhost:5002/team2f25**

### Stop & Cleanup

```bash
./scripts/cleanup.sh
```

To also delete local data:
```bash
./scripts/cleanup.sh --hard
```

---

## Docker Compose (Optional)

Create a `docker-compose.yml` for easier orchestration:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: team2f25-streamlit
    ports:
      - "5002:5002"
    environment:
      - PORT=5002
      - BASE_URL=/team2f25
    volumes:
      - ./data:/data
    restart: unless-stopped

  tests:
    build:
      context: .
      dockerfile: docker/Dockerfile.test
    image: team2f25-tests
    environment:
      - API_KEY=${API_KEY}
    command: pytest tests
```

### Usage

```bash
# Run app
docker-compose up app

# Run tests
docker-compose run --rm tests

# Run specific test type
docker-compose run --rm tests pytest tests/unit

# Stop and cleanup
docker-compose down
```

---

## Environment Variables

| Variable | Required | Description | Default |
|-----------|-----------|-------------|----------|
| `PORT` | No | Internal app port | `5002` |
| `BASE_URL` | No | Web path for app | `/team2f25` |
| `API_KEY` | Optional | Used for integration or E2E tests | — |

---

## Container Management

```bash
# List running containers
docker ps

# View logs
docker logs -f team2f25

# Stop/start
docker stop team2f25
docker start team2f25

# Remove container
docker rm team2f25

# Remove image
docker rmi team2f25-streamlit
```

---

## Updating the App

```bash
# Pull latest code
git pull origin main

# Rebuild & restart
docker stop team2f25 || true
docker rm team2f25 || true
docker build -f docker/Dockerfile -t team2f25-streamlit .
docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit
```

Or simply rerun:
```bash
./scripts/cleanup.sh
./scripts/startup.sh
```

---

## Troubleshooting

### Port Already in Use

```bash
docker run -d -p 5003:5002 --name team2f25 team2f25-streamlit
```

### Permission Denied on Scripts

On Windows:
```bash
chmod +x scripts/*.sh
```

### Container Exits Immediately

```bash
docker logs team2f25
```

### Integration Tests Skipped

```bash
# Ensure API_KEY is provided
docker run --rm -e API_KEY="your-key" team2f25-tests pytest tests/integration
```

---

## Best Practices

1. **Build once:**
   ```bash
   docker build -f docker/Dockerfile -t team2f25-streamlit .
   docker build -f docker/Dockerfile.test -t team2f25-tests .
   ```

2. **Run unit tests frequently:**
   ```bash
   docker run --rm team2f25-tests
   ```

3. **Run integration tests before pushing changes:**
   ```bash
   docker run --rm -e API_KEY="your-key" team2f25-tests pytest tests
   ```

4. **Test the app locally:**
   ```bash
   docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit
   ```

---

## Quick Reference

```bash
# Production
docker build -f docker/Dockerfile -t team2f25-streamlit .
docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit

# Tests (unit)
docker build -f docker/Dockerfile.test -t team2f25-tests .
docker run --rm team2f25-tests

# Tests (integration)
docker run --rm -e API_KEY="key" team2f25-tests pytest tests/integration

# Tests (all)
docker run --rm -e API_KEY="key" team2f25-tests pytest tests

# Cleanup
docker system prune -a
```

---

## Hosted Access

- **CSUSB Hosted:** [https://sec.cse.csusb.edu/team2f25](https://sec.cse.csusb.edu/team2f25)  
- **Google Colab:** [https://colab.research.google.com/drive/1ziLOvU7CpqMwhXzOjQ0TL9gU5D9jNCgo#scrollTo=CByYPUtDGy-L](https://colab.research.google.com/drive/1ziLOvU7CpqMwhXzOjQ0TL9gU5D9jNCgo#scrollTo=CByYPUtDGy-L)
