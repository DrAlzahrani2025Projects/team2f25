🧱 Build the Docker Image

Run this inside your project folder (TEAM2F25):

docker build -t internship-finder:team2f25 .

🚀 Run the Container
docker run --rm -p 5002:5002 \
  -v "$(pwd)/data:/app/data" \
  --name team2f25 \
  internship-finder:team2f25


👉 Windows Git Bash users:
If you get a volume mount error, use:

docker run --rm -p 5002:5002 \
  -v "//c/Users/<YourUsername>/Downloads/TEAM2F25/data:/app/data" \
  --name team2f25 \
  internship-finder:team2f25


Then open in your browser:
👉 http://localhost:5002/team2f25

or (if deployed on CSUSB server):
👉 https://sec.cse.csusb.edu/team2f25









////////////////////
# from the project root (where Dockerfile is)
docker build -t internship-finder:team2f25 .

# run
docker run --rm -p 5002:5002 \
  -v "$(pwd)/data:/app/data" \
  --name team2f25 \
  internship-finder:team2f25


# stop/clean
./cleanup.sh
///////////////////

# RUN WITHOUT DOCKER
////////
Run without Docker (native)
# 1 create & activate venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2 install deps
pip install -r requirements.txt
python -m playwright install chromium   # installs browser

# 3 run streamlit on port 5002 with base path /team2f25
streamlit run app.py --server.address 0.0.0.0 --server.port 5002 --server.baseUrlPath team2f25
////////