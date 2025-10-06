## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from its official website.
2. **Docker**: [Install Docker](https://www.docker.com) from its official website.
3. **Linux/MacOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/windows/wsl/).

---

### Step 1: Remove the existing code directory completely

Because the local repository can't been updated correctly, need to remove the directory first.

```bash
rm -rf team2f25
```

### Step 2: Clone the Repository

Clone the GitHub repository to your local machine:

```
git clone https://github.com/DrAlzahrani2025Projects/team2f25.git
```

### Step 3: Navigate to the Repository

Change to the cloned repository directory:

```
cd team2f25
```

### Step 4: Pull the Latest Version

Update the repository to the latest version:

```
git pull origin main
```


### Step 5: Build the docker image:

Run the setup script to build and start the Docker container:

```
docker build -t team2f25-streamlit:latest .
```

### Step 6: Run the container:

```
docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit:latest streamlit run app.py --server.port=5002 --server.address=0.0.0.0 --server.enableCORS=false --server.baseUrlPath=/team2f25

```

If you're using git bash run the below command
```
docker run -d -p 5002:5002 --name team2f25 team2f25-streamlit:latest
```

### Optional Step : Error: port is already allocated
If you're encountering error: port is already allocated
```
docker stop $(docker ps -q --filter "publish=5002")
docker ps -a -q | xargs -r docker rm

```


### Step 7: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://localhost:5002/team2f25

  

---

### Hosted on CSE department web server

For Streamlit:

Open browser at https://sec.cse.csusb.edu/team2f25 

## Google Colab Notebook  
[Open in Colab](https://colab.research.google.com/drive/1icOiUzhhm0l7PkDoCxUdDMqpX1eua8ug?usp=sharing)
