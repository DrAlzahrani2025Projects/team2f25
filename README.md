## Prerequisites

Before setting up the project, ensure you have the following installed:

Python 3.10+

pip (Python package installer)

Docker Desktop or Docker CLI
Install Docker

Visual Studio Code (VS Code)
Install VS Code

Git for version control
Install Git

Jupyter Notebook (for documentation and demos)
Install Jupyter
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
