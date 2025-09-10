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
rm -rf team1f25
```

### Step 2: Clone the Repository

Clone the GitHub repository to your local machine:

```
git clone https://github.com/DrAlzahrani2025Projects/team1f25.git
```

### Step 3: Navigate to the Repository

Change to the cloned repository directory:

```
cd team1f25
```

### Step 4: Pull the Latest Version

Update the repository to the latest version:

```
git pull origin main
```

### Step 5: Enable execute permissions for the Docker build and cleanup script:

Run the setup script to build and start the Docker container:

```
chmod +x docker-setup.sh
```

### Step 6: Run Build Script (enter your Groq API Key when prompted):

```
./docker-setup.sh
```

### Step 7: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://localhost:2501/team1f25
  

### Step 8: Run the script to stop and remove the Docker image and container :

```
./docker-cleanup.sh
```

---

### Hosted on CSE department web server

For Streamlit:

Open browser at https://sec.cse.csysb.edu/team1f25 

## Google Colab Notebook  

We have integrated a Google Colab notebook for easy access and execution.

[Open in Colab]()

