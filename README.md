## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from its official website.
2. **Docker**: [Install Docker](https://www.docker.com) from its official website.
3. **Linux/MacOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/windows/wsl/).


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


### Step 5: Make the startup file executable:

Run the setup script to build and start the Docker container:

```
sed -i 's/\r$//' startup.sh cleanup.sh entrypoint.sh
chmod +x startup.sh cleanup.sh entrypoint.sh
```

### Step 6: Run the startup file:

This will automatically build, start and run the container

```
./startup.sh
```

### Step 7: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://localhost:5002/team2f25

  

---
### Step 9: Clean Up

When you are finished, run the cleanup script to stop and remove the Docker container and image.

```bash
./cleanup.sh
```

---

### Hosted on CSE department web server

For Streamlit:

Open browser at https://sec.cse.csusb.edu/team2f25 

## Google Colab Notebook  
[Open in Colab]([https://colab.research.google.com/drive/1icOiUzhhm0l7PkDoCxUdDMqpX1eua8ug?usp=sharing])
