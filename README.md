## Prerequisites

Before setting up the project, ensure you have the following installed:

Python 3.10+

pip (Python package installer)

Docker Desktop or Docker CLI: [Install Docker](https://www.docker.com) from its official website.

Visual Studio Code (VS Code): [Install VS code](https://code.visualstudio.com/download) from its official website.

Git for version control: [Install Git](https://git-scm.com/) from its official website.

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
chmod +x startup.sh
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

### Hosted on CSE department web server

For Streamlit:

Open browser at https://sec.cse.csusb.edu/team2f25 

## Google Colab Notebook  
[Open in Colab](https://colab.research.google.com/drive/1icOiUzhhm0l7PkDoCxUdDMqpX1eua8ug?usp=sharing)
