#  CSUSB Internship Finder — User Guide

Your conversational assistant for discovering internship opportunities from the CSUSB website.  
This guide explains installation, configuration, running with Docker, and using the chat experience effectively.

---

##  What it is
A **Streamlit web app** that chats with you to clarify your internship interests, then searches the CSUSB internship listings and presents results in a clean table with direct links.

**Built with**
- Python & Streamlit  
- Docker for containerized deployment  
- CSUSB Internship Data (API/Scraper)  

---

##  Key Features
-  Chat interface to refine internship field, type, and timeframe  
-  Smart filters for department and major  
-  Time-based search (e.g., “Spring 2025”, “next summer”)  
-  Table view with title, department, type, location, and link  
-  Suggestions when no results are found  

---

##  Prerequisites
To run the app locally with Docker, you need:
- Docker Desktop  
- Internet connection  
- (Windows) WSL2 for Docker integration  

---

##  Running the Application
The repository includes a production-ready Dockerfile that sets a base URL path for easy hosting.

Follow these steps to build and run the **CSUSB Internship Finder** app from scratch on your local machine.  
This method ensures a clean setup, even if you’ve previously cloned or run the project.

---

## 🧹 Step 1: Remove any old version

If you already have the `team2f25` folder, delete it to avoid conflicts.

```bash
rm -rf team2f25
```

---

## 📥 Step 2: Clone the repository

Download the latest version of the project from GitHub.

```bash
git clone https://github.com/DrAlzahrani2025Projects/team2f25.git
```

---

## 📂 Step 3: Go to the project directory

```bash
cd team2f25
```

---

## 🔄 Step 4: Update to the latest code

Make sure your local copy is fully up to date.

```bash
git pull origin main
```

---

## ⚙️ Step 5: Make scripts executable

Only required once — allows the helper scripts to run properly.  
*(Windows users: run this in Git Bash or WSL.)*

```bash
chmod +x scripts/startup.sh scripts/cleanup.sh
```

---

## ▶️ Step 6: Build and start the Docker container

This automatically builds, starts, and runs the app inside Docker.

```bash
./scripts/startup.sh
```

Docker will:
- Build the container image  
- Start the Streamlit web app  
- Expose it on port 5002  

---

## 🌐 Step 7: Access the app

Once the container starts successfully, open your browser and go to:

**Local URL:**  
👉 [http://localhost:5002/team2f25](http://localhost:5002/team2f25)

You’ll see the **CSUSB Internship Finder** chat interface — start typing to search for internships!

---

## 🧹 Step 8: Stop and clean up

When you’re done, stop and remove the running container:

```bash
./scripts/cleanup.sh
```

To fully remove all images and cached data:

```bash
./scripts/cleanup.sh --hard
```

---

## 🌎 Hosted Version (no setup required)

You can also use the version hosted on the CSUSB CSE server:

👉 **https://sec.cse.csusb.edu/team2f25**

---

##  Google Colab Notebook

Try the project interactively in Google Colab:  
 [Open in Colab](https://colab.research.google.com/drive/1ziLOvU7CpqMwhXzOjQ0TL9gU5D9jNCgo#scrollTo=CByYPUtDGy-L)

---

##  Using the App
Open the app: **http://localhost:5002/team2f25**

### Start Chatting
Describe your interests naturally:
> “Find computer science internships for Spring 2025”  
> “Show paid on-campus internships in business”

The assistant may ask brief follow-ups before showing results.

### Filters
You can specify:
- Type → paid | remote | on-campus | volunteer  
- Field/Department → Computer Science, Marketing, etc.  
- Result count → “Show 5 internships” (default 10)

### Date Examples
- “This semester”   •  “Next summer”   •  “Spring 2025”  
- “Fall internships”   •  “Any time”

### Results
* The app displays a table:
>  > Title | Department | Type | Location | 🔗 Link |
* If nothing matches, the assistant suggests broader searches.

---

##  Tips
- Be specific (“paid CS internships Spring 2025”)  
- Include semester or season  
- Limit to 5–20 results  
- Add filters (“remote”, “on-campus”)  

---

##  Troubleshooting
* **Port already in use (0.0.0.0:5002)**  
Change port mapping: `-p 5003:5002`

* **No results found**  
Try broader keywords or remove filters.


* **Network error**  
Check your internet; the app auto-retries temporary failures.

---

##  Advanced Configuration
These environment variables tune app behavior.

**Variables**
- `PORT` → default `5002`  
- `BASE_URL` → default `/team2f25`

**Example**
```powershell
docker run -d `
  -p 5002:5002 `
  -e PORT=5002 `
  -e BASE_URL="/team2f25" `
  --name team2f25 `
  team2f25-streamlit
```

**Custom Example**
```powershell
docker run -d `
  -p 8080:5002 `
  -e PORT=5002 `
  -e BASE_URL="/internships" `
  --name csusb-internship `
  team2f25-streamlit
```
Then open → **http://localhost:8080/internships**

---

##  Where Else It’s Available
- Hosted (CSUSB): [https://sec.cse.csusb.edu/team2f25](https://sec.cse.csusb.edu/team2f25)  
- Google Colab Notebook: [Open in Colab](https://colab.research.google.com/drive/1ziLOvU7CpqMwhXzOjQ0TL9gU5D9jNCgo#scrollTo=CByYPUtDGy-L)

---

##  Uninstall / Cleanup
```powershell
docker stop team2f25
docker rm team2f25
# optional
docker rmi team2f25-streamlit
```

---

##  FAQ
* **What data leaves my machine?**  
Only your search queries to the CSUSB site; no personal data is stored.


* **Does it support REST APIs?**  
No — this is a Streamlit web app for browser use.


* **Can I change the number of results?**  
Yes — say “show 5 internships” (default 10).

* **Can I filter by type?**  
Yes — “paid”, “remote”, “on-campus”, or “volunteer”.


---
