import streamlit as st
import os, time, math, re, json
import pandas as pd

from backend import navigate_career_page
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# --- Ollama warm-up (once) ---
@st.cache_resource(show_spinner=False)
def ensure_ollama_ready():
    """Ping local Ollama and pull the model if missing (one-time)."""
    import os, json, urllib.request
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        names = [m.get("name","") for m in (data.get("models", []) or [])]
        if model not in names:
            st.info(f"Pulling model `{model}`… (one-time)")
            body = json.dumps({"name": model}).encode("utf-8")
            req = urllib.request.Request(f"{host}/api/pull", data=body, headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=300).read()
    except Exception:
        st.warning("Ollama isn't reachable; general chat may be slow or fail.")

@st.cache_resource
def initialize_llm():
    """Initialize Ollama LLM"""
    try:        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.3,
            streaming=False,
            model_kwargs={"num_ctx": 1536, "num_predict": 180}
        )
        
        return llm
        
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return "I'm here to help! What would you like to know?"

def llm_query(llm, user_text: str) -> str:
    """
    Query the LLM with a user message and optional conversation history.
    
    Args:
        llm: ChatOllama instance (from initialize_llm())
        user_text: Current user message
        history_messages: Optional list of previous SystemMessage/HumanMessage objects
    
    Returns:
        String response from the LLM
    """
    try:
        # System prompt
        sys_message = SystemMessage(
            content="You are a helpful and friendly assistant for CSUSB internship search. Keep replies concise and encouraging."
        )
        
        # Build message list
        messages = [sys_message]
        
        # Add current user message
        st.session_state.messages.append(HumanMessage(content=user_text))

        # Add conversation history
        messages.extend(st.session_state.messages[1:][-5:])
        
        
        # Invoke LLM
        response = llm.invoke(messages)
        st.session_state.messages.append(AIMessage(content=response.content.strip()))

        return response.content.strip()
        
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return "I'm here to help! What would you like to know?"


# ---------- LLM general reply ----------
def llm_general_reply(user_text: str, history: str) -> str:
    """Call Ollama LLM for general chat."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.3,
            streaming=False,
            model_kwargs={"num_ctx": 1536, "num_predict": 180}
        )
        
        sys = "You are a helpful and friendly assistant for CSUSB internship search. Keep replies concise and encouraging."
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys),
            ("human", "History:\n{history}\n\nUser: {q}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"history": history, "q": user_text})
        return response.content.strip()
        
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return "I'm here to help! What would you like to know?"

# ---------- NEW: Extract preference from user response ----------
def extract_preference_from_response(user_response: str, pref_key: str) -> list:
    """Extract structured preference from user's freeform response."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.1,
            streaming=False,
            model_kwargs={"num_ctx": 1024, "num_predict": 100}
        )
        
        sys_extract = f"""You are an internship preference extractor. Extract key terms from the user's response about {pref_key}.

Return ONLY a JSON array of strings. Example: ["item1", "item2", "item3"]

Be faithful to what they said. Don't invent terms."""

        prompt_extract = ChatPromptTemplate.from_messages([
            ("system", sys_extract),
            ("human", f"User response: {user_response}\n\nReturn JSON array now.")
        ])
        
        response = (prompt_extract | llm).invoke({})
        raw = response.content.strip()
        
        # Extract JSON array
        json_match = re.search(r'\[[\s\S]*\]', raw)
        if json_match:
            extracted = json.loads(json_match.group(0))
            return extracted if isinstance(extracted, list) else []
        return []
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return []


# ---------- LLM-directed internship search ----------
def llm_internship_search_directed(user_text: str, csusb_links_df: pd.DataFrame, max_hops: int, user_prefs: dict = None) -> tuple[str, pd.DataFrame]:


    """
    1. LLM receives CSUSB career page links
    2. LLM decides which to navigate (up to 5), filtered by user preferences
    3. Backend navigates each one using LLM guidance
    4. LLM filters and presents best results
    """
    if csusb_links_df.empty:
        return "Sorry, no company career pages available at the moment.", pd.DataFrame()
    
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.2,
            streaming=False,
            model_kwargs={"num_ctx": 3072, "num_predict": 300}
        )
        
        # Step 1: Get unique companies and build list
        links_list = csusb_links_df[['company', 'link']].drop_duplicates().to_dict('records')
        
        # Filter out None values and build list
        available_companies = []
        for item in links_list:
            company = item.get('company') or 'Unknown'
            link = item.get('link') or ''
            if link and link.startswith('http'):
                available_companies.append(f"{company}: {link}")
        
        if not available_companies:
            return "No valid company links found in the database.", pd.DataFrame()
        
        links_text = "\n".join(available_companies)
        
        st.sidebar.info(f"📋 Available companies: {len(available_companies)}")
        
        # Include user preferences in the prompt
        prefs_text = ""
        if user_prefs:
            interests = user_prefs.get('interests', [])
            roles = user_prefs.get('roles', [])
            location = user_prefs.get('location', 'Any')
            skills = user_prefs.get('skills', [])
            
            # Handle both list and string formats
            interests_str = ', '.join(interests) if isinstance(interests, list) else str(interests)
            roles_str = ', '.join(roles) if isinstance(roles, list) else str(roles)
            skills_str = ', '.join(skills) if isinstance(skills, list) else str(skills)
            
            prefs_text = f"\nUser Preferences:\n- Interests: {interests_str}\n- Roles: {roles_str}\n- Location: {location}\n- Skills: {skills_str}"
        
        sys_step1 = """You are an internship search assistant. You have a list of companies with career page URLs.

Your task: Match the user's query to companies in the available list, prioritizing their stated preferences.

Rules:
1. Return ONLY valid JSON (no markdown, no explanations)
2. Use EXACT company names and URLs from the list provided
3. If a company name in the query doesn't match any in the list, DON'T make it up
4. If no matches found, return empty array
5. Prioritize companies matching user preferences
6. Select up to 5 companies to navigate

Return JSON with two fields: "companies_to_navigate" (array of objects with "company" and "url" fields) and "reasoning" (string).""" + prefs_text

        prompt1 = ChatPromptTemplate.from_messages([
            ("system", sys_step1),
            ("human", "Available companies and URLs:\n{links}\n\nUser query: {query}\n\nFind matching companies. Return ONLY valid JSON as response, nothing else.")
        ])
        
        chain1 = prompt1 | llm
        response1 = chain1.invoke({
            "links": links_text,
            "query": user_text
        })
        
        # Debug output
        st.sidebar.text_area("LLM Response (Debug)", response1.content[:300], height=100)
        
        try:
            raw_response = response1.content.strip()
            
            # Try to parse as direct JSON first
            try:
                parsed = json.loads(raw_response)
                # If it's an array, wrap it in the expected structure
                if isinstance(parsed, list):
                    scrape_decision = {"companies_to_navigate": parsed, "reasoning": "Companies matched"}
                else:
                    scrape_decision = parsed
            except json.JSONDecodeError:
                # Try to find JSON object or array in the response
                json_match = re.search(r'[\{\[][\s\S]*[\}\]]', raw_response)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        scrape_decision = {"companies_to_navigate": parsed, "reasoning": "Companies matched"}
                    else:
                        scrape_decision = parsed
                else:
                    scrape_decision = {"companies_to_navigate": [], "reasoning": "Could not extract JSON from response"}
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"LLM response was: {response1.content[:500]}")
            scrape_decision = {"companies_to_navigate": [], "reasoning": f"Parse error: {str(e)}"}
        
        companies_to_navigate = scrape_decision.get("companies_to_navigate", [])[:5]
        
        if not companies_to_navigate:
            reasoning = scrape_decision.get("reasoning", "No matching companies found")
            available_list = ", ".join([
                item.get('company') or 'Unknown' 
                for item in links_list 
                if item.get('company')
            ][:10])
            return f"I couldn't find matching companies. {reasoning}\n\n**Available companies on CSUSB:**\n{available_list}", pd.DataFrame()
        
        # Step 2: Navigate each company
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        all_navigation_results = []
        
        for idx, company_info in enumerate(companies_to_navigate, 1):
            company_name = company_info.get("company", "Unknown")
            company_url = company_info.get("url", "")
            
            if not company_url or not company_url.startswith("http"):
                continue
            
            with status_placeholder:
                st.info(f"🔍 Navigating {company_name} ({idx}/{len(companies_to_navigate)})...")
            
            progress_bar.progress(int((idx - 1) / len(companies_to_navigate) * 100))
            
            # Call backend to navigate
            nav_result = navigate_career_page(company_url, user_text, max_hops)
            
            # Store result even if not "successful"
            found_links = nav_result.get("found_links", [])
            visited_urls = nav_result.get("visited_urls", [])
            
            # Use the last visited URL as final_url if none provided
            final_url = nav_result.get("final_url") or (visited_urls[-1] if visited_urls else company_url)
            
            all_navigation_results.append({
                "company": company_name,
                "final_url": final_url,
                "visited_urls": visited_urls,
                "found_links": found_links,
                "success": nav_result.get("success", False)
            })
            
            time.sleep(1)
        
        status_placeholder.empty()
        progress_bar.empty()
        
        # Count how many had links
        results_with_links = [r for r in all_navigation_results if r.get("found_links")]
        
        if not results_with_links:
            # Still provide the career page URLs
            career_pages = []
            df_results = []
            for r in all_navigation_results:
                company = r.get("company", "Unknown")
                url = r.get("final_url", "")
                if url:
                    career_pages.append(f"- [{company}]({url})")
                    df_results.append({
                        "title": f"{company} Career Page",
                        "company": company,
                        "url": url,
                        "link": url
                    })
            
            if career_pages:
                pages_text = "\n".join(career_pages)
                result_df = pd.DataFrame(df_results) if df_results else pd.DataFrame()
                return f"I found the career pages but couldn't automatically extract job listings. Here are the pages you can visit:\n\n{pages_text}", result_df
            
            return f"I navigated {len(companies_to_navigate)} companies but couldn't find job listing pages. The companies may have dynamic job sites.", pd.DataFrame()
        
        # Step 3: LLM analyzes and presents results
        results_text = json.dumps(all_navigation_results, indent=2)
        
        sys_step2 = """You are a helpful internship search assistant. You have results from navigating company career pages.

For each company, you have:
- The career page they reached (final_url)
- Intermediate pages visited
- Links found on the final page

Analyze and provide:
1. Brief summary of what was found
2. Highlight the most relevant internship opportunities
3. Provide direct links when available

Be conversational, encouraging, and concise. Focus on actionable next steps."""

        prompt2 = ChatPromptTemplate.from_messages([
            ("system", sys_step2),
            ("human", "Navigation results:\n{results}\n\nUser query: {query}\n\nAnalyze and recommend:")
        ])
        
        chain2 = prompt2 | llm
        response2 = chain2.invoke({
            "results": results_text,
            "query": user_text
        })
        
        result_text = response2.content.strip()
        
        # Add stats
        total_links = sum(len(r.get("found_links", [])) for r in all_navigation_results)
        successful_navs = sum(1 for r in all_navigation_results if r.get("success"))
        
        stats_text = f"\n\n📊 **Navigation Summary:** Explored {len(all_navigation_results)} companies, found {total_links} links across {successful_navs} successful navigations."
        result_text += stats_text
        
        # Convert to dataframe for display
        df_results = []
        for nav_result in all_navigation_results:
            company = nav_result.get("company") or "Unknown"
            found_links = nav_result.get("found_links") or []
            
            for link in found_links[:10]:
                if not isinstance(link, dict):
                    continue
                
                link_url = link.get("url") or ""
                link_text = link.get("text") or ""
                
                if link_url and link_url.startswith('http'):
                    df_results.append({
                        "title": link_text if link_text else "Link",
                        "company": company,
                        "url": link_url,
                        "link": link_url
                    })
        
        result_df = pd.DataFrame(df_results) if df_results else pd.DataFrame()
        
        return result_text, result_df
        
    except Exception as e:
        st.error(f"Search Error: {str(e)}")
        print(f"Error in llm_internship_search_directed: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error during the search. Please try again.", pd.DataFrame()
