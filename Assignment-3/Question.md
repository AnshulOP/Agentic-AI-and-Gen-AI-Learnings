# 📝 Assignment 3 — AI Agent Workflow (with Nodes & Validation)

---

## 1. Supervisor Node
**Purpose:** Analyze the user query and classify it into one of the three routes:

- **LLM Response**
- **RAG (Retrieval-Augmented Generation)**
- **Web Scraping (Real-time Internet)**

---

## 2. Router Function
**Input:** Output of the Supervisor

**Logic:**

- If classified as **"General Query"** → route to **LLM Node**
- If classified as **"Data Structures"** → route to **RAG Node**
- If classified as **"Web Search"** → route to **Web Scraper Node**

---

## 3. Task Nodes (Based on Classification)

### 3.1 LLM Node
- **Performs:** Pure LLM response using models like Mistral or GPT
- **Input:** The user question
- **Output:** Generated answer using the model

### 3.2 RAG Node
- **Performs:** Context-based answer using RAG pipeline (Retriever + Prompt + LLM)
- **Input:** The original user question
- **Output:** Structured and relevant answer based on internal documents

### 3.3 Web Scraper Node
- **Performs:** Real-time internet search using tools like Tavily or custom web scraper
- **Input:** The user question
- **Output:** Extracted and summarized web content

---

## 4. Validation Node
**Purpose:** Validate the generated answer for:

- Completeness
- Relevance
- Factual accuracy
- Proper structure

**Techniques:** Use tools like:

- Pydantic for structured checks
- Regex for pattern validation
- LLM-based self-verification

**Output:**

- **Pass** → Proceed to Final Output
- **Fail** → Route back to Supervisor for retry

---

## 5. Retry Loop (If Validation Fails)
If Validation Node fails:

- Send the **original query + feedback** back to **Supervisor Node**
- Supervisor reclassifies and re-routes to correct path (LLM/RAG/Web)
- Continue retrying until:
  - **Validation passes**, OR
  - **Retry limit exceeds** (optional stop condition)

---

## 6. Final Output Node
If validation passes:

- Return the final generated answer to the user
- **End the workflow**

---
