# Script Explanation: `13) app.md`

## 1. Overview
The primary role of `frontend/app.py` is serving as the **Enterprise Multi-Tenant User Interface** for our hybrid RAG system using the Streamlit framework. 

We designed the application with a **Two-Screen Architecture with In-Memory Multi-Conversation Management**:
1. **Screen 1: Enterprise Credentials & Authentication Gate**: Enforces multi-tenant scoping by capturing `tenant_id` and `user_id` before entering the application.
2. **Screen 2: Enterprise ChatGPT-Style Workspace**: Features:
   * **In-Memory Multi-Conversation Manager**: Maintains multiple separate chats in RAM (`chat_1`, `chat_2`, etc.) with dynamic auto-titling from the first user question and one-click sidebar switching.
   * **Vertical Message Feed**: Native `st.chat_message("user")` and `st.chat_message("assistant")`.
   * **Sticky Bottom Input**: Native `st.chat_input`.
   * **Drag-and-Drop Document Indexing**: Indexes PDF tables and text into Qdrant under the active tenant and user.
   * **Real-Time SSE Token Streaming**: Streams tokens with an animated cursor (`▌`).
   * **Verification & FinOps Drawers**: Expandable accordions under each assistant response displaying verified citations, context chunks with `chunk_id` tags, and live metrics (TTFT in ms, latency, token counts, and cost in USD).

---

## 2. System Architecture & UI Flow Diagram

```
                     Browser Opens: http://localhost:8501
                                      │
                                      ▼
                        Check Session Authentication
                    st.session_state.authenticated == True?
                    ├── No  ──► SCREEN 1: Enterprise Login Gate
                    │           [ Tenant ID: tenant_default ]
                    │           [ User ID:   usr_admin      ]
                    │           [ 🚀 Enter Enterprise Workspace ]
                    │                         │ (On Click)
                    │                         ▼
                    │           st.session_state.authenticated = True
                    │           st.rerun()
                    │                         │
                    └── Yes ──► SCREEN 2: Main Workspace
                                      │
       ┌──────────────────────────────┴──────────────────────────────┐
       ▼                                                             ▼
  LEFT SIDEBAR                                                 MAIN CHAT FEED
  • Active Identity: usr_admin @ tenant_default               • Scoped Workspace Header
  • [➕ New Chat] / [🚪 Logout]                              • Active Chat: [Title]
  • 💬 Conversations List:                                    • Vertical Message Feed
    - 👉 What is Machine Learning...                            (Displays active chat's messages)
    - 💬 Explain Deep Learning...                             • Sticky st.chat_input()
  • Drag & Drop File Ingest (.pdf / .txt)                            │
    (Runs asyncio.run(ingest_documents))                             ▼ (User types prompt)
                                                              httpx.stream POST /chat/stream
                                                                     │
                                      ┌──────────────────────────────┼──────────────────────────────┐
                                      ▼                              ▼                              ▼
                               Metadata Event                   Token Event                  Telemetry Event
                           (Citations & Chunk IDs)          (Streaming text ▌)             (TTFT, Cost, Latency)
                                      │                              │                              │
                                      └──────────────────────────────┼──────────────────────────────┘
                                                                     ▼
                                                      Render Assistant Chat Bubble
                                                      + Expanders: [📚 Sources] [📊 FinOps]
```

---

## 3. How Multi-Conversation Switching Works in Memory (Without a Database)

### The In-Memory Session State Schema
Even without an external PostgreSQL or SQLite database, Streamlit's `st.session_state` maintains isolated data structures in RAM for the lifetime of the active browser session:

```python
st.session_state.chats = {
    "chat_1": {
        "title": "What is Machine Learning...",
        "messages": [
            {"role": "user", "content": "What is Machine Learning?"},
            {"role": "assistant", "content": "Backend offline (Preview Mode)"}
        ]
    },
    "chat_2": {
        "title": "Explain Deep Learning...",
        "messages": [
            {"role": "user", "content": "Explain Deep Learning and Neural Nets"},
            {"role": "assistant", "content": "Backend offline (Preview Mode)"}
        ]
    }
}
st.session_state.current_chat_id = "chat_2"
```

### What Happens When You Switch Between Chats?
1. **When in Chat 2 (DL):** The UI reads `st.session_state.chats["chat_2"]["messages"]` and renders your Deep Learning question and its response.
2. **Clicking Chat 1 (ML) in Sidebar:**  
   The button sets `st.session_state.current_chat_id = "chat_1"` and calls `st.rerun()`.
3. **Instant Re-Render:**  
   Because `current_chat_id` is now `"chat_1"`, the UI reads `st.session_state.chats["chat_1"]["messages"]`.  
   **You see your exact Machine Learning question and its response, exactly as you left it!**
4. **Zero Data Loss Across Switches:**  
   Neither chat overwrites the other. Each conversation keeps its own private message list in memory.
5. **Auto-Titling:**  
   When a new chat starts, its title is `"New Chat"`. The moment you type your first question, the script extracts the first 26 characters and dynamically renames the chat title in the sidebar!

---

## 4. Deep Line-by-Line Code Walkthrough

---

### Part A: Page Configuration & Multi-Chat Session State Setup (Lines 1–40)

```python
1: import json
2: import os
3: import asyncio
4: import streamlit as st
5: import httpx
```
* Standard imports including `asyncio` for non-blocking document ingestion and `httpx` for SSE HTTP streaming.

```python
18: if "authenticated" not in st.session_state:
19:     st.session_state.authenticated = False
20: if "tenant_id" not in st.session_state:
21:     st.session_state.tenant_id = "tenant_default"
22: if "user_id" not in st.session_state:
23:     st.session_state.user_id = "usr_admin"
```
* Tracks login status and multi-tenant credentials.

```python
28: if "chats" not in st.session_state:
29:     st.session_state.chats = {
30:         "chat_1": {
31:             "title": "New Chat",
32:             "messages": []
33:         }
34:     }
35:     st.session_state.current_chat_id = "chat_1"
36:     st.session_state.chat_counter = 1
```
* **Lines 28–36 (Multi-Chat In-Memory Manager)**:
  * Initializes the `chats` dictionary mapping unique chat IDs (`chat_1`, `chat_2`) to their title and message list.
  * Tracks `current_chat_id` and a numeric `chat_counter`.

---

### Part B: Screen 1 — Authentication Gate (Lines 44–80)

```python
44: if not st.session_state.authenticated:
...
68:         if st.button("🚀 Enter Enterprise Workspace", use_container_width=True, type="primary"):
69:             st.session_state.tenant_id = input_tenant.strip() or "tenant_default"
70:             st.session_state.user_id = input_user.strip() or "usr_admin"
71:             st.session_state.authenticated = True
72:             st.rerun()
76:     st.stop()
```
* Renders a centered enterprise card.
* Captures `tenant_id` and `user_id`, sets `authenticated = True`, and reruns.
* `st.stop()` prevents unauthorized rendering of the main chat workspace.

---

### Part C: Screen 2 — Sidebar Controls & Multi-Chat List (Lines 86–175)

```python
86: active_chat = st.session_state.chats[st.session_state.current_chat_id]
87: active_messages = active_chat["messages"]
```
* Grabs the active conversation object and message list based on `current_chat_id`.

```python
96:     if st.button("➕ New Chat", use_container_width=True):
97:         st.session_state.chat_counter += 1
98:         new_id = f"chat_{st.session_state.chat_counter}"
99:         st.session_state.chats[new_id] = {
100:             "title": "New Chat",
101:             "messages": []
102:         }
103:         st.session_state.current_chat_id = new_id
104:         st.rerun()
```
* **Lines 96–104 (Creating a New Chat)**:
  * Increments counter (`chat_2`, `chat_3`).
  * Creates a new chat entry with empty messages.
  * Switches `current_chat_id` to the new ID and reruns to present a fresh chat window.

```python
116:     # --- CHAT SESSIONS LIST (ChatGPT Style Multi-Chat) ---
117:     st.markdown("### 💬 Conversations")
118:     for c_id, c_data in list(st.session_state.chats.items()):
119:         is_active = (c_id == st.session_state.current_chat_id)
120:         btn_label = f"👉 {c_data['title']}" if is_active else f"💬 {c_data['title']}"
121:         if st.button(btn_label, key=f"session_btn_{c_id}", use_container_width=True, type="primary" if is_active else "secondary"):
122:             st.session_state.current_chat_id = c_id
123:             st.rerun()
```
* **Lines 116–123 (Sidebar Chat Switcher)**:
  * Iterates over all saved chats.
  * Displays active chat with a primary highlight (`👉 [Title]`).
  * Clicking any past chat switches `current_chat_id` and re-renders that conversation instantly.

```python
140:     if uploaded_files and st.button("📥 Index to Qdrant", use_container_width=True, type="primary"):
...
154:         from backend.injest_documents import ingest_documents
155:         asyncio.run(ingest_documents(
156:             target_dir, 
157:             tenant_id=st.session_state.tenant_id, 
158:             user_id=st.session_state.user_id
159:         ))
```
* Runs async ingestion with progress feedback, tagging chunks with active tenant and user.

---

### Part D: Screen 2 — Chat Feed & Auto-Titling (Lines 185–290)

```python
216: user_query = st.chat_input("Ask a question about your enterprise documents...")
218: if user_query:
219:     if active_chat["title"] == "New Chat":
220:         clean_title = user_query.strip()
221:         active_chat["title"] = clean_title[:26] + ("..." if len(clean_title) > 26 else "")
```
* **Auto-Titling Logic:** If the active chat title is still `"New Chat"`, automatically sets its title to the first 26 characters of the user's question.

```python
223:     active_messages.append({"role": "user", "content": user_query})
224:     with st.chat_message("user"):
225:         st.markdown(user_query)
...
274:         except httpx.ConnectError:
275:             response_placeholder.warning(
276:                 "⚠️ **Backend API Server is Offline**\n\n"
277:                 "The FastAPI backend is not running on `http://localhost:8000`.\n\n"
278:                 "**To run end-to-end with live AI generation:**\n"
279:                 "```bash\nuvicorn backend.main:app --reload\n```\n"
280:                 "*You are currently exploring the frontend UI in standalone mode.*"
281:             )
282:             full_response = "Backend offline (Preview Mode)"
...
286:         active_messages.append({
287:             "role": "assistant",
288:             "content": full_response,
289:             "citations": citations,
290:             "contexts": contexts,
291:             "chunk_ids": chunk_ids,
292:             "telemetry": telemetry
293:         })
294:         st.rerun()
```
* Appends both user question and streamed assistant response to `active_messages`.
* Handles offline status gracefully without crashing, enabling standalone frontend exploration.

---

### Part E: Interactive Citations Drawer & FinOps Telemetry Metrics (Lines 220–246)

```python
236:                     with st.expander("📊 Latency & FinOps Telemetry"):
237:                         m1, m2 = st.columns(2)
238:                         m1.metric("TTFT", f"{telemetry.get('ttft_ms', 0)} ms")
239:                         m2.metric("Total Time", f"{telemetry.get('total_latency_sec', 0)} s")
240:                         
241:                         m3, m4 = st.columns(2)
242:                         total_tok = telemetry.get("total_tokens", 0)
243:                         q_tok = telemetry.get("prompt_tokens", 0)
244:                         r_tok = telemetry.get("completion_tokens", 0)
245:                         m3.metric(
246:                             "Tokens", 
247:                             f"{total_tok}", 
248:                             delta=f"Q: {q_tok} | R: {r_tok}", 
249:                             delta_color="off"
250:                         )
251:                         m4.metric("Cost (USD)", f"${telemetry.get('estimated_cost_usd', 0.0):.6f}")
```
* **TTFT (`ttft_ms`)**: Measures Time-To-First-Token in milliseconds (the universal industry standard for measuring perceived latency).
* **Token Breakdown (`delta=f"Q: {q_tok} | R: {r_tok}"`)**:
  * Displays the total token consumption as the primary metric.
  * In the sub-badge, explicitly breaks down:
    * **`Q` (Query/Prompt Tokens)**: System instructions + 3 retrieved document chunks + user question. In RAG systems, prompt tokens represent $85\%-95\%$ of total token usage!
    * **`R` (Response/Completion Tokens)**: Tokens generated by the LLM in its response.
  * `delta_color="off"` renders the sub-badge in clean neutral gray without altering column widths.

---

## 5. Summary of Multi-Chat Behavior

| User Action | What Happens in Background State | What You See on Screen |
| :--- | :--- | :--- |
| **Types 1st question in Chat 1** (*"What is ML?"*) | `chat_1["title"]` becomes `"What is ML..."`. Appends prompt + answer to `chat_1["messages"]`. | Sidebar updates to `👉 What is ML...`. Chat bubbles appear. |
| **Clicks "➕ New Chat"** | Increments counter. Creates `chat_2` with `messages = []`. Sets `current_chat_id = "chat_2"`. | Sidebar shows both chats. Chat area clears to a fresh window. |
| **Types question in Chat 2** (*"Explain DL..."*) | `chat_2["title"]` becomes `"Explain DL..."`. Appends prompt + answer to `chat_2["messages"]`. | Sidebar updates to `👉 Explain DL...`. Chat bubbles appear. |
| **Clicks "💬 What is ML..."** | Sets `current_chat_id = "chat_1"`. | **Instantly displays your exact ML question and answer!** |
| **Clicks "💬 Explain DL..."** | Sets `current_chat_id = "chat_2"`. | **Instantly displays your exact DL question and answer!** |
