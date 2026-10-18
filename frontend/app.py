import json
import os
import asyncio
import streamlit as st
import httpx

# Configure Streamlit Page
st.set_page_config(
    page_title="Enterprise Hybrid RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint Configurations
API_STREAM_URL = "http://localhost:8000/chat/stream"
API_CHAT_URL = "http://localhost:8000/chat"

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = "tenant_default"
if "user_id" not in st.session_state:
    st.session_state.user_id = "usr_admin"

# CONCEPT: In-Memory Multi-Conversation Session State Manager
# Stores multiple independent chat sessions in RAM without needing a database.
# Each chat has a unique ID, custom title (auto-generated from first prompt), and message history.
if "chats" not in st.session_state:
    st.session_state.chats = {
        "chat_1": {
            "title": "New Chat",
            "messages": []
        }
    }
    st.session_state.current_chat_id = "chat_1"
    st.session_state.chat_counter = 1

# Safety fallback if current_chat_id becomes invalid
if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]


# =====================================================================
# --- SCREEN 1: ENTERPRISE AUTHENTICATION & CREDENTIALS GATE ---
# =====================================================================
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="font-size: 2.4rem; margin-bottom: 5px;">⚡ Enterprise Hybrid RAG</h1>
                <p style="color: #6c757d; font-size: 1.05rem;">
                    768D Dense + Sparse BM25 Search | Multi-Tenant Data Isolation | Real-Time SSE
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        with st.container(border=True):
            st.subheader("🔐 Workspace Authentication")
            st.caption("Enter your tenant organization and user identifier to access scoped document partitions.")

            input_tenant = st.text_input(
                "Tenant ID (Organization Scope)", 
                value=st.session_state.tenant_id,
                help="Physically scopes Qdrant queries to this organization partition."
            )
            input_user = st.text_input(
                "User ID (Security Tag)", 
                value=st.session_state.user_id,
                help="Applies user-level RBAC filtering and security audit logging."
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Enter Enterprise Workspace", use_container_width=True, type="primary"):
                st.session_state.tenant_id = input_tenant.strip() or "tenant_default"
                st.session_state.user_id = input_user.strip() or "usr_admin"
                st.session_state.authenticated = True
                st.rerun()

        st.caption("ℹ️ *Demo defaults: `tenant_default` / `usr_admin`. Data is isolated at the physical vector database layer.*")

    st.stop()


# =====================================================================
# --- SCREEN 2: MAIN ENTERPRISE CHAT & RAG WORKSPACE ---
# =====================================================================

# Get active conversation dictionary
active_chat = st.session_state.chats[st.session_state.current_chat_id]
active_messages = active_chat["messages"]

# --- SIDEBAR: IDENTITY, SESSIONS & DOCUMENT INGESTION ---
with st.sidebar:
    st.markdown("### 👤 Active Identity")
    st.markdown(f"**Tenant:** `{st.session_state.tenant_id}`")
    st.markdown(f"**User:** `{st.session_state.user_id}`")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.chat_counter += 1
            new_id = f"chat_{st.session_state.chat_counter}"
            st.session_state.chats[new_id] = {
                "title": "New Chat",
                "messages": []
            }
            st.session_state.current_chat_id = new_id
            st.rerun()
    with col_btn2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.chats = {
                "chat_1": {"title": "New Chat", "messages": []}
            }
            st.session_state.current_chat_id = "chat_1"
            st.session_state.chat_counter = 1
            st.rerun()

    st.divider()

    # --- CHAT SESSIONS LIST (ChatGPT Style Multi-Chat) ---
    st.markdown("### 💬 Conversations")
    for c_id, c_data in list(st.session_state.chats.items()):
        is_active = (c_id == st.session_state.current_chat_id)
        btn_label = f"👉 {c_data['title']}" if is_active else f"💬 {c_data['title']}"
        if st.button(btn_label, key=f"session_btn_{c_id}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_chat_id = c_id
            st.rerun()

    st.divider()

    # Document Ingestion Section
    st.markdown("### 📄 Document Ingestion")
    st.caption("Upload `.pdf` (with Markdown tables) or `.txt` files to index into Qdrant under your active tenant.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("📥 Index to Qdrant", use_container_width=True, type="primary"):
        target_dir = "data"
        os.makedirs(target_dir, exist_ok=True)
        
        progress = st.progress(0)
        status_box = st.empty()

        for idx, file in enumerate(uploaded_files):
            file_path = os.path.join(target_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            progress.progress(int((idx + 1) / len(uploaded_files) * 60))
            status_box.text(f"Saving {file.name}...")

        status_box.text("⚡ Extracting tables & calculating 768D embeddings...")
        try:
            from backend.injest_documents import ingest_documents
            asyncio.run(ingest_documents(
                target_dir, 
                tenant_id=st.session_state.tenant_id, 
                user_id=st.session_state.user_id
            ))
            progress.progress(100)
            status_box.empty()
            st.success(f"✅ Ingestion complete! Chunks indexed into Qdrant for `{st.session_state.tenant_id}`.")
        except Exception as e:
            status_box.empty()
            st.error(f"Ingestion Error: {e}")

    st.divider()
    st.markdown("### ⚙️ System Status")
    st.caption("• **Vector Dimension:** 768D (`bge-base-en-v1.5`)")
    st.caption("• **Keyword Engine:** BM25Okapi Scoped")
    st.caption("• **Reranker:** `ms-marco-MiniLM-L-6-v2`")
    st.caption("• **Streaming:** Server-Sent Events (SSE)")


# --- MAIN WORKSPACE HEADER ---
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e9ecef; padding-bottom: 10px; margin-bottom: 20px;">
        <div>
            <h2 style="margin: 0;">⚡ Enterprise Hybrid RAG Workspace</h2>
            <span style="color: #6c757d; font-size: 0.95rem;">Scoped Partition: <b>{st.session_state.tenant_id}</b> | User: <b>{st.session_state.user_id}</b></span>
        </div>
        <div>
            <span style="background-color: #e3f2fd; color: #0d47a1; padding: 6px 12px; border-radius: 20px; font-weight: 500; font-size: 0.85rem;">
                🟢 768D Dual Security Pre-Filtering Active
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Render Chat History for Active Session
for msg in active_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If assistant response has metadata/telemetry, render interactive drawers
        if msg["role"] == "assistant":
            citations = msg.get("citations", [])
            contexts = msg.get("contexts", [])
            chunk_ids = msg.get("chunk_ids", [])
            telemetry = msg.get("telemetry", {})

            col_meta1, col_meta2 = st.columns([1, 1])

            with col_meta1:
                if citations or contexts:
                    with st.expander("📚 Sources & Grounded Contexts"):
                        if citations:
                            st.markdown("**Cited Files:**")
                            for c in citations:
                                st.markdown(f"- 📄 `{c}`")
                        if contexts:
                            st.markdown("**Retrieved Chunks:**")
                            for idx, ctx in enumerate(contexts):
                                cid = chunk_ids[idx] if idx < len(chunk_ids) else "N/A"
                                st.markdown(f"*Chunk ID: `{cid}`*")
                                st.info(ctx)

            with col_meta2:
                if telemetry:
                    with st.expander("📊 Latency & FinOps Telemetry"):
                        m1, m2 = st.columns(2)
                        m1.metric("TTFT", f"{telemetry.get('ttft_ms', 0)} ms")
                        m2.metric("Total Time", f"{telemetry.get('total_latency_sec', 0)} s")
                        
                        m3, m4 = st.columns(2)
                        m3.metric("Tokens", f"{telemetry.get('total_tokens', 0)}")
                        m4.metric("Cost (USD)", f"${telemetry.get('estimated_cost_usd', 0.0):.6f}")
                        st.caption(f"Engine: `{telemetry.get('model_used', 'N/A')}`")


# --- CHAT INPUT & REAL-TIME STREAMING ---
user_query = st.chat_input("Ask a question about your enterprise documents...")

if user_query:
    # Auto-generate chat title from the first question if currently "New Chat"
    if active_chat["title"] == "New Chat":
        clean_title = user_query.strip()
        active_chat["title"] = clean_title[:26] + ("..." if len(clean_title) > 26 else "")

    # 1. Append & render user message to active session
    active_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Open assistant chat container for live streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        citations = []
        contexts = []
        chunk_ids = []
        telemetry = {}

        try:
            with httpx.stream(
                "POST",
                API_STREAM_URL,
                json={
                    "prompt": user_query,
                    "tenant_id": st.session_state.tenant_id,
                    "user_id": st.session_state.user_id
                },
                timeout=60.0
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                payload = json.loads(data_str)
                                event_type = payload.get("type")

                                if event_type == "metadata":
                                    citations = payload.get("citations", [])
                                    contexts = payload.get("contexts", [])
                                    chunk_ids = payload.get("chunk_ids", [])
                                elif event_type == "token":
                                    token = payload.get("content", "")
                                    full_response += token
                                    response_placeholder.markdown(full_response + "▌")
                                elif event_type == "telemetry":
                                    telemetry = payload
                            except json.JSONDecodeError:
                                pass
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.error(f"API Error: HTTP {response.status_code}")
                    full_response = f"⚠️ Server Error (HTTP {response.status_code})"

        except httpx.ConnectError:
            # Friendly Standalone / Offline Notice
            response_placeholder.warning(
                "⚠️ **Backend API Server is Offline**\n\n"
                "The FastAPI backend is not running on `http://localhost:8000`.\n\n"
                "**To run end-to-end with live AI generation:**\n"
                "```bash\nuvicorn backend.main:app --reload\n```\n"
                "*You are currently exploring the frontend UI in standalone mode.*"
            )
            full_response = "Backend offline (Preview Mode)"
        except Exception as err:
            response_placeholder.error(f"Connection Exception: {err}")
            full_response = f"Error: {err}"

        # 3. Save assistant message and metadata to active session
        active_messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": citations,
            "contexts": contexts,
            "chunk_ids": chunk_ids,
            "telemetry": telemetry
        })
        st.rerun()



# =====================================================================
# --- PREVIOUS IMPLEMENTATION (Commented for reference) ---
# =====================================================================
# import json
# import os
# import streamlit as st
# import httpx
# 
# st.set_page_config(page_title="Enterprise Hybrid RAG Dashboard", layout="wide")
# 
# st.title("⚡ Enterprise Hybrid RAG Platform")
# st.caption("768D Dense + Sparse Lexical Hybrid Search | Multi-Tenant Data Isolation | Real-Time SSE Token Streaming")
# 
# API_STREAM_URL = "http://localhost:8000/chat/stream"
# API_CHAT_URL = "http://localhost:8000/chat"
# 
# # --- SIDEBAR: MULTI-TENANCY & DOCUMENT INGESTION ---
# with st.sidebar:
#     st.header("🔑 Multi-Tenant Credentials")
#     tenant_id = st.text_input("Tenant Identity Tag", value="tenant_default")
#     user_id = st.text_input("User Identity Tag", value="usr_admin")
#     
#     st.divider()
#     st.header("📄 Drag & Drop Document Ingestion")
#     uploaded_files = st.file_uploader(
#         "Upload PDF or TXT Documents", 
#         type=["pdf", "txt"], 
#         accept_multiple_files=True
#     )
#     
#     if uploaded_files and st.button("Ingest Uploaded Documents"):
#         progress_bar = st.progress(0)
#         status_text = st.empty()
#         
#         target_dir = os.path.join("data", "rag_concepts")
#         os.makedirs(target_dir, exist_ok=True)
#         
#         for idx, file in enumerate(uploaded_files):
#             file_path = os.path.join(target_dir, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
#             progress_bar.progress((idx + 1) / len(uploaded_files))
#             status_text.text(f"Saved {file.name}...")
#             
#         status_text.text("✅ Document files saved. Triggering multi-tenant 768D indexing...")
#         
#         try:
#             from backend.injest_documents import ingest_documents
#             ingest_documents(target_dir, tenant_id=tenant_id, user_id=user_id)
#             st.success("✅ Ingestion complete! Chunks & Markdown tables indexed in Qdrant.")
#         except Exception as e:
#             st.warning(f"Note: Run backend ingestion manually if server running: {e}")
# 
# # --- MAIN CHAT & STREAMING INTERFACE ---
# query = st.text_input("Ask a question (e.g. What is hybrid search, or questions about uploaded PDFs):")
# 
# if query:
#     col1, col2 = st.columns([3, 1])
#     
#     with col1:
#         st.subheader("🤖 AI Response")
#         response_placeholder = st.empty()
#         full_response = ""
#         
#         citations = []
#         contexts = []
#         telemetry = {}
#         
#         try:
#             with httpx.stream(
#                 "POST", 
#                 API_STREAM_URL, 
#                 json={"prompt": query, "tenant_id": tenant_id, "user_id": user_id},
#                 timeout=60.0
#             ) as response:
#                 if response.status_code == 200:
#                     for line in response.iter_lines():
#                         if line.startswith("data: "):
#                             data_str = line[6:]
#                             try:
#                                 payload = json.loads(data_str)
#                                 event_type = payload.get("type")
#                                 
#                                 if event_type == "metadata":
#                                     citations = payload.get("citations", [])
#                                     contexts = payload.get("contexts", [])
#                                 elif event_type == "token":
#                                     token = payload.get("content", "")
#                                     full_response += token
#                                     response_placeholder.markdown(full_response + "▌")
#                                 elif event_type == "telemetry":
#                                     telemetry = payload
#                             except json.JSONDecodeError:
#                                 pass
#                     response_placeholder.markdown(full_response)
#                 else:
#                     st.error(f"Backend API Error: HTTP {response.status_code}")
#         except Exception as err:
#             st.error(f"Connection Failed: {err}")
#             
#     with col2:
#         st.subheader("📊 Execution Metrics")
#         if telemetry:
#             st.metric("Time To First Token (TTFT)", f"{telemetry.get('ttft_ms', 0)} ms")
#             st.metric("Total Latency", f"{telemetry.get('total_latency_sec', 0)} s")
#             st.metric("Total Tokens", f"{telemetry.get('total_tokens', 0)}")
#             st.metric("Estimated Query Cost", f"${telemetry.get('estimated_cost_usd', 0.0):.6f}")
#             st.caption(f"Model: `{telemetry.get('model_used', 'N/A')}`")
#         
#         st.subheader("📚 Verified Citations")
#         if citations:
#             for cite in citations:
#                 st.info(f"📄 {cite}")
#         else:
#             st.write("No citations available.")
# 
#     if contexts:
#         with st.expander("🔍 View Retrieved Context Chunks"):
#             for i, ctx in enumerate(contexts, 1):
#                 st.markdown(f"**Chunk {i}:**")
#                 st.text(ctx)
#                 st.divider()
# =====================================================================