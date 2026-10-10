# Script Explanation: `7) semantic_retriever.md`

## 1. Overview
The primary role of `semantic_retriever.py` is executing dense vector similarity search with **Dual Security Pre-Filtering (Tenant ID + User ID)** against Qdrant. It converts user queries into 768D vectors using `BAAI/bge-base-en-v1.5`, applies database-level payload filters for both `tenant_id` and `user_id`, and returns high-confidence semantic chunks.

> [!IMPORTANT]
> **Data Leakage Prevention (Pre-Filtering):** Filtering documents in Python *after* retrieval (post-filtering) is a severe security flaw because unauthorized chunks waste retrieval slots or leak metadata. By injecting `query_filter=Filter(must=[tenant_id, user_id])` into `client.query_points()`, Qdrant's vector search engine ignores unauthorized tenant OR user data at the physical database index layer.

---

## 2. Implemented Concepts & Code Walkthrough

### CONCEPT: Dual Security Pre-Filtering (Tenant + User Level Isolation)
```python
security_filter = Filter(
    must=[
        FieldCondition(
            key="tenant_id",
            match=MatchValue(value=tenant_id)
        ),
        FieldCondition(
            key="user_id",
            match=MatchValue(value=user_id)
        )
    ]
)

results = await client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=security_filter,
    limit=limit
)
```
- Restricts vector similarity calculation strictly to points where `payload.tenant_id == active_tenant_id AND payload.user_id == active_user_id`.
- Prevents both cross-tenant organizational data leakage and cross-user RBAC data leakage.

---

## 3. Code History: Before vs. After

### Before (Tenant-Only Filter)
```python
# Filtered on tenant_id only (Risk of user-level data cross-retrieval)
tenant_filter = Filter(
    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
)
```

### After (Dual Tenant + User Security Pre-Filter)
```python
# Enforces both tenant_id AND user_id isolation
security_filter = Filter(
    must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]
)

results = await client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=security_filter,
    limit=limit
)
```

---

## 4. Execution Trace Flow

```
             User Query + Tenant Token + User Identity
                                │
                                ▼
               Generate 768D Dense Query Embedding
                                │
                                ▼
                 Build Qdrant Payload Pre-Filter
          (tenant_id == "acme_corp" AND user_id == "usr_123")
                                │
                                ▼
                 Async Qdrant HNSW Vector Search
            (Evaluates ONLY matching tenant/user points)
                                │
                                ▼
            Return Top 10 Isolated Semantic Chunks
```
