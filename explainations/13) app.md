# Script Explanation: `13) app.md`

## 1. Overview
The primary role of the `app.py` script is to implement the **Graphical User Interface (GUI)** for our chatbot using the **Streamlit** framework. It provides a visual chat window that connects to our backend REST API, coordinates user query submissions, displays loading spinner indicators during pipeline computations, handles errors gracefully, and prints the generated responses along with their document source citations in a formatted layout.

---

## 2. Code Walkthrough

### Imports and Configuration
```python
import streamlit as st
import httpx

# Page Title
st.title("Hybrid RAG Chatbot")

# Backend API URL
API_URL = "http://localhost:8000/chat"
```
- **Lines 1–8**:
  - We import `streamlit` (aliased as `st`) to build our UI controls and layouts in Python.
  - We import `httpx`, a modern HTTP client library for Python, to handle REST API communications.
  - We render the page heading using `st.title()`.
  - We set our backend query target address to `"http://localhost:8000/chat"` (where our FastAPI server receives requests).

---

### User Input and Spinner Integration
```python
# User Input
query = st.text_input("Ask a question")

# Query Processing
if query:

    # Loading Spinner
    with st.spinner(
        "Retrieving and generating answer..."
    ):
```
- **Lines 11–19**:
  - We render a text entry box labeled `"Ask a question"` using `st.text_input()`. The user's input string is saved in `query`.
  - `if query:` checks if the string is not empty. When a user writes a question and presses enter, this triggers the conditional block.
  - We wrap our execution block inside `with st.spinner(...)`. This displays an animated loading indicator to show the user that their request is being processed.

---