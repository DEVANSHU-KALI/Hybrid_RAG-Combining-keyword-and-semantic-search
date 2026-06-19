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

### API Request Submission
```python
        try:
            # Request Payload
            payload = {"prompt": query}

            # Send Async Request
            response = httpx.post(
                API_URL,
                json=payload,
                timeout=60.0
            )
```
- **Lines 21–31**:
  - We start a **try-except** block to catch network and server faults.
  - We package our prompt into a JSON payload dictionary: `{"prompt": query}`.
  - We send an HTTP POST request using `httpx.post()`. We pass `json=payload` to serialize the dict to a JSON body.
  - `timeout=60.0`: We set a 60-second limit. If the local language model is slow and doesn't respond within 60 seconds, the client aborts the request rather than hanging indefinitely.

---

### Response Rendering
```python
            # Successful Response
            if response.status_code == 200:
                data = response.json()

                # Display Answer
                st.subheader("Answer")
                st.write(data["answer"])
                st.subheader("Sources")

                for source in data["citations"]:
                    st.write(f"- {source}")
```
- **Lines 34–43**:
  - We verify if the server returned successfully: `if response.status_code == 200`.
  - We parse the JSON response body using `response.json()`.
  - We render the heading `"Answer"` and print the generated text response using `st.write(data["answer"])`.
  - We print the heading `"Sources"` and loop through `data["citations"]` to list the referenced filenames as bullet points on the screen.

---

### Error Handling
```python
            # API Error
            else:
                st.error(
                    f"Backend Error: "
                    f"{response.status_code}"
                )

        # Connection Error
        except Exception as error:
            st.error(
                f"Connection Failed: "
                f"{error}"
            )
```
- **Lines 46–57**:
  - The `else` statement handles cases where the backend server runs but encounters an internal error (e.g., returns `HTTP 500`). We display a red error notification card showing the status code.
  - The `except Exception` block intercepts connection errors (such as if the FastAPI server is completely offline). We capture the error details and display a red error message to guide the user.

---