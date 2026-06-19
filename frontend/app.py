import streamlit as st
import httpx

# Page Title
st.title("Hybrid RAG Chatbot")

# Backend API URL
API_URL = "http://localhost:8000/chat"

# User Input
query = st.text_input("Ask a question")

# Query Processing
if query:

    # Loading Spinner
    with st.spinner(
        "Retrieving and generating answer..."
    ):

        try:

            # Request Payload
            payload = {"prompt": query}

            # Send Async Request
            response = httpx.post(
                API_URL,
                json=payload,
                timeout=60.0
            )

            # Successful Response
            if response.status_code == 200:
                data = response.json()

                # Display Answer
                st.subheader("Answer")
                st.write(data["answer"])
                st.subheader("Sources")

                for source in data["citations"]:
                    st.write(f"- {source}")

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