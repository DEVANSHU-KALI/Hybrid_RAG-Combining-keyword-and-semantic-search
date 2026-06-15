# Script Explanation: `5) reset_qdrant.md`

## 1. Overview
The primary role of the `reset_qdrant.py` script is to wipe the vector database collection named `"rag_docs"`. During development, testing, or database schema changes, it is common to re-ingest documents from scratch. Running this script ensures that old document chunks, obsolete vector coordinates, and metadata are deleted, returning the database to a clean, empty state.

---

## 2. Code Walkthrough

### Imports and Database Connection
```python
from qdrant_client import QdrantClient

# Connect To Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)
```

- **Lines 1–7**: 
  - We import the `QdrantClient` to establish database communication.
  - We initialize the client by directing it to our local host (`localhost`) on port `6333`.

---

### Collection Reset Logic
```python
# Collection Name
COLLECTION_NAME = "rag_docs"

# Delete Collection
try:
    client.delete_collection(
        collection_name=COLLECTION_NAME
    )
    print(
        f"\nDeleted collection: "
        f"{COLLECTION_NAME}"
    )

except Exception as error:
    print(
        f"\nError deleting collection: "
        f"{error}"
    )
```

- **Lines 10–26**:
  - We define the collection name target: `"rag_docs"`.
  - We wrap the delete command inside a **try-except block** (an exception-handling structure used to run risky code and prevent application crashes if errors occur).
  - **The Try Block**: We attempt to run `client.delete_collection()`. If the collection exists and the database is running, Qdrant successfully deletes the collection and all its contents, and we print a confirmation message.
  - **The Except Block**: If an error occurs (such as the Qdrant server being offline, or the collection not existing in the first place), the code jumps directly to this block. We catch the exception as `error` and print the details to the console rather than crashing the script.

---

### Termination Feedback
```python
# Reset Complete
print(
    "\nQdrant reset complete."
)
```
- **Lines 29–31**: Prints a terminal message confirming the reset script has finished execution.

---

## 3. Execution Trace Flow & Step-by-Step Walkthrough

### Flow Diagram
```
                     Start: Run reset_qdrant.py
                                │
                                ▼
                   Connect to local Qdrant Host
                      (localhost:6333)
                                │
                                ▼
                  Begin try-except Block Execution
                                │
                                ▼
                  Call client.delete_collection()
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
        Success Case                           Failure Case
(Collection deleted successfully)       (Server offline or collection missing)
            │                                       │
            ▼                                       ▼
    Print "Deleted collection..."             Print "Error deleting..."
            │                                       │
            └───────────────────┬───────────────────┘
                                ▼
                   Print "Qdrant reset complete."
                                │
                                ▼
                               End
```

---

### Input and Output Specifications
* **Input**: An active connection to a local Qdrant instance.
* **Output**: Deletes the specified collection from Qdrant. Prints deletion success or error feedback messages to the terminal.

---

### Step-by-Step Variable Trace Walkthrough
Let's trace the execution steps under two different scenarios:

#### Scenario A: Collection exists and Qdrant is online
1. **Client Connection**: Connects to `localhost:6333`.
2. **Delete Request**: Enters the `try` block and invokes `client.delete_collection(collection_name="rag_docs")`.
3. **Database Action**: The Qdrant engine drops the index, clears the memory, deletes the underlying files, and returns a success response.
4. **Console Output**: Prints `"\nDeleted collection: rag_docs"`.
5. **Final Step**: Skips the `except` block and prints `"\nQdrant reset complete."`.

#### Scenario B: Qdrant server is offline
1. **Client Connection**: Attempts to initialize connection.
2. **Delete Request**: Enters the `try` block and calls `client.delete_collection(collection_name="rag_docs")`.
3. **Network Failure**: Since the port is unreachable, the client throws a connection error.
4. **Exception Handling**: The program catches the error in the `except` block.
5. **Console Output**: Prints `"\nError deleting collection: [Connection Refused...]"`.
6. **Final Step**: Prints `"\nQdrant reset complete."`.

---