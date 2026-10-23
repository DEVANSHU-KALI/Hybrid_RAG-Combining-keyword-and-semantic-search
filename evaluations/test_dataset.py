# Gold-Standard Benchmark Test Dataset for Multi-Tenant Hybrid RAG Evaluation
# Contains target domain questions directly mapped to the indexed technical corpus
# (machine_learning.pdf, deep_learning_test_corpus.pdf, and ai_concepts.txt).

evaluation_dataset = [
    {
        "question": "How do Hybrid Retrieval and Reciprocal Rank Fusion (RRF) work together to combine BM25 keyword search with dense vector search?",
        "ground_truth": (
            "Hybrid retrieval runs both keyword search (BM25) and semantic vector search for the same query. "
            "Because BM25 scores and cosine similarity scores have different numerical ranges and calculations, "
            "they cannot be directly added. Reciprocal Rank Fusion (RRF) resolves this by combining ranks instead of raw scores. "
            "It assigns each document a score based on its position in each result list, preventing one retriever from dominating "
            "and ensuring both exact keyword identifiers and semantic paraphrases are captured."
        )
    },
    {
        "question": "What is the difference between a Bi-Encoder and a Cross-Encoder Reranker, and why are Cross-Encoders not used for initial search over the entire corpus?",
        "ground_truth": (
            "A bi-encoder embeds queries and documents separately into vector space, allowing fast approximate nearest neighbor search over large collections. "
            "In contrast, a cross-encoder processes the query and candidate passage together through full transformer multi-head cross-attention, "
            "directly modeling word-level interactions to produce an accurate relevance score. "
            "Cross-encoders are computationally expensive because every query-passage pair requires a full transformer forward pass, "
            "making them impractical for searching millions of passages. Instead, bi-encoders or BM25 retrieve a candidate set, "
            "and the cross-encoder reranks the top candidates."
        )
    },
    {
        "question": "What causes hallucinations in Large Language Models, and how does grounded generation in RAG mitigate them?",
        "ground_truth": (
            "Hallucinations occur because generative language models predict statistically plausible next tokens rather than fact-checking claims against verified truth. "
            "In RAG pipelines, hallucinations can arise from missing evidence, poor retrieval, ambiguous passages, or the model extrapolating beyond the prompt. "
            "Grounded generation mitigates this by instructing the model to rely strictly on retrieved context passages, display source citations, "
            "and explicitly state when the requested information is absent from the provided documents."
        )
    },
    {
        "question": "How do forward propagation, loss functions, backpropagation, and optimizers work together to train a neural network?",
        "ground_truth": (
            "During the forward pass, input values are transformed layer by layer using trainable weights and biases to produce a prediction. "
            "A loss function measures the numerical discrepancy between the model's prediction and the true target. "
            "Backpropagation calculates gradients using the chain rule to determine how parameter changes affect the loss. "
            "Finally, an optimizer uses those gradients to iteratively update the network's weights and biases across batches of training examples."
        )
    },
    {
        "question": "What are the trade-offs between very large chunks and very small chunks in document chunking, and why do tables require special care?",
        "ground_truth": (
            "Very large chunks can contain multiple unrelated topics, which degrades retrieval precision and fills up the language model's context window with irrelevant text. "
            "Very small chunks risk removing necessary background context or separating a definition from its explanation. "
            "Tables require special handling because naive text splitters can divide text between rows or columns, destroying the structural relationships "
            "between header labels and cell values."
        )
    },
    {
        "question": "What are the common dataset split strategies in machine learning, and why should the test set not be used during development?",
        "ground_truth": (
            "A common workflow divides data into training, validation, and test sets. "
            "The training set fits model parameters; the validation set tunes hyperparameters and guides architecture decisions; "
            "and the test set provides an unbiased final estimate of generalization performance. "
            "If the test set is repeatedly used to make design or hyperparameter choices, it effectively becomes part of the development process "
            "and no longer provides an independent evaluation on unseen data."
        )
    },
    {
        "question": "How do Dropout Regularization and Early Stopping help prevent overfitting in machine learning models?",
        "ground_truth": (
            "Overfitting occurs when a model memorizes specific training examples and noise instead of learning generalizable underlying patterns, leading to poor performance on unseen data. "
            "Dropout regularization mitigates this in neural networks by randomly deactivating a subset of neurons during training batches, preventing co-adaptation and forcing more robust representations. "
            "Early stopping monitors validation loss during training and halts optimization before the network begins memorizing the training set."
        )
    }
]
