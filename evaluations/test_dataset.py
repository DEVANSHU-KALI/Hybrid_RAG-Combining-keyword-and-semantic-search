evaluation_dataset = [
    {
        "question": "What is overfitting?",
        "ground_truth": "Overfitting occurs when a machine learning model memorizes training data instead of learning general patterns, causing poor performance on unseen data."
    },
    {
        "question": "How do Vector Embeddings and Cosine Similarity work together to enable Semantic Search?",
        "ground_truth": "Semantic Search retrieves information based on meaning rather than exact keywords. It relies on Vector Embeddings to convert text into numerical vectors that represent semantic meaning. Once queries and documents are represented as vectors, Cosine Similarity is used to measure how closely their directions align in vector space. Documents with embeddings that are most similar to the query embedding are considered the most relevant results. Therefore, Vector Embeddings provide the representation and Cosine Similarity provides the comparison mechanism that enables Semantic Search."
    },
    {
        "question": "How can Dropout Regularization help reduce Overfitting in Neural Networks?",
        "ground_truth": "Overfitting occurs when a model memorizes training data instead of learning general patterns, causing poor performance on new data. Dropout Regularization helps reduce overfitting by randomly disabling a percentage of neurons during training. This prevents certain neurons from becoming overly dependent on one another and forces the network to learn more robust and generalized features. As a result, neural networks become better at handling unseen data and are less likely to memorize the training dataset."
    }
]