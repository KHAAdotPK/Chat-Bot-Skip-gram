### Development of a Chatbot Using Skip-Gram Trained Word Embeddings: A C++ Implementation
---
```C++
/*
    Extract the corresponding word embedding from the weight matrix 𝑊1.
    Instead of directly using a one-hot input vector, this implementation uses a linked list of word pairs.
    Each pair provides the index of the center word, which serves to extract the relevant embedding from 𝑊1.
    The embedding for the center word is stored in the hidden layer vector h.
 */
T* h_ptr = cc_tokenizer::allocator<T>().allocate(W1.getShape().getNumberOfColumns()); // Allocate memory for the hidden layer vector h.
// Loop through the columns of W1 to extract the embedding for the center word
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
{
    *(h_ptr + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];

    // Check if the value retrieved for h contains an invalid number (NaN), and throw an exception if true.
    if (_isnanf(h_ptr[i]))
    {        
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: Hidden layer at ") + cc_tokenizer::String<char>("(W1 row) center word index ") +  cc_tokenizer::String<char>(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + cc_tokenizer::String<char>(" and (column index) i -> ") + cc_tokenizer::String<char>(i) + cc_tokenizer::String<char>(" -> [ ") + cc_tokenizer::String<char>("_isnanf() was true") + cc_tokenizer::String<char>("\" ]"));
    } 
}

// Create a collective object h to store the hidden layer vector with dimensions (embedding size, 1).        
Collective<T> h = Collective<T>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};
/*	
    Represents an intermediat gradient.	 
    This vector has shape (1, len(vocab)), similar to y_pred. 
    It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
    The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
    numerical representation of how likely each word in the vocabulary is to be a context word of a given target word (within the skip-gram model).

    The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying the “softmax” function to generate the predicted probabilities. It represents internal state in the neural network during the working of "forward pass". This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         
    The dot product gives us the logits or unnormalized probabilities (u), which can then be transformed into probabilities using a softmax function
 */            
Collective<T> u = Numcy::dot(h, W2);

/*
    The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
    The softmax function converts the raw scores into probabilities.

    In `Skip-gram`, this output represents the likelihood of each word being one of the context words for the given center word.
 */
Collective<T> y_pred = softmax(u);

/* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
Collective<T> oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});

/*
    The following code block, iterates through the context word indices (left and right) from the pair object.
    For each valid context word index (i), it sets the corresponding element in the oneHot vector to 1.
    This effectively creates a one-hot encoded representation of the context words.
 */
for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
{       
    if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
    {
        oneHot[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
    }
}
for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
{
    if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
    {
        oneHot[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE] = 1;
    }        
}

/* The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) */
Collective<T> grad_u;
/* 
    The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency).
    Compute the gradient for the center word's embedding by subtracting the one-hot vector of the actual context word from the predicted probability distribution.
    `fp.predicted_probabilities` contains the predicted probabilities over all words in the vocabulary.
    `oneHot` is a one-hot vector representing the true context word in the vocabulary.
    The result, `grad_u`, is the error signal for updating the center word's embedding in the Skip-gram model.
    
    what is an error signal?
    -------------------------
    1. For the correct context word (where oneHot is 1), the gradient is (predicted_probabilities - 1), meaning the model's prediction was off by that much.
    2. For all other words (where oneHot is 0), the gradient is simply predicted_probabilities, meaning the model incorrectly assigned a nonzero probability to these words(meaning the model's prediction was off by that much, which the whole of predicted_probability for that out of context word).        
 */        
grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);

/*
    Dimensions of W2 is (SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, len(vocab) without redundency)
    Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)        
 */
Collective<T> W2_T;
W2_T = Numcy::transpose(W2);

/*
    Dimensions of grad_u is (1, len(vocab) without redundency)
    Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)

    Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
 */
Collective<T> grad_h;
grad_h = Numcy::dot(grad_u, W2_T);

/*
    Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
 */
Collective<T> grad_W1;
grad_W1 = Numcy::zeros(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});

/*
    Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
    Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
    
    Update the gradient for W1 (the center word's embedding) by adding the gradient 
    of the hidden layer (grad_h) to the corresponding row in grad_W1.
    This effectively adjusts the embedding for the center word based on the error signal.
 */
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getNumberOfColumns(); i++)
{
    grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] += grad_h[i];
    /*
        The above line should instead be... 
        grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] = grad_h[i];

        The old line adds the value of grad_h[i] to the existing value in grad_W1. This implies that you are accumulating the gradients, i.e.,
        you are adding to the current gradient values rather than replacing them.
        This makes sense if you are accumulating gradients over multiple training examples or word pairs in a batch. 
        This technique is common in mini-batch gradient descent where the gradient is averaged over multiple samples before updating the weights.

        Alternative Line (grad_W1[(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + i] = grad_h[i];):
        This would replace the existing value in grad_W1 with grad_h[i]. This is the correct approach if you want to set the gradient directly without accumulating it. 
        If you are processing one word pair at a time and updating the gradients immediately, this would be the appropriate line.

        Conclusion:
        - For batch processing: Use += to accumulate the gradient contributions across multiple word pairs.
        - For single word pair updates: Use = to directly set the gradient for the current word pair.
     */
} 

W1 -= bp.grad_weights_input_to_hidden * lr;
```
