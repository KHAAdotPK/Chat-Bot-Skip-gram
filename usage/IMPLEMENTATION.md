## Overview

The files `main.hh` and `main.cpp` contains the source code of a command-line application that loads pretrained word embeddings, processes a list of words from command-line arguments, and calculates the cosine similarity between the embeddings of these words.

### Code Breakdown

1. **Load Pretrained Embeddings**: 
   - The embeddings are loaded from two files (`w1trained.txt` and `w2trained.txt`) using a custom parser.
   - The weights are stored in two `Collective<double>` objects, `W1` and `W2`.

2. **Command-line Parsing**:
   - The program parses the command-line arguments, looking for a list of words provided with the `--words` option.
   - If no words are given, the program outputs a help message and exits.

3. **Finding Word Indices**:
   - For each word provided in the arguments, the program searches the loaded embeddings for the corresponding row (i.e., the word’s vector).
   - A linked list (`INDEX_PTR`) is used to store the positions of the found words in the embedding matrix.

4. **Cosine Similarity Calculation**:
   - For each pair of word vectors, it computes the cosine similarity using the `Numcy::Spatial::Distance::cosine` function.
   - The similarity between two words is printed to the console.

5. **Memory Management**:
   - After processing, the program deallocates the memory used for the linked list of indices.

#### Key Points about `W1trained.txt` and `w2trained.txt` fles:
The files contain word embeddings in a numerical format. Each line consists of a word from the vocabulary followed by a series of floating-point numbers, which represent the vector of that word in a high-dimensional space. The vectors have **100 dimensions** (the number of floating-point numbers following each word). The dimensionality directly affects the performance and accuracy of similarity computations: **higher dimensionality can capture more nuanced relationships between words, but too high a dimensionality can also increase computational cost and risk overfitting**. In general, **similar words** are represented by **vectors** that are **close to each other** in this **high-dimensional** space, meaning the **values** of each floating-point number in their vectors are **not too far apart** from each other.
