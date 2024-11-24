/*
    src/main.hh
    Q@khaa.pk
 */

#include <iostream>

#ifndef READ_TRAINED_SKIP_GRAM_WEIGHTS_TEST_APP_HH
#define READ_TRAINED_SKIP_GRAM_WEIGHTS_TEST_APP_HH

#define DEFAULT_CHAT_BOT_SKIP_GRAM_VOCABULARY_FILE_NAME "INPUT.txt"

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#define GRAMMAR_END_OF_TOKEN_MARKER ' '

#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#ifdef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#undef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 
#endif
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 64

#ifdef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#undef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#endif
#define SKIP_GRAM_CONTEXT_WINDOW_SIZE 4

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/read_write_weights/header.hh"
#include "./../lib/sundry/cooked_read_new.hh"
#include "./../lib/corpus/corpus.hh"
#include "../lib/Numcy/header.hh"
#include "../lib/pairs/src/header.hh"
#include "../lib/Skip-gram/lib/WordEmbedding-Algorithms/Word2Vec/Skip-gram/hyper-parameters.hh"

typedef struct index
{
    cc_tokenizer::string_character_traits<char>::size_type i;

    struct index* next;
    struct index* prev;
} INDEX;

typedef INDEX* INDEX_PTR;

#define COMMAND "h -h help --help ? /? (Displays the help screen)\n\
v -v version --version /v (Displays the version number)\n\
words --words (Expects a list of words from the vocabulary)\n\
w1 --w1 (Specifies the file containing trained input weights)\n\
w2 --w2 (Specifies the file containing the vocabulary)\n\
vocab --vocab (Name of the file which has the vocabulary)\n\
average (Acts as a flag to be used with the [w1 | --w1] command-line option; when used, the specified file is an average of w1 and w2 trained weights)\n\
show --show show_pairs showPairs (Displays pairs of target/center words and their surrounding context words. The number of context words for each target word is determined by the macro SKIP_GRAM_EMBEDDING_VECTOR_SIZE, which is a configurable hyperparameter)\n"

#define COMMAND_average "do (Used with the \"average\" command; optionally expects a numeric argument. This command implies that the \"W1\" and \"W2\" matrices will be averaged, and the program will proceed with processing the resulting matrix. If an optional numeric argument is provided, it acts as a multiplier, with the \"W2\" matrix as the multiplicand before averaging)\n"

#endif