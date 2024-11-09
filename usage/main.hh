/*
    src/main.hh
    Q@khaa.pk
 */

#include <iostream>

#include "./../lib/sundry/cooked_read_new.hh"
#include "./../lib/corpus/corpus.hh"
#include "../lib/Numcy/header.hh"
#include "../lib/Skip-gram/lib/WordEmbedding-Algorithms/Word2Vec/Skip-gram/hyper-parameters.hh"

#ifndef READ_TRAINED_SKIP_GRAM_WEIGHTS_TEST_APP_HH
#define READ_TRAINED_SKIP_GRAM_WEIGHTS_TEST_APP_HH

#define DEFAULT_CHAT_BOT_SKIP_GRAM_VOCABULARY_FILE_NAME "INPUT.txt"

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif

#define GRAMMAR_END_OF_TOKEN_MARKER ' '
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#ifdef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#undef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#endif
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 50

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/Numcy/header.hh"
#include "../lib/sundry/cooked_read_new.hh"
#include "../lib/read_write_weights/header.hh"

typedef struct index
{
    cc_tokenizer::string_character_traits<char>::size_type i;

    struct index* next;
    struct index* prev;
} INDEX;

typedef INDEX* INDEX_PTR;

#define COMMAND "h -h help --help ? /? (Displays help screen)\n\
v -v version --version /v (Displays version number)\n\
words --words (This option expects a list of words from the vocabulary)\n\
w1 --w1 (Name of the file which has trained weights. It could be w1 input trained weights, or average of w1 input and w2 output trained weights)\n\
w2 --w2 (Name of the file which has trained weights. It could be w2 output trained weights, or average of w1 input and w2 output trained weights)\n\
vocab --vocab (Name of the file which has the vocabulary)\n"

#endif