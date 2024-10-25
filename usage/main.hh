/*
    src/main.hh
    Q@khaa.pk
 */

#include <iostream>

#ifndef READ_TRAINED_SKIP_GRAM_WEIGHTS_TEST_APP_HH
#define READ_TRAINED_SKIP_GRAM_WEIGHTS_TEST_APP_HH

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif

#define GRAMMAR_END_OF_TOKEN_MARKER ' '
#define GRAMMAR_END_OF_LINE_MARKER '\n'

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

#define COMMAND "h -h help --help ? /? (Displays help screen)\nv -v version --version /v (Displays version number)\nwords --words (This option expects a list of words from the vocabulary)\n"

#endif