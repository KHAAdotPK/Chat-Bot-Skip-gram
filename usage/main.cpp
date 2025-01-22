/*
    usage/main.cpp
    Q@khaa.pk
 */

/*
    This code is designed to:
    1. Read pretrained word embeddings from files.
    2. Take a list of words from the command line.
    3. Find the corresponding word vectors.
    4. Calculate cosine similarities between the vectors of the words provided.
    5. Clean up memory after processing.
 */

#include "main.hh"

int main(int argc, char* argv[])
{
    INDEX_PTR head = NULL, ptr = NULL;
    ARG arg_common, arg_words, arg_w1, arg_w2, arg_help, arg_vocab, arg_average, arg_pairs, arg_proper;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser_average(cc_tokenizer::String<char>(COMMAND_average));
    cc_tokenizer::String<char> vocab_file_name;
    
    if (argc < 2)
    {              
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help);

        return 0;                     
    }
    
    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "proper", arg_proper);
               
    FIND_ARG(argv, argc, argsv_parser, "--vocab", arg_vocab);
    if (arg_vocab.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_vocab);

        if (arg_vocab.argc)
        {            
            vocab_file_name = cc_tokenizer::String<char>(argv[arg_vocab.i + 1]);
        }
        else
        {
            ARG arg_vocab_help;
            HELP(argsv_parser, arg_vocab_help, "vocab");                
            HELP_DUMP(argsv_parser, arg_vocab_help); 

            return 0;
        }
    }
    else
    {
        vocab_file_name = cc_tokenizer::String<char>(DEFAULT_CHAT_BOT_SKIP_GRAM_VOCABULARY_FILE_NAME); 
    }
    
    GET_FIRST_ARG_INDEX(argv, argc, argsv_parser,  arg_common);            
    FIND_ARG(argv, argc, argsv_parser, "--words", arg_words);
    if (!(arg_words.i))
    {   
        if (!(arg_common.argc))
        {        
            std::cout<< "Words are not given, can't go further. Please use \"help\" command line option." << std::endl;

            return 0;
        }
    }
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_words);
    if (!arg_words.argc) 
    {
        if (arg_common.argc)
        {                         
            arg_words = arg_common;
            //arg_words.i -= 1;
        }
        else
        {
            ARG arg_words_help;
            HELP(argsv_parser, arg_words_help, "words");                
            HELP_DUMP(argsv_parser, arg_words_help); 

            return 0;
        }       
    }

    FIND_ARG(argv, argc, argsv_parser, "w1", arg_w1);
    if (arg_w1.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_w1);

        if (!arg_w1.argc)
        {
            ARG arg_w1_help;
            HELP(argsv_parser, arg_w1_help, "--w1");                
            HELP_DUMP(argsv_parser, arg_w1_help); 

            return 0;
        }
    }
    else if (arg_proper.i)
    {        
        std::cerr<< "Error: The command-line option \"--proper\" requires a file name specifying the trained input weights. Please provide the appropriate file name and try again." << std::endl;

        ARG arg_w1_help;
        HELP(argsv_parser, arg_w1_help, "--w1");                
        HELP_DUMP(argsv_parser, arg_w1_help); 

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "--w2", arg_w2);
    if (arg_w2.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_w2);

        if (!arg_w2.argc)
        {
            ARG arg_w2_help;
            HELP(argsv_parser, arg_w2_help, "w2");                
            HELP_DUMP(argsv_parser, arg_w2_help); 

            return 0;
        }
    }
    else
    {
        ARG arg_w2_help;
        HELP(argsv_parser, arg_w2_help, "--w2");                
        HELP_DUMP(argsv_parser, arg_w2_help); 

        return 0;    
    }
        
    /*std::cout<< "--->>>> " << arg_words.argc << " - " << arg_words.i << " < - > " << arg_words.j << std::endl;

    for (int i = 1; i <= arg_words.argc; i++)
    {
        std::cout<< argv[arg_words.i + i] << ", ";
    }

    std::cout<< std::endl;
     
    return 0;*/

    cc_tokenizer::String<char> vocab_text = cc_tokenizer::cooked_read<char>(vocab_file_name);
    CORPUS vocab(vocab_text); 

    FIND_ARG(argv, argc, argsv_parser, "showPairs", arg_pairs);
    if (arg_pairs.i)
    {
        PAIRS grow_pairs_dude_ask_her_if_she_want_to_marry_you(vocab, true);
    }

    Collective<double> W1 /* = Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens() vocab.numberOfUniqueTokens(), NULL, NULL}}*/;
    Collective<double> W2 = Collective<double>{NULL, DIMENSIONS{/*vocab.numberOfUniqueTokens()*/ vocab.numberOfTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};

    if (arg_w1.argc)
    {
        W1 = Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, /*vocab.numberOfUniqueTokens()*/ vocab.numberOfTokens(), NULL, NULL}};
        READ_W_BIN(W1, argv[arg_w1.i + 1], double);

        std::cout<< "W1: " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " <<  W1.getShape().getNumberOfColumns() << std::endl;
    }
    
    READ_W_BIN(W2, argv[arg_w2.i + 1], double);
    
    std::cout<< "W2: " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " <<  W2.getShape().getNumberOfColumns() << std::endl;
    Collective<double> W2_transposed = Numcy::transpose(W2); 
    //W2 = Numcy::transpose(W2);    
    std::cout<< "W2 transposed: " << W2_transposed.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " <<  W2_transposed.getShape().getNumberOfColumns() << std::endl;

    if (!arg_proper.i)
    {
        FIND_ARG(argv, argc, argsv_parser, "average", arg_average);
        if (arg_average.i && arg_w1.i && arg_w1.argc)
        {
            FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_average);
            if (arg_average.argc)
            {
                ARG arg_average_do;

                std::cout<< "argc = " << argc << std::endl;
                std::cout<< "-> argc = " << arg_average.argc << " -> average i = " << arg_average.i << " -> average j = " << arg_average.j << std::endl;

                FIND_ARG((argv + arg_average.i), arg_average.argc + 1/*+ 1*/, argsv_parser_average, "do", arg_average_do);            
                //if ((W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() == W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays()) && (W2.getShape().getNumberOfColumns() == W1.getShape().getNumberOfColumns()))
                if (arg_average_do.i)
                {                  
                    std::cout<< *(argv + arg_average.i + arg_average_do.i) << " -> do argc = " << arg_average_do.argc << " -> do i = " << arg_average_do.i << " -> do j = " << arg_average_do.j << " (argc - arg_average.i) = " << (argc - arg_average.i) << std::endl; 
                    /*  */
                    //FIND_ARG_BLOCK((argv + arg_average.i /*+ arg_average_do.i*/), /*2*/ (argc - (arg_average.i /*+ arg_average_do.i*/)) /*+ 1*/ /*arg_average.j*/ /*argc*/, argsv_parser_average, arg_average_do);
                    FIND_ARG_BLOCK((argv + arg_average.i /*+ arg_average_do.i*/), /*2*/ arg_average.argc + 1 /*+ 1*/ /*arg_average.j*/ /*argc*/, argsv_parser_average, arg_average_do); 
                    std::cout<< *(argv + arg_average.i + arg_average_do.i) << " -> do argc = " << arg_average_do.argc << " -> do i = " << arg_average_do.i << " -> do j = " << arg_average_do.j << " (argc - arg_average.i) = " << (argc - arg_average.i) << std::endl;
                    //std::cout<< *(argv + arg_average.i + arg_average_do.i) << "->argc =  " << arg_average_do.argc << "-> i = " << arg_average_do.i << "-> j =" << arg_average_do.j << std::endl;                
                    if (W2_transposed.getShape() == W1.getShape())
                    {
                        if (arg_average_do.argc > 1)
                        {
                            unsigned int m = std::atoi(argv[arg_average.i + arg_average_do.i + 1]);
                            unsigned int d = std::atoi(argv[arg_average.i + arg_average_do.i + 2]);

                            std::cout<< "m = " << m << ", d = " << d << std::endl;

                            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getN(); i++)
                            {
                                //double f = W2[i];
                                //f = f*m;
                                //f = W1[i] + f;

                                //std::cout<< f/d << std::endl;

                                W1[i] = (((W1[i] + (W2_transposed[i]*m)))/d);
                            }                        
                        }
                        else
                        {
                            std::cout<< "No m no d" << std::endl;
                            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getN(); i++)
                            {
                                W1[i] = (W1[i] + W2_transposed[i]);
                            }
                        }
                    }
                    else
                    {
                        std::cout<< "main() Error: Shape of W1 and W2 are not same. To take avaerage of two matrices, their shapes must be same." << std::endl;

                        return 0;
                    }
                }
            }
        } 
    }  
    
    /*
        Finding Word Indices in Embeddings
        Get the word embeddings for a list of words
     */

    std::cout<< "\"Number of target words\"(arg_words.argc) = " << arg_words.argc << std::endl;
    std::cout<< "Target Instances and their indices in vocabulary..." << std::endl;

    bool found = false;
    
    for (int i = 0 /*arg_words.i*/; i < arg_words.argc; i++)
    {    
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < /*vocab.numberOfUniqueTokens()*/ vocab.numberOfTokens(); j++)
        {            
            if (/*!vocab[j + INDEX_ORIGINATES_AT_VALUE].compare(cc_tokenizer::String<char>(argv[i + 1]))*/ !vocab(j + INDEX_ORIGINATES_AT_VALUE, true).compare(argv[i + 1]))
            {
                found = true;

                /*std::cout<< "j = " << j + INDEX_ORIGINATES_AT_VALUE << ", " << argv[1 + i] << std::endl;*/
                
                //std::cout<< vocab(j + INDEX_ORIGINATES_AT_VALUE, true).c_str() << ", ";
                                
                if (head == NULL)
                {
                    try 
                    {
                        head = reinterpret_cast<INDEX_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(INDEX)));
                        head->next = NULL;
                        head->prev = NULL;

                        head->i = j;
                    }
                    catch(const std::bad_alloc& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    catch(const std::length_error& e)
                    {
                        std::cerr << e.what() << '\n';
                    }                    
                }
                else
                {
                    ptr = head;

                    while (ptr->next != NULL)
                    {
                        ptr = ptr->next;
                    }

                    try
                    {
                        ptr->next = reinterpret_cast<INDEX_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(INDEX)));
                        ptr->next->next = NULL;
                        ptr->next->prev = NULL;

                        ptr = ptr->next;

                        ptr->i = j;
                    }
                    catch(const std::bad_alloc& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    catch(const std::length_error& e)
                    {
                        std::cerr << e.what() << '\n';
                    }                    
                }
                
                std::cout<< vocab(j + INDEX_ORIGINATES_AT_VALUE, true).c_str() << "#";

                if (ptr != NULL)
                {                    
                    COMPOSITE_PTR composite_ptr;
                    LINETOKENNUMBER_PTR linetokennumber_ptr;

                    try
                    {
                        composite_ptr = vocab.get_composite_ptr(ptr->i + INDEX_ORIGINATES_AT_VALUE, true);
                        linetokennumber_ptr = vocab.get_line_token_number(composite_ptr, ptr->i + INDEX_ORIGINATES_AT_VALUE);
                    }
                    catch(ala_exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }

                    std::cout<< ptr->i << "(" << linetokennumber_ptr->l << ", " << linetokennumber_ptr->t << ") ";                 
                }
                else
                {                    
                    COMPOSITE_PTR composite_ptr;
                    LINETOKENNUMBER_PTR linetokennumber_ptr;

                    try
                    {
                        composite_ptr = vocab.get_composite_ptr(head->i + INDEX_ORIGINATES_AT_VALUE, true);
                        linetokennumber_ptr = vocab.get_line_token_number(composite_ptr, head->i + INDEX_ORIGINATES_AT_VALUE);
                    }
                    catch(ala_exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    } 

                    std::cout<< head->i << "(" << linetokennumber_ptr->l << ", " << linetokennumber_ptr->t << ") ";                    
                }
            }            
        }
        if (!found)
        {
            if (arg_proper.i)
            {
                if (head == NULL)
                { 
                     try 
                    {
                        head = reinterpret_cast<INDEX_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(INDEX)));
                        head->next = NULL;
                        head->prev = NULL;

                        head->i = CHAT_BOT_SKIP_GRAM_UNKNOWN_TOKEN_NUMERIC_VALUE;
                        head->actual_unknown_token = cc_tokenizer::String<char>(argv[i + 1]);
                    }
                    catch(const std::bad_alloc& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    catch(const std::length_error& e)
                    {
                        std::cerr << e.what() << '\n';
                    }              
                }
                else
                {
                    ptr = head;

                    while (ptr->next != NULL)
                    {
                        ptr = ptr->next;
                    }

                    try
                    {
                        ptr->next = reinterpret_cast<INDEX_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(INDEX)));
                        ptr->next->next = NULL;
                        ptr->next->prev = NULL;

                        ptr = ptr->next;

                        ptr->i = CHAT_BOT_SKIP_GRAM_UNKNOWN_TOKEN_NUMERIC_VALUE;
                        ptr->actual_unknown_token = cc_tokenizer::String<char>(argv[i + 1]);
                    }
                    catch(const std::bad_alloc& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    catch(const std::length_error& e)
                    {
                        std::cerr << e.what() << '\n';
                    }                                        
                }
                std::cout<< argv[i + 1] << "(" << CHAT_BOT_SKIP_GRAM_UNKNOWN_TOKEN_STRING_LITERAL << ") ";                
            }
        }
        else
        {
            found = false;
        }        
    }

    if (head)
    {
        std::cout<< std::endl;
    }

    //////////

    ptr = head;

    if (!arg_proper.i)
    {
        while (ptr)
        {
            //std::cout<< ptr->i << std::endl;

            //for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; j++)
            //{
                //std::cout<< W1[ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j] << ", ";            
            //}

            INDEX_PTR next_ptr = ptr->next;

            while (next_ptr)
            {
                //std::cout<< ptr->i << ", " << next_ptr->i << std::endl;

                //w1trainedParser.get_line_by_number(ptr->i + 1);
                //std::cout<< "Line number = " << ptr->i << " - " << w1trainedParser.get_token_by_number(1).c_str() << ", ";
                //w1trainedParser.get_line_by_number(next_ptr->i + 1);
                //std::cout<< "Line number = " << next_ptr->i << " - " << w1trainedParser.get_token_by_number(1).c_str() << std::endl;

                //std::cout<< vocab[ptr->i + INDEX_ORIGINATES_AT_VALUE].c_str() << " -> " << vocab[next_ptr->i + INDEX_ORIGINATES_AT_VALUE].c_str() << std::endl;

                std::cout<< vocab(ptr->i + INDEX_ORIGINATES_AT_VALUE, true).c_str() << "(" << ptr->i << ")"<< " -> " << vocab(next_ptr->i + INDEX_ORIGINATES_AT_VALUE, true).c_str() << "(" << next_ptr->i << ") " << std::endl;

                try
                {

                    Collective<double> u, v;

                    if (arg_average.i && arg_w1.i && arg_w1.argc)
                    {                    
                        //u = Collective<double>{W1.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL}};
                        u = W1.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL});
                    }
                    else 
                    {
                        /*
                            This line directly prints the cosine similarity between two vectors.
                            The cosine similarity value ranges between -1 (completely opposite vectors) and 1 (identical vectors).
                            A value of 0 would indicate orthogonality (no similarity).
                        */
                        //u = Collective<double>{W2.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL}};
                        u = W2_transposed.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL});
                    }

                    /*for (int i = 0; i < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; i++)
                    {
                        std::cout<< u[i] << ", ";
                    }
                    std::cout<< std::endl;*/
                    if (arg_average.i && arg_w1.i && arg_w1.argc)
                    {
                        //v = Collective<double>{W1.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
                        v = W1.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE,DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
                    }
                    else 
                    {
                        //v = Collective<double>{W2.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
                        v = W2_transposed.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
                    }
                    /*for (int i = 0; i < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; i++)
                    {
                        std::cout<< v[i] << ", ";
                    }
                    std::cout<< std::endl;*/

                    std::cout<< "Cosine Similarity = " << Numcy::Spatial::Distance::cosine(u, v) << ", ";

                    /*
                        This line calculates the cosine distance, which is 1 - cosine similarity.
                        Cosine distance transforms the similarity measure into a metric where 0 represents identical vectors,
                        and 1 represents vectors that are completely dissimilar (at 90° or 180°).
                    */
                    //Collective<double> u1 = Collective<double>{W1.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL}};
                    //Collective<double> v1 = Collective<double>{W1.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
                    /*
                         If the cosine similarity is negative (which can happen when two vectors point in opposite directions),
                        then deducting it from 1 (i.e., 1 - cosine similarity) would result in a value greater than 1.
                        This is problematic if you're expecting a distance or similarity metric that should be constrained 
                        between 0 and 1.
                        If you don't want negative values to affect your metric,
                        you can consider taking the absolute value of cosine similarity for a meaningful distance metric.
                        This way, you're ensuring that even opposite vectors are treated in a range that makes sense for your application
                    */
                    std::cout<< "Cosine Distance = " << 1 -  std::abs(Numcy::Spatial::Distance::cosine(u, v)) << std::endl;
                }
                catch(const std::bad_alloc& e)
                {
                    std::cerr << e.what() << '\n';
                }
                catch(const std::length_error& e)
                {
                    std::cerr << e.what() << "\n";
                }
                catch(ala_exception& e)
                {
                    std::cerr << e.what() << "\n";
                }
            
                /*double similarity = 1 - */ //Numcy::Spatial::Distance::cosine();
           
                next_ptr = next_ptr->next;
            }

            //std::cout<< std::endl;

            //Numcy::Spatial::Distance::cosine();

            ptr = ptr->next;
        }
    }
    else 
    {
        try
        {                    
            class proper<double> chatbot(head, vocab, W1, W2);
            
            chatbot.dsplay_list_of_context_words_for_each_target_word(head, vocab);

            /*
                Soni: WORKING HERE.
             */
            chatbot.predict_next_token(head, vocab);
        }
        catch (ala_exception& e)
        {
            std::cerr<< "main() -> " << e.what() << std::endl;
        }
    }

    /////////


    // Memory Management
    
    if (head != NULL)
    {
        ptr = head;

        while (ptr->next != NULL)
        {
            ptr = ptr->next;            
        }

        while (1)
        {
            if (ptr->prev == NULL)
            {
                cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(ptr), sizeof(INDEX)); 

                break; 
            }
            else
            {
                ptr = ptr->prev;

                cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(ptr->next), sizeof(INDEX));

                ptr->next = NULL;
            }   
        }
    }

    return 0;
}