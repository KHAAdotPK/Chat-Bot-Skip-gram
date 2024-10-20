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
    ARG arg_words;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));

    /*while (argsv_parser.go_to_next_line() != cc_tokenizer::string_character_traits<char>::eof())
    {
        std::cout<< argsv_parser.get_current_line().c_str() << std::endl;

        while (argsv_parser.go_to_next_token() != cc_tokenizer::string_character_traits<char>::eof())
        {
            std::cout<< argsv_parser.get_current_token().c_str() << std::endl;
        }
    }

    exit(0);*/

    // Command-line Argument Parsing
    FIND_ARG(argv, argc, argsv_parser, "--words", arg_words);
    if (!(arg_words.i))
    {
        std::cout<< "Words are not given, can't go further." << std::endl;
        return 0;
    }
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_words);
    if (!arg_words.argc) 
    {
        ARG arg_words_help;
        HELP(argsv_parser, arg_words_help, "words");                
        HELP_DUMP(argsv_parser, arg_words_help); 

        return 0;
    }

    // Reading Pretrained Embeddings
    cc_tokenizer::String<char> w1trained = cc_tokenizer::cooked_read<char>(cc_tokenizer::String<char>("w1trained.txt"));
    cc_tokenizer::String<char> w2trained = cc_tokenizer::cooked_read<char>(cc_tokenizer::String<char>("w2trained.txt"));

    Collective<double> W1;
    Collective<double> W2;

    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> w1trainedParser(w1trained);
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> w2trainedParser(w2trained);

    std::cout<< "Total number of lines in file \"w1trained.txt\" : " << w1trainedParser.get_total_number_of_lines() << std::endl;
    w1trainedParser.go_to_next_line();    
    std::cout<< "Total number of tokens per line in file \"w1trained.txt\" : " << w1trainedParser.get_total_number_of_tokens() << std::endl;

    READ_W1(w1trainedParser, W1);
    READ_W2(w2trainedParser, W2);
    
    std::cout<< "Dimensions of W1 = " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W1.getShape().getNumberOfColumns() << std::endl;
    std::cout<< "Dimensions of W2 = " << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " << W2.getShape().getNumberOfColumns() << std::endl;

    /*
        Finding Word Indices in Embeddings
        Get the word embeddings for a list of words
     */
    for (int i = 1; i <= arg_words.argc; i++)
    {
        w1trainedParser.reset(LINES);
        w1trainedParser.reset(TOKENS);

        cc_tokenizer::string_character_traits<char>::size_type j = 0;    

        while (w1trainedParser.go_to_next_line() != cc_tokenizer::string_character_traits<char>::eof())
        {
            if (!w1trainedParser.get_token_by_number(1).compare(cc_tokenizer::String<char>(argv[arg_words.i + i])))
            {
                //std::cout<< w1trainedParser.get_token_by_number(1).c_str() << std::endl;

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
            }

            j++;
        }
    }
    
    // Calculating Cosine Similarity:
    ptr = head;
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

            w1trainedParser.get_line_by_number(ptr->i + 1);
            std::cout<< "Line number = " << ptr->i << " - " << w1trainedParser.get_token_by_number(1).c_str() << ", ";
            w1trainedParser.get_line_by_number(next_ptr->i + 1);
            std::cout<< "Line number = " << next_ptr->i << " - " << w1trainedParser.get_token_by_number(1).c_str() << std::endl;


            try
            {
                /*
                    This line directly prints the cosine similarity between two vectors.
                    The cosine similarity value ranges between -1 (completely opposite vectors) and 1 (identical vectors).
                    A value of 0 would indicate orthogonality (no similarity).
                 */
                Collective<double> u = Collective<double>{W1.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL}};

                /*for (int i = 0; i < SKIP_GRAM_EMBEDDNG_VECTOR_SIZE; i++)
                {
                    std::cout<< u[i] << ", ";
                }
                std::cout<< std::endl;*/

                Collective<double> v = Collective<double>{W1.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
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
    
    // Memory Management
    // Deallocate
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