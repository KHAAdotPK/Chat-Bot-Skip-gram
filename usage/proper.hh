/*
    proper.hh
    Q@khaa.pk
 */

#include "main.hh"

#ifndef SKIP_GRAM__CHAT_BOT_PROPER_HH
#define SKIP_GRAM__CHAT_BOT_PROPER_HH

#define DEFAULT_NUMBER_OF_CONTEXT_WORDS 5 // Default number of context words to consider for each target word

template <typename E = double>
struct context_word_indices
{
    cc_tokenizer::string_character_traits<char>::size_type i_target_word; // W1 index
    cc_tokenizer::string_character_traits<char>::size_type i_context_word;  // W2 index
    E cosine_similarity; // Cosine similarity between the target word and the context word

    struct context_word_indices* next;
    struct context_word_indices* prev;
};

template <typename E = double>
class proper
{            
    public:
        typedef struct context_word_indices<E> CONTEXT_WORD_INDICES;
        typedef struct CONTEXT_WORD_INDICES* CONTEXT_WORD_INDICES_PTR;

        proper(INDEX_PTR head_target_word_indices, CORPUS& vocab, Collective<E>& W1, Collective<E>& W2, cc_tokenizer::string_character_traits<char>::size_type n = DEFAULT_NUMBER_OF_CONTEXT_WORDS) throw (ala_exception)
        {
            head_context_word_indices = NULL;
            CONTEXT_WORD_INDICES_PTR context_word_index_ptr = NULL;  
            INDEX_PTR target_index_ptr = head_target_word_indices; // Linked list of indices into W1

            try
            {            
                while (target_index_ptr != NULL)
                {
                    // Holds a slice of W1 trained weights file. W1 file, which contains word embeddings for vocabulary
                    Collective<E> vocabulary_word_embedding;
                    Collective<E> context_word_embedding;
                    /*
                        Check if the token is part of the vocabulary.
                        The index `CHAT_BOT_SKIP_GRAM_UNKNOWN_TOKEN_NUMERIC_VALUE` is used to indicate 
                        that the token is not recognized or is out of vocabulary
                     */   
                    if (target_index_ptr->i != CHAT_BOT_SKIP_GRAM_UNKNOWN_TOKEN_NUMERIC_VALUE)
                    {                        
                        vocabulary_word_embedding = W1.slice(target_index_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL});
                    }

                    if (vocabulary_word_embedding.getShape().getN())
                    {   
                        bool sorted = false;
                        cc_tokenizer::string_character_traits<char>::size_type count = 0;
                        CONTEXT_WORD_INDICES_PTR ptr = NULL;

                        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getNumberOfColumns(); i++)
                        {                                                             
                            context_word_embedding = W2.slice(i, DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}, AXIS_NONE);
                            
                            E cs =Numcy::Spatial::Distance::cosine<E>(vocabulary_word_embedding, context_word_embedding);

                            //std::cout<< "Cosine Similarity = " << cs << std::endl;

                            if (count < (n /*- 1*/))
                            {
                                if (head_context_word_indices == NULL)
                                {
                                    head_context_word_indices = reinterpret_cast<CONTEXT_WORD_INDICES_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(CONTEXT_WORD_INDICES)));
                                    head_context_word_indices->i_target_word = target_index_ptr->i;
                                    head_context_word_indices->i_context_word = i;
                                    head_context_word_indices->cosine_similarity = cs /*Numcy::Spatial::Distance::cosine<E>(vocabulary_word_embedding, context_word_embedding)*/;

                                    head_context_word_indices->next = NULL;
                                    head_context_word_indices->prev = NULL;

                                    context_word_index_ptr = head_context_word_indices;

                                    ptr = head_context_word_indices;                                    
                                }
                                else
                                {   
                                    context_word_index_ptr->next = reinterpret_cast<CONTEXT_WORD_INDICES_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(CONTEXT_WORD_INDICES)));
                                    context_word_index_ptr->next->prev = context_word_index_ptr;
                                    context_word_index_ptr->next->next = NULL;
                                    context_word_index_ptr = context_word_index_ptr->next;

                                    context_word_index_ptr->i_target_word = target_index_ptr->i;
                                    context_word_index_ptr->i_context_word = i; 
                                    context_word_index_ptr->cosine_similarity = cs /*Numcy::Spatial::Distance::cosine<E>(vocabulary_word_embedding, context_word_embedding)*/;

                                    if (ptr == NULL)
                                    {
                                        ptr = context_word_index_ptr;
                                    }
                                }

                                count = count + 1;
                            }
                            else 
                            {
                                // Sort from ptr to NULL, using bubble sort which sorts in ascending order                                
                                while (1)
                                {
                                    bool swapped = false;
                                    CONTEXT_WORD_INDICES_PTR current = ptr; 
                                    CONTEXT_WORD_INDICES_PTR next = ptr->next; 

                                    while (next != NULL)
                                    {
                                        // Compare cosine_similarity values and swap if out of order
                                        if (current->cosine_similarity > next->cosine_similarity)
                                        {    
                                            // Swap cosine_similarity
                                            E cosine_similarity = current->cosine_similarity;
                                            current->cosine_similarity = next->cosine_similarity;
                                            next->cosine_similarity = cosine_similarity;

                                            // Swap i_target_word
                                            cc_tokenizer::string_character_traits<char>::size_type i_target_word = current->i_target_word;
                                            current->i_target_word = next->i_target_word;
                                            next->i_target_word = i_target_word;

                                            // Swap i_context_word   
                                            cc_tokenizer::string_character_traits<char>::size_type i_context_word = current->i_context_word;
                                            current->i_context_word = next->i_context_word;
                                            next->i_context_word = i_context_word;

                                            swapped = true;
                                        }

                                        current = next;
                                        next = next->next;
                                    }

                                    if (!swapped)
                                    {   
                                        break;
                                    }                                        
                                }

                                /* Find if this cosine similarity is greater than any of the n cosine similarities already in links */                                
                                CONTEXT_WORD_INDICES_PTR current = /*head_context_word_indices*/ ptr; 
                                while (current)
                                {                                    
                                    if (cs > current->cosine_similarity)
                                    {
                                        if (current->next)
                                        {
                                            if (current->next->cosine_similarity > cs)
                                            {
                                                current->cosine_similarity = cs;
                                                current->i_target_word = target_index_ptr->i;
                                                current->i_context_word = i;                                                  
                                            }                                           
                                        }
                                        else
                                        {
                                            current->cosine_similarity = cs;
                                            current->i_target_word = target_index_ptr->i;
                                            current->i_context_word = i;
                                        }
                                    }

                                    current = current->next;
                                }
                            }                            
                        }                        
                    }                    
                    target_index_ptr = target_index_ptr->next;
                }
            }
            catch(const std::bad_alloc& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("proper::proper() Error: ") + cc_tokenizer::String<char>(e.what()));
            }
            catch(const std::length_error& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("proper::proper() Error: ") + cc_tokenizer::String<char>(e.what()));
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("proper::proper() -> ") + e.what());
            }
        }

        void dsplay_list_of_context_words_for_each_target_word(CORPUS& vocab, cc_tokenizer::string_character_traits<char>::size_type n = DEFAULT_NUMBER_OF_CONTEXT_WORDS)
        {
            COMPOSITE_PTR composite_ptr = NULL;
            LINETOKENNUMBER_PTR linetokennumber_ptr = NULL;
            CONTEXT_WORD_INDICES_PTR ptr = head_context_word_indices;

            while (ptr)
            {
                //std::cout<< "Target Word Index = " << ptr->i_target_word << ", Context Word Index = " << ptr->i_context_word << ", Cosine Similarity = " << ptr->cosine_similarity << std::endl;

                std::cout<< vocab(ptr->i_target_word + INDEX_ORIGINATES_AT_VALUE, true).c_str() << "(" << ptr->i_target_word << ", ";
                composite_ptr = vocab.get_composite_ptr(ptr->i_target_word + INDEX_ORIGINATES_AT_VALUE, true);
                linetokennumber_ptr = vocab.get_line_token_number(composite_ptr, ptr->i_target_word + INDEX_ORIGINATES_AT_VALUE);
                std::cout<< linetokennumber_ptr->index <<", " << linetokennumber_ptr->l << ", " << linetokennumber_ptr->t << ")" << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < n; i++)
                {
                    std::cout<< "------> " << vocab(ptr->i_context_word + INDEX_ORIGINATES_AT_VALUE, true).c_str() << "(" << ptr->i_context_word << ", ";
                    composite_ptr = vocab.get_composite_ptr(ptr->i_context_word + INDEX_ORIGINATES_AT_VALUE, true);
                    linetokennumber_ptr = vocab.get_line_token_number(composite_ptr, ptr->i_context_word + INDEX_ORIGINATES_AT_VALUE);
                    std::cout<< linetokennumber_ptr->index <<", " << linetokennumber_ptr->l << ", " << linetokennumber_ptr->t << ")" << std::endl;

                    ptr = ptr->next;
                }                
            }
        }

        ~proper()
        {
            if (head_context_word_indices == NULL)
            {
                return;
            }

            CONTEXT_WORD_INDICES_PTR context_word_index_ptr = head_context_word_indices;

            while (context_word_index_ptr != NULL)
            {
                CONTEXT_WORD_INDICES_PTR next = context_word_index_ptr->next;

                cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(context_word_index_ptr), sizeof(CONTEXT_WORD_INDICES));

                context_word_index_ptr = next;
            }
        }

        private:
            CONTEXT_WORD_INDICES_PTR head_context_word_indices;
};

#endif