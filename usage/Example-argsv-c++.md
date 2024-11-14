**argsv-c++** also handles sub-commands or sub-options, the following code block documents that behaviour of **argsv-c++**
---

```C++
#define COMMAND "h -h help --help ? /? (Displays the help screen)\n\
v -v version --version /v (Displays the version number)\n\
average --average (Acts as a flag to be used with the [w1 | --w1] command-line option; when used, the specified file is an average of w1 and w2 trained weights)\n"

#define SUB_COMMAND_average "do (Used with the \"average\" command; optionally expects a numeric argument. This command implies that the \"W1\" and \"W2\" matrices will be averaged, and the program will proceed with processing the resulting matrix. If an optional numeric argument is provided, it acts as a multiplier, with the \"W2\" matrix as the multiplicand before averaging)\n"

ARG arg_common, arg_words, arg_w1, arg_w2, arg_help, arg_vocab, arg_average;
cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser_average(cc_tokenizer::String<char>(SUB_COMMAND_average));

/* 
    Processes all arguments that are common to all commands listed in the \"COMMAND\" string.
    Starts at index 1, and end at an index corresponding to the first command-line argument in the \"COMMAND\" string.
 */
GET_FIRST_ARG_INDEX(argv, argc, argsv_parser,  arg_common);            s
if (arg_common.argc)
{  
    /*                       
        argv[1:arg_common.argc] 
    */
}

/*
    In argv array, index 0 is an anchor index for COMMAND string, when you process the COMMAND string using argsv-c++.
    By default at index 0 in argv array of main() function, the pointer to program name gets stored.     
 */
FIND_ARG(argv, argc, argsv_parser, "average", arg_average);
FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_average);
if (arg_average.argc)
{
    /*
        When a command-line option includes sub-commands, as in this case where \"average\" has a sub-command,
        the keyword \"average |--average\" serves as an anchor, similar to the program name in the argv array of the C main() function.
     */
    ARG arg_average_do; 
    
    FIND_ARG((argv + arg_average.i) /* Anchor index 0 for \"SUB_COMMAND_average\" strng */, arg_average.argc, argsv_parser_average, "do", arg_average_do); 
    FIND_ARG_BLOCK((argv + arg_average.i) /* Anchor index 0 for \"SUB_COMMAND_average\" strng */, (arg_average.argc + 1) /* Adjusted to replicate the behavior of argc in the main() function in C */, argsv_parser_average, arg_average_do); 
}
```