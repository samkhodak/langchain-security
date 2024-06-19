#include <stdio.h>
#include <stdlib.h>

int main()
{
    char input[100];
    scanf('%s', input);

    if (input[0] == 'c')
    {
        // enough space for 2 digits + a space + input-1 chars
        auto out = malloc(sizeof(input) - 1 + 3);
        sprintf(out, "%lu ", strlen(input) - 1);
        strcat(out, input + 1);
        printf("%s", out);
        free(out);
    }
    else if ((input[0] == 'e') && (input[1] == 'c'))
    {
        // echo input
        printf(input + 2);
    }
    else if (strncmp(input, "head", 4) == 0)
    {
        // truncate string at specified position
        if (strlen(input) > 5)
        {
            input[input[4]] = '\0';
            printf("%s", input + 4);
        }
        else
        {
            fprintf(stderr, "head input was too small\n");
        }
    }

    return 0;
}