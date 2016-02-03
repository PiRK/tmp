#include "mymax.h"
float mymax(float num1, float num2)
{
    /* local variable declaration */
    float result;

    if (num1 > num2)
        result = num1;
    else
        result = num2;

    return result;
}
