#include <stdio.h>
#include <math.h>
#include "mymax.h"

int main()
{
	float MYPI = acos(-1);
	float MYE = exp(1);

    printf("Hello World\n");
    printf("%f %f\n", MYPI, MYE);
    printf("max of e and pi is approximately %f\n", mymax(MYPI, MYE));
    return 0;
}
