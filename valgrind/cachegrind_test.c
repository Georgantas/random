#include <stdio.h>
#include <stdlib.h>

int main(void)
{
	int r = 40000000, c = 30; 
	int* disp = malloc((r * c) * sizeof(int));

	for(int i = 0; i < r; i++) {
		for(int j = 0; j < c; j++) {
			// disp[j * r + i] = 3; // 12.2% D1 miss rate
			disp[i * c + j] = 3; // 0.8% D1 miss rate
		}
	}

	return 0;
}
