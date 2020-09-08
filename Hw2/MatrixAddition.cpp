// MatrixAddition.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	int row, col;
	int* A = NULL;
	int* B = NULL;
	int* C = NULL;
	int result = 1;
	const int dimension = 10000;
	time_t start, end;

	size_t datasize = sizeof(int) * dimension * dimension;

	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);

	for (row = 0; row < dimension; row++) {
		for (col = 0; col < dimension; col++) {
			if( A )
				*(A + row*dimension + col) = row + col;
			if( B )
				*(B + row*dimension + col) = row + col;
		}
	}

	start = clock();
	
	for (row = 0; row < dimension; row++) {
		for (col = 0; col < dimension; col++) {
			if( A )
				if( B )
					if( C )
						*(C + row*dimension + col) = *(A + row * dimension + col) + *(B + row * dimension + col);
		}
	}

	end = clock();

	for (row = 0; row < dimension; row++) {
		for (col = 0; col < dimension; col++) {
			if( C ) {
				if (*(C + row * dimension + col) != ((row + col) * 2)) {
					result = 0;
					break;
				}
			}
		}
	}

	if (result) {
		printf("Results verified!!! (%ld)\n", ( long )( end - start ) );
	}
	else {
		printf("Wrong results!!!\n");
	}

	free(A);
	free(B);
	free(C);

	return 0;
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
