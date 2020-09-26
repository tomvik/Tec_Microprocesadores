#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

constexpr int kDefaultElements = 52428800;
constexpr int kDefaultRuns = 1;
constexpr bool kDefaultType = true;

void initializeData(float* A, float* B, const int elements) {
    for (int i = 0; i < elements; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }
}

bool validateData(float* C, const int elements) {
    for (int i = 0; i < elements; i++) {
        if (C[i] != i + i) {
            return false;
        }
    }
    return true;
}

void printFirstN(float* C, const int elements_to_print) {
    for (int i = 0; i < elements_to_print; i++) {
        printf("C[%d]=%10.2lf\n", i, C[i]);
    }
}

int64_t normalAddition(float* A, float* B, float* C, const int elements) {
    int i;
    time_t start, end;
    start = clock();
    for (i = 0; i < elements; i++) C[i] = A[i] + B[i];
    end = clock();

    return static_cast<int64_t>(end - start);
}

int64_t vectorizedAddition(float* A, float* B, float* C, const int elements) {
    int i;
    time_t start, end;
    start = clock();
    // This loop can be optimized using Intrinsics
    for (i = 0; i < elements; i++) C[i] = A[i] + B[i];
    end = clock();

    return static_cast<int64_t>(end - start);
}

void runArrayAddition(const int elements, const int elements_to_print, const bool normal_sum) {
    int i;
    float* A = NULL;
    float* B = NULL;
    float* C = NULL;
    // Array creation
    size_t datasize = sizeof(float) * elements;
    A = reinterpret_cast<float*>(malloc(datasize));
    B = reinterpret_cast<float*>(malloc(datasize));
    C = reinterpret_cast<float*>(malloc(datasize));

    initializeData(A, B, elements);

    const int64_t total_time =
        normal_sum ? normalAddition(A, B, C, elements) : vectorizedAddition(A, B, C, elements);

    printFirstN(C, elements_to_print);

    if (validateData(C, elements)) {
        printf("Results verified!!! (%ld)\n", total_time);
    } else {
        printf("Wrong results!!!\n");
    }

    // Memory deallocation
    free(A);
    free(B);
    free(C);
}

int main() {
    int options = -1, elements = kDefaultElements, runs = kDefaultRuns;
    bool type = kDefaultType;
    // Dumb UI, but works.
    printf(
        "Hello! Welcome to the super duper complex quantum program that adds two arrays. Enjoy "
        ":)\n\n");
    while (options != 0) {
        printf("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n");
        printf("The code will run with the following setting:\n");
        printf(
            "*\tElements per array: %d,\n"
            "*\tExecute %s code,\n"
            "*\tTotal runs: %d\n\n",
            elements, type ? "vectorized" : "unoptimized", runs);

        printf("If you want to customize something, select your option:\n");
        printf(
            "*\tEnter 0 to skip the customization and run the code.\n"
            "*\tEnter 1 to change the run type (vectorized or unoptimized).\n"
            "*\tEnter 2 to change the amount of elements in the array.\n"
            "*\tEnter 3 to change the amount of times the code will run.\n"
            "*\tEnter 4 to fall back to default values.\n");
        printf("Select your option: ");
        scanf("%d", &options);
        printf("\n");

        switch (options) {
            case 0:
                break;
            case 1:
                type = !type;
                break;
            case 2:
                printf("Select the amount of elements per array: ");
                scanf("%d", &elements);
                break;
            case 3:
                printf("Select the amount of total runs: ");
                scanf("%d", &runs);
                break;
            case 4:
                elements = kDefaultElements;
                runs = kDefaultRuns;
                type = kDefaultType;
                break;
            default:
                printf("[WARNING] Wrong number of option. Try again.\n");
                break;
        }
    }

    printf("\n\nRunning the code with the following settings:\n");
    printf(
        "*\tElements per array: %d,\n"
        "*\tExecute %s code,\n"
        "*\tTotal runs: %d\n\n",
        elements, type ? "vectorized" : "unoptimized", runs);

    for (int i = 0; i < runs; ++i) {
        runArrayAddition(elements, 0, type);
    }

    return 0;
}