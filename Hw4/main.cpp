#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

constexpr int kDefaultElements = 104857600;
constexpr int kDefaultRuns = 1;
constexpr int kDefaultType = 0;
constexpr int kAlignedBytes = 64;
constexpr int kAmountTypes = 3;

const char* const types[] = {"Unoptimized", "Vectorized with SSE", "Vectorized with AVX"};

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
    printf("NormalSum\n\n");
    int i;
    time_t start, end;
    start = clock();
    for (i = 0; i < elements; i++) C[i] = A[i] + B[i];
    end = clock();

    return static_cast<int64_t>(end - start);
}

int64_t vectorizedSSEAddition(float* A, float* B, float* C, const int elements) {
    printf("vectorizedSSEAddition\n\n");
    int i;
    const int step = 4;
    time_t start, end;
    __m128 a, b, c;
    start = clock();

#pragma vector aligned
    for (i = 0; i < elements / 4; ++i) {
        a = _mm_load_ps(A + i * 4);
        b = _mm_load_ps(B + i * 4);

        c = _mm_add_ps(a, b);
        _mm_store_ps(C + i * 4, c);
    }
    end = clock();

    return static_cast<int64_t>(end - start);
}

int64_t vectorizedAVXAddition(float* A, float* B, float* C, const int elements) {
    printf("vectorizedAVXAddition\n\n");
    int i;
    const int step = 4;
    time_t start, end;
    __m256 a, b, c;
    start = clock();

#pragma vector aligned
    for (i = 0; i < elements / 8; ++i) {
        a = _mm256_load_ps(A + i * 8);
        b = _mm256_load_ps(B + i * 8);

        c = _mm256_add_ps(a, b);
        _mm256_store_ps(C + i * 8, c);
    }
    end = clock();

    return static_cast<int64_t>(end - start);
}

void runArrayAddition(const int elements, const int elements_to_print, const int sum_type) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    // Array creation
    const size_t datasize = sizeof(float) * elements;

    switch (sum_type) {
        case 0:
            A = reinterpret_cast<float*>(malloc(datasize));
            B = reinterpret_cast<float*>(malloc(datasize));
            C = reinterpret_cast<float*>(malloc(datasize));
            break;
        case 1:
        case 2:
            A = reinterpret_cast<float*>(_mm_malloc(datasize, kAlignedBytes));
            B = reinterpret_cast<float*>(_mm_malloc(datasize, kAlignedBytes));
            C = reinterpret_cast<float*>(_mm_malloc(datasize, kAlignedBytes));
            break;
        default:
            break;
    }

    initializeData(A, B, elements);

    int64_t total_time = -1;

    switch (sum_type) {
        case 0:
            total_time = normalAddition(A, B, C, elements);
            break;
        case 1:
            total_time = vectorizedSSEAddition(A, B, C, elements);
            break;
        case 2:
            total_time = vectorizedAVXAddition(A, B, C, elements);
            break;
        default:
            break;
    }

    printFirstN(C, elements_to_print);

    if (validateData(C, elements)) {
        printf("Results verified!!! (%ld)\n", total_time);
    } else {
        printf("Wrong results!!!\n");
    }

    // Memory deallocation
    switch (sum_type) {
        case 0:
            free(A);
            free(B);
            free(C);
            break;
        case 1:
        case 2:
            _mm_free(A);
            _mm_free(B);
            _mm_free(C);
            break;
        default:
            break;
    }
}

int main() {
    int options = -1, elements = kDefaultElements, runs = kDefaultRuns, type = kDefaultType;
    // Dumb UI, but works.
    printf(
        "Hello! Welcome to the super duper complex quantum program that adds two arrays. Enjoy "
        ":)\n\n");
    while (options != 0) {
        printf("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n");
        printf("The code will run with the following setting:\n");
        printf(
            "*\tElements per array: %d,\n"
            "*\tExecute: %s code,\n"
            "*\tTotal runs: %d\n\n",
            elements, types[type], runs);

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
                type = -1;
                while (type < 0 || type >= kAmountTypes) {
                    printf("Select the type you wish to run:\n");
                    for (int i = 0; i < kAmountTypes; ++i) {
                        printf("*\tEnter %d for '%s'\n", i, types[i]);
                    }
                    printf("Type: ");
                    scanf("%d", &type);
                }
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
        "*\tExecute: %s code,\n"
        "*\tTotal runs: %d\n\n",
        elements, types[type], runs);

    for (int i = 0; i < runs; ++i) {
        runArrayAddition(elements, 0, type);
    }

    return 0;
}