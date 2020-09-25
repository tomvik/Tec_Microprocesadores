#include <ctype.h>
#include <pmmintrin.h>
#include <stdio.h>

void lp(float *x, float *y, int n) {
    __m128 div, v1, v2, avg;
    div = _mm_set1_ps(2.0);

    for (int i = 0; i < n / 8; ++i) {
        v1 = _mm_load_ps(x + i * 8);      // [a0=x0] [a1=x1] [a2=x2] [a3=x3]
        v2 = _mm_load_ps(x + 4 + i * 8);  // [b0=x4] [b1=x5] [b2=x6] [b3=x7]
        avg = _mm_hadd_ps(v1, v2);        // [a0+a1] [a2+a3] [b0+b1] [b2+b3]
        avg = _mm_div_ps(avg, div);       // [(a0+a1)/2.0] [(a2+a3)/2.0] [(b0+b1)/2.0] [(b2+b3)/2.0]
        _mm_store_ps(y + i * 4, avg);     // [(a0+a1)/2.0] [(a2+a3)/2.0] [(b0+b1)/2.0] [(b2+b3)/2.0]
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }
    const int kSize = atoi(argv[1]) * 8;

    printf("The numbers will go from 0 to %d\n", kSize);

    float nums[kSize] = {0};
    float result[kSize / 2] = {0};

    for (int i = 0; i < kSize; ++i) {
        nums[i] = i + 0.0;
    }

    lp(nums, result, kSize);

    printf("Result: ");
    for (int i = 0; i < kSize / 2; ++i) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}