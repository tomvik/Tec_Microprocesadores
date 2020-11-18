#include <GPUMatrix/GPUMatrix.h>
#include <stdio.h>

namespace {
    __global__
    void H() {
        printf("Hello World from GPU! %d %d\n", blockIdx.x, threadIdx.x);
    }
}

namespace GPUMatrix {
    
void HelloThreadIdx() { H<<<2, 4>>>(); }

}  // namespace GPUMatrix