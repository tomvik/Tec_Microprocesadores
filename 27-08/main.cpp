#include <Windows.h>
#include <stdio.h>

#define kArraySize 4

DWORD WINAPI helloFunc(LPVOID pArg) {
    int* param = reinterpret_cast<int*>(pArg);
    printf("Hello Thread %d\n", *param);
    return 0;
}

int main() {
    HANDLE hThread[kArraySize];
    int ids[kArraySize];
    for (int i = 0; i < kArraySize; ++i) {
        ids[i] = i;
        hThread[i] = CreateThread(NULL, 0, helloFunc, (ids+i), 0, NULL);
    }
    WaitForMultipleObjects(kArraySize, hThread, true, INFINITE);
    return 0;
}
