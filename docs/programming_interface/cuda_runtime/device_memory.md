# デバイスメモリ
CUDAのプログラムモデルは各々のメモリを持つホストとデバイスからなるシステムを仮定している。カーネルはデバイスメモリの外から操作するので、ランタイムはホストメモリとデバイスメモリ間のデータ転送を行う関数だけでなく、デバイスメモリの確保、解放、デバイスメモリ内のコピーを行う関数も提供している。

デバイスメモリは線形メモリ(linear memory)かCUDA配列(CUDA array)のどちらかとして確保される。

CUDA配列はテクスチャフェッチのために最適化された不透明なメモリレイアウトである。詳しくは[テクスチャとサーフェスメモリー](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)にて。

線形メモリは一つの統一されたメモリ空間にアロケートされる。つまり、例えば2分木やリンクリストで、別々にアロケーションされたものをポインターを通してお互いに参照することが出来る。
アドレス空間の大きさはホストシステム(CPU)と、使っているGPUのcompute capabilityに依存する。

| compute capability | x86_64(AMD64) | ARM64 |
| ---- | ---- | ---- |
| >= 6.0 (Pascal) | up to 47bit | up to 48bit |
| <= 5.3 (Maxwell) | 40bit | 40bit |

*線型メモリのアドレス空間*

線形メモリは一般的に`cudaMalloc()`を使って確保され、`cudaFree()`で解放される。ホストメモリとデバイスメモリ間のデータ転送は`cudaMemcpy()`を用いて行われることが多い。ベクトル和のコードサンプルでは、ベクトルはホストメモリからデバイスメモリへコピーされなければならない。

```cpp
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    ...
}
```

線形メモリは[cudaMallocPitch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)と[cudaMalloc3D](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g188300e599ded65c925e79eab2a57347)を通しても確保出来る。これらの関数は2次元または3次元配列のメモリ確保がアライメントの要件に合うように適切にパディングされるようにしたいときにおすすめ。詳しくは[デバイスメモリへのアクセス](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)にて。返されたピッチ(pitch)(またはストライド(stride))は配列要素にアクセスするように使わなければならない。以下のコードサンプルでは、`width` x `height`の2次元配列をアロケートし、デバイスコード内で配列上をループする方法を示す。
```cpp
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

以下のコードは、要素数が`width` * `height` * `depth`の3次元配列を確保するコードである。

```cpp
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```

リファレンスマニュアルには、`cudaMalloc()`で確保された線形メモリや`cudaMallocPitch()`や`cudaMalloc3D()`で確保された線形メモリ、CUDA配列、グローバルまたは定数メモリ空間で宣言された変数に対して確保されたメモリ間のコピーに使われる関数が色々ある。

以下のコードは、ランタイムAPIを使った、グローバル変数へアクセスする様々な方法を示す。

```cpp
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

`cudaGetSymbolAddress()`はグローバルメモリ空間で宣言された変数に対して確保されたメモリを指すアドレスを取り出すために使われる。例は[Understanding of CUDA's cudaGetSymbolAddress](https://stackoverflow.com/questions/60759486/understanding-of-cudas-cudagetsymboladdress)にて。確保されたメモリのサイズは`cudaGetSymbolSize()`で得られる。

