---
title: "プログラミングインターフェイス"
---

CUDAランタイムは低級なC APIであるCUDAドライバーAPIの上に作られている。CUDAドライバーAPIもアプリケーションから利用できる。このAPIは、

- CUDAコンテキスト - ホストプロセスのデバイス版
- CUDAモジュール - 動的リンクライブラリのデバイス版

といった低級な概念を提供する。詳しくは[ドライバーAPI](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)にて。

## NVCCを使ったコンパイル
カーネルはPTXと呼ばれる、CUDAの命令セットを使って書くこともできる。しかし、普通はC++のような高級言語を使ったほうがより効率的である。どちらにしろ、カーネルは`nvcc`を使ってデバイス上で実行されるバイナリコードにコンパイルされる必要がある。

`nvcc`はC++やPTXコードのコンパイルを単純にするコンパイルドライバーである。これは単純で親しみ深いコマンドを提供し、異なるコンパイルステージを実装するツールを呼び出して実行する。この節では`nvcc`ワークフローの概要とコマンドオプションの概要を述べる。

詳しくは[NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)にて。

### コンパイルワークフロー

![](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/_images/cuda-compilation-from-cu-to-executable.png)

#### オフラインコンパイル
`nvcc`によってコンパイルされるソースファイルはホストコードとデバイスコードを含みうる。`nvcc`の基本的なワークフローは

1. ホストコードからデバイスコードを分け、
1. デバイスコードをアセンブリ形式(PTXコード)かバイナリ形式(cubinオブジェクト)にコンパイルし、
1. ホストコードでカーネルに指定された`<<<...>>>`を、PTXコードやcubinオブジェクトからコンパイルされたカーネルを読み込んで起動するために必要なCUDAランタイム関数の呼び出しに置き換える

ことである。

修正されたホストコードは他のツールを使ってコンパイルできるようにC++コードとして出力されるか、最後のコンパイルステージで`nvcc`にホストコンパイラを呼び出させることでオブジェクトコードとして直接出力される。

アプリケーションは、

- コンパイルされたホストコードをリンクする(最もありふれたケース)か、
- 修正されたホストコードを無視し、PTXコードかcubinオブジェクトをロードし、実行するためにCUDAドライバーAPIを使う (詳しくは[ドライバーAPI](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)にて)。

#### Just-in-Timeコンパイル
実行時にアプリケーションにロードされるPTXコードは、デバイスドライバーによってさらにバイナリコードにコンパイルされる。これはjust-in-timeコンパイルと呼ばれる。just-in-timeコンパイルはアプリケーションのロードタイムを増やすが、アプリケーションが新しいコンパイラーの改善を享受できるようになる。また、この方法はアプリケーションがコンパイルされた時になかったデバイス上で実行するための唯一の方法である。詳しくは[アプリケーション互換性](#アプリケーション互換性)にて。

デバイスドライバーはアプリケーションのPTXコードをjust-in-timeコンパイルする時、アプリケーションの呼び出し毎にコンパイルしないように、生成したバイナリコードのコピーを自動的にキャッシュする。(compute cacheと呼ばれる)このキャッシュはデバイスドライバーがアップグレードされた時に自動的に無効になるので、アプリケーションはデバイスドライバーに組み込まれた新しいjust-in-timeコンパイラーの改善を享受することが出来る。

環境変数を使うと、just-in-timeコンパイルをコントロールできる。詳しくは[CUDAの環境変数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)にて。

CUDA C++デバイスコードをコンパイルするために`nvcc`を使う代わりとして、実行時にCUDA C++デバイスコードをPTXにコンパイルするNVRTCを使うことが出来る。NVRTCはCUDA C++の実行時コンパイルライブラリである。

### バイナリ互換性
バイナリコードはアーキテクチャ特有である。cubinオブジェクトは対象アーキテクチャを指定するコンパイラーオプション`-code`を使って生成される。例えば、`-code=sm_80`を付けてコンパイルすると、compute capability 8.0のデバイスに対するバイナリコードが作られる。compute capability X.yに対して生成されたcubinオブジェクトはcompute capability X.z(z>=y)のデバイス上でしか実行できない。

:::message
バイナリ互換性はデスクトップに対してしかサポートされない。Tegraに対してはサポートされず、デスクトップとTegra間のバイナリ互換性もサポートされない。
:::

### PTX互換性
いくつかのPTX命令は高いcompute capabilityを持つデバイス上でしかサポートされない。例えば、warpシャッフル関数はcompute capabilityが5.0以上のデバイス上でしかサポートされない。`-arch`コンパイラーオプションにはC++をPTXコードにコンパイルする時に必要なcompute capabilityを指定する。例えば、warpシャッフルを含むコードは`-arch=compute_50`(かそれ以上)を付けてコンパイルされなければならない。

ある特定のcompute capabilityに対して作られたPTXコードは必ずそれ以上のcompute capabilityを持つバイナリコードにコンパイルされる。以前のPTXのバージョンからコンパイルされたバイナリはいくつかのハードウェアの特徴を利用しないかもしれないので、最終的なバイナリは最新のPTXを使って生成されたバイナリより性能が悪いかもしれない。

compute capabilityは仮想アーキテクチャとも呼ばれ、プリプロセスとPTXへのコンパイルをコントロールするために使われる。そのため、`-arch`オプションだけを指定しても実行ファイルやライブラリを作れず、`-code`オプションで物理アーキテクチャを指定しなければならない。

### アプリケーション互換性
特定のcompute capabilityのデバイス上でコードを実行するためには、アプリケーションはこのcompute capabilityと互換性のあるバイナリかPTXコードをロードしなければならない。特に、高いcompute capabilityを持つ、将来のアーキテクチャ上でコードを実行できるようにするためには、アプリケーションはこれらのデバイスに対してjust-in-timeでコンパイルされるPTXコードをロードしなければならない。

どのPTXとバイナリコードがCUDA C++アプリケーションに埋め込まれるかはコンパイルオプション`-arch`と`-code`または`gencode`によって制御される。例えば、

```
nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

とすると、最初と2番目の`-gencode`オプションからcompute capability 5.0と6.0に互換性があるバイナリコードと、3番目の`-gencode`オプションからcompute capability 7.0と互換性があるPTXとバイナリコードを埋め込む。

ホストコードは実行時にロードし実行する、最も適したコードを自動的に選ぶよう生成される。上の例では以下のようになる。

- compute capability 5.0と5.2を持つデバイスに対する5.0バイナリコード
- compute capability 6.0と6.1を持つデバイスに対する6.0バイナリコード
- compute capability 7.0と7.5を持つデバイスに対する7.0バイナリコード
- compute capabilityが8.0以上のデバイスに対する、実行時にバイナリコードにコンパイルされるPTXコード

compute capabilityに基づいた様々なコードパスを区別するために`__CUDA_ARCH__`マクロを使うことが出来る。ただし、このマクロはデバイスコードでのみ定義されている。例えば、`-arch=compute_80`を付けてコンパイルするとき、`__CUDA_ARCH__`の値は`800`である。

ドライバーAPIを使うアプリケーションは実行時に最適なファイルを明示的にロードし実行できるようにコードをコンパイルしなければならない。

VoltaアーキテクチャはスレッドがGPU上でスケジュールされる方法を変えるIndependent Thread Schedulingを導入している。以前のアーキテクチャ内のSIMTスケジューリングの特定の振る舞いに依存するコードに対して、Independent Thread Schedulingは関係するスレッドの集合を変えるかもしれず、不正確な結果をもたらす。
Independent Thread Schedulingを実装しつつ移植するためには、Volta開発者はコンパイルオプション`-arch=compute_60 -code=sm_70`をつけてPascalのスレッドスケジューリングに最適化すればよい。

`nvcc`ユーザーマニュアルは`-arch`、`-code`、`-gencode`コンパイラーオプションに対する様々な短縮形をリストにまとめている。例えば、`-arch=sm_70`は`-arch=compute_70 -code=compute_70,sm_70`の略である。

### C++互換性
コンパイラのフロントエンドはCUDAソースファイルをC++の文法に従って処理する。全てのC++の機能はホストコードに対してはサポートされるが、デバイスコードに対しては一部のみサポートされる。詳しくは[C++言語サポート](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support)にて。

### 64ビット互換性
64bitバージョンの`nvcc`は64bitモード(つまりポインターが64bit)でデバイスコードをコンパイルする。64bitモードでコンパイルされたデバイスコードは64bitモードでコンパイルされたホストコードでしかサポートされない。

## CUDAランタイム
ランタイムは`cudart`ライブラリに実装されていて、`cudart.lib`または`libcudart.a`を使って静的にリンクするか、`cudart.dll`または`libcudart.so`を使って動的にリンクする。

`cudart`の全てのエントリーポイントは`cuda`から始まる。

[デバイスメモリ](#デバイスメモリ)では、デバイスメモリの管理に使われる関数の概要を述べる。

[共有メモリ](#共有メモリ)では、性能を最大化するために共有メモリの使い方を述べる。

[ページロックドホストメモリ](#ページロックドホストメモリ)では、ホストとデバイス間のデータ転送を行うカーネルの実行をオーバーラップするために使われるページロックドホストメモリを紹介する。

[非同期並列実行](#非同期並列実行)では、コンセプトといろいろなレベルでの非同期並列実行を行うためのAPIを述べる。

[マルチデバイスシステム](#マルチデバイスシステム)では、プログラミングモデルが同じホストに付けられた複数のデバイスを持つシステムにどのように拡張されるかを述べる。

[エラーチェック](#エラーチェック)では、ランタイムによって発生したエラーを適切に確認する方法を述べる。

[コールスタック](#コールスタック)では、CUDA C++コールスタックを管理するために使われる関数を述べる。

[テクスチャとサーフェスメモリ](#テクスチャとサーフェスメモリ)では、デバイスメモリにアクセスする別の方法を提供するテクスチャとサーフェスメモリ空間について述べる。これらはGPUのテクスチャハードウェアのサブセットも提供する。

[グラフィックとの相互運用性](#グラフィックとの相互運用性)では、ランタイムがOpenGLとDirect3Dと相互運用するために提供する、さまざまな関数を紹介する。

### 初期化
CUDA 12.0から、[cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419)関数と[cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb)関数は指定されたデバイスに紐付いた、ランタイムと[プライマリーコンテキスト](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)を初期化する。これらの関数呼び出しがないと、他のランタイムAPIリクエストを処理する必要がある時にランタイムは暗にデバイス`0`を使い、自身を初期化する。ランタイムの関数呼び出しの時間計測とランタイムへの最初の関数呼び出しからのエラーコードを解釈するときに心に留めておく必要がある。12.0以前では、`cudaSetDevice`関数はランタイムを初期化せず、アプリケーションは(時間計測とエラーハンドリングのために)他のAPI呼び出しからランタイムの初期化を切り離す、何もしない関数呼び出し`cudaFree(0)`をよく使っていた。

compute capabilityが2.0以上のデバイスの数は[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f)関数でわかる。デバイス番号は0から始まり、デバイス数未満までである。

プライマリーコンテキストは、確保されたメモリのリストやデバイスコードを含む、ロードされたモジュールなどデバイスを制御するデータ全てを持つ^[https://stackoverflow.com/questions/43244645/what-is-a-cuda-context]。プライマリーコンテキストはデバイス毎に1つしかなく、CUDAランタイムAPIと共有される。つまり、アプリケーションのすべてのホストスレッドで共有される。このコンテキストを作成する時、必要ならデバイスコードをjust-in-timeコンパイルする。詳しくは、[コンテキスト](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)にて。

ホストスレッドが[cudaDeviceReset](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217)関数を呼び出した場合、ホストスレッドが現在操作しているデバイスのプライマリーコンテキストを破棄する。

:::message
CUDAインターフェイスはホストプログラムの開始時に初期化され、ホストプログラムの終了時に破棄されるグローバルな状態を持つ。CUDAランタイムとドライバーはこの状態が無効かどうかを検出できないので、明示的か暗黙的かに関わらず、プログラムの開始時と終了時にこれらのインターフェイスを使うと未定義動作になる。

CUDA 12.0時点では、`cudaSetDevice()`はホストスレッドの現在のデバイスを変更後にランタイムを明示的に初期化する。以前のバージョンのCUDAでは、`cudaSetDevice()`が呼び出されてから初めてランタイムの関数が呼ばれるまで、新しいデバイス上のランタイムの初期化を遅延していた。この変更により、初期化のエラーを`cudaSetDevice()`の戻り値で確認することがとても重要になった。

エラー処理とバージョン管理のランタイム関数はランタイムを初期化しない。
:::

### デバイスメモリ
CUDAのプログラムモデルはそれぞれ自身のメモリを持つホストとデバイスからなるシステムを仮定している。カーネルはデバイスメモリの外から操作するので、ランタイムはホストメモリとデバイスメモリ間のデータ転送を行う関数だけでなく、デバイスメモリの確保、解放、デバイスメモリのコピーを行う関数も提供している。

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

### デバイスメモリのL2アクセス管理
CUDAカーネルがグローバルメモリ内のデータ領域に繰り返しアクセスするとき、そのようなデータアクセスは持続している(persisting)と考えられる。一方で、一回しかデータにアクセスしないなら、そのようなデータアクセスはストリーミングと考えられる。

CUDA 11.0から、compute capabilityが8.0以上のデバイスはL2キャッシュのデータの持続性に影響を与えることが出来る。これはグローバルメモリへの高い帯域幅と低レイテンシなアクセスを提供する可能性がある。

#### 持続的なアクセスのためのL2キャッシュ
L2キャッシュの一部はグローバルメモリへの持続的なデータアクセスのために取っておくことが出来る。持続的なアクセスはこの分けられたL2キャッシュの一部を優先的に使え、グローバルメモリへの通常のアクセスとストリーミングアクセスは持続的なアクセスで使われない時だけしか、このキャッシュを利用できない。

持続的なアクセスのためのL2キャッシュのサイズは制限内で調整できる。

```cpp
cudaGetDeviceProperties(&prop, device_id);                
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/ 
```

GPUがマルチインスタンスGPU(MIG)モードの場合はこの機能を利用できない。

マルチプロセスサービス(MPS)を使っている時、分けられたL2キャッシュのサイズを`cudaDeviceSetLimit`で変更できない。その代わりに、このサイズは環境変数`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`を使ってMPSサーバーの立ち上げ時にのみ指定できる。

#### 持続的なアクセスのためのL2キャッシュポリシー
アクセスポリシーウィンドウはグローバルメモリの連続した領域とその領域内へのアクセスに対するL2キャッシュの持続性を指定する。

以下のコード例はCUDAストリームを使ったL2キャッシュへの持続的なアクセスのウィンドウの設定の仕方を示す。

**CUDA Stream Example**
```cpp
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

この後にカーネルがCUDAストリームで実行しているとき、グローバルメモリの`[ptr, ptr + num_byte)`内のメモリアクセスはグローバルメモリの他の場所へのアクセスよりL2キャッシュ内で持続しやすくなる。

以下の例のように、L2持続性はCUDAグラフカーネルノードに対しても設定できる。

```cpp
cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

`hitRatio`パラメータは`accessPolicyWindow`の`num_bytes`ほどのデータを持続的アクセスのために取っておいたキャッシュに載せる割合を表す。

例えば、取っておいたL2キャッシュが16KBで、`accessPolicyWindow`の`num_bytes`が32KBとする。

- `hitRatio`が0.5の時、ハードウェアはランダムに32KBのウィンドウから16KBを選び、それらを持続すると指定し、取っておいたL2キャッシュにキャッシュする。
- `hitRatio`が1の時、ハードウェアは32KBのウィンドウ全体を取っておいたL2キャッシュにキャッシュしようとする。取っておいたキャッシュはウィンドウより小さいので、キャッシュラインは32KBのデータの内、最も最近に使われた16KBを保持するよう削除される。

あるメモリアクセスが持続的かどうかは`hitRatio`の確率でランダムで、確率分布はハードウェアアーキテクチャとそのメモリサイズに依存する。

`hitRatio`はキャッシュのスラッシングを避けるために使われる。

1.0未満の`hitRatio`は、並列なCUDAストリームの異なる`accessPolicyWindow`がL2キャッシュでキャッシュできるデータ量をコントロールするために使うことが出来る。例えば、取っておいたL2キャッシュが16KBで、2つの異なるCUDAストリームの並列なカーネルが16KBの`accessPolicyWindow`を持ち、どちらも`hitRatio`が1.0とすると、共有するL2キャッシュで競合した時にお互いのキャッシュラインを削除するかもしれない。しかし、両方の`hitRatio`が0.5なら、自分自身やお互いの持続的なキャッシュラインを消しにくくなる。

#### L2アクセスプロパティ
3種類のアクセスプロパティがグローバルメモリの異なるデータへのアクセスに対して定義されている。

| アクセスプロパティ | 効果 |
| ---- | ---- |
| `cudaAccessPropertyStreaming` | L2キャッシュに残りにくくする |
| `cudaAccessPropertyPersisting` | L2キャッシュに残りやすくする |
| `cudaAccessPropertyNormal` | 以前適用された、持続的なアクセスプロパティを通常の状態に戻す |

`cudaAccessPropertyPersisting`を持つメモリアクセスは要らなくなってもL2キャッシュに残るため、利用可能なL2キャッシュの量が減ってしまう。そのため、`cudaAccessPropertyNormal`でリセットする。

#### L2持続の例
以下の例は持続的なアクセスのためのL2キャッシュの取り方とCUDAストリームを使ったCUDAカーネルでの取っておいたL2キャッシュの使い方、そしてそのキャッシュのリセット方法を示す。

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // Create CUDA stream

cudaDeviceProp prop;                                                                        // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // This data1 is used by a kernel multiple times
}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // A different kernel in the same stream can also benefit
                                                                                            // from the persistence of data1

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode
```

#### L2アクセスを通常へリセット
リセットする方法は3つある。

1. 以前の持続的なメモリ領域を`cudaAccessPropertyNormal`アクセスプロパティでリセットする。
2. `cudaCtxResetPersistingL2Cache()`を呼び出して、全ての持続的なL2キャッシュラインをリセットする。
3. たまたま触っていないラインが自動的に通常に戻る。自動で戻るのにどれくらい時間がかかるかわからないので非推奨。

#### 取っておいたL2キャッシュの使用管理
異なるCUDAストリームで並列に実行している複数のCUDAカーネルは自身のストリームに割り当てられた、異なるアクセスポリシーウィンドウを持つかもしれない。しかし、取っておいたL2キャッシュはこれら並列で実行されている全てのCUDAカーネルに共有される。よって、持続的なアクセスの量が取っておいたL2キャッシュの容量を超えた場合、メモリアクセスを持続的と指定したメリットが消える。

取っておいたL2キャッシュの利用を管理するため、アプリケーションは以下を考えなければならない。
- 取っておくL2キャッシュのサイズ
- 並列に実行するかもしれないCUDAカーネルとそのアクセスポリシーウィンドウ
- 通常またはストリーミングなアクセスが以前取っておいたL2キャッシュを同じ優先度で使えるように、いつ、どのようなL2リセットが必要か。

#### L2キャッシュプロパティのクエリ
L2キャッシュに関連したプロパティは`cudaDeviceProp`構造体の一部にあり、`cudaGetDeviceProperties`関数で取得できる。

CUDAデバイスプロパティは以下を含む。

| プロパティ名 | 意味 |
| ---- | ---- |
| `l2CacheSize` | GPU上で利用可能なL2キャッシュの容量 |
| `persistingL2CacheMaxSize` | 持続的なメモリアクセスのために取っておけるL2キャッシュの最大容量 |
| `accessPolicyMaxWindowSize` | アクセスポリシーウィンドウの最大サイズ |

#### 持続的なメモリアクセスのために取っておくL2キャッシュのサイズの管理
持続的なメモリアクセスのために取っておくL2キャッシュのサイズは`cudaDeviceGetLimit`関数を使って取得でき、`cudaDeviceSetLimit`関数に[cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651)として渡すことで設定できる。最大値は`cudaDeviceProp::persistingL2CacheMaxSize`。

```cpp
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};
```

### 共有メモリ
[メモリ空間指定子](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers)で詳しく述べるが、共有メモリは`__shared__`メモリ空間指定子を使って確保される。

共有メモリはグローバルメモリよりかなり早いとされ、CUDAブロックによるグローバルメモリへのアクセスを最小化するためのスクラッチパッドメモリとして使える。

以下のコード例は共有メモリを活用しない行列積の実装である。各スレッドはAのある行とBのある列を読んで、対応するCの要素を計算する。以下の図のように、Aの各行はBの列数だけグローバルメモリから読まれ、Bの各行もAの行数だけ読まれる。

```cpp
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
```

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png)
*共有メモリを使わない行列積*

以下のコード例は共有メモリを活用した行列積の実装である。この実装では、各スレッドブロックはCのある正方部分行列Csubを計算し、そのブロックの各スレッドはCsubの各要素を計算する。以下の図のように、Csubは次元が(ブロックサイズ, Aの列数)のAの部分行列と次元が(Bの行数, ブロックサイズ)のBの部分行列の積で計算できるので、この2つの部分行列をグローバルメモリから共有メモリにロードすればAの各行は(Bの列数) / (ブロックサイズ)だけ、Bの各行は(Aの行数) / (ブロックサイズ)だけ読めばよい。

```cpp
// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png)
*共有メモリを使った行列積*

### 分散共有メモリ
Compute capability 9.0で導入されたスレッドブロッククラスターによって、スレッドブロッククラスター内のスレッドはクラスター内の全てのスレッドブロックの共有メモリにアクセス出来るようになった。この分割された共有メモリを分散共有メモリと呼び、対応するアドレス空間を分散共有メモリアドレス空間と呼ぶ。スレッドブロッククラスター内のスレッドは分散共有メモリのアドレス空間が自身が属するスレッドブロックか他のスレッドブロックかに関わらず、そのメモリへの読み書きと不可分操作を行うことが出来る。カーネルが分散共有メモリを使うかどうかに関わらず、共有メモリのサイズ指定(静的か動的か)は未だスレッドブロック毎である。分散共有メモリのサイズは(クラスター毎のスレッドブロックの数) * (スレッドブロック毎の共有メモリのサイズ)である。

分散共有メモリ内のデータへアクセスするには、全てのスレッドブロックがなければならない。ユーザーは[クラスターグループ](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg)APIの`cluster.sync()`を使うと、全てのスレッドブロックが実行を開始したことを保証できる。また、ユーザーは全ての分散共有メモリへの操作がスレッドブロックが終わる前に行われることも保証しなければならない。

スレッドブロッククラスターを使った、GPU上での単純なヒストグラムの計算の最適化の仕方を見てみよう。ヒストグラムの計算の標準的な方法は各スレッドブロックの共有メモリで計算し、グローバルメモリで不可分操作を行う方法である。この方法の限界は共有メモリの容量である。ヒストグラムのビン(縦棒)が共有メモリに収まらなくなると、ユーザーはグローバルメモリ上で不可分操作をしながら直接ヒストグラムを計算する必要がある。分散共有メモリを使うと、ヒストグラムのビンの数によるが、ヒストグラムを共有メモリや分散共有メモリ、グローバルメモリで計算できる。

以下のCUDAカーネルはヒストグラムのビンの数によって、共有メモリか分散共有メモリでヒストグラムを計算するやり方を示している。

```cpp
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(
    int *bins,
    const int nbins,
    const int bins_per_block,
    const int *__restrict__ input,
    size_t array_size
) {
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //Initialize shared memory histogram to zeros
  }

  // cluster synchronization ensures that shared memory is initialized to zero in
  // all thread blocks in the cluster. It also ensures that all thread blocks
  // have started executing and they exist concurrently.
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    //Find the right histogram bin.
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    //Find destination block rank and offset for computing
    //distributed shared memory histogram
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    //Pointer to target block shared memory
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    //Perform atomic update of the histogram bin
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // cluster synchronization is required to ensure all distributed shared
  // memory operations are completed and no thread block exits while
  // other thread blocks are still accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

上のカーネルは必要な分散共有メモリの総量に応じたクラスターサイズを持って実行時に開始できる。もしヒストグラムが1ブロック分の共有メモリに収まるほど十分小さいなら、ユーザーはクラスター数を1にしてカーネルを開始できる。以下のコードは必要な共有メモリ数に応じて動的にクラスターカーネルを開始する方法を示す。

```cpp
// Launch via extensible launch
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // cluster_size depends on the histogram size.
  // ( cluster_size == 1 ) implies no distributed shared memory,
  // just thread block local shared memory
  int cluster_size = 2; // size 2 is an example here
  int nbins_per_block = nbins / cluster_size;

  //dynamic shared memory size is per block.
  //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute(
    (void *)clusterHist_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    config.dynamicSmemBytes
  ));

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(
    &config,
    clusterHist_kernel,
    bins,
    nbins,
    nbins_per_block,
    input,
    array_size
  );
}
```

### ページロックされたホストメモリ
CUDAランタイムは、(`malloc()`によって確保された、普通のページング可能なホストメモリとは反対に、)ページロックされた(またはピンされた)ホストメモリを使うことができる関数を提供している。例えば、

- `cudaHostAlloc()`と`cudaFreeHost()`はページロックされたホストメモリの確保と解放を行う。
- `cudaHostRegister()`は`malloc()`によって確保された、ある範囲のメモリをページロックする。(ただし、制限がある模様。詳しくはリファレンスマニュアルを参照。)

ページロックされたホストメモリの使用にはいくつか利点がある。

- ページロックされたホストメモリとデバイスメモリ間のコピーを[非同期並列実行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)で述べるように、デバイスに対するカーネルの実行と並列に行うことができる。
- あるデバイスでは、ページロックされたホストメモリはデバイスのアドレス空間にマップされうる。これによってデバイスメモリとのコピーを無くすことができる。詳しくは[マップされたメモリ](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory)を参照。
- 


### 非同期並列実行

### マルチデバイスシステム

### エラーチェック

### コールスタック

### テクスチャとサーフェスメモリ

### グラフィックとの相互運用性
