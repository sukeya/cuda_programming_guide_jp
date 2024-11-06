# デバイスメモリ
CUDAのプログラムモデルは各々のメモリを持つホストとデバイスからなるシステムを仮定している。
カーネルはデバイスメモリの外から操作するので、ランタイムはホストメモリとデバイスメモリ間のデータ転送を行う関数だけでなく、デバイスメモリの確保、解放、デバイスメモリ内のコピーを行う関数も提供している。

デバイスメモリは線形メモリ(linear memory)かCUDA配列(CUDA array)のどちらかとして確保される。

CUDA配列はテクスチャフェッチのために最適化された不透明なメモリレイアウトである。
詳しくは[テクスチャとサーフェスメモリー](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)にて。

線形メモリは一つの統一されたメモリ空間にアロケートされる。
つまり、例えば2分木やリンクリストで、別々にアロケーションされたものをポインターを通してお互いに参照することが出来る。
アドレス空間の大きさはホストシステム(CPU)と、使っているGPUのcompute capabilityに依存する。

| compute capability | x86_64(AMD64) | ARM64 |
| ---- | ---- | ---- |
| >= 6.0 (Pascal) | up to 47bit | up to 48bit |
| <= 5.3 (Maxwell) | 40bit | 40bit |

*線型メモリのアドレス空間*

線形メモリは一般的に`cudaMalloc()`を使って確保され、`cudaFree()`で解放される。
ホストメモリとデバイスメモリ間のデータ転送は`cudaMemcpy()`を用いて行われることが多い。
ベクトル和のコードサンプルでは、ベクトルはホストメモリからデバイスメモリへコピーされなければならない。

```cpp title="/src/programming_interface/device_memory/add_2_vectors.cu" linenums="1"
--8<-- "./src/programming_interface/device_memory/add_2_vectors.cu"
```

線形メモリは[cudaMallocPitch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)と[cudaMalloc3D](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g188300e599ded65c925e79eab2a57347)を通しても確保出来る。
これらの関数は2次元または3次元配列のメモリ確保がアライメントの要件に合うように適切にパディングされるようにしたいときにおすすめ。
詳しくは[デバイスメモリへのアクセス](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)にて。
返されたピッチ(pitch)(またはストライド(stride))は配列要素にアクセスするように使わなければならない。
以下のコードサンプルでは、`column` * `row`の2次元配列をアロケートし、デバイスコード内で配列上をループする方法を示す。

```cpp title="/src/programming_interface/device_memory/add_2_matrices.cu" linenums="3"
--8<-- "./src/programming_interface/device_memory/add_2_matrices.cu:3:35"
```

以下のコードは、要素数が`width` * `height` * `depth`の3次元配列を確保するコードである。

```cpp title="/src/programming_interface/device_memory/add_2_tensors.cu" linenums="3"
--8<-- "./src/programming_interface/device_memory/add_2_tensors.cu:3:39"
```

リファレンスマニュアルには、`cudaMalloc()`で確保された線形メモリや`cudaMallocPitch()`や`cudaMalloc3D()`で確保された線形メモリ、CUDA配列、グローバルまたは定数メモリ空間で宣言された変数に対して確保されたメモリ間のコピーに使われる関数が色々ある。

以下のコードは、ランタイムAPIを使った、グローバル変数へアクセスする様々な方法を示す。

```cpp title="/src/programming_interface/device_memory/global_memory.cu" linenums="1"
--8<-- "./src/programming_interface/device_memory/global_memory.cu:1:20"
```

`cudaGetSymbolAddress()`はグローバルメモリ空間で宣言された変数に対して確保されたメモリを指すアドレスを取り出すために使われる。
例は[Understanding of CUDA's cudaGetSymbolAddress](https://stackoverflow.com/questions/60759486/understanding-of-cudas-cudagetsymboladdress)にて。
確保されたメモリのサイズは`cudaGetSymbolSize()`で得られる。
