# デバイスメモリのL2アクセス管理
CUDAカーネルがグローバルメモリ内のデータ領域に繰り返しアクセスするとき、そのようなデータアクセスは持続している(persisting)と考えられる。一方で、一回しかデータにアクセスしないなら、そのようなデータアクセスはストリーミングと考えられる。

CUDA 11.0から、compute capabilityが8.0以上のデバイスはL2キャッシュのデータの持続性に影響を与えることが出来る。これはグローバルメモリへの高い帯域幅と低レイテンシなアクセスを提供する可能性がある。

## 持続的なアクセスのためのL2キャッシュ
L2キャッシュの一部はグローバルメモリへの持続的なデータアクセスのために取っておくことが出来る。持続的なアクセスはこの分けられたL2キャッシュの一部を優先的に使え、グローバルメモリへの通常のアクセスとストリーミングアクセスは持続的なアクセスで使われない時だけしか、このキャッシュを利用できない。

持続的なアクセスのためのL2キャッシュのサイズは制限内で調整できる。

```cpp
cudaGetDeviceProperties(&prop, device_id);                
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/ 
```

GPUがマルチインスタンスGPU(MIG)モードの場合はこの機能を利用できない。

マルチプロセスサービス(MPS)を使っている時、分けられたL2キャッシュのサイズを`cudaDeviceSetLimit`で変更できない。その代わりに、このサイズは環境変数`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`を使ってMPSサーバーの立ち上げ時にのみ指定できる。

## 持続的なアクセスのためのL2キャッシュポリシー
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

## L2アクセスプロパティ
3種類のアクセスプロパティがグローバルメモリの異なるデータへのアクセスに対して定義されている。

| アクセスプロパティ | 効果 |
| ---- | ---- |
| `cudaAccessPropertyStreaming` | L2キャッシュに残りにくくする |
| `cudaAccessPropertyPersisting` | L2キャッシュに残りやすくする |
| `cudaAccessPropertyNormal` | 以前適用された、持続的なアクセスプロパティを通常の状態に戻す |

`cudaAccessPropertyPersisting`を持つメモリアクセスは要らなくなってもL2キャッシュに残るため、利用可能なL2キャッシュの量が減ってしまう。そのため、`cudaAccessPropertyNormal`でリセットする。

## L2持続の例
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

## L2アクセスを通常へリセット
リセットする方法は3つある。

1. 以前の持続的なメモリ領域を`cudaAccessPropertyNormal`アクセスプロパティでリセットする。
2. `cudaCtxResetPersistingL2Cache()`を呼び出して、全ての持続的なL2キャッシュラインをリセットする。
3. たまたま触っていないラインが自動的に通常に戻る。自動で戻るのにどれくらい時間がかかるかわからないので非推奨。

## 取っておいたL2キャッシュの使用管理
異なるCUDAストリームで並列に実行している複数のCUDAカーネルは自身のストリームに割り当てられた、異なるアクセスポリシーウィンドウを持つかもしれない。しかし、取っておいたL2キャッシュはこれら並列で実行されている全てのCUDAカーネルに共有される。よって、持続的なアクセスの量が取っておいたL2キャッシュの容量を超えた場合、メモリアクセスを持続的と指定したメリットが消える。

取っておいたL2キャッシュの利用を管理するため、アプリケーションは以下を考えなければならない。
- 取っておくL2キャッシュのサイズ
- 並列に実行するかもしれないCUDAカーネルとそのアクセスポリシーウィンドウ
- 通常またはストリーミングなアクセスが以前取っておいたL2キャッシュを同じ優先度で使えるように、いつ、どのようなL2リセットが必要か。

## L2キャッシュプロパティのクエリ
L2キャッシュに関連したプロパティは`cudaDeviceProp`構造体の一部にあり、`cudaGetDeviceProperties`関数で取得できる。

CUDAデバイスプロパティは以下を含む。

| プロパティ名 | 意味 |
| ---- | ---- |
| `l2CacheSize` | GPU上で利用可能なL2キャッシュの容量 |
| `persistingL2CacheMaxSize` | 持続的なメモリアクセスのために取っておけるL2キャッシュの最大容量 |
| `accessPolicyMaxWindowSize` | アクセスポリシーウィンドウの最大サイズ |

## 持続的なメモリアクセスのために取っておくL2キャッシュのサイズの管理
持続的なメモリアクセスのために取っておくL2キャッシュのサイズは`cudaDeviceGetLimit`関数を使って取得でき、`cudaDeviceSetLimit`関数に[cudaLimit](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4c4b34c054d383b0e9a63ab0ffc93651)として渡すことで設定できる。最大値は`cudaDeviceProp::persistingL2CacheMaxSize`。

```cpp
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};
```

