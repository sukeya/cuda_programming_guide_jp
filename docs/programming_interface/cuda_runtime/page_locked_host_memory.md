# ページロックされたホストメモリ
CUDAランタイムは、(`malloc()`によって確保された、普通のページング可能なホストメモリとは反対に、)ページロックされた(またはピンされた)ホストメモリを使うことができる関数を提供している。例えば、

- `cudaHostAlloc()`と`cudaFreeHost()`はページロックされたホストメモリの確保と解放を行う。
- `cudaHostRegister()`は`malloc()`によって確保された、ある範囲のメモリをページロックする。(ただし、制限がある模様。詳しくはリファレンスマニュアルを参照。)

ページロックされたホストメモリの使用にはいくつか利点がある。

- ページロックされたホストメモリとデバイスメモリ間のコピーを[非同期並列実行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)で述べるように、デバイスに対するカーネルの実行と並列に行うことができる。
- あるデバイスでは、ページロックされたホストメモリはデバイスのアドレス空間にマップされうる。これによってデバイスメモリとのコピーを無くすことができる。詳しくは[マップされたメモリ](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory)を参照。
