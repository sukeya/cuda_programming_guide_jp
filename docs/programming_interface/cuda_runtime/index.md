# CUDAランタイム
ランタイムは`cudart`ライブラリに実装されていて、`cudart.lib`または`libcudart.a`を使って静的にリンクするか、`cudart.dll`または`libcudart.so`を使って動的にリンクする。

`cudart`の全てのエントリーポイントは`cuda`から始まる。

[デバイスメモリ](./device_memory.md)では、デバイスメモリの管理に使われる関数の概要を述べる。

[共有メモリ]()では、性能を最大化するために共有メモリの使い方を述べる。

[ページロックドホストメモリ]()では、ホストとデバイス間のデータ転送を行うカーネルの実行をオーバーラップするために使われるページロックドホストメモリを紹介する。

[非同期並列実行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)では、コンセプトといろいろなレベルでの非同期並列実行を行うためのAPIを述べる。

[マルチデバイスシステム](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)では、プログラミングモデルが同じホストに付けられた複数のデバイスを持つシステムにどのように拡張されるかを述べる。

[エラーチェック](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking)では、ランタイムによって発生したエラーを適切に確認する方法を述べる。

[コールスタック](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#call-stack)では、CUDA C++コールスタックを管理するために使われる関数を述べる。

[テクスチャとサーフェスメモリ](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)では、デバイスメモリにアクセスする別の方法を提供するテクスチャとサーフェスメモリ空間について述べる。これらはGPUのテクスチャハードウェアのサブセットも提供する。

[グラフィックとの相互運用性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability)では、ランタイムがOpenGLとDirect3Dと相互運用するために提供する、さまざまな関数を紹介する。
