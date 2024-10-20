# 共有メモリ
[メモリ空間指定子](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers)で詳しく述べるが、共有メモリは`__shared__`メモリ空間指定子を使って確保される。

共有メモリはグローバルメモリよりかなり早いとされ、CUDAブロックによるグローバルメモリへのアクセスを最小化するためのスクラッチパッドメモリとして使える。

以下のコード例は共有メモリを活用しない行列積の実装である。各スレッドはAのある行とBのある列を読んで、対応するCの要素を計算する。以下の図のように、Aの各行はBの列数だけグローバルメモリから読まれ、Bの各行もAの行数だけ読まれる。

```cpp title="/src/programming_interface/shared_memory/not_using_shared_memory.cu" linenums="1"
--8<-- "./src/programming_interface/shared_memory/not_using_shared_memory.cu:7:70"
```

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png)
*共有メモリを使わない行列積*

以下のコード例は共有メモリを活用した行列積の実装である。この実装では、各スレッドブロックはCのある正方部分行列Csubを計算し、そのブロックの各スレッドはCsubの各要素を計算する。以下の図のように、Csubは次元が(ブロックサイズ, Aの列数)のAの部分行列と次元が(Bの行数, ブロックサイズ)のBの部分行列の積で計算できるので、この2つの部分行列をグローバルメモリから共有メモリにロードすればAの各行は(Bの列数) / (ブロックサイズ)だけ、Bの各行は(Aの行数) / (ブロックサイズ)だけ読めばよい。

```cpp title="/src/programming_interface/shared_memory/using_shared_memory.cu" linenums="1"
--8<-- "./src/programming_interface/shared_memory/using_shared_memory.cu:7:137"
```

![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png)
*共有メモリを使った行列積*
