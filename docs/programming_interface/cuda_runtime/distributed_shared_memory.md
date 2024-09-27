# 分散共有メモリ
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
