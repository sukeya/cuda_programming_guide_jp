#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

/*
 * 分散共有メモリを用いてヒストグラムを計算するカーネル
 *
 * 方針:
 * 全てのビンを各スレッドブロックで分けて共有メモリに保持し、計算が全部終わったらグローバルメモリに書き込む。
 * 入力はスレッド数でmodを取り、自身のスレッドIDの同値類のそれぞれの値を配列のインデックスと見てビンに足す。
*/
__global__ void clusterHist_kernel(
    int *bins,
    const int nbins,
    const int bins_per_block,
    const int *__restrict__ input,
    size_t array_size
) {
  // 共有メモリを動的に確保するため、extern宣言する
  extern __shared__ int smem[];

  namespace cg = cooperative_groups;

  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
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
    if (ldata < 0) {
      binid = 0;
    } else if (ldata >= nbins) {
      binid = nbins - 1;
    }

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
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&bins[i], smem[i]);
  }
}


int main() {
  constexpr int array_size = 64;
  constexpr int threads_per_block = 16;
  constexpr int nbins = 16;

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

  // 動的共有メモリの最大サイズを変更
  auto e = cudaFuncSetAttribute(
    (void *)clusterHist_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    config.dynamicSmemBytes
  );
  assert(e == cudaSuccess);

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  int* dev_bins;

  e = cudaMalloc(&dev_bins, nbins * sizeof(int));
  assert(e == cudaSuccess);

  std::vector<int> input;
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());

  // 範囲外のチェックをしているか確認するために、わざと範囲外を生成するように指定
  std::uniform_int_distribution<int> dist(-1, nbins);

  for (std::size_t i = 0; i < array_size; ++i) {
    input.push_back(dist(engine));
  }

  int* dev_input;
  e = cudaMalloc(&dev_input, input.size() * sizeof(int));
  assert(e == cudaSuccess);

  e = cudaMemcpy(dev_input, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);
  assert(e == cudaSuccess);

  e = cudaLaunchKernelEx(
    &config,
    clusterHist_kernel,
    dev_bins,
    nbins,
    nbins_per_block,
    dev_input,
    array_size
  );
  assert(e == cudaSuccess);

  std::unique_ptr<int[]> bins = std::make_unique<int[]>(nbins);

  e = cudaMemcpy(bins.get(), dev_bins, nbins * sizeof(int), cudaMemcpyDeviceToHost);
  assert(e == cudaSuccess);

  std::vector<int> answer;
  answer.resize(nbins, 0);
  for (const auto i : input) {
    if (i < 0) {
      ++answer.front();
    } else if (i >= array_size) {
      ++answer.back();
    } else {
      ++answer[i];
    }
  }

  for (int i = 0; i < nbins; ++i) {
    assert(answer[i] == bins[i]);
  }

  e = cudaFree(dev_input);
  assert(e == cudaSuccess);
  e = cudaFree(dev_bins);
  assert(e == cudaSuccess);
}
