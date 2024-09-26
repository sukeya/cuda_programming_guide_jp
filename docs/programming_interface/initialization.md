### 初期化
CUDA 12.0から、[cudaInitDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gac04a5d82168676b20121ca870919419)関数と[cudaSetDevice](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb)関数は指定されたデバイスに紐付いた、ランタイムと[プライマリーコンテキスト](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)を初期化する。これらの関数呼び出しがないと、他のランタイムAPIリクエストを処理する必要がある時にランタイムは暗にデバイス`0`を使い、自身を初期化する。ランタイムの関数呼び出しの時間計測とランタイムへの最初の関数呼び出しからのエラーコードを解釈するときに心に留めておく必要がある。12.0以前では、`cudaSetDevice`関数はランタイムを初期化せず、アプリケーションは(時間計測とエラーハンドリングのために)他のAPI呼び出しからランタイムの初期化を切り離す、何もしない関数呼び出し`cudaFree(0)`をよく使っていた。

compute capabilityが2.0以上のデバイスの数は[cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f)関数でわかる。デバイス番号は0から始まり、デバイス数未満までである。

プライマリーコンテキストは、確保されたメモリのリストやデバイスコードを含む、ロードされたモジュールなどデバイスを制御するデータ全てを持つ[^1]。プライマリーコンテキストはデバイス毎に1つしかなく、CUDAランタイムAPIと共有される。つまり、アプリケーションのすべてのホストスレッドで共有される。このコンテキストを作成する時、必要ならデバイスコードをjust-in-timeコンパイルする。詳しくは、[コンテキスト](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)にて。

ホストスレッドが[cudaDeviceReset](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217)関数を呼び出した場合、ホストスレッドが現在操作しているデバイスのプライマリーコンテキストを破棄する。

!!! note "注"

    CUDAインターフェイスはホストプログラムの開始時に初期化され、ホストプログラムの終了時に破棄されるグローバルな状態を持つ。CUDAランタイムとドライバーはこの状態が無効かどうかを検出できないので、明示的か暗黙的かに関わらず、プログラムの開始時と終了時にこれらのインターフェイスを使うと未定義動作になる。
    CUDA 12.0時点では、`cudaSetDevice()`はホストスレッドの現在のデバイスを変更後にランタイムを明示的に初期化する。以前のバージョンのCUDAでは、`cudaSetDevice()`が呼び出されてから初めてランタイムの関数が呼ばれるまで、新しいデバイス上のランタイムの初期化を遅延していた。この変更により、初期化のエラーを`cudaSetDevice()`の戻り値で確認することがとても重要になった。
    エラー処理とバージョン管理のランタイム関数はランタイムを初期化しない。


[^1]: [What is a CUDA context?](https://stackoverflow.com/questions/43244645/what-is-a-cuda-context)
