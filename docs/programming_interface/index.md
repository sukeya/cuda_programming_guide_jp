# プログラミングインターフェイス
CUDAランタイムは低級なC APIであるCUDAドライバーAPIの上に作られている。CUDAドライバーAPIもアプリケーションから利用できる。このAPIは、

- CUDAコンテキスト - ホストプロセスのデバイス版
- CUDAモジュール - 動的リンクライブラリのデバイス版

といった低級な概念を提供する。詳しくは[ドライバーAPI](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)にて。

