

这个仓库被我用作 [data-calibrator](https://github.com/BrevityD/data-calibrator) 的 submodule。

原始[README](./README.old.md)，这个文档用于记录我个人复现的操作：

# 数据

使用 `data-calibrator/datasets/download_data.py` 下载，下载了如下的数据集（默认 `./` 目录是 `data-calibrator` 项目目录：

```json
{
    "allenai/tulu-3-sft-personas-code": "./datasets/code_domain/tulu-3-sft-personas-code",
    "allenai/tulu-3-sft-personas-algebra": "./datasets/math_domain/tulu-3-sft-personas-algebra",
    "allenai/tulu-3-sft-personas-instruction-following": "./datasets/general_domain/tulu-3-sft-personas-instruction-following",
    "allenai/tulu-3-sft-personas-math-grade-filtered": "./datasets/math_domain/tulu-3-sft-personas-math-grade-filtered",
    "allenai/tulu-3-sft-personas-math-filtered": "./datasets/math_domain/tulu-3-sft-personas-math-filtered",
}
```

处理数据就直接使用了 [TULU3-load脚本](./data_preprocess/tulu3-sft/load_data.py)，对此进行了一些修改，主要是考虑这么多的数据一次处理压力会很大，也容易出错丢数据，所以每处理一些就增量保存一次，如果中断需要手动添加末尾：

```shell
$ sed -i '$ c}\n]' xxx.json
```

采样数据用了 `./data_preprocess/data_mixing/sample_data.py`，唯一的考虑是200M token的数据集大小太夸张了，我选择的TULU的这些数据集很多量不太够，训练压力也比较大，所以减成了20M token，且加了个上采样（重复采样）凑token。最多的code重复了四遍，除此之外只有 IF 几乎重复了两遍，此外都没什么重复采样的情况。

用于估计几个参数的数据用了更少的2M base，也就是差不多6k条数据。

# 训练

我用uv管理，不用conda，并且是裸金属连服务器运行，不需要设置集群环境变量。再加上仓库原作者对训练参数的部分设置让我产生疑虑
（比如 `exp2` 中的普通实验都是 linear 调度器 + gradient_steps=2，
而标记了 `-optim` 的实验都是 cosine 调度器 + gradient_steps=4），
所以我自己实现了训练脚本（都在 `./commands/reprod_bd` 中）。

```shell-session
foo@bar:~/data-calibrator/samples/experiment-reprod-DMO$ . .venv/bin/activate
foo@bar:~/data-calibrator/samples/experiment-reprod-DMO$ cd Data-Mixing-Optimization/
foo@bar:~/data-calibrator/samples/experiment-reprod-DMO/Data-Mixing-Optimization$ CUDA_VISIBLE_DEVICES=0,2,5,6 python commands/reprod_bd/train_sampling.py --template_yaml commands/reprod_bd/sft_full.yaml --base_token 2000000 --model_name Qwen3-1.7B-Base
```

4*H100，单卡batch size设置为4（还可以更大，全程显存占用小于40G，懒得调了，反正挺快的），大约十分钟能训三轮。

顺带提一下，仓库原作者给出的内嵌llamafactory方案是无法运行的（`src/train.py` 需要导入 data template 相关的模块，而原作者并没有嵌入）。
因此我把 `Llama-Factory` 作为 submodule 引入，跑起来了。
