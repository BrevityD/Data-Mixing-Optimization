

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
所以我自己实现了训练脚本（都在 `./commands/reprod_bd` 中，包括train_sampling训扫点的baseline，train_calibrator训用于标定参数的ckpt）。

```shell-session
foo@bar:~/data-calibrator/samples/experiment-reprod-DMO$ . .venv/bin/activate
foo@bar:~/data-calibrator/samples/experiment-reprod-DMO$ cd Data-Mixing-Optimization/
foo@bar:~/data-calibrator/samples/experiment-reprod-DMO/Data-Mixing-Optimization$ CUDA_VISIBLE_DEVICES=0,2,5,6 python commands/reprod_bd/train_sampling.py --template_yaml commands/reprod_bd/sft_full.yaml --base_token 2000000 --model_name Qwen3-1.7B-Base
```

4*H100，单卡batch size设置为4（还可以更大，全程显存占用小于40G，懒得调了，反正挺快的），大约十分钟能训三轮。

顺带提一下，仓库原作者给出的内嵌llamafactory方案是无法运行的（`src/train.py` 需要导入 data template 相关的模块，而原作者并没有嵌入）。
因此我把 `Llama-Factory` 作为 submodule 引入，跑起来了。

跑完了，并且进行了批量的PPL计算和统计，PPL计算放在 `./commands/reprod_bd/ppl_runner.py`

但此时我发现一个问题：我对原文的理解，Li（或者PPL）是针对每个领域i单独的测试集而言的，
但观看作者实现的 `./domain_weights/beta_calibration.py` 不难发现，
实际上的PPL是针对同一个测试集的（所有PPL差距很小，所有1111配比的PPL都一模一样）。

有点疑惑，接着用 `./domain_weights/beta_calibration.py` 做做看。

以及这个地方对参数比较敏感，尤其是取的A上界，会严重影响最终结果。
（下面的domain1是math-grade，domain2是insfo，domain3是code，domain4是algebra，domain5是math）

a_upper_ratio: 5.0
=== Running beta calibration for reprod ===
[Domain 1] Beta=0.0030, A=0.0007, Loss=0.000003
[Domain 2] Beta=0.0067, A=0.1237, Loss=0.000001
[Domain 3] Beta=0.0314, A=0.0133, Loss=0.000024
[Domain 4] Beta=0.0043, A=1.0966, Loss=0.000003
[Domain 5] Beta=0.0020, A=0.2483, Loss=0.000006

a_upper_ratio: 1.0
=== Running beta calibration for reprod ===
[Domain 1] Beta=0.0031, A=0.0012, Loss=0.000003
[Domain 2] Beta=0.0067, A=0.1197, Loss=0.000001
[Domain 3] Beta=0.0307, A=0.0124, Loss=0.000024
[Domain 4] Beta=0.0038, A=0.9979, Loss=0.000003
[Domain 5] Beta=0.0023, A=0.3158, Loss=0.000006


a_upper_ratio: 0.8

=== Running beta calibration for reprod ===
[Domain 1] Beta=0.0025, A=0.0030, Loss=0.000031
[Domain 2] Beta=0.0066, A=0.1114, Loss=0.000006
[Domain 3] Beta=0.0302, A=0.0124, Loss=0.000024
[Domain 4] Beta=0.0040, A=1.0453, Loss=0.000029
[Domain 5] Beta=0.0018, A=0.2256, Loss=0.000059

考虑到A上界的可能取值，这里先按照0.8继续处理。

继续跑 `./domain_weights/param_calibration.py`，会报错，报错问题出在 residuals 是一个数组，而huber loss实现是按照float实现的

--- Summary ---
llama-3.2-3b:
{'beta': 0.051, 'c': 1.1684086283497737, 'gamma': 0.2028506624087608, 'alpha': 0.4835430572187707, 'e': 1.0850955259279713, 'loss': 5.27289724525855e-06}
{'beta': 0.043, 'c': 1.0400851550373114, 'gamma': 0.30771406857207306, 'alpha': 0.48615169878903874, 'e': 1.21897530701965, 'loss': 3.1662223580762807e-06}
{'beta': 0.0439, 'c': 1.1943096075416968, 'gamma': 0.31982181823990985, 'alpha': 0.48608269597594067, 'e': 1.0672524326328172, 'loss': 2.616888091466727e-06}
orca:
{'beta': 0.0663, 'c': 0.2862807935222693, 'gamma': 0.002847823167745079, 'alpha': 0.2393850912392418, 'e': 1.7022100069734682, 'loss': 7.293530876850732e-06}
{'beta': 0.0583, 'c': 1.000064063739959, 'gamma': 1.1967188479138744, 'alpha': 0.586790339867987, 'e': 1.0419043858508832, 'loss': 1.6144187881281025e-05}
{'beta': 0.0907, 'c': 0.21092234330219178, 'gamma': 0.00012156143964921182, 'alpha': 0.3842510890342751, 'e': 1.7793320126218384, 'loss': 1.5672137841258513e-05}
reprod:
{'beta': 0.0025, 'c': 0.6438822970090471, 'gamma': 0.04298530162957, 'alpha': 0.2977915700418385, 'e': 0.8415645944959887, 'loss': 2.610253979330633e-08}
{'beta': 0.0066, 'c': 1.0049334248113198, 'gamma': 0.07784550996776536, 'alpha': 0.23974814961733348, 'e': 0.484284137477415, 'loss': 6.504852319686068e-09}
{'beta': 0.0302, 'c': 0.11491131047305071, 'gamma': 0.002084546537675324, 'alpha': 0.28382230031041555, 'e': 1.3723727842828402, 'loss': 1.5935130213300665e-06}
{'beta': 0.004, 'c': 0.9441718507018281, 'gamma': 0.38347775747630475, 'alpha': 0.4905550013695846, 'e': 0.5444781808424246, 'loss': 4.4661569559995846e-08}
{'beta': 0.0018, 'c': 1.163021071515249, 'gamma': 0.18555491737003274, 'alpha': 0.31375376650764425, 'e': 0.3229289039292824, 'loss': 8.585062590385621e-08}


最终跑出来的结果（我用代码里的原始数据拟合出来的参数不一样，没改什么，标记成mine了）：

=== Optimizing weights for llama-3.2-3b ===
N=   5: weights=[0.41164194 0.26835549 0.32000257], objective=6.660530
N=  10: weights=[0.41683454 0.2616712  0.32149426], objective=6.576103
N=  20: weights=[0.42004145 0.25726491 0.32269364], objective=6.492995
N=  50: weights=[0.42229458 0.25371654 0.32398887], objective=6.385765
N= 100: weights=[0.42300334 0.25216866 0.324828  ], objective=6.306931
N= 200: weights=[0.42313732 0.25127053 0.32559215], objective=6.230204
N= 500: weights=[0.42270761 0.25075929 0.3265331 ], objective=6.132112

=== Optimizing weights for orca ===
N=   5: weights=[0.51288044 0.25065322 0.23646634], objective=6.731690
N=  10: weights=[0.48729187 0.27459122 0.23811691], objective=6.614433
N=  20: weights=[0.47629393 0.28764629 0.23605978], objective=6.502949
N=  50: weights=[0.47473223 0.29257227 0.2326955 ], objective=6.365483
N= 100: weights=[0.47884906 0.29387499 0.22727595], objective=6.269664
N= 200: weights=[0.48666732 0.29281987 0.22051282], objective=6.181215
N= 500: weights=[0.50097326 0.28898854 0.2100382 ], objective=6.075645

=== Optimizing weights for llama-3.2-3b-mine ===
N=   5: weights=[0.42941187 0.2562098  0.31437833], objective=6.654115
N=  10: weights=[0.4113701  0.26667021 0.32195969], objective=6.561578
N=  20: weights=[0.39764257 0.27506418 0.32729324], objective=6.469035
N=  50: weights=[0.38748144 0.28078845 0.33173011], objective=6.347895
N= 100: weights=[0.38208706 0.28386077 0.33405216], objective=6.257857
N= 200: weights=[0.37800244 0.28619672 0.33580083], objective=6.169622
N= 500: weights=[0.37394673 0.28851607 0.3375372 ], objective=6.056162

=== Optimizing weights for orca-mine ===
N=   5: weights=[0.4050532  0.19169279 0.40325401], objective=5.921527
N=  10: weights=[0.34105151 0.32591727 0.33303122], objective=5.869820
N=  20: weights=[0.3015087  0.4110261  0.28746519], objective=5.817528
N=  50: weights=[0.26563726 0.48250616 0.25185658], objective=5.749042
N= 100: weights=[0.24846266 0.51890014 0.23263721], objective=5.698446
N= 200: weights=[0.2363312  0.54534324 0.21832556], objective=5.649268
N= 500: weights=[0.22548318 0.57016764 0.20434918], objective=5.586741

=== Optimizing weights for reprod ===
N=   5: weights=[0.10481775 0.45889574 0.25356327 0.09873307 0.08399017], objective=7.430506
N=  10: weights=[0.10189011 0.43255174 0.23145039 0.13391852 0.10018923], objective=7.419636
N=  20: weights=[0.09972289 0.4153476  0.21653926 0.15902393 0.10936631], objective=7.408455
N=  50: weights=[0.0983949  0.39980909 0.2032305  0.18198346 0.11658206], objective=7.393385
N= 100: weights=[0.09325609 0.38904802 0.1992424  0.19707436 0.12137913], objective=7.381873
N= 200: weights=[0.09321377 0.38510639 0.19360857 0.20517845 0.12289281], objective=7.370319
N= 500: weights=[0.09208204 0.38268149 0.18790563 0.213406   0.12392484], objective=7.355048


最接近的扫点配比是：

insfo 0.38268149
math 0.12392484
code 0.18790563
algebra 0.213406
math-grade 0.09208204

即：./saves/exp2_grid/Qwen3-1.7B-Base_2000000/insfo0.375_math0.125_code0.125_algebra0.25_math-grade0.125

那么这个ckpt的效果是：epoch2效果最好，eval loss 0.4223394989967346

>>> scores.sort()
>>> scores
[0.4192710518836975, 0.4194924831390381, 0.41975873708724976, 0.420850932598114, 0.4210183620452881, 0.4210820198059082, 0.4211217761039734, 0.4216400384902954, 0.42174434661865234, 0.4219423234462738, 0.4223394989967346, 0.4224286675453186, 0.42268168926239014, 0.4235580563545227, 0.42414501309394836, 0.42419496178627014, 0.4242250919342041, 0.4242370128631592, 0.4243707060813904, 0.42447352409362793, 0.42453616857528687, 0.4246982932090759, 0.424852579832077, 0.42530521750450134, 0.42589613795280457, 0.4273248016834259, 0.4273611307144165, 0.42745208740234375, 0.4274604916572571, 0.4277729094028473, 0.42794591188430786, 0.4279499053955078, 0.4281059205532074, 0.4282401502132416, 0.4283432960510254]
>>> scores.index(0.4223394989967346)
10
>>> len(scores)
35

hh，前三分之一啊嗯

不过最好的是：
"insfo0.375_math0.125_code0.25_algebra0.125_math-grade0.125"
"insfo0.25_math0.125_code0.375_algebra0.125_math-grade0.125"
"insfo0.125_math0.125_code0.5_algebra0.125_math-grade0.125"
"insfo0.125_math0.125_code0.375_algebra0.125_math-grade0.25"

至少可以说，提高insfo和code是没什么毛病的。

然后同样用2M token，在这个最优配比上训练了一下，最终loss是0.41025376319885254。
确实比上面扫点出来的都强一些。