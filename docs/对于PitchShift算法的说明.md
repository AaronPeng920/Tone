# 对于 Pitch Shift 算法的说明

## 音高频率表

![image-20230423205626999](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230423205626999.png)

任意相邻两个音符的音高的比值是 $\sqrt[12]{2}\approx 1.0594631$。因此，对于一个音符 $a$，其音高频率是 $f_a$，对其升高 $n$ 个音，只需要将其频率调整为 $\sqrt[12]{2}^n=2^{n/12}\times f_a$。

## 时间扩展

> 时间扩展(time_stretch)需要指定一个比例 rate，如果原来的信号长度是 `len`，那么经过时间扩展之后信号长度就成了 `len / rate`，其实就相当于对时间进行拉伸。

指定原始信号 `y` 和时间扩展比例(加速比) `rate`

1. 对信号 `y` 进行 STFT 变换得到复数元素频谱图

```python
stft = librosa.stft(y)
```

2. 用相位声码器运算上述频谱图 `stft` 和加速比 `rate` 来重新生成改变速率后的信号

```python
stft_stretch = librosa.phase_vocoder(stft, rate=rate)
```

> 在相位声码器内部是这么进行工作的，给定一个 STFT 矩阵 `D`，其中 `D.shape=[n_fft//2+1, 帧数量]`，通过因子 `rate` 进行加速，如果 `rate > 1` 表示加速，如果 `rate < 1` 表示减速。
>
> ```python
> phase_vocoder(D, *, rate, hop_length=None, n_fft=None):
> ```
>
> 1. 以 `rate` 做步长采取从 `0` 到 `帧数量` 的时间步：
>
> ```
> time_steps = np.arange(0, D.shape[-1], rate)
> ```
>
> 2. 创建一个维度是 `[n_fft//2+1, 时间步数量]` 的零数组 `d_stretch`
>
> ```python
> d_stretch = np.zeros_like(D, shape=[n_fft//2+1, 时间步数量])
> ```
>
> 3. 每个 fft_bin 中的预期相位提前
>
> ```python
> phi_advance = np.linspace(0, np.pi * hop_length, n_fft//2+1)
> ```
>
> 4. 设置相位累加器，初始化为第一个样本的相位
>
> ```python
> phase_acc = np.angle(D[..., 0])
> ```
>
> 5. 对 `D` 的最后一个维度的高索引处(即帧数量的维度)填充 2 列 0，以简化边界逻辑
>
> 6. 对每个时间步进行迭代，迭代的当前时间步是 `step`
>
> > * 获取 `D` 的 `int(step):int(step+2)` 的两帧
> > * 计算当前时间步 `step` 的小数部分  `alpha=np.mod(step, 1.0)` 
> >
> > 
>
> 
>
> 