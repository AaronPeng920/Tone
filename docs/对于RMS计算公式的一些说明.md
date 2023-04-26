# 对于 RMS 计算公式的一些说明

## 由信号进行计算

> 1. 如果指定 `center=True`，那么会对信号 `y` 的两端分别填充 `frame_length // 2` 的 0，目的是确保信号位于窗口的中间，即使得所有的信号数据都在分帧后计算的过程中使用到
> 2. 按照 `frame_length` 和 `hop_length` 进行分帧，得到 `n` 个长度是 `frame_length` 的帧，`shape:(n, frame_length)`
> 3. 对于每一帧 $x\in[-1,1]^{frame\_length}$，计算其中元素的绝对值的平方，即 `np.abs(x) ** 2`，然后计算其平均值，即 `power=np.mean(np.abs(x) ** 2)`，得到的就是能量
> 4. 最后对 `power` 进行开平方，得到的就是

下面是一个例子，指定信号 `[1,2,3,4,5,6,7,8,9,10]`，根据此 API 计算得到的结果如下：

```python
signal = np.array([1,2,3,4,5,6,7,8,9,10])
rms = librosa.feature.rms(y=signal, frame_length=5, hop_length=2)
print(rms)
```

结果：

```python
[[1.67332005 3.31662479 5.19615242 7.14142843 7.66811581]]
```

内部的计算是这样的：

1. 因为默认指定 `center=True`，那么对原信号进行左右填充，填充的尺寸是 `frame_length // 2 = 2`，填充的数值是 0，现在信号是：

```python
[0,0,1,2,3,4,5,6,7,8,9,10,0,0]
```

2. 分帧(这里每一行表示一帧，在 librosa 中是每一列表示一帧)

```python
[
  [0,0,1,2,3],
  [1,2,3,4,5],
  [3,4,5,6,7],
  [5,6,7,8,9],
  [7,8,9,10,0]
]
```

3. 对每帧的数据 `x`  计算 `np.mean(np.abs(x) ** 2)`，得到：

```python
power = [[ 2.8 11.  27.  51.  58.8]]
```

4. 对 `power` 进行开方

```python
result = [[1.67332005 3.31662479 5.19615242 7.14142843 7.66811581]]
```

## 由频谱图进行计算

> 1. 频谱图 `S` 的维度是 `(n_fft//2+1, 帧数)`，并且 `S` 需要是 `librosa.stft` 直接返回的结果，即元素是复数
> 2. 计算频谱图的每个复数元素的绝对值的平方，即 `x=np.abs(S) ** 2`
> 3. 对第一个频率对应的所有帧的的值乘以 0.5，即 `x[0, :] *= 0.5`，如果帧长度 `frame_length` 是偶数，那么对最后一个频率进行同样的处理，即 `x[-1, :] *= 0.5`，目的在于调整直流(第一个频率，其实就是频率=0) 和 `sr/2` 的部分
> 4. 对每帧的 `x` 数据计算和，然后乘以 2，然后除以 $frame\_length^2$，即 `power = 2 * np.sum(x) / frame_length ** 2`
> 5. 对 `power` 进行开方

下面是一个例子，指定频谱图 `arr = np.array([[1., 2., 3.],[4., 5., 6.], [7., 8., 9.]])`，由此 API 计算得到的 RMS 如下：

```python
# n_fft = 5
S = np.array([[1., 2., 3.],[4., 5., 6.], [7., 8., 9.]])
rms = librosa.feature.rms(y=None, S=S, frame_length=5)
print(rms)
```

结果：

```python
[[2.28910463 2.69814751 3.11769145]]
```

内部的计算是这样的：

1. 计算频谱图每个复数元素的绝对值的平方，即 `x = np.abs(S) ** 2`，得到：

```python
[
  [1, 4, 9],
  [16, 25, 36],
  [49, 64, 81]
]
```

2. 对第一个所有帧的值乘以 0.5，由于 `frame_length` 不是偶数，所以不对最后一个帧处理，得到：

```python
[
  [0.5, 2, 4.5],
  [16, 25, 36],
  [49, 64, 81]
]
```

3. 对每帧的数据计算和，然后乘以 2，然后除以 $frame\_length^2$，即 `power = 2 * np.sum(x) / frame_length ** 2`

```python
[
  [5.24],
  [7.28],
  [9.72]
]
```

4. 对 `power` 开平方

```python
[
  [2.28910463],
  [2.69814751], 
  [3.11769145]
]
```



设原来的计算



![image-20230424193047631](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230424193047631.png)

![image-20230424193842745](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230424193842745.png)

