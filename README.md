# 腔

## 项目结构

|    目录     |                 内容                 |
| :---------: | :----------------------------------: |
|   audios    |          一些示例的音频样本          |
|    docs     |        一些说明性的文档和教程        |
|  features   |           音频特征分析模块           |
|   flagged   |  gradio 进行 flag 的文件夹，可忽略   |
| gradio_temp |     存储生成的临时文件，不可删除     |
|  glissando  |             滑音处理模块             |
|  toolkits   | 一些有用的工具，如泛音分析、均衡器等 |
|   vibrato   |             颤音处理模块             |

## 日志

`2023.5.7 19:31` 组合了新的用户界面在 `main.py` 中

`2023.5.7 10:32` 增加了五种颤音模拟函数 

## 快速开始

* 执行 `python main.py` 即可生成一个本地 URL，包括了颤音处理、滑音处理和微调的工具

### 颤音处理

* 选择多种颤音波动函数：标准的正弦波、频率均匀分布的正弦波、频率正态分布的正弦波、频率逐渐提高的正弦波、频率逐渐降低的正弦波

### 滑音处理

* 输入要处理的音频和参考音频即可模拟滑音