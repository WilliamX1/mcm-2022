# mcm-2022

## 几何布朗运动

设 $S_t$ 是 $t$ 时刻某种股票的价格，它满足如下随机微分方程

$$
\begin{aligned}
dS_t &= S_t\mu dt + S_t\sigma dB_t \\
S_0 &= s_0 \\
\end{aligned}
$$

其中 $\mu, \sigma$ 为常数（$\mu$ 为收益率，$\sigma$  为波动率）。$B_t$ 是正态分布的概率密度函数的随机值。

其解析解是

$$
S_t = s_0 \exp[(\mu - \frac{1}{2}\sigma^2)t + \sigma B_t]
$$

两边取对数，得到一个线性模型

$$
\ln(S_t) = \ln(s_0) + (\mu - \frac{1}{2}\sigma^2)t + \sigma B_t
$$

由于 $B_t$ 是布朗运动函数，是随机值且波动性太大，因此将其省略。

当 $t \in [T_1, T_2]$ 时的价格 $B^{ge}_t$。我们先求出 $\sigma$ 的确定值，再用机器学习线性拟合（逻辑回归）$\mu$ 值。

$$
\sigma^2 = \frac{1}{T_2 - T_1}\sum_{j = 0}^{m - 1}[\ln(\frac{B^{ge}_{t_{j + 1}}}{B^{ge}_{t_j}})]^2
$$

最终的机器学习模型应该是：

$$
\ln(S_t) - \ln(s_0) + \frac{1}{2}\sigma^2 t - \sigma B_t = \mu t
$$

### 直接估算 $\mu$ 和 $\sigma^2$

https://parsiad.ca/blog/2020/maximum-likelihood-estimation-of-gbm-parameters/

$$
\mu = \frac{\delta X}{\delta t} + \frac{1}{2}\sigma^2 \\
\sigma^2 = -\frac{1}{N}\frac{(\delta X)^2}{\delta t} + \frac{1}{\delta t}\sum_{n = 1}^N\Delta X_n^2
$$

其中

$$
X_t = \log(S_t) \\
\delta X = X_{t_N} - X_{t_0} \\
\delta t = t_N - t_0 \\
\Delta X_n = X_{t_n} - X_{t_{n - 1}} \\
\Delta t_n = t_n - t_{n - 1}
$$

解析解是

$$
S_t = s_0 \exp[(\mu - \frac{1}{2}\sigma^2)t + \sigma B_t]
$$

### 计算 $S_t$ 的期望和方差

$$
E(S_t) = e^{\log{S_{t - 1} + \mu - \frac{1}{2}\sigma^2} + \frac{1}{2}\sigma^2} = S_{t - 1} \times e^\mu
$$

$$
D(S_t) = e^{2 \times (\log(S_{t - 1}) + \mu - \frac{1}{2}\sigma^2) + \sigma^2} \times (e^{\sigma^2} - 1) = S_{t - 1} ^ 2 \times e^{2\mu} \times (e^{\sigma^2} - 1)
$$

收益率则是 $R$ 是

$$
R = \frac{S_{t - 1} \times e^\mu - S_{t - 1}}{S_{t - 1}} = e^\mu - 1
$$

### $S_t$ 的上下误差图（5 % 和 95 %）

$$
E(S_t) \times e^{-\frac{1}{2} \times \sigma^2 \times t \pm 1.96\times\sigma}
$$

所以 **正误差** 是：

$$
+error = E(S_t) \times (e^{-\frac{1}{2} \times \sigma^2 \times t + 1.96\times\sigma} - 1)
$$

**负误差** 是：

$$
-error = E(S_t) \times (1 - e^{-\frac{1}{2} \times \sigma^2 \times t - 1.96\times\sigma})
$$

## 决策

比特币买入卖出均须乘以 $0.98$，黄金买入卖出均须乘以 $0.99$。

相对误差计算

$$
err = 1 - \frac{\text{昨天预测值}}{\text{昨天真实值}}
$$

将每天进行分类

1. 前一天收益率 $R_{t - 1} \gt 0$，且 $\text{前一天真实价格} \gt \text{预测价格}$，则为 $1$ 类，全部买入。
2. 前一天收益率 $R_{t - 1} \gt 0$，且 $\text{前一天真实价格} \lt \text{预测价格}$，则为 $2$ 类，部分卖出。
3. 前一天收益率 $R_{t - 1} \lt 0$，且 $\text{前一天真实价格} \gt \text{预测价格}$，则为 $3$ 类，部分买入。
4. 前一天收益率 $R_{t - 1} \lt 0$，且 $\text{前一天真实价格} \lt \text{预测价格}$，则为 $4$ 类，全部卖出。

