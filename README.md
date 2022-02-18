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
\ln(S_t) - \ln(s_0) + \frac{1}{2}\sigma^2 t = \mu t
$$