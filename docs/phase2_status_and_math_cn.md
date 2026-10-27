# Phase 2 项目说明：我做了什么、理解了什么、下一步怎么做

这份文件是项目进度、可运行方法、BO/GP 数学和 Peter meeting 提纲的合并版。重点区分“benchmark 已验证”和“BO/stopping rule 还没比较”，避免把工程准备说成研究结论。

## 一、现在是什么阶段

项目核心问题：在目标函数评价昂贵时，Bayesian Optimization (BO) 何时可以停止？不同景观、GP 假设和 stopping rule 会不会造成过早停止，错过足够好的解？

| 阶段/工作 | 现在做完的内容 | 意义与边界 |
|---|---|---|
| Phase 1：基础与闭环 | 论文阅读清单、概念图/数学例子、BoTorch 1D notebook 和脚本、length-scale sensitivity、结果、PPT/讲稿 | 学会 GP 后验、acquisition 与 BO loop；一个玩具例子表现好，不代表普遍可靠。 |
| Phase 2：目标函数库 | 24 个 2D/3D benchmark、函数文档、单元测试、可复现的 2D 热图/3D 切片生成脚本 | 造一套可重复的“考题/地图”，之后让不同算法在同样问题上比较；目前还没有完成 acquisition/stopping 策略比较。 |
| 下一阶段 | 依 workplan 进入 acquisition、stopping rule、pilot 与可靠性分析 | 真正回答方法何时可靠、何时早停。 |

Phase 2 的文件在 phase2-start 分支。最新 GitHub Actions 对该分支提交完成并通过：测试、Phase 1 产物重建与产物检查都成功。这次 Phase 2 新增没有合并到 main。CI: https://github.com/jinchengwang-August/stophere/actions/runs/40472234583

## 二、我现在做的每件事有什么意义

- 24 个 benchmark：构造具有不同难点的确定性目标函数，避免只在一个简单函数上判断算法。
- 单元测试：检查函数输入输出、边界、批量计算、解析最优点与随机场固定性，防止后面优化器测到的是代码错误。
- 2D 热图与 3D 切片：帮助人看出峰、谷、窄峰、各向异性和局部波动；它们是目标函数可视化，不是 BO 的优化结果。
- 固定的随机场：field_seed 决定某张地图，地图在整个优化过程保持固定；未来 optimizer_seed 决定算法从哪里开始尝试。每次 query 重抽地图会改变问题，等同于把真实目标和测量噪声混淆。
- Phase 1 的 toy BO：用低维例子核对 GP→acquisition→query→更新数据闭环；不是对真实复杂问题的性能保证。

Benchmark 的 11 类函数分别做 2D 与 3D，再加 Branin (2D) 和 Hartmann-3 (3D)，共 24 个：

| 类别 | 测试的性质 |
|---|---|
| quadratic peak | 宽而平滑的单峰，基础 sanity check |
| rotated ellipsoid | 坐标耦合、某些方向变化特别快（anisotropy） |
| rippled peak | 整体趋势加局部波纹 |
| competing Gaussian peaks | 多个高度相近、彼此分离的峰 |
| hidden narrow peak | 容易看到的宽峰旁边藏着更高、更窄、较远的峰 |
| Rosenbrock | 狭窄弯曲的最优谷 |
| Ackley / Rastrigin | 平坦区、波动与多个局部峰 |
| GRF long / short / rough | 固定随机地图的平滑尺度/粗糙程度不同 |
| Branin / Hartmann-3 | 常用的标准优化锚点，分别是 2D 和 3D |

目标输入统一归一化到 [0,1]^d，函数统一按最大化表达。常见的最小化函数经坐标映射/取负号，使峰值代表最佳解。不同函数的原始输出不可直接比较；后续应用每个函数的 reference optimum 计算 regret，并标明解析最优值或数值参考值。GRF 是有限随机 Fourier 特征的 Matérn-like 近似，不是精确的 Matérn sample path；数值参考最优点也不是对连续域的证明。

## 三、1D、2D、3D 各自有什么意义

- 1D：Phase 1 教学例子。能完整画出曲线，适合理解 posterior、uncertainty、acquisition 与采样点的关系；不是 Phase 2 当前的 24 个函数。
- 2D：可以看完整热图/等高线，是先做端到端 pilot 的好起点；能展示多峰、窄峰及各方向尺度差异。
- 3D：检查结论是否依赖二维几何。当前图只固定 x3=0.5 画一个二维 slice，不能把一张切片当成整个三维函数。
- 维度越高，同样数量的 query 越难覆盖输入空间。比较维度时需明确相同预算、初始设计、候选点与 scoring 规则。

## 四、我能自己运行吗？怎么运行？

可以。到仓库根目录安装环境并运行：

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
python -m pytest tests/test_benchmarks.py
python -m examples.phase2_gallery
~~~

完整项目测试可以运行 python -m pytest -q。依赖安装后，gallery 会在 results/phase2/gallery/ 生成 2D 热图和 x3=0.5 的 3D 切片。单元测试检查 24 个案例的维度、标量/batch 调用、输入范围、有限值、已知最优点与随机场复现性。若本机提示 No module named pytest，说明当前环境缺 pytest；先激活项目虚拟环境并安装依赖，不是 benchmark 代码本身失败。

目前 Phase 2 图片由脚本生成；如果 PNG 没有显示在 GitHub，重新运行上面的 gallery 命令即可。Phase 2 热图只显示测试函数的形状，不能当作“优化器已找到最优点”的证据。

## 五、结果里有什么值得注意，不该过度解读什么

Phase 1 固定 seed 的 1D 例子：3 个初始观察 + 6 轮 LogEI = 9 次目标函数调用。返回 x=0.7494546、f(x)=1.2997290；事后密集网格参考约 x=0.7511500、f=1.3002251；simple regret 约 0.0004961。对这个具体玩具函数，这个结果很接近参考峰值。

更有研究意义的观察是 best-so-far 在一段迭代中停滞，后面又有改善。因此，“几轮没有刷新最好值”不等于“真实函数没有更好的区域”，单独靠 stagnation 可能早停。但这只是一个低维、无噪声、固定 seed 的演示，不能说 LogEI 总是好，也不能说任何 stopping rule 已经可靠。Phase 2 CI 测的是目标函数实现/项目已有测试，不是算法性能排名。

## 六、BO、GP、acquisition、recommendation、stopping 的关系

未知的是目标函数 f(x)，不是一个独立的 x。观察模型可写为 y_i=f(x_i)+ε_i，ε_i~Normal(0, σ_n²)。GP 根据已经观察的数据 D={(x_i,y_i)} 建立未知函数的后验概率分布：

~~~text
f(x) | D ~ Normal( μ_D(x), σ_D²(x) )
~~~

μ 是预测平均值，σ² 是模型对未知函数值的不确定性。BO 每轮先基于观测拟合/更新 GP，再在便宜的 acquisition 上选择下一个 x 去真实查询，取得 y 后更新。acquisition 不是目标函数，也不是 GP；GP 是函数分布模型，acquisition 是选下一个实验点的规则。

记 K_y=K_θ(X,X)+σ_n²I+ηI，其中 η 是很小的 numerical jitter。对一个候选 x*：

~~~text
μ_D(x*) = m(x*) + k_*ᵀ K_y⁻¹ (y - m(X))
σ_D²(x*) = k(x*,x*) - k_*ᵀ K_y⁻¹ k_*
~~~

recommendation 是最终返回哪个点（例如当前观测中 y 最大的 x）；stopping rule 决定是否继续花预算；regret 是最大化问题中 r_t=f(x*)-f(x_best,t)。因此四件事分别回答：用什么模型描述未知函数、下一次去哪试、最后交哪个点、什么时候交卷。

## 七、GP 超参数的数学形式、意义、影响及实验测试

Phase 3 workplan 计划用 scaled ARD Matérn-5/2。其 kernel 为：

~~~text
k(x,x') = σ_f² (1 + √5 r + 5r²/3) exp(-√5 r)
r = sqrt( Σ_j ((x_j-x'_j)/ℓ_j)² )
~~~

注意：GRF benchmark 里的 length scale 控制“测试地图”；GP 的 ℓ_j 控制“算法以为地图有多平滑”。这是两个不同的参数，不能混为一谈。

| 参数符号 | 数学位置 | 直观意思 | 设错可能的影响 | 后续怎么测 |
|---|---|---|---|---|
| ℓ_j，ARD length scale | 距离 r 中除以 ℓ_j | 第 j 维走多远，函数才明显变化 | 太小：附近点也被看成不相似，模型过度起伏；太大：平滑掉窄峰/快变化，可能自信地预测错 | 先以 MLL 拟合为 baseline；再固定在 fitted ℓ 的 0.25×、1×、4×或预设短/中/长值，其他条件不变 |
| σ_f²，signal/output variance | kernel 前乘的振幅 | 函数先验变化的垂直尺度 | 太小压低函数变化；太大可能夸大振幅与潜在改进 | 记录拟合值，再单独改 prior/bounds 做敏感性测试；统一输出标准化 |
| σ_n²，observation noise variance | K_y 对角线加噪声 | 同一位置重复测量的波动程度 | 太小追逐噪声并过度自信；太大把真实差异当噪声、抑制探索 | noiseless pilot 先设并记录固定 numerical noise floor；之后单独加入已知观测噪声水平与独立 noise seed |
| m(x)，mean function | 后验里减去/加回均值 | 观测前的基线趋势假设 | 均值模型错会影响稀疏数据与远离观测处预测 | 先固定简单均值；只在单独实验比较，不能同时乱改其他超参数 |
| ν，Matérn smoothness | kernel 形式 | 允许目标函数有多粗糙/平滑 | 先验太平滑漏掉粗糙结构；太粗糙可能增加拟合难度 | 主实验先固定 Matérn-5/2；之后将 ν 作为单独因素测试 |
| η，jitter | K_y 加的小常数 | 防矩阵病态的数值垫片，不是生物/测量噪声 | 太小可能矩阵运算失败；太大扭曲模型 | 记录设定、失败与边界，不将它随意当科学因素 |
| β_t，UCB 权重 | UCB=μ+√β_t σ | 对不确定区额外奖励多少 | 太小偏利用、可能陷局部峰；太大偏探索、耗预算 | 固定其余因素，预先设少数低/中/高，配对 seeds 比 regret 与评估次数 |
| ξ，EI/PI 改进 margin | EI=E[(f(x)-f_best-ξ)_+] | 认为超过当前最好多少才算值得追 | 小 ξ 更关注微小改进；大 ξ 偏好潜在较大改进和探索 | 固定 GP/预算/seed，只改变预先定义的 ξ levels |
| ε，acceptable regret | 停止/成功判据，如 r_t≤ε | 我们接受离最优差多远 | 设太宽会把差解判成功；太严则预算内很难成功 | 按函数尺度声明 absolute 或 normalized regret，并报告阈值敏感性 |
| α 或 δ，置信/失败概率 | 概率型停止界 | 可容忍多少“其实还没够好”的风险 | 越保守通常多跑；保证依赖 GP 校准与条件 | 在相同轨迹上改变阈值，报告 false-stop rate 和 query 数 |

超参数不是算法每轮在输入空间里搜索的 x；它们定义 GP 的形状与噪声假设，通常由数据拟合。常见 marginal log likelihood (MLL)：

~~~text
log p(y|X,θ) = -1/2 (y-m)ᵀK_y⁻¹(y-m) - 1/2 log|K_y| - n/2 log(2π)
~~~

第一项奖励贴合观测；第二项惩罚过于复杂的协方差；末项是归一化常数。程序对 θ 做带约束的数值最大化，得到 θ_hat。新观测进来后再拟合。MLL 优化可能有局部最优、初值敏感、参数碰边界或数值不稳定，所以应保存每轮拟合参数与失败，不要只看最终一条曲线。

简单直觉：如果很近的点观测值差很大，小 ℓ 或较大噪声可能解释得通；如果远处点的趋势仍平稳，大 ℓ 可能合理。点少时 ℓ 与 noise 可能解释同一个差异，所以“拟合出来”并不自动代表参数真值正确。hyperparameter stress test 应一次只改变一个因素，保持函数、initial points、预算、candidate set、noise 和 seed 配对一致。

## 八、下一步实验的具体计划

Phase 2 完成的是 benchmark bank。按 workplan，Phase 3 实现 5 种 acquisition；Phase 4 实现 stopping rules；Phase 5 跑 pilot matrix；Phase 6 才分析 regret、过早停止、成本和 GP diagnostics。

建议先做最小端到端 pilot：

1. 选三个有代表性的 2D 函数：quadratic_peak、competing_gaussian_peaks、hidden_narrow_peak。
2. 固定同一候选 catalogue、初始设计、评价预算、recommendation 和配对 seeds。
3. 先比较 LogEI 和 random search baseline，确认整个数据记录/评分 pipeline 正确。
4. 再加少数 stopping rules；记录每轮 x,y、拟合 ℓ/σ_f²/σ_n²、best-so-far、regret、停止轮数、false stop、拟合失败。
5. 小试后至少用 5 个 optimizer seeds 做 pilot；稳定后再扩展 acquisition、3D、noise 与超参数 stress tests。

指标建议：最终 simple regret、regret 随 query 次数曲线、停止轮数/评价成本、false-stop rate（停下时 regret 仍大于预先声明的可接受阈值）、GP calibration 和拟合失败。若相对有限 catalogue 算最优值，结论就是 catalogue 内结论，不能直接宣称连续域全局保证。

## 九、明天和 Peter 说什么（直接一点）

### 开场 + Strang

> Hi Peter. I want to give you a quick update on Phase 2, make sure I’m interpreting the project correctly, and agree on the next concrete task. Also, have you talked with Professor Alexander Strang about the sponsorship?
>
> He said he could sponsor my research credit with limited involvement and was open to meeting with both of us. If you haven’t connected yet, I can send a short intro email. What still needs to happen for the credit sponsorship, and what is separately required for LBNL onboarding?

如果 Peter 已联系：

> What did you agree on, and is there anything you need me to follow up on?

### 进度和理解

> I’ve completed the Phase 2 benchmark bank: 24 two- and three-dimensional cases, tests, documentation, and a reproducible plotting script. It’s on phase2-start, and the latest CI run passed. I understand this phase gives us controlled landscapes; it doesn’t yet tell us which acquisition or stopping rule is better.
>
> My current understanding is that the GP gives a posterior over the unknown function, acquisition chooses the next evaluation, recommendation chooses what we return, and stopping decides when another evaluation is no longer worth the cost. A GP can be confidently wrong, so a flat best-so-far curve alone is not enough evidence that it is safe to stop.

### 确认下一步并请 Peter 指方向

> I’d like to start with a small end-to-end pilot on a smooth peak, competing peaks, and the hidden narrow peak. I’d compare LogEI with random search using the same initial points, candidate set, and budget, then add a small set of stopping rules. Does that sound right, or should I prioritize something else?
>
> For hyperparameters, would you prefer a fitted GP as the baseline and a separate length-scale misspecification stress test? How should we define “good enough” across functions with different scales—absolute regret or normalized regret?
>
> I can follow the loop, but I’m still working on the math of marginal-likelihood fitting and how uncertainty should translate into a reliable stopping decision. What is one derivation or small experiment you’d like me to do independently next?
>
> Before we wrap up, can we pin down the next deliverable, what validation is sufficient, and who should coordinate with Professor Strang?

## 十、见面前确保能自己解释这四点

1. Benchmark 是固定的测试目标函数，不是优化算法，也不是 BO 的结果。
2. GP 给未知函数的后验分布；超参数影响预测均值/不确定性，再影响 acquisition 和停止判断。
3. Phase 1 的 9-query 示例接近密集网格参考，但只证明这一小例子能跑通；平台期后又改善，说明“近期没进步”可能早停。
4. Phase 2 已把考题准备好。下一阶段才是在相同预算和 seeds 下对比方法，并测出在哪些 landscape/hyperparameter 条件下会失败。

Branch: https://github.com/jinchengwang-August/stophere/tree/phase2-start
