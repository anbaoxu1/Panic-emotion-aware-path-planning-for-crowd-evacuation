# Panic-emotion-aware-path-planning-for-crowd-evacuation
<img width="2103" height="821" alt="image" src="https://github.com/user-attachments/assets/009a5f5c-51cd-46e1-b900-471e5f7cb3e3" />

Crowd evacuation simulations provide crucial guidance for emergency response in public venues. However, many existing path planning methods do not fully account for the impact of crowd panic, which limits their effectiveness in real-world evacuations. To address this issue, we propose a panic‑emotion‑aware path planning method for crowd evacuation. First, we design a ResNet‑RP (Residual Neural Network – Recognition Panic) model with spatiotemporal similarity constraints to accurately identify individual panic emotions and compute the crowd panic level based on the degree of individual aggregation. Second, using the crowd panic level, we apply a mean‑field method to predict the proportion of panicked individuals. Finally, we incorporate the predicted proportion of panicked people into the reward function of a multi‑agent deep deterministic policy gradient algorithm (P‑MAD) to enable emotion‑aware evacuation path planning. Experimental results demonstrate that the proposed method can effectively perceive panic emotions and generate panic‑avoiding evacuation paths for crowds.

### Publication

- SCI 三区，CCF C 类期刊：[`Applied Intelligence`](https://link.springer.com/journal/10489)
- 引用信息：  
  An, B., Zhang, G., Zhao, C. et al. *Panic emotion aware path planning for crowd evacuation*. **Applied Intelligence** 56, 125 (2026).  
  [https://doi.org/10.1007/s10489-026-07154-9](https://doi.org/10.1007/s10489-026-07154-9)

# 项目说明

## 1. 项目整体在做什么

这个项目整体上是一个基于强化学习的路径规划 / 避障仿真实验工程。  
代码主线可以概括为：

1. 用 `ArmEnv` 这一类环境脚本定义智能体、目标点、障碍物、奖励函数和状态观察。
2. 用 `model.py` 里的 actor / critic 网络做策略学习。
3. 用 `replay_buffer.py` 存训练过程中的经验。
4. 用 `main_openai*.py` 这几类训练入口脚本跑不同场景下的训练实验。
5. 用 `enjoy_split*.py`、`test.py`、`tracker.py`、`figtest.py` 等脚本做推理、轨迹回放、效果对比和画图。

从当前代码的引用关系看，这个项目不是只有一套最终版本，而是包含了多轮实验过程中的多个环境版本、训练版本和测试版本。

---

## 2. 顶层目录说明

| 路径 | 作用 |
| --- | --- |
| `.idea/` | PyCharm 工程配置目录，保存 IDE 的项目设置、模块信息、工作区配置，不属于核心算法代码。 |
| `__pycache__/` | Python 运行后生成的字节码缓存目录，不需要手动维护。 |
| `models/` | 训练好的模型参数目录，里面每个 `office_xxx_xxx_xxxxxx` 子目录都是一次训练保存下来的权重。 |
| `four_train/` | 一套单独的实验分支，包含自己的训练脚本、环境脚本、数据文件和画图脚本。 |
| 根目录下各 `.py` 文件 | 主实验代码，包含环境定义、训练入口、模型结构、经验池、可视化和测试脚本。 |

---

## 3. `models/` 文件夹说明

`models/` 用来存放训练结果，也就是训练好的神经网络参数。

每个子目录名字类似：

- `office_2208_211335_102000`
- `office_2209_232357_502000`

这类目录一般表示：

- `office`：场景名
- 中间时间串：训练保存时的时间
- 最后一段数字：保存时对应的训练步数

每个模型目录里常见的文件有：

| 文件名 | 作用 |
| --- | --- |
| `a_c_0.pt` | 当前 actor 网络参数 |
| `a_t_0.pt` | target actor 网络参数 |
| `c_c_0.pt` | 当前 critic 网络参数 |
| `c_t_0.pt` | target critic 网络参数 |

如果是多智能体实验，理论上会出现多个编号，例如 `a_c_1.pt`、`a_c_2.pt` 这种形式；当前已有模型目录里主要保留了编号为 `0` 的文件。

---

## 4. 根目录核心公共代码说明

这部分文件是多个训练脚本共用的基础组件。

| 文件 | 作用 |
| --- | --- |
| `arguments.py` | 统一管理训练参数，例如训练轮数、每轮步数、学习率、batch size、模型保存路径、是否恢复旧模型等。 |
| `model.py` | 定义神经网络结构，包括 `openai_actor`、`openai_critic` 以及基础版的 `actor_agent`、`critic_agent`。这是整个强化学习策略网络的核心。 |
| `replay_buffer.py` | 经验回放池。负责存储 `obs -> action -> reward -> next_obs -> done` 这些训练样本，并在训练时随机采样。 |
| `RVO.py` | 实现 RVO（Reciprocal Velocity Obstacles）相关函数，用于速度障碍避碰计算，是环境动力学 / 避障逻辑的重要辅助模块。 |

这四个文件可以理解为项目的“公共底座”。

---

## 5. 根目录训练入口脚本说明

这些脚本负责真正发起训练，只是训练所使用的环境和 critic 输入方式不一样。

| 文件 | 作用 |
| --- | --- |
| `main_openai.py` | 主训练入口之一，使用 `mofang_env.py` 里的 `ArmEnv`。这是单智能体场景下的一套训练流程，训练时会记录奖励并画出曲线。 |
| `main_openai2.py` | 使用 `hard_env2.py` 的训练入口。相较基础版，它对 critic 的输入做了重新组织，不是直接看全量状态，而是根据邻近关系拼接局部相关 agent 的观测和动作，属于更复杂的多智能体实验版本。 |
| `main_openai4.py` | 使用 `suiji_env.py` 的训练入口，可以看成随机场景下的一版基线训练脚本。 |
| `main_openai_suiji.py` | 同样基于 `suiji_env.py`，但 critic 输入组织方式更像一个实验性改进版，会重新构造局部观察和局部动作再训练 critic。 |

这些脚本的共同流程基本一致：

1. 读参数
2. 创建环境
3. 初始化 actor / critic
4. 与环境交互采样
5. 把样本写入 `ReplayBuffer`
6. 周期性更新网络
7. 定期把模型保存到 `models/`

---

## 6. 根目录环境脚本说明

这一组文件都是环境定义文件，核心都是实现 `ArmEnv` 类，并提供：

- `step()`：执行动作并返回下一状态、奖励、结束标记
- `reset()`：重置环境
- `get_obs()`：构造观察量
- `render()`：渲染显示
- `sample_action()`：生成随机动作或调试动作

它们的主要区别是场景复杂度、智能体数量、是否带障碍物、是否随机化。

| 文件 | 作用 |
| --- | --- |
| `env.py` | 较早期的基础环境原型，内部直接维护多个 agent 的位置和目标，属于最早的一版环境实现。 |
| `10.19env.py` | 2020 年 10 月重写的一版环境，代码结构比 `env.py` 更完整，已经引入 `Agent` 类和更清晰的 `ArmEnv` 封装，可视为中间版本。 |
| `mofang_env.py` | 根目录里较重要的单智能体环境之一。包含 `Agent`、`Obstacle`、`Viewer` 和 `ArmEnv`，并会读取 `test.xlsx` 中的数据。`main_openai.py` 就是基于它训练。 |
| `suiji_env.py` | 随机场景环境。相较 `mofang_env.py`，它更强调障碍物或场景初始化的随机性，`main_openai4.py` 和 `main_openai_suiji.py` 都使用它。 |
| `hard_env.py` | 一版更难的环境实验，保留了复杂场景逻辑，用于难场景调试和试验。 |
| `hard_env2.py` | 多智能体难场景环境，是 `main_openai2.py` 的直接训练环境。 |
| `hard_env3.py` | 另一版 hard 场景环境，保留了障碍物和单独的观察/奖励设计，更多像实验分支。 |
| `simple_env.py` | 更简单的多智能体场景版本，适合做基础验证。 |
| `simple_env2.py` | `simple_env.py` 的扩展版，场景规模更大，代码里能看到 8 个 agent 的可视化和状态更新逻辑。 |
| `tenenv.py` | 更大规模的环境版本，用于批量推理或性能测试。`enjoy_split.py` 会在这个环境里一次性构造大量 agent。 |
| `test_env.py` | 测试用环境分支，保留了较完整的 8-agent 场景逻辑，主要服务于验证和对比。 |
| `test_env3.py` | 另一版测试环境，`test.py` 会加载它做批量模型推理与渲染。 |

简单理解就是：  
`simple / test / hard / suiji / mofang / ten` 这些命名，本质上是在表示不同难度、不同规模、不同随机程度、不同实验目的的环境版本。

---

## 7. 根目录推理、测试与辅助脚本说明

| 文件 | 作用 |
| --- | --- |
| `enjoy_split.py` | 模型加载与推理脚本。会从旧模型目录中加载 policy，然后放到 `tenenv.py` 环境里跑，重点更偏向推理速度和大规模场景测试。 |
| `enjoy_split_suiji.py` | 在 `suiji_env.py` 环境中加载训练好的模型做回放和渲染，属于随机环境的测试脚本。 |
| `test.py` | 在 `test_env3.py` 环境中加载已训练模型并渲染，主要用于验证模型在测试环境中的行为。 |
| `fig.py` | 一个独立的小脚本，用来检查当前机器是否能识别 CUDA / GPU，并打印设备信息。 |
| `atest.py` | 很小的临时测试脚本，目前内容只是简单循环打印，属于调试残留。 |
| `game.py` | 一个独立的 5x5 迷宫小游戏环境，用 OpenCV 做可视化。它和上面的 `ArmEnv` 主线不是同一套系统，更像是早期强化学习练习代码。 |

---

## 8. 根目录数据文件说明

| 文件 | 作用 |
| --- | --- |
| `test.xlsx` | 被 `mofang_env.py` 读取的 Excel 数据文件，通常用于轨迹、场景点位或对比数据输入。 |
| `students001.txt` | 一份数值文本数据。从当前代码引用关系看，没有发现 Python 脚本直接读取它，更像是历史数据记录或实验导出文件。 |

---

## 9. `four_train/` 文件夹整体说明

`four_train/` 不是简单的备份目录，而是一套单独维护的实验代码。  
它拥有自己的：

- 参数文件
- 模型结构
- 经验池
- RVO 实现
- 环境定义
- 训练入口
- 推理与画图脚本
- 实验生成的数据文件

可以把它理解成“根目录主实验之外的另一套独立实验工程”。

---

## 10. `four_train/` 内部文件说明

| 文件 | 作用 |
| --- | --- |
| `four_train/arguments.py` | `four_train` 这套实验自己的参数配置，和根目录 `arguments.py` 类似，但超参数取值不同。 |
| `four_train/model.py` | `four_train` 的网络结构定义，和根目录版本作用相同。 |
| `four_train/replay_buffer.py` | `four_train` 的经验回放池。 |
| `four_train/RVO.py` | `four_train` 自己使用的 RVO 避碰计算模块。 |
| `four_train/main_openai.py` | `four_train` 的训练主入口。使用 `four_train/mofang_env.py` 进行训练，并在训练结束后把奖励曲线横纵坐标保存为 `x.txt`、`y.txt`。 |
| `four_train/enjoy_split.py` | `four_train` 的模型回放脚本。会加载训练好的模型，在环境中跑轨迹，并把轨迹坐标保存到 `l1.txt`、`l2.txt`。 |
| `four_train/mofang_env.py` | `four_train` 的主环境定义，结构和根目录版类似，但这里读取的是 `test2.xlsx`。 |
| `four_train/hard_env2.py` | `four_train` 保留的复杂环境版本，用于这套实验里的难场景扩展。 |
| `four_train/tracker.py` | 轨迹对比脚本。它会读取 `test2.xlsx` 的真实轨迹数据，再读取 `l1.txt`、`l2.txt` 的仿真轨迹，对比真实路径和仿真路径。 |
| `four_train/figtest.py` | 奖励曲线画图脚本。会读取 `eps.txt`、`e1.txt`、`e2.txt`，画出不同方法的 reward 对比曲线。 |
| `four_train/rew.py` | 一个独立的对比画图脚本，用固定数组画出 `RVO` 和 `Our method` 的时间对比。 |
| `four_train/test2.xlsx` | `four_train/mofang_env.py` 和 `four_train/tracker.py` 会读取的 Excel 数据文件。 |
| `four_train/eps.txt` | 画奖励曲线时使用的横坐标数据，一般表示 episode 序号。 |
| `four_train/e1.txt` | 一组奖励实验结果数据。 |
| `four_train/e2.txt` | 另一组奖励实验结果数据，通常用于和 `e1.txt` 对比。 |
| `four_train/l1.txt` | 回放后导出的轨迹 x 坐标或第一维轨迹数据。 |
| `four_train/l2.txt` | 回放后导出的轨迹 y 坐标或第二维轨迹数据。 |
| `four_train/x.txt` | 训练或轨迹处理后导出的中间数据文件。 |
| `four_train/y.txt` | 训练或轨迹处理后导出的中间数据文件。 |
| `four_train/yy.txt` | 另一份轨迹/实验中间结果文本，主要用于实验记录。 |
| `four_train/models/` | 当前只保留了 `__init__.py`，更像是预留模型目录或轻量占位目录。 |

---

## 11. `four_train/` 这套代码和根目录代码的关系

两者关系可以理解成：

- 根目录：主实验区，包含更多环境版本和更多训练入口。
- `four_train/`：独立实验区，更聚焦于一套特定环境、训练和轨迹对比流程。

它们并不是简单“主代码 + 子模块”的关系，而更像是“同一课题下的两套实验脚本集合”。

---

## 12. `.idea/` 和 `__pycache__/` 说明

### `.idea/`

这是 PyCharm 自动生成的工程配置目录，里面包括：

- `misc.xml`
- `modules.xml`
- `workspace.xml`
- `inspectionProfiles/`

这些文件只影响 IDE，不影响算法本身。

### `__pycache__/`

这是 Python 自动生成的缓存目录，里面的 `.pyc` 文件是：

- `arguments.cpython-39.pyc`
- `model.cpython-39.pyc`
- `replay_buffer.cpython-39.pyc`
- `RVO.cpython-39.pyc`
- `suiji_env.cpython-39.pyc`

这些都不是手写业务代码，可以忽略。

---

## 13. 如果按“功能链路”理解整个项目

为了让目录关系更直观，可以按下面这条链路理解：

| 功能阶段 | 对应文件 |
| --- | --- |
| 参数配置 | `arguments.py`、`four_train/arguments.py` |
| 环境定义 | `mofang_env.py`、`suiji_env.py`、`hard_env*.py`、`simple_env*.py`、`tenenv.py`、`test_env*.py`、`four_train/mofang_env.py` |
| 避障逻辑 | `RVO.py`、`four_train/RVO.py` |
| 网络结构 | `model.py`、`four_train/model.py` |
| 经验回放 | `replay_buffer.py`、`four_train/replay_buffer.py` |
| 正式训练 | `main_openai.py`、`main_openai2.py`、`main_openai4.py`、`main_openai_suiji.py`、`four_train/main_openai.py` |
| 模型测试 / 回放 | `enjoy_split.py`、`enjoy_split_suiji.py`、`test.py`、`four_train/enjoy_split.py` |
| 结果分析 / 画图 | `fig.py`、`four_train/figtest.py`、`four_train/tracker.py`、`four_train/rew.py` |
| 模型结果存档 | `models/`、`four_train/models/` |

---

## 14. 总结

这个项目的本质不是单一脚本，而是一组围绕“强化学习路径规划与避障”展开的实验代码集合。

最核心的代码主线是：

- 环境：`ArmEnv` 系列文件
- 算法：`model.py` + `replay_buffer.py` + `main_openai*.py`
- 避障辅助：`RVO.py`
- 结果验证：`enjoy_split*.py`、`test.py`、`tracker.py`、`figtest.py`
- 训练产物：`models/`
