# IsaacSim 后端

UniLab 的 `isaacsim` 后端在独立的 Python 3.12 worker 进程中通过 IsaacSim 6.0 与
IsaacLab 3（develop `4ecd0b036da1`）运行 PhysX。主进程保留标准 `SimBackend`
NumPy contract，并使用 MuJoCo 状态布局：共享内存传输 `qpos`/`qvel`、body 坐标系
与 contact 传感器数据，管道传输生命周期命令。

## 运行时边界

IsaacSim 6 需要 Python 3.12，因此运行时位于 UniLab 环境之外。安装入口是
`scripts/tools/setup_isaacsim_env.sh`：它在 `$UNISIM_ISAACSIM_HOME`（默认
`$HOME/.cache/unisim/isaacsim`）下创建 venv 和钉住的 IsaacLab 源码树，并运行
IsaacLab 自带的安装器，因此使用该 IsaacLab 提交钉住的 Isaac Sim、torch 与
Newton 版本。

- `UNISIM_ISAACSIM_HOME`：运行时根目录。
- `UNISIM_ISAACSIM_PYTHON`：覆盖 worker 解释器路径。
- `OMNI_KIT_ACCEPT_EULA=YES`：worker 非交互启动（适配器会为 worker 设置）。

主进程需要 `unisim-core` 的 `isaacsim` extra，它会安装 MuJoCo。

## 场景如何进入 PhysX

MJCF 场景是唯一事实来源：

1. 主进程用 MuJoCo 编译场景（以及每个同布局 fixed variant），拒绝 PhysX 无法表达
   的特性：等式约束、tendon、flex、mocap body、ball joint、关节弹簧、
   ellipsoid/height field geom、geom margin、1 或 3 以外的接触维度，以及非单位
   传动比位置伺服的执行器。
2. MuJoCo weld 组成为 articulation link。带多个关节的 body 变成一串 link，中间
   用小质量辅助 link 连接；固定基座的 actor 增加一个固定到世界的锚点 link。
   单个自由 body 成为 rigid object。
3. 质量属性、关节坐标系与限位、armature、`frictionloss`（PhysX 关节摩擦力）、
   `damping`（粘性关节摩擦）、重力补偿、碰撞体与碰撞过滤（父子、`<exclude>`、
   `contype`/`conaffinity`、显式 pair）都从编译后的模型导出。
4. MuJoCo 按 geom 对决定摩擦，PhysX 按材质 combine mode 决定。主进程在所有可碰撞
   对上选择偏差最小的 combine mode 分配，并记录无法复现的 geom 对。

worker 为每个 variant 生成一份 USD layer，引用到每个环境中，并绑定 IsaacLab
articulation 与 rigid object。

## 支持的能力

- 位置伺服控制、按行 reset、fixed model variant（同布局）。
- body 质量、质心、惯量、关节 armature、geom 摩擦（可动 body）与 kp/kd 的 reset
  随机化；作用在 body 质心的 interval 力与力矩。
- 按刚体对上报的 `<contact>` 传感器。geom 选择比刚体级上报更窄的传感器会在构造
  时报错。reset 之后这些传感器行为零，直到下一次物理步。
- 由发布状态计算的 frame、gyro、velocimeter 与关节传感器。

## 与 MuJoCo 的差异

以下是适配器无法消除的引擎边界，这里如实列出，而不是藏在某种近似背后：

- **摩擦**：MuJoCo 按 geom 对决定摩擦（显式 pair、priority、最大值），PhysX 按材质
  combine mode 决定。无法复现的 geom 对会在构造时报告。
- **接触力**：传感器力是 PhysX 的法向力加上同一对刚体的摩擦力。PhysX 的摩擦锚点按
  刚体对而不是按接触点给出，因此单接触归约（`mindist`、`maxforce`）把该对的摩擦力
  投影到它选中的那个接触的坐标系里。接触力仍小于 MuJoCo：同一状态下指尖对读到
  0.44 N，MuJoCo 是 0.58 N，因为 PhysX 没有 `solref`/`solimp` 那样的接触柔度。
- **reset**：随机化的 body 质量、质心与惯量晚一个物理步生效：reset 之后的第一步仍按
  上一次 reset 的值积分，之后每一步用新值。被 reset 行的接触传感器在下一次物理步之前
  读到 0；body 被瞬移之后，PhysX 最多要两步才报告它的接触。
- **求解器**：PhysX TGS 的迭代次数来自 MJCF 的求解迭代预算；MuJoCo 的
  `solref`/`solimp` 接触柔度在 PhysX 没有对应。

## 回放

worker 不做渲染。`record` 与 `interactive` 回放和 `mjwarp` 一样，通过 MuJoCo
renderer 重放物理快照，也支持 `ghost_model` 等 debug overlay：

```bash
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
uv run eval --algo ppo --task g1_walk_flat --sim isaacsim --load-run <run-id> \
  --render-mode record training.play_steps=300
```
