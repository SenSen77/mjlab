---
name: mjlab对齐G1-23DOF到rl_sar
overview: 在mjlab中新增一个“G1 23DOF Motion Imitation/Tracking”任务配置，使训练时的观测/动作接口与 rl_sar/policy/g1_23（124维输入、23维输出、joint顺序与default pose）一致，并在训练保存checkpoint时自动导出可被rl_sar（LibTorch TorchScript）直接加载的 `.pt` 策略文件。
todos:
  - id: add-g1-23-robot-cfg
    content: 把 `rl_sar_zoo/g1_23_description` 的 MJCF+meshes 复制进 mjlab（不手改模型），新增 `g1_23_constants.py` 直接从该 MJCF 加载 spec，并把 init_state/default_joint_pos 对齐 rl_sar default_dof_pos
    status: completed
  - id: slice-motioncommand-23dof
    content: 扩展 `MotionCommandCfg` 支持 `command_joint_names`，让 command 与 joint误差只作用在 23DOF 子集
    status: completed
  - id: add-tracking-task-g1-23
    content: 新增 Tracking 任务 `Mjlab-Tracking-Flat-Unitree-G1-23DoF`：obs=124/action=23 且顺序对齐 rl_sar
    status: completed
  - id: export-torchscript-pt
    content: 在 Tracking runner.save() 里新增导出 `policy.pt`（TorchScript 单输入/单输出，rl_sar 可直接加载）
    status: completed
  - id: add-interface-test
    content: 新增单测/小脚本校验 obs/action 维度与 joint 顺序对齐
    status: completed
isProject: false
---

### 可行性结论

- **策略文件导出**：mjlab 已内置 TorchScript 与 ONNX 导出器（见 `src/mjlab/utils/lab_api/rl/exporter.py`），因此最终输出 `**.pt`（TorchScript）** 或 `**.onnx**` 都可行。
- **当前不匹配点**：
  - mjlab 的 Unitree G1 资产是 **29DOF**（`g1.xml` 的 model 名为 `g1_29dof_rev_1_0`），而 rl_sar 的 sim2real 约定是 **23DOF**（`policy/g1_23/base.yaml`）。
  - mjlab Tracking 的默认 ONNX 导出是 **双输入（`obs`,`time_step`）+多输出**（`src/mjlab/tasks/tracking/rl/exporter.py`），而 rl_sar 的 C++ 推理仅支持 **单输入/单输出**；不过你选择的是 **TorchScript(.pt)**，其接口正好是单输入/单输出（`rl_sar` 的 `TorchModel::forward()`只喂一个输入向量并取一个 tensor 输出）。
  - **资产来源建议**：你已经在 `rl_sar/src/rl_sar_zoo/g1_23_description/mjcf/` 提供了稳定的 **23DOF 物理模型**（`g1_23dof.xml` + `meshes/`）。为避免手工裁剪/修改 MJCF 引入错误，mjlab 侧应 **直接拷贝并复用这套资产**。

### 目标接口（以 rl_sar g1_23 为准）

- **控制频率**：dt=0.005、decimation=4（mjlab Tracking 已是这个设置：`tracking_env_cfg.py`）。
- **动作维度与顺序**：23维，顺序严格对齐 `policy/g1_23/base.yaml: joint_names`。
- **观测维度与顺序**：124维，顺序对齐 `policy/g1_23/whole_body_tracking/config.yaml: observations`：
  - 46：motion_command（23 joint pos + 23 joint vel）
  - 6：motion_anchor_ori_b
  - 3：ang_vel
  - 23：dof_pos
  - 23：dof_vel
  - 23：actions
- **默认 pose（action offset）**：对齐 `policy/g1_23/base.yaml: default_dof_pos`。

### 实施方案（推荐："直接复用 rl_sar 的 23DOF MJCF 资产"）

该方案最稳：训练的物理结构就是 23DOF，且资源可直接同步到服务器训练。mjlab 仓库内会保留一份拷贝（不在 `.gitignore` 范围内），避免训练机还需要依赖外部路径。

1. **把 23DOF 资产复制进 mjlab（不手改模型；只拷贝 robot MJCF + meshes）**

- **资源拷贝**（建议落在 mjlab 资产树内，便于打包/同步）：
  - `src/mjlab/asset_zoo/robots/unitree_g1_23/mjcf/g1_23dof.xml`
  - `src/mjlab/asset_zoo/robots/unitree_g1_23/mjcf/meshes/*.STL`
- **git 追踪**：`.gitignore` 未忽略这些资源，可直接提交并同步到训练服务器。

1. **新增 G1 23DOF 机器人 constants（从拷贝过来的 MJCF 加载）**

- 新增文件（示例路径）：`src/mjlab/asset_zoo/robots/unitree_g1_23/g1_23_constants.py`\n+  - `spec_fn` 直接用 `mujoco.MjSpec.from_file()` 读取 `g1_23dof.xml`\n+  - 处理 mesh 资源：保持 `<compiler meshdir="meshes">` 生效，必要时将 `spec.assets` 填充为读取到的 STL bytes（参考 `src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` 的 `update_assets()` 用法）\n+  - `get_g1_23_robot_cfg()`：`EntityCfg(init_state.joint_pos=对齐 rl_sar default_dof_pos, spec_fn=..., articulation=...)`\n+  - `G1_23_JOINT_NAMES`：直接拷贝 `policy/g1_23/base.yaml` 的 23 joint 顺序（用于 obs/action slicing 与一致性测试）

1. **让 Tracking 的 MotionCommand 支持“只用 23DOF 做 command/误差/奖励”**

- 修改 `src/mjlab/tasks/tracking/mdp/commands.py`
  - 扩展 `MotionCommandCfg`：新增字段 `command_joint_names: tuple[str, ...] = ()`
  - 在 `MotionCommand.__init__()` 中根据 `command_joint_names` 计算 `command_joint_ids`（用 `robot.find_joints(..., preserve_order=True)`）
  - 修改 `MotionCommand.command`：用 `command_joint_ids` 取子集后再 `cat([pos, vel])`，确保 **46维**
  - 修改 `_update_metrics()`：`error_joint_pos/vel` 也只在子集上算，避免对未控制关节施加惩罚
  - 兼容性：若 `command_joint_names` 为空，保持现有 29DOF 行为不变

1. **新增 Tracking G1-23DOF 环境配置（obs/action 对齐 124/23）**

- 新增目录：`src/mjlab/tasks/tracking/config/g1_23/`
  - `env_cfgs.py`
    - `cfg.scene.entities = {"robot": get_g1_23_robot_cfg()}`
    - `JointPositionActionCfg(actuator_names=(".*",), use_default_offset=True)` 但由于 articulation 已裁剪，action_dim 会变成 **23**
    - `joint_pos_action.scale = G1_ACTION_SCALE`（或新建 `G1_23_ACTION_SCALE`），确保与 rl_sar 的 `action_scale`一致
    - 设置 `MotionCommandCfg.command_joint_names = G1_23_JOINT_NAMES`
    - 重写 `cfg.observations["policy"]` 的 terms 顺序与内容：
      - 保留：`command`、`motion_anchor_ori_b`、`base_ang_vel`、`joint_pos`、`joint_vel`、`actions`
      - 移除：`motion_anchor_pos_b`、`base_lin_vel`（以满足 124 维）
      - 对 `joint_pos/joint_vel` 传入 `asset_cfg=SceneEntityCfg("robot", joint_names=G1_23_JOINT_NAMES)`，确保只输出 23 维且顺序正确
    - **IMU 兼容性检查**：mjlab Tracking 目前使用 `robot/imu_ang_vel`、`robot/imu_lin_vel` 等内置传感器名；而 `g1_23dof.xml` 里是 `imu_gyro/imu_acc/imu_quat`。\n+      - 若 mjlab 的 builtin sensor 桥接依赖传感器名称，我们会在 mjlab 侧增加一个小适配（例如：在创建 BuiltinSensor 时将 `imu_gyro -> imu_ang_vel` 这类映射补齐），保证 Tracking 配置无需大改。
  - `rl_cfg.py` 可先复用现有 `unitree_g1_tracking_ppo_runner_cfg()`（后续可再按稳定性调整）
  - `__init__.py` 注册新 task：例如 `Mjlab-Tracking-Flat-Unitree-G1-23DoF`

1. **训练保存时自动导出 rl_sar 可直接加载的 TorchScript `.pt**`

- 修改 `src/mjlab/tasks/tracking/rl/runner.py`
  - 在 `save()` 里，除了现有 ONNX 导出外，新增调用：
    - `from mjlab.utils.lab_api.rl.exporter import export_policy_as_jit`
    - `export_policy_as_jit(self.alg.policy, normalizer, path=policy_path, filename="policy.pt")`
  - 由于你选择 TorchScript 后端，rl_sar 可直接用该 `policy.pt`（单输入/单输出）
  - （可选）保留现有多输出 ONNX 作为 debug；如果你不需要 ONNX，可在后续把 tracking runner 的 ONNX 导出设为可配置开关

1. **一致性校验（最关键的3项）**

- 训练前用一个小脚本/测试（建议新增 `tests/test_tracking_g1_23dof_interface.py`）：
  - 构建 `Mjlab-Tracking-Flat-Unitree-G1-23DoF` env，断言：
    - policy obs 维度 == 124
    - action_dim == 23
    - `MotionCommand.command.shape[-1] == 46`
- 训练后：确认导出的 `policy.pt` 用 rl_sar 的输入维度（124）喂入可得到 23 维输出。

### 训练与对接（你将怎么用）

- **motion 数据集**：你可以继续用 mjlab 现有 `src/mjlab/scripts/csv_to_npz.py` 处理 29DOF CSV；我们在 `MotionCommand` 内部做 23DOF slicing，因此无需强制改动 CSV/NPZ 格式。
- **训练**：`uv run train Mjlab-Tracking-Flat-Unitree-G1-23DoF --registry-name ...`
- **部署到 rl_sar**：把训练目录里导出的 `policy.pt` 拷贝/替换到 `rl_sar/policy/g1_23/whole_body_tracking/` 并在配置中指向它（或按你的目录规范放置）。

### 风险与后续（可选第二阶段）

- 当前方案已是 **真正 23DOF 物理模型**（直接复用 `rl_sar_zoo/g1_23_description`），主要风险转为：\n+  - **资源打包/路径**（meshdir 与 STL 读取是否在 mjlab 的 spec/assets 体系下正确工作）\n+  - **IMU 传感器命名**（mjlab builtin sensor 名称与 MJCF sensor 名的桥接是否一致）\n+ 这两点会在“接口单测/小脚本”中提前验证并在需要时加最小适配。

