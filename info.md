* **先给结论**

  * **VERL 对 agentic RL 的支持是够用的，而且关键抽象已经有了**：它已经提供了 server-based asynchronous rollout、多轮对话与 tool call、以及 `AgentLoop` 这层可插拔抽象；`AgentLoopBase.run()` 是核心扩展点，返回 `AgentLoopOutput`，其中要显式给出 `prompt_ids`、`response_ids`、`response_mask`。官方文档还明确写了：这层的目标就是“plugable user defined agent loop”，而“tool 怎么定义/怎么调用”本身不在框架强约束里。([Verl Documentation][1])
  * **但它对“任意外部黑盒 harness”还不是一等公民**：`Agent Loop` 文档仍标成 alpha；到 **2026-03-25**，官方仓库里关于 `RemoteAgentLoop` 的能力还是一个开放中的 feature request，里面点名的目标场景就包括第三方 agent / 外部集群 / Claude Code 这类外部 agent runtime。也就是说，**本地自定义 loop 很顺，远端黑盒 agent 还需要你自己补一层**。([Verl Documentation][2])

* **你这个问题里，最关键的是先区分两件事**

  * **“Harness / env”**：文件系统、bash、browser、MCP、外部 API、sandbox、tool execution、session 管理这些。
  * **“Policy model”**：RL 里真正被 VERL 更新参数的那个模型。
  * 在 VERL 的设计里，最自然的做法是：**policy model 继续走 VERL 的 rollout server；harness 只是环境层**。因为 VERL 很在意 token 级轨迹对齐，官方专门强调过训练时更偏向 token-in/token-out，而不是只用标准 chat-completion 文本接口。([GitHub][3])

* **所以，换 Harness 有两条路**

  * **路 1：把 Claude Agent SDK 当“环境能力层”，不要把它当 policy loop**

    * 这是 **更推荐** 的接法。
    * 做法是：你在 VERL 里写一个新的 `AgentLoopBase` 子类，模型生成仍然调用 `server_manager.generate(...)`；当模型决定执行某个动作时，你再去调用 Claude Agent SDK 暴露出来的工具 / MCP / 沙箱能力，或者调用一个你包好的 Claude-side runtime 服务。最后把工具返回内容作为 `response_ids` 的一部分塞回轨迹，并把这些 token 的 `response_mask` 设成 `0`。文档里对 `response_mask` 的定义就是：**模型生成 token 为 1，工具/环境反馈 token 为 0**。([Verl Documentation][2])
    * 这种方式下，**被 RL 的还是你的开源 policy model**；Claude Agent SDK 只是 env/harness。这个结构和 VERL 的抽象最一致。([Verl Documentation][2])

  * **路 2：把 Claude Agent SDK 当“完整 agent loop”，VERL 只负责训练外围**

    * 这条路能做，但复杂度高很多。
    * 原因是 Claude Agent SDK 自带自己的 agent loop、内置工具、hooks、custom tools、MCP 和 session 管理；它默认也是按 **Claude model** 去跑。官方 Python reference 里 `ClaudeAgentOptions.model` 的描述就是 “Claude model to use”。([Claude][4])
    * 只要 **真正出 token 的不是 VERL 正在训练的那个模型**，你就没法直接用 PPO/GRPO 去更新它的权重。也就是说：

      * **如果 Claude 真的是 agent 背后的模型**，那它更适合做 teacher / evaluator / data generator，不适合当 VERL 的可训练 policy。
      * **如果你只是喜欢 Claude Agent SDK 的运行时和工具体系**，那你需要把它的采样请求劫持到你自己的可训练模型上。这个思路官方也是允许做代理的：Claude Agent SDK 支持用 `ANTHROPIC_BASE_URL` 把 sampling 请求发到代理，而不是直接发 Claude API。([Claude][5])

* **如果你想用 Claude Agent SDK 替代 OpenHands，实际工程上最像样的方案是：`custom RemoteAgentLoop + sampling proxy`**

  * **整体结构**

    * VERL 训练集群里：

      * 自定义 `AgentLoopBase` 子类，比如 `ClaudeSDKRemoteLoop`
      * 一个 proxy，负责接 Claude Agent SDK 的采样请求
      * VERL rollout server / vLLM / SGLang
    * Agent 运行集群里：

      * Claude Agent SDK 进程
      * 受控文件系统 / bash / browser / MCP / 你自己的工具服务
  * **流量路径**

    * `ClaudeSDKRemoteLoop.run()` 为每条 trajectory 生成一个 `trial_id`
    * 启动或复用一个 sampling proxy
    * 给远端 Claude Agent SDK 进程注入：

      * `ANTHROPIC_BASE_URL=http://proxy/<trial_id>`
      * `allowed_tools` / `mcp_servers` / sandbox / hooks
    * Claude Agent SDK 的每次采样都先进 proxy
    * proxy 再把请求转给 VERL 的 rollout server，顺手记录：

      * 输入消息
      * 输出 token ids
      * logprobs
      * 每 turn 的 finish reason
    * rollout 结束后，按 turn 重建 `AgentLoopOutput`
  * 这和 VERL 社区里 **2026-03-25** 提的 `RemoteAgentLoop` 方案几乎是同一思路：它就是用代理去接第三方 agent 的 LLM 调用，再把会话重建成 VERL 兼容的 `AgentLoopOutput`。那个 issue 里甚至直接把外部 agent 场景写成了 `SWE-agent / Claude Code / separate Ray/K8s cluster`。([GitHub][6])

* **你需要在 VERL 里落地的最小接口，其实不多**

  * **1）注册一个新的 loop**

    * VERL 现在就是通过 `@register(agent_name)` 把 loop 放进 registry，然后 worker 根据 dataset 里的 `agent_name` 去实例化对应 loop。没有 `agent_name` 时，会走默认 loop。([GitHub][7])
  * **2）`run()` 里返回这几个核心字段**

    * `prompt_ids`
    * `response_ids`
    * `response_mask`
    * 可选再补 `response_logprobs`
    * 以及 `reward_score`、`num_turns`、`metrics` 等辅助字段。([Verl Documentation][2])
  * **3）重建规则**

    * Claude/你的 policy 每次生成出来的 token：

      * 追加到 `response_ids`
      * `response_mask` 填 `1`
      * `response_logprobs` 填真实值
    * 工具返回 / 环境反馈 / 用户模拟输入：

      * 也追加到 `response_ids`
      * `response_mask` 填 `0`
      * `response_logprobs` 通常填 `0.0` 或留空
    * 这个规则和 VERL 在 agent loop 文档里的定义是对齐的。([Verl Documentation][2])

* **用 Claude Agent SDK 时，你能直接复用的能力**

  * 它本身就有：

    * 内置 agent loop
    * built-in tools
    * hooks
    * custom tools
    * MCP
    * sessions / continuity
  * 自定义工具这块，官方推荐的是把你自己的 domain actions 包成 in-process MCP server / custom tools，Claude 在对话里调用这些工具。这个点对“把 harness 做成 env API”很方便。([Claude][8])
  * 所以如果你的 env 是代码仓库、浏览器、检索、内部 API，这套做法是自然的：

    * Claude Agent SDK 负责调工具
    * VERL 侧只负责捕获 token 轨迹、算 reward、做更新([Claude][9])

* **一个最小骨架大概会长这样**

  ```python
  from verl.experimental.agent_loop.agent_loop import (
      AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
  )

  @register("claude_sdk_remote")
  class ClaudeSDKRemoteLoop(AgentLoopBase):
      async def run(self, sampling_params, messages=None, instance_id=None, **kwargs):
          # 1. 为当前 rollout 创建唯一会话
          trial_id = f"{instance_id}"

          # 2. 准备 proxy 地址；Claude SDK 的采样流量走这里
          proxy_url = f"http://proxy-host:8080/{trial_id}"

          # 3. 启动远端 agent runtime
          #    远端进程环境变量里注入:
          #    ANTHROPIC_BASE_URL=proxy_url
          #    并配置 allowed_tools / mcp_servers / cwd / sandbox / hooks
          result = await launch_claude_agent_sdk_runtime(
              prompt=messages,
              env={"ANTHROPIC_BASE_URL": proxy_url},
              agent_kwargs=kwargs,
          )

          # 4. 从 proxy 拉回整段 session 记录
          session = await fetch_session_record(trial_id)

          # 5. 重建 prompt_ids / response_ids / response_mask / response_logprobs
          prompt_ids = tokenize_initial_messages(messages)
          response_ids = []
          response_mask = []
          response_logprobs = []

          for turn in session.turns:
              # LLM completion
              response_ids.extend(turn.completion_token_ids)
              response_mask.extend([1] * len(turn.completion_token_ids))
              response_logprobs.extend(turn.completion_logprobs)

              # tool/env feedback
              tool_ids = tokenize_tool_feedback(turn.tool_feedback_messages)
              response_ids.extend(tool_ids)
              response_mask.extend([0] * len(tool_ids))
              response_logprobs.extend([0.0] * len(tool_ids))

          reward = compute_reward(result, session, kwargs)

          return AgentLoopOutput(
              prompt_ids=prompt_ids,
              response_ids=response_ids,
              response_mask=response_mask,
              response_logprobs=response_logprobs,
              reward_score=reward,
              num_turns=len(session.turns),
              metrics=AgentLoopMetrics(),
          )
  ```

* **几个你会遇到的硬点**

  * **硬点 1：Claude Agent SDK 的代理接口是 Anthropic 风格，不是 OpenAI 风格**

    * 它官方支持的是 `ANTHROPIC_BASE_URL`，不是“给你一个 OpenAI-compatible base_url 就完事”。所以你的 proxy 要么：

      * 实现 Anthropic sampling API 语义，再转到 VERL rollout server；
      * 要么你自己做一层 Anthropic ↔︎ VERL 的协议翻译。([Claude][5])
  * **硬点 2：如果远端 runtime 里还有别的联网动作，HTTP 代理不一定全拦得住**

    * 官方文档明确提到：`HTTP_PROXY/HTTPS_PROXY` 只对遵守这些环境变量的程序有效；像部分 Node.js `fetch()` 默认就不会走。对这类场景，要上 TLS-terminating proxy / transparent proxy / iptables / proxychains 一类方案。([Claude][5])
  * **硬点 3：closed model 不能被 VERL 直接更新**

    * 这个不是框架问题，是训练对象问题。
    * 你若真用 Claude 作为 agent 背后的模型，那 VERL 最多能围绕它做：

      * 轨迹采集
      * reward 学习
      * 数据蒸馏
      * 对另一个开源 policy 的 imitation / RL
    * 但不能直接把 PPO/GRPO 的梯度打回 Claude 权重。这个逻辑上是绕不过去的。`ClaudeAgentOptions.model` 也说明了 SDK 默认面向 Claude 模型。([Claude][10])

* **所以，针对你的例子，我的建议是**

  * **如果你的目标是“训练一个开源 policy，让它在 Claude Agent SDK 风格的 runtime 里做事”**：

    * 选 **`custom AgentLoop + remote Claude runtime + ANTHROPIC proxy + trajectory reconstruction`**
    * 这是最贴近 VERL 当前设计的做法。([Verl Documentation][2])
  * **如果你的目标是“直接把 Claude Agent SDK 里的 Claude 当 policy 拿来做 RL”**：

    * 用 VERL 不合适，至少不是直接路子。
    * 这时更像是：

      * 用它做 teacher / evaluator / data collection
      * 再把数据回灌到你自己的可训练模型上。([Claude][10])
  * **如果你只是想换掉 OpenHands 的 runtime，而不是换掉被训练模型**：

    * 这是能做的，而且是比较合理的目标。关键不是“Claude Agent SDK 能不能接”，而是你要不要自己补那层 **proxy + session recorder + AgentLoopOutput reconstruction**。([GitHub][6])

* **一句更直接的话**

  * **VERL 已经支持“你自己定义 harness 作为 agent env”这件事；但对 Claude Agent SDK 这种完整外部 agent runtime，当前更像是“可以接，但要自己写 adapter/proxy”，不是开箱即用。**([Verl Documentation][2])

* **你现在若是走 SWE / code-agent 这个方向，我下一条可以直接给你一份 `claude_sdk_remote_loop.py` 的最小可运行骨架，包含 proxy、session 记录结构、以及 `AgentLoopOutput` 重建逻辑。**

[1]: https://verl.readthedocs.io/en/latest/start/agentic_rl.html?utm_source=chatgpt.com "Agentic RL Training - verl documentation - Read the Docs"
[2]: https://verl.readthedocs.io/en/latest/advance/agent_loop.html "Agent Loop — verl  documentation"
[3]: https://github.com/volcengine/verl/blob/main/docs/start/agentic_rl.rst "verl/docs/start/agentic_rl.rst at main · verl-project/verl · GitHub"
[4]: https://platform.claude.com/docs/en/agent-sdk/overview "Agent SDK overview - Claude API Docs"
[5]: https://platform.claude.com/docs/en/agent-sdk/secure-deployment "Securely deploying AI agents - Claude API Docs"
[6]: https://github.com/verl-project/verl/issues/5737 "Feature Request: RemoteAgentLoop - Support for External Distributed Agent Integration · Issue #5737 · verl-project/verl · GitHub"
[7]: https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py "verl/verl/experimental/agent_loop/agent_loop.py at main · verl-project/verl · GitHub"
[8]: https://platform.claude.com/docs/en/agent-sdk/agent-loop "How the agent loop works - Claude API Docs"
[9]: https://platform.claude.com/docs/en/agent-sdk/custom-tools "Give Claude custom tools - Claude API Docs"
[10]: https://platform.claude.com/docs/en/agent-sdk/python "Agent SDK reference - Python - Claude API Docs"
