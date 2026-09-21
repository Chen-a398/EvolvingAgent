"""Agent 启动入口：加载配置 -> 组装工具 -> 执行一个端到端验证任务

用法（项目根目录下）:
    uv run python run_agent.py
    uv run python run_agent.py 你的config.yaml路径
"""
import asyncio
import sys
from pathlib import Path

from my_min_agent.config import Config
from my_min_agent.retry import RetryConfig as EngineRetryConfig
from my_min_agent.schema import LLMProvider
from my_min_agent.LLM import LLMClient
from my_min_agent.agent import Agent
from my_min_agent.Tool.file_tools import ReadTool, WriteTool, EditTool
from my_min_agent.Tool.bash_tool import BashTool

# 第一个端到端验证任务：写文件 + 读回来，覆盖 LLM 调用和工具调用两条链路
TASK = "请在当前工作目录创建 hello.txt，内容为：我的 agent 跑通了。然后用读取工具读回来，确认内容无误后告诉我。"

# 项目里没有 system_prompt.md 时的兜底提示词
FALLBACK_SYSTEM_PROMPT = """你是一个带工具的命令行助手，可以使用提供的工具读写文件和执行命令。
逐步完成用户交给的任务；任务完成后，直接用自然语言回复最终结果，不要再调用工具。
所有文件操作的相对路径都基于当前工作目录。"""


def find_config() -> Path:
    """找配置文件：命令行参数 > 脚本同目录的 config.yaml"""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {p}")
        return p
    p = Path(__file__).parent / "config.yaml"
    if p.exists():
        return p
    raise FileNotFoundError(
        "没找到 config.yaml。把它放到 run_agent.py 同目录，"
        "或者运行: uv run python run_agent.py 你的配置文件路径"
    )


async def main() -> None:
    cfg_path = find_config()
    print(f"[1/4] 加载配置: {cfg_path}")
    config = Config.from_yaml(cfg_path)

    # config.yaml 里的 retry 配置(pydantic版) -> retry.py 的引擎版
    r = config.llm.retry
    llm = LLMClient(
        api_key=config.llm.api_key,
        provider=LLMProvider(config.llm.provider),
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=EngineRetryConfig(
            enable=r.enabled,
            max_retries=r.max_retries,
            initial_delay=r.initial_delay,
            max_delay=r.max_delay,
            exponential_base=r.exponential_base,
        ),
    )
    print(f"[2/4] LLM 客户端就绪: {config.llm.model} @ {config.llm.api_base}")

    workspace = Path(config.agent.workspace_dir).resolve()
    tools = [
        ReadTool(str(workspace)),
        WriteTool(str(workspace)),
        EditTool(str(workspace)),
        BashTool(str(workspace)),
    ]

    sp_path = Path(config.agent.system_prompt_path)
    if not sp_path.exists():
        sp_path = Path(__file__).parent / sp_path
    if sp_path.exists():
        system_prompt = sp_path.read_text(encoding="utf-8")
        print(f"[3/4] 系统提示词: {sp_path}")
    else:
        system_prompt = FALLBACK_SYSTEM_PROMPT
        print("[3/4] 没找到 system_prompt.md，先用内置最小提示词")

    agent = Agent(
        llm_client=llm,
        system_prompt=system_prompt,
        tools=tools,
        max_steps=config.agent.max_steps,
        workspace_dir=str(workspace),
    )

    print(f"[4/4] 启动任务: {TASK}\n")
    agent.add_user_message(TASK)
    result = await agent.run()
    print("\n===== 最终结果 =====")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
