"""自改错（重试）机制最小测试：不需要 API Key、不联网

场景1：失败2次第3次成功 -> 应看到重试日志 + 最终成功
场景2：一直失败 -> 应看到指数退避 + RetryExhaustedError
"""
import asyncio
import logging
import time

from retry import RetryConfig, async_retry, RetryExhaustedError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main():
    # --- 场景 1：连续失败 2 次后成功 ---
    # 用字典记录调用次数，模拟"前两次网络抖动、第三次成功"的典型不稳定 API 场景
    calls = {"n": 0}
    start = time.time()

    # 配置：最多重试 5 次，初始退避 0.5s，最大退避 2.0s
    @async_retry(config=RetryConfig(max_retries=5, initial_delay=0.5, max_delay=2.0))
    async def flaky_api():
        calls["n"] += 1
        print(f"  [场景1] 第 {calls['n']} 次尝试...")
        # 前 2 次抛出 ConnectionError，触发重试；第 3 次返回成功结果
        if calls["n"] < 3:
            raise ConnectionError("模拟网络抖动")
        return "成功拿到结果"

    # 调用被装饰的函数，重试机制会自动处理中间失败
    result = await flaky_api()
    print(f"PASS 场景1: {result}（共调用 {calls['n']} 次, 耗时 {time.time()-start:.1f}s）\n")

    # --- 场景 2：一直失败，验证重试耗尽 ---
    # 验证当所有重试都用完后，装饰器正确抛出 RetryExhaustedError
    start = time.time()

    # on_retry 回调：每次重试前触发，用于记录日志或触发告警
    def on_retry_cb(e, n):
        print(f"  [场景2] on_retry 回调触发: 第 {n} 次重试, 原因: {e}")

    # 配置：最多重试 3 次，初始退避 0.2s，最大退避 1.0s，并注册 on_retry 回调
    @async_retry(config=RetryConfig(max_retries=3, initial_delay=0.2, max_delay=1.0),
                 on_retry=on_retry_cb)
    async def always_fail():
        # 每次调用都抛出 ValueError，模拟服务持续不可用
        raise ValueError("这个服务今天就是坏的")

    try:
        await always_fail()
        print("FAIL 场景2: 没有抛出异常，重试机制没生效！")
    except RetryExhaustedError as e:
        # 捕获 RetryExhaustedError，验证其携带了重试次数和最后一次异常信息
        print(f"PASS 场景2: 正确抛出 RetryExhaustedError（耗时 {time.time()-start:.1f}s）")
        print(f"  attempts={e.attempts}, 最后错误={e.last_exception}")

    # --- 场景 3：默认参数（不传 config），LLM 客户端传 config 的对偶情况 ---
    # 不传 RetryConfig，使用默认配置（max_retries=3, initial_delay=1.0）
    # 验证装饰器在无自定义配置时也能正常工作
    @async_retry()
    async def instant_ok():
        # 一次就成功，不会触发任何重试逻辑
        return "一次就成"

    print(f"PASS 场景3: {await instant_ok()}（默认参数也正常）")


asyncio.run(main())