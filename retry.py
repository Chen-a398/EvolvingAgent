import asyncio
import functools
import logging
from typing import Any, Callable, Type, TypeVar

logger=logging.getLogger(__name__)
T=TypeVar("T")
class RetryConfig:
    def __init__(
        self,
        enable:bool=True,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),

    ):
        self.enabled = enable
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions
    def calculate_delay(self, attempt: int) -> float:
        delay=self.initial_delay*( self.exponential_base**attempt)
        return min(delay,self.max_delay)  
class RetryExhaustedError(Exception):
    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(f"Retry failed after {attempts} attempts. Last error: {str(last_exception)}")

def async_retry(
    config:RetryConfig |None=None,
    on_retry:Callable[[Exception,int],None]|None=None,

)->Callable:
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Any])->Callable[...,Any]:
        @functools.wraps(func)
        async def wrapper(*args,**kwargs)->Any:
            last_exception: Exception | None = None
            for attempt in range(config.max_retries+1):
                try:
                    return await func( *args,**kwargs)
                except config.retryable_exceptions as e:
                    last_exception =e

                if attempt >= config.max_retries:
                    logger.error(f"Function {func.__name__} retry failed, reached maximum retry count {config.max_retries}")
                    raise RetryExhaustedError(last_exception, attempt + 1)
                delay = config.calculate_delay(attempt)
                # Log
                logger.warning(
                    f"Function {func.__name__} call {attempt + 1} failed: {str(last_exception)}, "
                    f"retrying attempt {attempt + 2} after {delay:.2f} seconds"
                )
                if on_retry:
                    on_retry(last_exception,attempt+1)
                await asyncio.sleep(delay)

            if last_exception:
                raise last_exception
            raise Exception("unknown error")
        return wrapper
    return decorator
