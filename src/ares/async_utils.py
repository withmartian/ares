"""A collection of utilities for working with asyncio."""

import asyncio
import dataclasses


@dataclasses.dataclass(frozen=True)
class ValueAndFuture[ValType, FutureType]:
    value: ValType
    future: asyncio.Future[FutureType]
