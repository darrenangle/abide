"""Gemma 4-capable vLLM server wrapper for the legacy verifiers RL trainer.

This mirrors the upstream `verifiers_rl.rl.inference.server` behavior while
using the current vLLM import paths needed by the verified Gemma 4 runtime
overlay. It exposes the extra weight-sync endpoints expected by
`verifiers_rl.rl.inference.client.VLLMClient`.
"""

from __future__ import annotations

import asyncio
import os
import signal
from argparse import Namespace
from typing import TYPE_CHECKING, Any, cast

import torch
import uvloop
from fastapi import Body
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit

if TYPE_CHECKING:
    from collections.abc import Sequence

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAX_CONCURRENT_WEIGHT_UPDATES = 10
weight_update_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEIGHT_UPDATES)
background_tasks: set[asyncio.Task[Any]] = set()


class WeightSyncWorkerExtension:
    """Worker extension that receives live weight updates from the trainer."""

    pynccl_comm: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: int | None = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. Call close_communicator first."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        assert self.device is not None
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        torch_dtype = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)
        client_rank = self.client_rank
        assert client_rank is not None
        self.pynccl_comm.broadcast(weight, src=client_rank)
        self.pynccl_comm.group.barrier()
        cast("Any", self).model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    def create_background_task(coro: Any) -> None:
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "abide.verifiers_vllm_server.WeightSyncWorkerExtension"
    engine = AsyncLLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    supported_tasks = await engine.get_supported_tasks()
    app = cast("Any", build_app)(args, supported_tasks, engine.model_config)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def get_world_size() -> dict[str, int]:
        return {"world_size": args.tensor_parallel_size * args.data_parallel_size}

    @app.post("/init_communicator")
    async def init_communicator(data: dict[str, Any] = Body(...)) -> dict[str, str]:
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")
        create_background_task(
            engine.collective_rpc("init_communicator", args=(host, port, world_size))
        )
        return {"status": "ok"}

    @app.post("/update_named_param")
    async def update_named_param(data: dict[str, Any] = Body(...)) -> dict[str, str]:
        name = data.get("name")
        dtype = data.get("dtype")
        shape_data = data.get("shape")
        if not isinstance(shape_data, list):
            raise ValueError("shape must be a list")
        shape = tuple(shape_data)

        async def throttled_update() -> None:
            async with weight_update_semaphore:
                await engine.collective_rpc("update_named_param", args=(name, dtype, shape))

        create_background_task(throttled_update())
        return {"status": "ok"}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, str]:
        create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks() -> dict[str, int]:
        return {"num_background_tasks": len(background_tasks)}

    @app.post("/close_communicator")
    async def close_communicator() -> dict[str, str]:
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}

    await init_app_state(engine, app.state, args, supported_tasks)
    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    sock.close()


def main() -> None:
    parser = cast(
        "Any",
        FlexibleArgumentParser,
    )(description="Gemma 4-capable vLLM OpenAI server with legacy verifiers weight sync.")
    parser = make_arg_parser(parser)
    args = parser.parse_args() or Namespace()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
