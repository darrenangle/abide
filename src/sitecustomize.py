"""Runtime compatibility patches for subprocess-based training stacks.

This module is only injected into the prime-rl subprocess tree by
`scripts/train_prime_rl.py` via `PYTHONPATH=src`. It keeps the main project
code untouched while patching known third-party compatibility gaps.
"""

# mypy: disable-error-code="import-not-found,no-any-return"

from __future__ import annotations

import importlib
import inspect
from importlib import metadata
from typing import Any

from packaging import version


def _install_flash_attn_symbol_compat() -> None:
    try:
        import transformers.modeling_flash_attention_utils as flash_utils
    except Exception:
        return

    if hasattr(flash_utils, "is_flash_attn_greater_or_equal_2_10"):
        return

    def is_flash_attn_greater_or_equal_2_10() -> bool:
        try:
            flash_attn_version = metadata.version("flash-attn")
        except metadata.PackageNotFoundError:
            return False
        return version.parse(flash_attn_version) >= version.parse("2.10.0")

    cast_target = flash_utils
    cast_target_any: Any = cast_target
    cast_target_any.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10


def _install_prime_rl_gemma4_compat() -> None:
    try:
        vlm = importlib.import_module("prime_rl.utils.vlm")
    except Exception:
        return

    if "gemma4" in vlm.VLM_REGISTRY:
        return

    vlm.VLM_REGISTRY["gemma4"] = vlm.VLMModelInfo(
        vision_encoder_attr="model.vision_tower",
        language_model_attr="model.language_model",
    )


def _install_vllm_serve_symbol_compat() -> None:
    try:
        serve = importlib.import_module("vllm.entrypoints.cli.serve")
    except Exception:
        return

    if hasattr(serve, "run_api_server_worker_proc"):
        return

    try:
        from vllm.entrypoints.openai.api_server import run_server_worker
    except Exception:
        return

    def run_api_server_worker_proc(
        listen_address: str,
        sock: Any,
        args: Any,
        client_config: Any = None,
        **uvicorn_kwargs: Any,
    ) -> None:
        import uvloop

        uvloop.run(
            run_server_worker(
                listen_address,
                sock,
                args,
                client_config=client_config,
                **uvicorn_kwargs,
            )
        )

    serve_any: Any = serve
    serve_any.run_api_server_worker_proc = run_api_server_worker_proc


def _install_prime_rl_vllm_build_app_compat() -> None:
    try:
        server = importlib.import_module("prime_rl.inference.vllm.server")
        api_server = importlib.import_module("vllm.entrypoints.openai.api_server")
    except Exception:
        return

    original_build_app = getattr(server, "_original_build_app", None)
    router = getattr(server, "router", None)
    if original_build_app is None or router is None:
        return

    def custom_build_app(
        args: Any,
        supported_tasks: Any = None,
        model_config: Any = None,
    ) -> Any:
        app = original_build_app(args, supported_tasks, model_config)
        app.include_router(router)
        return app

    server_any: Any = server
    server_any.custom_build_app = custom_build_app
    api_server_any: Any = api_server
    api_server_any.build_app = custom_build_app


def _install_prime_rl_serving_chat_kwarg_compat() -> None:
    try:
        serving = importlib.import_module("prime_rl.inference.vllm.serving_chat_with_tokens")
    except Exception:
        return

    chat_cls = getattr(serving, "OpenAIServingChatWithTokens", None)
    if chat_cls is None:
        return

    original_init = chat_cls.__init__
    if getattr(original_init, "__abide_kwarg_compat__", False):
        return

    signature = inspect.signature(original_init)
    accepted_kwargs = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    def compat_init(self: Any, *args: Any, **kwargs: Any) -> None:
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted_kwargs}
        original_init(self, *args, **filtered_kwargs)

    compat_init_any: Any = compat_init
    compat_init_any.__abide_kwarg_compat__ = True
    chat_cls.__init__ = compat_init


_install_flash_attn_symbol_compat()
_install_prime_rl_gemma4_compat()
_install_vllm_serve_symbol_compat()
_install_prime_rl_vllm_build_app_compat()
_install_prime_rl_serving_chat_kwarg_compat()
