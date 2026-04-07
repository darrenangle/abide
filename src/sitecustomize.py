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


def _resolve_openai_serving_chat_kwargs(
    chat_init: Any,
    state: Any,
    args: Any,
    *,
    request_logger: Any,
    resolved_chat_template: Any,
) -> dict[str, Any]:
    signature = inspect.signature(chat_init)
    accepted_kwargs = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    structured_outputs_config = getattr(args, "structured_outputs_config", None)
    candidate_kwargs: dict[str, Any] = {
        "openai_serving_render": getattr(state, "openai_serving_render", None),
        "request_logger": request_logger,
        "chat_template": resolved_chat_template,
        "chat_template_content_format": getattr(args, "chat_template_content_format", None),
        "trust_request_chat_template": getattr(args, "trust_request_chat_template", False),
        "return_tokens_as_token_ids": getattr(args, "return_tokens_as_token_ids", False),
        "reasoning_parser": getattr(structured_outputs_config, "reasoning_parser", ""),
        "enable_auto_tools": getattr(args, "enable_auto_tool_choice", False),
        "exclude_tools_when_tool_choice_none": getattr(
            args, "exclude_tools_when_tool_choice_none", False
        ),
        "tool_parser": getattr(args, "tool_call_parser", None),
        "enable_prompt_tokens_details": getattr(args, "enable_prompt_tokens_details", False),
        "enable_force_include_usage": getattr(args, "enable_force_include_usage", False),
        "enable_log_outputs": getattr(args, "enable_log_outputs", False),
        "enable_log_deltas": getattr(args, "enable_log_deltas", True),
        "default_chat_template_kwargs": getattr(args, "default_chat_template_kwargs", None),
        "log_error_stack": getattr(args, "log_error_stack", False),
    }
    return {key: value for key, value in candidate_kwargs.items() if key in accepted_kwargs}


def _install_prime_rl_init_app_state_compat() -> None:
    try:
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.logger import RequestLogger

        api_server = importlib.import_module("vllm.entrypoints.openai.api_server")
        original_vllm_init_app_state = api_server.init_app_state
        server = importlib.import_module("prime_rl.inference.vllm.server")
        serving = importlib.import_module("prime_rl.inference.vllm.serving_chat_with_tokens")
    except Exception:
        return

    chat_cls = getattr(serving, "OpenAIServingChatWithTokens", None)
    if chat_cls is None or getattr(server, "__abide_init_app_state_compat__", False):
        return

    async def compat_init_app_state(
        engine_client: Any,
        state: Any,
        args: Any,
        supported_tasks: Any,
    ) -> None:
        await original_vllm_init_app_state(engine_client, state, args, supported_tasks)

        if getattr(args, "enable_log_requests", False):
            request_logger = RequestLogger(max_log_len=getattr(args, "max_log_len", None))
        else:
            request_logger = None

        resolved_chat_template = load_chat_template(getattr(args, "chat_template", None))
        chat_kwargs = _resolve_openai_serving_chat_kwargs(
            chat_cls.__init__,
            state,
            args,
            request_logger=request_logger,
            resolved_chat_template=resolved_chat_template,
        )
        serving_chat = chat_cls(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            **chat_kwargs,
        )
        state.openai_serving_chat = serving_chat if "generate" in supported_tasks else None
        state.openai_serving_chat_with_tokens = (
            serving_chat if "generate" in supported_tasks else None
        )

    server_any: Any = server
    server_any.custom_init_app_state = compat_init_app_state
    server_any.__abide_init_app_state_compat__ = True
    api_server_any: Any = api_server
    api_server_any.init_app_state = compat_init_app_state


def _install_prime_rl_perf_counter_compat() -> None:
    try:
        perf = importlib.import_module("prime_rl.trainer.perf")
    except Exception:
        return

    perf_counter = getattr(perf, "PerfCounter", None)
    original_get_active_mm_params = getattr(perf_counter, "get_active_mm_params", None)
    if (
        perf_counter is None
        or original_get_active_mm_params is None
        or getattr(perf, "__abide_perf_counter_compat__", False)
    ):
        return

    def compat_get_active_mm_params(config: Any) -> float:
        text_config = getattr(config, "text_config", config)
        patched_attrs: dict[str, Any] = {}
        for attr_name in (
            "num_experts",
            "n_routed_experts",
            "num_shared_experts",
            "num_experts_per_tok",
        ):
            if hasattr(text_config, attr_name) and getattr(text_config, attr_name) is None:
                patched_attrs[attr_name] = None
                setattr(text_config, attr_name, 0)

        try:
            return original_get_active_mm_params(config)
        finally:
            for attr_name, original_value in patched_attrs.items():
                setattr(text_config, attr_name, original_value)

    perf_counter_any: Any = perf_counter
    perf_counter_any.get_active_mm_params = staticmethod(compat_get_active_mm_params)
    perf_any: Any = perf
    perf_any.__abide_perf_counter_compat__ = True


_install_flash_attn_symbol_compat()
_install_prime_rl_gemma4_compat()
_install_prime_rl_perf_counter_compat()
_install_vllm_serve_symbol_compat()
_install_prime_rl_init_app_state_compat()
_install_prime_rl_vllm_build_app_compat()
