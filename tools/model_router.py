#!/usr/bin/env python3
"""
Hermes tool wrapper that forwards model calls to the local model-router proxy.
Register this in tools/ and use it from skills/other modules.
"""
from tools.registry import registry
import requests
import os

MODEL_ROUTER_URL = os.environ.get('MODEL_ROUTER_URL', 'http://127.0.0.1:6000')


def model_router_call(prompt: str, task_type: str = 'minor', timeout_total: int = 30):
    payload = {
        'prompt': prompt,
        'task_type': task_type,
        'timeout_total': timeout_total,
    }
    r = requests.post(MODEL_ROUTER_URL + '/v1/chat', json=payload, timeout=timeout_total+2)
    if r.status_code != 200:
        raise RuntimeError(f"model-router error: {r.status_code} {r.text}")
    return r.json()


registry.register(
    name='model_router',
    toolset='model',
    schema={
        'name': 'model_router',
        'description': 'Route model calls through local model-router proxy with fallback and rotation',
        'parameters': {
            'type': 'object',
            'properties': {
                'prompt': {'type': 'string'},
                'task_type': {'type': 'string'},
                'timeout_total': {'type': 'number'},
            },
            'required': ['prompt']
        }
    },
    handler=lambda args, **kw: model_router_call(
        args.get('prompt'), args.get('task_type', 'minor'), args.get('timeout_total', 30)
    ),
    check_fn=lambda: True,
)
