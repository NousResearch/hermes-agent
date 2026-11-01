"""Shared tokenizer identity and conversation counting for the offline dataset pipeline."""

import hashlib
import json
import logging
import re
from pathlib import Path


logger = logging.getLogger(__name__)
_ROLES = {"human": "user", "gpt": "assistant"}


def resolve_tokenizer_revision(name, revision=None):
    """Resolve a Hub ref once so workers and compression cannot load different revisions."""
    if Path(name).is_dir() or (revision and re.fullmatch(r"[0-9a-f]{40}", revision)):
        return revision
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    resolved = get_tokenizer_config(name, revision=revision).get("_commit_hash")
    if not resolved:
        raise ValueError(f"Cannot resolve tokenizer revision for {name!r}; use a pinned commit or a local tokenizer directory")
    return resolved


def load_trajectory_tokenizer(name, revision=None, trust_remote_code=True):
    from transformers import AutoTokenizer

    revision = resolve_tokenizer_revision(name, revision)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code)
    # A missing/ambiguous template is an error, never an excuse to estimate a dataset budget.
    template = tokenizer.get_chat_template()
    metadata = {
        "name": name,
        "revision": revision,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "special_token_ids": {kind: getattr(tokenizer, f"{kind}_token_id") for kind in ("bos", "eos", "unk")},
    }
    print(f"Tokenizer: {json.dumps(metadata, sort_keys=True)}")
    return tokenizer, metadata


def count_conversation_tokens(tokenizer, trajectory):
    """Count the complete training serialization for ShareGPT or role/content messages."""
    if not trajectory:
        return 0
    messages = []
    for turn in trajectory:
        message = dict(turn)
        role = message.pop("from", message.get("role"))
        message["role"] = _ROLES.get(role, role)
        if "value" in message:
            message["content"] = message.pop("value")
        messages.append(message)
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False,
    )
    # Templates choose their own BOS/EOS conventions; there is no universal expected count.
    special_counts = {
        kind: token_ids.count(token_id) if (token_id := getattr(tokenizer, f"{kind}_token_id", None)) is not None else 0
        for kind in ("bos", "eos", "unk")
    }
    logger.debug("Conversation tokens: total=%d, special=%s", len(token_ids), special_counts)
    if special_counts["unk"]:
        logger.warning("Conversation contains %d UNK tokens with the selected tokenizer", special_counts["unk"])
    return len(token_ids)
