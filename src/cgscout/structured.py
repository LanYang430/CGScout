"""JSON-schema helper for the older `(system_prompt, user_message, schema_dict) -> str` call convention (CallFn).

`_schema(model_cls)` is the pydantic JSON schema of a response model with the class-level `description`s dropped; the CallFn classifiers
(terminology.expand_terminology) pass it as the call's third argument. llm.make_call_fn ignores that argument and validates
against the response class itself, so on the live path this value has no consumer (kept unchanged on purpose -- see the plan's item B10).
"""
from __future__ import annotations

from pydantic import BaseModel


def _schema(model_cls: type[BaseModel]) -> dict:
    schema = model_cls.model_json_schema()
    schema.pop("description", None)
    for defn in schema.get("$defs", {}).values():
        defn.pop("description", None)
    return schema
