"""Request recorder: logs EVERY search the pipeline asks of OpenAlex (also those answered from a cache), with the route that asked.

Production code is not modified: `search_and_dedupe` is wrapped in each module namespace that imported it.  The route is read from the CALLER's
qualified name (Python >= 3.11 `co_qualname`) -- flat / need_semantic / 3A / hop1 / hop2 / canonical -- hop1 vs hop2 share one closure and are split
by search mode (Hop-1 is semantic, Hop-2 lexical).  Flat queries additionally get their (route, role) labels from the deterministic plan.
Each record: kind, mode, limit, query, served_from (http | cache), n_results, seed_rank, the DOIs/titles of the returned papers in rank order, errors.
"""
from __future__ import annotations
import sys, time, importlib

MODULES = ["cgscout.retrieval.search_adapters", "cgscout.retrieval.orchestrator", "cgscout.retrieval.direct.from_planner",
           "cgscout.retrieval.direct.route_plan"]
_KINDS = [("need_semantic_search_fn", "need_semantic"), ("flat_search_fn", "flat"), ("_run_direct_concept_search", "3A"),
          ("structural_hop1_search_fn", "hop1"), ("structural_hop2_search_fn", "hop2"),
          ("_default_flat_search", "flat")]


def classify(qualname: str, mode: str) -> str:
    for needle, kind in _KINDS:
        if needle in qualname:
            return kind
    if "_make_search_fn" in qualname:                  # Hop-1 / Hop-2 closure of transfer_retrieval_bridge
        return "hop1" if mode == "semantic" else "hop2"
    return "other:" + qualname


class Recorder:
    def __init__(self, usage_snapshot, is_seed, flat_roles=None):
        """usage_snapshot(): {'http_requests': int, 'cache_hits': int} of the current run label;  is_seed(paper) -> bool;
        flat_roles: {query: [[route, role], ...]} from the deterministic plan."""
        self.snap, self.is_seed, self.flat_roles = usage_snapshot, is_seed, flat_roles or {}
        self.records: list[dict] = []
        self._orig: dict[str, object] = {}

    def install(self):
        for name in MODULES:
            m = sys.modules.get(name) or importlib.import_module(name)
            orig = getattr(m, "search_and_dedupe", None)
            if orig is None or getattr(orig, "_is_recorder", False):
                continue
            self._orig[name] = orig
            setattr(m, "search_and_dedupe", self._wrap(orig))
        return self

    def uninstall(self):
        for name, orig in self._orig.items():
            setattr(sys.modules[name], "search_and_dedupe", orig)
        self._orig.clear()

    def _wrap(self, orig):
        rec = self

        def wrapper(query, limit=20, *, sources=("openalex",), mode="lexical"):
            caller = sys._getframe(1).f_code.co_qualname
            before = rec.snap()
            t0 = time.time()
            err = None
            try:
                papers = orig(query, limit=limit, sources=sources, mode=mode)
            except BaseException as e:               # budget exhaustion must still propagate; record it first
                rec._add(caller, query, limit, mode, before, [], f"{type(e).__name__}: {str(e)[:120]}", t0)
                raise
            wrapper.last_errors = list(getattr(orig, "last_errors", []))
            rec._add(caller, query, limit, mode, before, papers, "; ".join(wrapper.last_errors) or None, t0)
            return papers

        wrapper._is_recorder = True
        wrapper.last_errors = []
        return wrapper

    def _add(self, caller, query, limit, mode, before, papers, err, t0):
        after = self.snap()
        kind = classify(caller, mode)
        served = "http" if after["http_requests"] > before["http_requests"] else "cache"
        seed_rank = next((i for i, p in enumerate(papers, 1) if self.is_seed(p)), None)
        self.records.append({
            "n": len(self.records) + 1, "kind": kind, "mode": mode, "limit": limit, "query": query, "served_from": served,
            "n_results": len(papers), "seed_rank": seed_rank, "error": err, "seconds": round(time.time() - t0, 2),
            "plan_labels": self.flat_roles.get(query) if kind in ("flat", "need_semantic") else None,
            "results": [{"rank": i, "doi": getattr(p, "doi", None), "title": (p.title or "")[:110], "year": getattr(p, "year", None)}
                        for i, p in enumerate(papers, 1)],
        })
