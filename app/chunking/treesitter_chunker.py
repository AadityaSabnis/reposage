"""AST-aware chunking via tree-sitter.

Emits one chunk per top-level function, per top-level class, and per
method inside a top-level class. Decorators (`@app.get(...)`) and
`export` keywords are folded into the chunk's line range so a citation
points at the whole construct, not just the body.

Design notes
------------
* The per-language behaviour lives in `LANG_CONFIG`, a small registry of
  tree-sitter node types. Adding a language = adding one entry there,
  not touching the walker.
* Heavy import (`tree_sitter_languages`) is lazy so the rest of the app
  (and the fallback chunker) imports fine on machines without the native
  grammars; `is_available()` reports whether AST chunking can run.
* If parsing yields zero symbols (e.g. a script that is all top-level
  statements), the caller (`registry.chunk_file`) falls back to the
  line-window chunker so content is never silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.models import Chunk

# tree_sitter_languages uses these names for get_parser().
TS_PARSER_NAME = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
}


@dataclass(frozen=True)
class LangConfig:
    func_types: Set[str]                 # -> symbol_type "function"
    class_types: Set[str]                # -> symbol_type "class" (descend for methods)
    method_types: Set[str]               # node types that are methods inside a class body
    var_decl_types: Set[str] = field(default_factory=set)   # const f = () => {}
    func_value_types: Set[str] = field(default_factory=set)  # arrow_function, ...
    type_decl_types: Dict[str, str] = field(default_factory=dict)  # node type -> symbol_type
    wrappers: Set[str] = field(default_factory=set)          # export/decorated wrappers


LANG_CONFIG: Dict[str, LangConfig] = {
    "python": LangConfig(
        func_types={"function_definition"},
        class_types={"class_definition"},
        method_types={"function_definition"},
        wrappers={"decorated_definition"},
    ),
    "javascript": LangConfig(
        func_types={"function_declaration", "generator_function_declaration"},
        class_types={"class_declaration"},
        method_types={"method_definition"},
        var_decl_types={"lexical_declaration", "variable_declaration"},
        func_value_types={"arrow_function", "function_expression", "function"},
        wrappers={"export_statement"},
    ),
    "typescript": LangConfig(
        func_types={"function_declaration", "generator_function_declaration"},
        class_types={"class_declaration", "abstract_class_declaration"},
        method_types={"method_definition"},
        var_decl_types={"lexical_declaration", "variable_declaration"},
        func_value_types={"arrow_function", "function_expression", "function"},
        type_decl_types={
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "type_alias_declaration": "type",
        },
        wrappers={"export_statement"},
    ),
}
LANG_CONFIG["tsx"] = LANG_CONFIG["typescript"]

# Reuse parsers across calls — they are expensive to build.
_PARSER_CACHE: Dict[str, object] = {}


def is_available() -> bool:
    try:
        import tree_sitter_languages  # noqa: F401
        return True
    except Exception:
        return False


def _get_parser(language: str):
    name = TS_PARSER_NAME[language]
    if name not in _PARSER_CACHE:
        try:
            from tree_sitter_languages import get_parser
        except ImportError as e:
            raise RuntimeError(
                f"tree-sitter-languages not installed; using line-window fallback for {language}. "
                "Install with: pip install tree-sitter-languages"
            ) from e
        _PARSER_CACHE[name] = get_parser(name)
    return _PARSER_CACHE[name]


def _node_name(node) -> Optional[str]:
    name_node = node.child_by_field_name("name")
    if name_node is not None and name_node.text:
        return name_node.text.decode("utf-8", "replace")
    return None


def _unwrap(node, cfg: LangConfig):
    """Resolve export/decorated wrappers to (inner_def, span_node).

    `span_node` is the outer wrapper so decorators / `export` are part of
    the chunk's line range; `inner_def` is what we read the name/type from.
    """
    if node.type in cfg.wrappers:
        known = (
            cfg.func_types | cfg.class_types | cfg.var_decl_types | set(cfg.type_decl_types)
        )
        for child in node.named_children:
            if child.type in known:
                return child, node
    return node, node


def _make_chunk(
    span, name: str, symbol_type: str, file_path: str, language: str, source_lines: List[str]
) -> Chunk:
    start_line = span.start_point[0] + 1
    end_line = span.end_point[0] + 1
    # Build text from full source lines (not node.text) so the chunk equals the
    # exact cited lines — node.text drops the leading indentation of line 1.
    chunk_text = "\n".join(source_lines[start_line - 1:end_line])
    return Chunk(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=name,
        symbol_type=symbol_type,
        language=language,
        text=chunk_text,
    )


def _class_methods(
    class_node, class_name: str, cfg: LangConfig, file_path: str, language: str,
    source_lines: List[str],
) -> List[Chunk]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    out: List[Chunk] = []
    for member in body.named_children:
        inner, span = _unwrap(member, cfg)
        if inner.type in cfg.method_types or inner.type in cfg.func_types:
            mname = _node_name(inner)
            if mname:
                out.append(
                    _make_chunk(span, f"{class_name}.{mname}", "method",
                                file_path, language, source_lines)
                )
    return out


def chunk(text: str, file_path: str, language: str) -> List[Chunk]:
    """Parse `text` and return AST-level chunks. Empty list => caller falls back."""
    if language not in LANG_CONFIG:
        return []

    try:
        parser = _get_parser(language)
        tree = parser.parse(text.encode("utf-8"))
    except Exception:
        return []

    cfg = LANG_CONFIG[language]
    root = tree.root_node
    source_lines = text.splitlines()
    chunks: List[Chunk] = []

    for top in root.named_children:
        inner, span = _unwrap(top, cfg)
        ntype = inner.type

        if ntype in cfg.class_types:
            cname = _node_name(inner)
            if not cname:
                continue
            chunks.append(_make_chunk(span, cname, "class", file_path, language, source_lines))
            chunks.extend(_class_methods(inner, cname, cfg, file_path, language, source_lines))

        elif ntype in cfg.func_types:
            fname = _node_name(inner)
            if fname:
                chunks.append(_make_chunk(span, fname, "function", file_path, language, source_lines))

        elif ntype in cfg.type_decl_types:
            tname = _node_name(inner)
            if tname:
                chunks.append(
                    _make_chunk(span, tname, cfg.type_decl_types[ntype],
                                file_path, language, source_lines)
                )

        elif ntype in cfg.var_decl_types:
            # const handler = () => {...}  /  let foo = function () {...}
            for decl in inner.named_children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                if value is not None and value.type in cfg.func_value_types:
                    vname = _node_name(decl)
                    if vname:
                        chunks.append(
                            _make_chunk(span, vname, "function", file_path, language, source_lines)
                        )

    return chunks
