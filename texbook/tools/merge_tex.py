#!/usr/bin/env python3
"""
merge_tex.py -- assemble a set of standalone LaTeX documents into one book.

For every source .tex it:
  * splits preamble / body
  * harvests the preamble (packages, macros, tikz libraries, colors, ...) into a
    single generated preamble, warning on conflicting duplicates
  * turns the document body into a chapter file (\\chapter{...} + content),
    demoting any internal \\chapter/\\part
  * namespaces every \\label and internal \\ref so cross-file collisions vanish
  * pulls every thebibliography entry out, merges them, deduplicates and
    rewrites the \\cite keys accordingly
  * writes chapters/<slug>.tex and a book.tex with \\part / \\include

Usage
-----
    python3 tools/merge_tex.py --src src --out . --manifest manifest.yaml
    python3 tools/merge_tex.py --src src --out . --scan      # no manifest, alphabetical

Nothing in the source folder is modified.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# preamble commands we know how to carry over verbatim
# --------------------------------------------------------------------------- #

VERBATIM_CMDS = [
    "newcommand", "renewcommand", "providecommand", "DeclareMathOperator",
    "DeclareRobustCommand", "newenvironment", "renewenvironment",
    "newtheorem", "theoremstyle", "definecolor", "colorlet",
    "newcolumntype", "DeclareSIUnit", "sisetup", "lstset", "lstdefinestyle",
    "captionsetup", "pgfplotsset", "tikzset", "tikzstyle", "hypersetup",
    "graphicspath", "setlength", "setcounter", "renewcommandx",
    "AtBeginDocument", "numberwithin",
]

LIB_CMDS = ["usetikzlibrary", "usepgfplotslibrary", "usepackage"]

# body-level commands that make no sense inside a chapter
STRIP_LINE_CMDS = [
    "maketitle", "tableofcontents", "listoffigures", "listoftables",
    "printindex", "printbibliography",
]
STRIP_ARG_CMDS = [
    "bibliographystyle", "bibliography", "thispagestyle", "pagestyle",
    "pagenumbering", "title", "author", "date", "hypersetup", "addbibresource",
]

SECTION_DEMOTION = [
    ("subsubsection", "paragraph"),
    ("subsection", "subsubsection"),
    ("section", "subsection"),
    ("chapter", "section"),
]

REF_CMDS = [
    "ref", "eqref", "autoref", "pageref", "cref", "Cref", "crefrange",
    "Crefrange", "nameref", "vref", "labelcref",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "chapter"


def strip_comments(text: str) -> str:
    """Remove % comments but keep \\% and keep the newline structure."""
    out = []
    for line in text.split("\n"):
        res, i, n = [], 0, len(line)
        while i < n:
            c = line[i]
            if c == "\\" and i + 1 < n:
                res.append(line[i:i + 2])
                i += 2
                continue
            if c == "%":
                break
            res.append(c)
            i += 1
        out.append("".join(res))
    return "\n".join(out)


def grab_command(text: str, start: int) -> tuple[str, int]:
    """
    Starting at index `start` (which must point at a backslash), consume the
    command name plus every immediately following [..] / {..} group, honouring
    nesting and escaped braces. Returns (raw_text, end_index).
    """
    i = start + 1
    n = len(text)
    while i < n and (text[i].isalpha() or text[i] == "@"):
        i += 1
    # consume argument groups, allowing whitespace/newline between them
    while True:
        j = i
        while j < n and text[j] in " \t\n":
            j += 1
        if j >= n or text[j] not in "[{":
            break
        opener, closer = ("[", "]") if text[j] == "[" else ("{", "}")
        depth, k = 0, j
        while k < n:
            if text[k] == "\\":
                k += 2
                continue
            if text[k] == opener:
                depth += 1
            elif text[k] == closer:
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        if depth != 0:
            break  # unbalanced -- give up quietly
        i = k
    return text[start:i], i


def find_commands(text: str, names: list[str]) -> list[tuple[str, int, int]]:
    """Return [(raw, start, end)] for every occurrence of the given commands."""
    hits = []
    pattern = re.compile(r"\\(" + "|".join(re.escape(n) for n in names) + r")(?![a-zA-Z@])")
    for m in pattern.finditer(text):
        raw, end = grab_command(text, m.start())
        hits.append((raw, m.start(), end))
    return hits


def first_arg(raw: str) -> str:
    m = re.search(r"\{", raw)
    if not m:
        return ""
    depth, i, n = 0, m.start(), len(raw)
    while i < n:
        if raw[i] == "\\":
            i += 2
            continue
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[m.start() + 1:i]
        i += 1
    return ""


def opt_arg(raw: str) -> str:
    m = re.match(r"\\[A-Za-z@]+\s*\[([^\]]*)\]", raw)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #

@dataclass
class Doc:
    path: str
    slug: str
    title: str
    part: str = ""
    body: str = ""
    packages: dict[str, str] = field(default_factory=dict)   # name -> options
    macros: list[str] = field(default_factory=list)
    libs: dict[str, set] = field(default_factory=dict)       # cmd -> {items}
    bib: list[tuple[str, str]] = field(default_factory=list) # (key, text)
    leftovers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)  # emitted at chapter start
    restores: list[str] = field(default_factory=list)   # emitted at chapter end


# --------------------------------------------------------------------------- #
# stage 1: split and harvest the preamble
# --------------------------------------------------------------------------- #

def split_document(raw: str) -> tuple[str, str]:
    m = re.search(r"\\begin\s*\{document\}", raw)
    if not m:
        return "", raw
    pre = raw[:m.start()]
    rest = raw[m.end():]
    e = re.search(r"\\end\s*\{document\}", rest)
    body = rest[:e.start()] if e else rest
    return pre, body


def harvest_preamble(doc: Doc, pre: str) -> None:
    pre_nc = strip_comments(pre)
    consumed = []

    for raw, s, e in find_commands(pre_nc, ["usepackage", "RequirePackage"]):
        opts = opt_arg(raw)
        for name in [p.strip() for p in first_arg(raw).split(",") if p.strip()]:
            if name in doc.packages and doc.packages[name] != opts:
                doc.warnings.append(
                    f"package '{name}' requested twice with different options "
                    f"('{doc.packages[name]}' vs '{opts}')")
            doc.packages[name] = opts
        consumed.append((s, e))

    for cmd in ["usetikzlibrary", "usepgfplotslibrary"]:
        for raw, s, e in find_commands(pre_nc, [cmd]):
            items = {x.strip() for x in first_arg(raw).split(",") if x.strip()}
            doc.libs.setdefault(cmd, set()).update(items)
            consumed.append((s, e))

    for raw, s, e in find_commands(pre_nc, VERBATIM_CMDS):
        doc.macros.append(raw.strip())
        consumed.append((s, e))

    # intentionally dropped: the book has its own title page
    for raw, s, e in find_commands(pre_nc, [
            "documentclass", "input", "include", "title", "author", "date",
            "institute", "keywords", "makeatletter", "makeatother"]):
        consumed.append((s, e))

    # whatever is left and not blank needs a human eye
    consumed.sort()
    cursor, chunks = 0, []
    for s, e in consumed:
        chunks.append(pre_nc[cursor:s])
        cursor = max(cursor, e)
    chunks.append(pre_nc[cursor:])
    rest = "\n".join(c.strip() for c in chunks if c.strip())
    for line in rest.split("\n"):
        if line.strip():
            doc.leftovers.append(line.strip())


# --------------------------------------------------------------------------- #
# stage 2: clean the body
# --------------------------------------------------------------------------- #

def extract_title(pre: str, body: str) -> str:
    for src in (pre, body):
        for raw, _, _ in find_commands(strip_comments(src), ["title"]):
            t = first_arg(raw).strip()
            if t:
                return re.sub(r"\\\\|\\thanks\{[^}]*\}", " ", t).strip()
    return ""


def extract_bibliography(doc: Doc, body: str) -> str:
    out = body
    for m in list(re.finditer(
            r"\\begin\s*\{thebibliography\}(?:\{[^}]*\})?(.*?)\\end\s*\{thebibliography\}",
            out, re.S)):
        inner = m.group(1)
        items = re.split(r"\\bibitem", inner)
        for it in items[1:]:
            it = it.strip()
            if it.startswith("["):
                it = it[it.index("]") + 1:].strip()
            if not it.startswith("{"):
                continue
            key = first_arg(it)
            text = it[it.index("}") + 1:].strip()
            text = re.sub(r"\s+", " ", text)
            if key:
                doc.bib.append((key, text))
        out = out.replace(m.group(0), "")
    return out


def clean_body(doc: Doc, body: str) -> str:
    out = body

    # remove one-shot commands
    for cmd in STRIP_LINE_CMDS:
        out = re.sub(r"\\" + cmd + r"(?![a-zA-Z@])\s*", "", out)
    for raw, _, _ in sorted(find_commands(out, STRIP_ARG_CMDS),
                            key=lambda t: -t[1]):
        out = out.replace(raw, "", 1)

    # abstract -> a chapter-opening summary environment defined in the preamble
    out = re.sub(r"\\begin\s*\{abstract\}", r"\\begin{chapabstract}", out)
    out = re.sub(r"\\end\s*\{abstract\}", r"\\end{chapabstract}", out)

    # \appendix inside a chapter would restart numbering for the whole book
    if re.search(r"\\appendix(?![a-zA-Z@])", out):
        doc.warnings.append("\\appendix found inside the document -- replaced by a plain section")
        out = re.sub(r"\\appendix(?![a-zA-Z@])", r"\\section*{Appendix}", out)

    # demote internal sectioning if the file already uses \chapter or \part
    if re.search(r"\\(chapter|part)\*?\s*[\[{]", out):
        for old, new in SECTION_DEMOTION:
            out = re.sub(r"\\" + old + r"(\*?)(\s*[\[{])", r"\\" + new + r"\1\2", out)
        out = re.sub(r"\\part(\*?)(\s*[\[{])", r"\\section\1\2", out)
        doc.warnings.append("internal \\chapter/\\part demoted one level")

    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def namespace_labels(doc: Doc, body: str) -> str:
    pfx = doc.slug + ":"
    labels = {first_arg(raw) for raw, _, _ in find_commands(body, ["label"])}
    labels = {l for l in labels if l}
    if not labels:
        return body

    out = re.sub(r"\\label\s*\{([^}]*)\}",
                 lambda m: r"\label{" + pfx + m.group(1) + "}", body)

    def fix_refs(m):
        cmd, keys = m.group(1), m.group(2)
        new = ",".join((pfx + k.strip()) if k.strip() in labels else k.strip()
                       for k in keys.split(","))
        return "\\" + cmd + "{" + new + "}"

    out = re.sub(r"\\(" + "|".join(REF_CMDS) + r")\s*\{([^}]*)\}", fix_refs, out)
    return out


# --------------------------------------------------------------------------- #
# stage 3: global bibliography merge
# --------------------------------------------------------------------------- #

def merge_bibliography(docs: list[Doc]) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns (ordered entries, warnings). Rewrites \\cite keys in place."""
    canonical: dict[str, str] = {}   # normalised text -> global key
    by_key: dict[str, str] = {}      # global key -> text
    warnings: list[str] = []
    order: list[tuple[str, str]] = []

    for doc in docs:
        remap: dict[str, str] = {}
        for key, text in doc.bib:
            norm = re.sub(r"[^a-z0-9]+", "", text.lower())[:180]
            if norm in canonical:
                remap[key] = canonical[norm]
                continue
            gkey = key
            if gkey in by_key:
                gkey = f"{key}-{doc.slug}"
                warnings.append(
                    f"[{doc.slug}] bibitem key '{key}' already used for a different "
                    f"entry -> renamed '{gkey}'")
            by_key[gkey] = text
            canonical[norm] = gkey
            remap[key] = gkey
            order.append((gkey, text))

        if remap:
            def fix_cite(m):
                cmd, keys = m.group(1), m.group(2)
                new = ",".join(remap.get(k.strip(), k.strip()) for k in keys.split(","))
                return "\\" + cmd + (m.group(0)[len(cmd) + 1:m.group(0).index("{")]) + "{" + new + "}"

            doc.body = re.sub(
                r"\\(cite|citep|citet|autocite)((?:\s*\[[^\]]*\])*)\s*\{([^}]*)\}",
                lambda m: "\\" + m.group(1) + m.group(2) + "{" +
                          ",".join(remap.get(k.strip(), k.strip())
                                   for k in m.group(3).split(",")) + "}",
                doc.body)

        # flag citations with no matching bibitem anywhere in this file
        for m in re.finditer(r"\\cite\w*(?:\s*\[[^\]]*\])*\s*\{([^}]*)\}", doc.body):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in by_key:
                    warnings.append(f"[{doc.slug}] \\cite{{{k}}} has no matching \\bibitem")
    return order, warnings


# --------------------------------------------------------------------------- #
# stage 4: emit
# --------------------------------------------------------------------------- #

PREAMBLE_HEADER = r"""% ------------------------------------------------------------------
% preamble_generated.tex -- produced by tools/merge_tex.py
% Union of the preambles of every source document.
% Edit preamble.tex (hand written) rather than this file.
% ------------------------------------------------------------------
"""


REDEFINABLE = ("\\newcommand", "\\renewcommand", "\\providecommand")


def as_renew(defn: str) -> str:
    for pfx in REDEFINABLE:
        if defn.startswith(pfx):
            return "\\renewcommand" + defn[len(pfx):]
    return defn


def resolve_macro_conflicts(docs: list[Doc]) -> list[str]:
    """
    A macro defined with the same name but different bodies in two documents is
    kept globally with its first definition; every later chapter gets a local
    \\renewcommand at its start and the canonical definition restored at its end,
    so no chapter silently uses another chapter's notation.
    """
    canonical: dict[str, tuple[str, str]] = {}   # name -> (slug, definition)
    warnings: list[str] = []
    for doc in docs:
        for m in doc.macros:
            if not m.startswith(REDEFINABLE):
                continue
            name = first_arg(m)
            if not name:
                continue
            body = re.sub(r"\s+", "", m)
            if name not in canonical:
                canonical[name] = (doc.slug, m)
                continue
            if re.sub(r"\s+", "", canonical[name][1]) == body:
                continue
            doc.overrides.append(as_renew(m))
            doc.restores.append(as_renew(canonical[name][1]))
            warnings.append(
                f"macro {name}: '{canonical[name][0]}' definition kept globally, "
                f"'{doc.slug}' redefines it locally for its own chapter")
    return warnings


def write_preamble(path: str, docs: list[Doc]) -> list[str]:
    packages: dict[str, str] = {}
    warnings: list[str] = []
    skip = {"geometry", "hyperref", "inputenc", "fontenc", "babel", "biblatex",
            "lmodern", "fancyhdr"}
    for doc in docs:
        for name, opts in doc.packages.items():
            if name in skip:
                packages.setdefault(name, opts)
                continue
            if name in packages and packages[name] != opts:
                warnings.append(
                    f"package '{name}': options '{packages[name]}' vs '{opts}' "
                    f"(from {doc.slug}) -- keeping the first, check by hand")
                continue
            packages.setdefault(name, opts)

    libs: dict[str, set] = {}
    for doc in docs:
        for cmd, items in doc.libs.items():
            libs.setdefault(cmd, set()).update(items)

    macros, seen = [], set()
    for doc in docs:
        for m in doc.macros:
            key = re.sub(r"\s+", "", m)
            if key in seen:
                continue
            seen.add(key)
            macros.append((doc.slug, first_arg(m), m))

    lines = [PREAMBLE_HEADER]
    lines.append("% --- packages ---")
    for name in sorted(packages):
        if name in skip:
            lines.append(f"% (handled in preamble.tex) \\usepackage{{{name}}}")
            continue
        opts = packages[name]
        lines.append(f"\\usepackage{{{name}}}" if not opts
                     else f"\\usepackage[{opts}]{{{name}}}")
    lines.append("")
    for cmd in sorted(libs):
        if libs[cmd]:
            lines.append(f"\\{cmd}{{{','.join(sorted(libs[cmd]))}}}")
    lines.append("")
    lines.append("% --- macros and settings collected from the sources ---")
    emitted = set()
    for slug, name, m in macros:
        if name in emitted and name:
            lines.append(f"% [dropped duplicate from {slug}] {m.splitlines()[0]}")
            continue
        emitted.add(name)
        lines.append(m)
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return warnings


def write_chapter(out_dir: str, doc: Doc) -> str:
    rel = os.path.join("chapters", doc.slug + ".tex")
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    head = [
        f"% chapters/{doc.slug}.tex",
        f"% generated from {os.path.basename(doc.path)}",
        f"\\chapter{{{doc.title}}}",
        f"\\label{{ch:{doc.slug}}}",
        "",
    ]
    if doc.overrides:
        head.append("% notation local to this chapter (conflicts with another chapter)")
        head += doc.overrides
        head.append("")
    tail = []
    if doc.restores:
        tail = ["", "% restore the book-wide notation", *doc.restores]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head) + doc.body + "\n" + "\n".join(tail) + "\n")
    return rel.replace(os.sep, "/")[:-4]  # \include wants no extension


def write_bibliography(path: str, entries: list[tuple[str, str]]) -> None:
    width = "9" * max(1, len(str(len(entries))))
    lines = [
        "% bibliography.tex -- merged from every source document",
        f"\\begin{{thebibliography}}{{{width}}}",
        "",
    ]
    for key, text in entries:
        lines.append(f"\\bibitem{{{key}}} {text}")
        lines.append("")
    lines.append("\\end{thebibliography}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_book(path: str, docs: list[Doc], includes: dict[str, str]) -> None:
    lines = [
        "% book.tex -- master file (generated section below is rewritten by merge_tex.py)",
        "\\documentclass[11pt,a4paper,oneside]{book}",
        "\\input{preamble}",
        "\\input{preamble_generated}",
        "",
        "\\begin{document}",
        "\\frontmatter",
        "\\input{frontmatter}",
        "\\tableofcontents",
        "",
        "\\mainmatter",
        "% >>> BEGIN GENERATED CHAPTER LIST",
    ]
    current_part = None
    for doc in docs:
        if doc.part and doc.part != current_part:
            current_part = doc.part
            lines.append("")
            lines.append(f"\\part{{{doc.part}}}")
        lines.append(f"\\include{{{includes[doc.slug]}}}")
    lines += [
        "% <<< END GENERATED CHAPTER LIST",
        "",
        "\\backmatter",
        "\\input{bibliography}",
        "\\end{document}",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def read_manifest(path: str) -> list[dict]:
    """
    Minimal reader for the manifest format (no PyYAML dependency):

        - part: Power semiconductor devices
        - file: power_diode.tex
          title: The PIN power diode
          slug: power-diode
    """
    entries, cur, part = [], None, ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            s = raw.strip()
            if s.startswith("- part:"):
                part = s.split(":", 1)[1].strip()
                cur = None
                continue
            if s.startswith("- file:"):
                cur = {"file": s.split(":", 1)[1].strip(), "part": part}
                entries.append(cur)
                continue
            if cur is not None and ":" in s:
                k, v = s.split(":", 1)
                cur[k.strip()] = v.strip()
    return entries


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="src", help="folder holding the source .tex files")
    ap.add_argument("--out", default=".", help="book root (book.tex lives here)")
    ap.add_argument("--manifest", default="manifest.yaml",
                    help="ordering file; ignored with --scan")
    ap.add_argument("--scan", action="store_true",
                    help="ignore the manifest, take every .tex in --src alphabetically")
    args = ap.parse_args()

    if args.scan or not os.path.exists(args.manifest):
        if not args.scan:
            print(f"note: {args.manifest} not found, falling back to --scan")
        files = sorted(f for f in os.listdir(args.src) if f.endswith(".tex"))
        entries = [{"file": f, "part": ""} for f in files]
    else:
        entries = read_manifest(args.manifest)

    docs: list[Doc] = []
    missing: list[str] = []
    for ent in entries:
        path = os.path.join(args.src, ent["file"])
        if not os.path.exists(path):
            missing.append(ent["file"])
            continue
        raw = open(path, encoding="utf-8", errors="replace").read()
        pre, body = split_document(raw)
        title = ent.get("title") or extract_title(pre, body) \
            or os.path.splitext(ent["file"])[0].replace("_", " ").title()
        slug = ent.get("slug") or slugify(os.path.splitext(ent["file"])[0])
        doc = Doc(path=path, slug=slug, title=title, part=ent.get("part", ""))
        harvest_preamble(doc, pre)
        body = extract_bibliography(doc, body)
        body = clean_body(doc, body)
        doc.body = namespace_labels(doc, body)
        docs.append(doc)

    if not docs:
        print("nothing to do: no source files found", file=sys.stderr)
        return 1

    bib_entries, bib_warn = merge_bibliography(docs)
    macro_warn = resolve_macro_conflicts(docs)

    includes = {d.slug: write_chapter(args.out, d) for d in docs}
    pre_warn = write_preamble(os.path.join(args.out, "preamble_generated.tex"), docs)
    write_bibliography(os.path.join(args.out, "bibliography.tex"), bib_entries)
    write_book(os.path.join(args.out, "book.tex"), docs, includes)

    # ------------------------------------------------------------------ report
    print(f"chapters written : {len(docs)}")
    print(f"bib entries      : {len(bib_entries)}")
    if missing:
        print("\nMISSING SOURCE FILES")
        for f in missing:
            print(f"  - {f}")
    problems = False
    for doc in docs:
        msgs = list(doc.warnings)
        if doc.leftovers:
            msgs.append("preamble lines not carried over: "
                        + " | ".join(doc.leftovers[:6])
                        + (" ..." if len(doc.leftovers) > 6 else ""))
        if msgs:
            problems = True
            print(f"\n[{doc.slug}]")
            for m in msgs:
                print(f"  ! {m}")
    globals_ = pre_warn + macro_warn + bib_warn
    if globals_:
        problems = True
        print("\n[book-wide]")
        for m in globals_:
            print(f"  ! {m}")
    if not problems:
        print("\nno conflicts detected")
    print("\nnow run:  pdflatex book && pdflatex book && pdflatex book")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
