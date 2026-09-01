"""Coverage XML path normalizer.

Detects coverage format (Clover/Cobertura/JaCoCo) and normalizes Docker-internal
paths so diff-cover can match them against git-relative file paths.

Also exposes extract_reported_files()/is_in_report(), which answer a question
diff-cover cannot: is a file present in the coverage report at all?
"""

from __future__ import annotations

import os
import tempfile
from enum import Enum
from pathlib import Path
from xml.etree import ElementTree as ET


class CoverageFormat(Enum):
    CLOVER = "clover"
    COBERTURA = "cobertura"
    JACOCO = "jacoco"
    UNKNOWN = "unknown"


# LCOV tracefiles are plain text, not XML — handled separately by the
# report-presence extractor. diff-cover reads them natively, so they are never
# path-normalized here.
_LCOV_SUFFIXES = (".info", ".lcov", ".dat")


def detect_format(filepath: str) -> CoverageFormat:
    """Detect the coverage XML format by inspecting root structure.

    Returns CoverageFormat.UNKNOWN for non-XML files or unrecognized formats.
    """
    if not filepath.endswith(".xml"):
        return CoverageFormat.UNKNOWN

    try:
        tree = ET.parse(filepath)  # noqa: S314
    except ET.ParseError:
        return CoverageFormat.UNKNOWN

    root = tree.getroot()

    # Clover: <coverage> root with <project> child
    if root.tag == "coverage" and root.find("project") is not None:
        return CoverageFormat.CLOVER

    # JaCoCo: <report> root with <package>/<sourcefile> structure
    if root.tag == "report":
        pkg = root.find(".//package")
        if pkg is not None and pkg.find("sourcefile") is not None:
            return CoverageFormat.JACOCO

    # Cobertura: <coverage> root with <packages> child
    if root.tag == "coverage" and root.find("packages") is not None:
        return CoverageFormat.COBERTURA

    return CoverageFormat.UNKNOWN


def detect_path_prefix(xml_paths: list[str], diff_files: list[str]) -> str:
    """Auto-detect the path prefix in coverage XML that maps to git-relative paths.

    Uses suffix-matching: for each absolute XML path, checks if any diff_file
    is a suffix. The prefix is whatever remains before the matched suffix.

    Returns the prefix string (including trailing separator) or "" if no prefix detected.
    """
    if not xml_paths or not diff_files:
        return ""

    diff_set = set(diff_files)

    absolute_xml_paths = [
        p
        for p in xml_paths
        if p.startswith("/") or (len(p) >= 3 and p[1] == ":" and p[2] in ("/", "\\"))
    ]
    if not absolute_xml_paths:
        return ""

    for xml_path in absolute_xml_paths:
        normalized = xml_path.replace("\\", "/")
        for diff_file in diff_set:
            if normalized.endswith("/" + diff_file):
                return xml_path[: len(xml_path) - len(diff_file)]

    return ""


def _extract_clover_paths(root: ET.Element) -> list[str]:
    """Extract all file paths from a Clover XML tree."""
    paths: list[str] = []
    for file_elem in root.findall(".//file"):
        name = file_elem.get("name") or file_elem.get("path") or ""
        if name:
            paths.append(name)
    return paths


def _extract_clover_report_paths(root: ET.Element) -> list[str]:
    """Extract every path candidate from a Clover XML tree.

    Deliberately returns BOTH attributes of each <file>, where
    _extract_clover_paths() prefers "name": reporters disagree on which one is
    fully qualified. PHPUnit puts the full path in name=; Jest's clover reporter
    puts only the basename there and the real path in path=. A presence check
    needs whichever attribute is qualified, so it takes both and lets
    is_in_report() do the matching.
    """
    paths: list[str] = []
    for file_elem in root.findall(".//file"):
        paths.extend(val for attr in ("name", "path") if (val := file_elem.get(attr)))
    return paths


def _is_absolute_path(path: str) -> bool:
    """Check for a POSIX ("/x") or Windows ("C:/x", "C:\\x") absolute path."""
    return path.startswith("/") or (len(path) >= 3 and path[1] == ":" and path[2] in ("/", "\\"))


def _extract_cobertura_paths(root: ET.Element) -> list[str]:
    """Extract all source paths from a Cobertura XML tree.

    <class filename="..."> is relative to the <sources><source> roots whenever
    the report was scoped to a subdirectory: coverage.py run as `--cov=src`
    emits filename="main.py" plus source=".../src", never "src/main.py". Each
    root is joined on so the result is comparable to a git path. Filenames that
    are already absolute are left alone.
    """
    sources = [
        stripped
        for stripped in (
            (s.text or "").replace("\\", "/").rstrip("/") for s in root.findall("sources/source")
        )
        if stripped
    ]

    paths: list[str] = []
    for clazz in root.findall(".//class"):
        fname = (clazz.get("filename") or "").replace("\\", "/")
        if not fname:
            continue
        paths.append(fname)
        if _is_absolute_path(fname):
            continue
        paths.extend(f"{source}/{fname}" for source in sources)
    return paths


def _extract_jacoco_paths(root: ET.Element) -> list[str]:
    """Extract package-qualified source paths from a JaCoCo XML tree.

    JaCoCo splits the path: <package name="com/foo"> holds <sourcefile
    name="Bar.java">. Joined here so the result is comparable to a git path.
    """
    paths: list[str] = []
    for pkg in root.findall(".//package"):
        pkg_name = (pkg.get("name") or "").replace("\\", "/").strip("/")
        for src in pkg.findall("sourcefile"):
            name = src.get("name") or ""
            if name:
                paths.append(f"{pkg_name}/{name}" if pkg_name else name)
    return paths


def _extract_lcov_paths(filepath: str) -> list[str]:
    """Extract source paths from an LCOV tracefile (one per SF: record)."""
    paths: list[str] = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("SF:"):
                    name = line[3:].strip()
                    if name:
                        paths.append(name)
    except OSError:
        return []
    return paths


def _rewrite_clover(filepath: str, prefix: str) -> str:
    """Rewrite Clover XML stripping prefix from all <file> name/path attributes.

    Also ensures diff-cover compatibility:
    - Root <coverage> gets a 'clover' attribute (diff-cover uses .[@clover] detection)
    - Each <file> gets a 'path' attribute (diff-cover reads file_tree.get("path"))
    """
    tree = ET.parse(filepath)  # noqa: S314
    root = tree.getroot()

    # diff-cover detects Clover via xml_document.findall(".[@clover]")
    # PHPUnit 10+ may omit this attribute — ensure it's present
    if root.get("clover") is None:
        root.set("clover", "test-guard")

    for file_elem in tree.findall(".//file"):
        for attr in ("name", "path"):
            val = file_elem.get(attr)
            if val and val.startswith(prefix):
                file_elem.set(attr, val[len(prefix) :])

        # diff-cover's Clover parser reads file_tree.get("path"), but PHPUnit
        # uses "name". Ensure "path" is always set for diff-cover compatibility.
        if file_elem.get("path") is None:
            name_val = file_elem.get("name")
            if name_val:
                file_elem.set("path", name_val)

    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="tg_clover_")
    os.close(fd)
    tree.write(tmp_path, encoding="unicode", xml_declaration=True)
    return tmp_path


def _normalize_cobertura(filepath: str, diff_files: list[str]) -> str:
    """Normalize Cobertura XML with absolute filenames."""
    try:
        tree = ET.parse(filepath)  # noqa: S314
    except ET.ParseError:
        return filepath

    root = tree.getroot()
    classes = root.findall(".//class")
    if not classes:
        return filepath

    filenames = [c.get("filename", "") for c in classes]
    has_absolute = any(
        f.startswith("/") or (len(f) >= 3 and f[1] == ":" and f[2] in ("/", "\\"))
        for f in filenames
        if f
    )
    if not has_absolute:
        return filepath

    prefix = detect_path_prefix(filenames, diff_files)
    if not prefix:
        return filepath

    print(
        f"::warning::Coverage path normalization: stripped prefix '{prefix}' "
        f"from {sum(1 for f in filenames if f.startswith(prefix))} filename(s) "
        f"in {Path(filepath).name} (Docker/container paths detected)"
    )

    for clazz in classes:
        fname = clazz.get("filename", "")
        if fname.startswith(prefix):
            clazz.set("filename", fname[len(prefix) :])

    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="tg_cobertura_")
    os.close(fd)
    tree.write(tmp_path, encoding="unicode", xml_declaration=True)
    return tmp_path


def normalize_coverage_file(filepath: str, diff_files: list[str]) -> str:
    """Normalize a coverage file's paths for diff-cover compatibility.

    Detects format, auto-detects Docker path prefixes, and rewrites if necessary.
    Returns the path to use (original or a temp rewritten copy).
    """
    if Path(filepath).suffix.lower() != ".xml":
        return filepath

    fmt = detect_format(filepath)

    if fmt == CoverageFormat.CLOVER:
        try:
            tree = ET.parse(filepath)  # noqa: S314
        except ET.ParseError:
            return filepath
        xml_paths = _extract_clover_paths(tree.getroot())
        prefix = detect_path_prefix(xml_paths, diff_files)
        if prefix:
            print(
                f"::warning::Coverage path normalization: stripped prefix '{prefix}' "
                f"from {len(xml_paths)} file path(s) in {Path(filepath).name} "
                f"(Docker/container paths detected)"
            )
            return _rewrite_clover(filepath, prefix)
        return filepath

    if fmt == CoverageFormat.COBERTURA:
        return _normalize_cobertura(filepath, diff_files)

    return filepath


def extract_reported_files(filepath: str) -> set[str]:
    """Return every source path a coverage report mentions, "/"-normalized.

    This answers a question diff-cover cannot. diff-cover's src_stats lists
    only files with changed *executable* lines, so a file whose diff touches
    nothing executable — type declarations, interface members, doc comments —
    is absent from src_stats even when it is fully instrumented and reported at
    100%. Telling "absent because there was nothing to measure" apart from
    "absent because it was never instrumented" needs the report itself.

    Returns an empty set for unknown or unparseable formats, which leaves
    callers on their conservative pre-existing behaviour.
    """
    suffix = Path(filepath).suffix.lower()

    if suffix in _LCOV_SUFFIXES:
        raw = _extract_lcov_paths(filepath)
    elif suffix == ".xml":
        try:
            root = ET.parse(filepath).getroot()  # noqa: S314
        except (ET.ParseError, OSError):
            return set()
        fmt = detect_format(filepath)
        if fmt == CoverageFormat.CLOVER:
            raw = _extract_clover_report_paths(root)
        elif fmt == CoverageFormat.COBERTURA:
            raw = _extract_cobertura_paths(root)
        elif fmt == CoverageFormat.JACOCO:
            raw = _extract_jacoco_paths(root)
        else:
            return set()
    else:
        return set()

    reported: set[str] = set()
    for path in raw:
        cleaned = path.replace("\\", "/").strip()
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if cleaned:
            reported.add(cleaned)
    return reported


def is_in_report(diff_file: str, reported_files: set[str]) -> bool:
    """Check whether a git-relative path appears in a coverage report.

    Exact match first, then suffix matching in both directions, because
    reporters disagree on path shape:
    - the report path may keep a prefix normalization could not strip
      ("/app/src/x.ts" for "src/x.ts")
    - the report path may be *shorter* than the git path (JaCoCo emits
      "com/foo/Bar.java" for "src/main/java/com/foo/Bar.java")

    The reverse direction requires a separator in the reported path, so a bare
    basename never matches a same-named file living in another directory.
    """
    if diff_file in reported_files:
        return True
    needle = "/" + diff_file
    for reported in reported_files:
        if reported.endswith(needle):
            return True
        if "/" in reported and diff_file.endswith("/" + reported):
            return True
    return False
