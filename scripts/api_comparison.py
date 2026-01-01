#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
API Comparison Script: cuCIM vs. scikit-image

This script generates a comprehensive report comparing the API availability
and parameter differences between cuCIM and scikit-image.

Usage:
    python api_comparison.py [--output-format markdown|html|json] [--output-file report.md]
"""

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from typing import Any

# List of submodules to compare (including nested submodules)
SUBMODULES = [
    "color",
    "data",
    "draw",
    "exposure",
    "feature",
    "filters",
    "filters.rank",
    "future",
    "future.graph",
    "graph",
    "io",
    "measure",
    "metrics",
    "morphology",
    "registration",
    "restoration",
    "segmentation",
    "transform",
    "util",
]

# Notes explaining why certain submodules have limited or no cuCIM coverage
SUBMODULE_NOTES = {
    "data": (
        "cuCIM does not intend to replicate the sample image datasets provided by "
        "scikit-image. To load one of these images onto the GPU, it is recommended "
        "to use `cupy.asarray` to transfer the output of the scikit-image function "
        "to the GPU."
    ),
    "io": ("cuCIM does not currently plan to implement the `skimage.io` module."),
    "draw": (
        "cuCIM does not currently plan to implement the `skimage.draw` module. "
        "For drawing shapes into an existing array on device using CUDA, there is "
        "some functionality available in CV-CUDA's cuOSD (CUDA On-Screen Display) "
        "operator."
    ),
    "graph": (
        "cuCIM does not currently have plans to implement the `skimage.graph` module."
    ),
    "future": (
        "cuCIM does not currently have plans to implement the `skimage.future` module."
    ),
    "future.graph": (
        "cuCIM does not currently have plans to implement the `skimage.future.graph` "
        "module."
    ),
}


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    kind: str  # positional, keyword, etc.
    default: Any = inspect.Parameter.empty
    annotation: Any = inspect.Parameter.empty

    def has_default(self) -> bool:
        return self.default is not inspect.Parameter.empty

    def default_repr(self) -> str:
        if self.default is inspect.Parameter.empty:
            return "<no default>"
        return repr(self.default)


@dataclass
class FunctionSignature:
    """Extracted function signature information."""

    name: str
    parameters: list[ParameterInfo] = field(default_factory=list)
    docstring: str | None = None

    @classmethod
    def from_callable(cls, func: callable, name: str) -> "FunctionSignature":
        """Create a FunctionSignature from a callable object."""
        try:
            sig = inspect.signature(func)
            params = []
            for param_name, param in sig.parameters.items():
                kind_map = {
                    inspect.Parameter.POSITIONAL_ONLY: "positional_only",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
                    inspect.Parameter.VAR_POSITIONAL: "var_positional",
                    inspect.Parameter.KEYWORD_ONLY: "keyword_only",
                    inspect.Parameter.VAR_KEYWORD: "var_keyword",
                }
                params.append(
                    ParameterInfo(
                        name=param_name,
                        kind=kind_map.get(param.kind, "unknown"),
                        default=param.default,
                        annotation=param.annotation,
                    )
                )
            docstring = inspect.getdoc(func)
            return cls(name=name, parameters=params, docstring=docstring)
        except (ValueError, TypeError):
            # Some built-in functions don't have signatures
            return cls(name=name, parameters=[], docstring=inspect.getdoc(func))


@dataclass
class FunctionComparison:
    """Comparison result for a single function."""

    name: str
    in_skimage: bool = False
    in_cucim: bool = False
    skimage_sig: FunctionSignature | None = None
    cucim_sig: FunctionSignature | None = None

    # Parameter differences
    params_only_in_skimage: list[str] = field(default_factory=list)
    params_only_in_cucim: list[str] = field(default_factory=list)
    params_with_different_defaults: list[tuple[str, Any, Any]] = field(
        default_factory=list
    )

    def compute_differences(self):
        """Compute parameter differences between signatures."""
        if not self.skimage_sig or not self.cucim_sig:
            return

        skimage_params = {p.name: p for p in self.skimage_sig.parameters}
        cucim_params = {p.name: p for p in self.cucim_sig.parameters}

        skimage_names = set(skimage_params.keys())
        cucim_names = set(cucim_params.keys())

        self.params_only_in_skimage = sorted(skimage_names - cucim_names)
        self.params_only_in_cucim = sorted(cucim_names - skimage_names)

        # Check for different defaults in common parameters
        common_params = skimage_names & cucim_names
        for param_name in sorted(common_params):
            sk_param = skimage_params[param_name]
            cu_param = cucim_params[param_name]

            # Compare defaults (handle empty case)
            sk_default = sk_param.default
            cu_default = cu_param.default

            # Both have no default - that's equal
            if (
                sk_default is inspect.Parameter.empty
                and cu_default is inspect.Parameter.empty
            ):
                continue

            # One has default, other doesn't
            if (sk_default is inspect.Parameter.empty) != (
                cu_default is inspect.Parameter.empty
            ):
                self.params_with_different_defaults.append(
                    (param_name, sk_param.default_repr(), cu_param.default_repr())
                )
                continue

            # Both have defaults - compare them
            try:
                if sk_default != cu_default:
                    self.params_with_different_defaults.append(
                        (param_name, sk_param.default_repr(), cu_param.default_repr())
                    )
            except (ValueError, TypeError):
                # Some defaults can't be compared (e.g., numpy arrays)
                if repr(sk_default) != repr(cu_default):
                    self.params_with_different_defaults.append(
                        (param_name, sk_param.default_repr(), cu_param.default_repr())
                    )

    @property
    def has_differences(self) -> bool:
        """Check if there are any parameter differences."""
        return bool(
            self.params_only_in_skimage
            or self.params_only_in_cucim
            or self.params_with_different_defaults
        )


@dataclass
class SubmoduleComparison:
    """Comparison result for a submodule."""

    name: str
    skimage_available: bool = False
    cucim_available: bool = False
    functions: list[FunctionComparison] = field(default_factory=list)

    @property
    def functions_only_in_skimage(self) -> list[str]:
        return [f.name for f in self.functions if f.in_skimage and not f.in_cucim]

    @property
    def functions_only_in_cucim(self) -> list[str]:
        return [f.name for f in self.functions if f.in_cucim and not f.in_skimage]

    @property
    def functions_in_both(self) -> list[FunctionComparison]:
        return [f for f in self.functions if f.in_skimage and f.in_cucim]

    @property
    def functions_with_differences(self) -> list[FunctionComparison]:
        return [f for f in self.functions_in_both if f.has_differences]


def get_public_callables(module) -> dict[str, callable]:
    """Get all public callable objects from a module."""
    callables = {}

    # Get __all__ if defined, otherwise use dir()
    if hasattr(module, "__all__"):
        names = module.__all__
    else:
        names = [n for n in dir(module) if not n.startswith("_")]

    for name in names:
        try:
            obj = getattr(module, name)
            # Include functions and classes (but not submodules)
            if callable(obj) and not inspect.ismodule(obj):
                callables[name] = obj
        except AttributeError:
            continue

    return callables


def compare_submodule(submodule_name: str) -> SubmoduleComparison:
    """Compare a submodule between cuCIM and scikit-image."""
    result = SubmoduleComparison(name=submodule_name)

    # Try to import scikit-image submodule
    skimage_module = None
    skimage_callables = {}
    try:
        skimage_module = importlib.import_module(f"skimage.{submodule_name}")
        result.skimage_available = True
        skimage_callables = get_public_callables(skimage_module)
    except ImportError as e:
        print(f"Warning: Could not import skimage.{submodule_name}: {e}")

    # Try to import cuCIM submodule
    cucim_module = None
    cucim_callables = {}
    try:
        cucim_module = importlib.import_module(f"cucim.skimage.{submodule_name}")
        result.cucim_available = True
        cucim_callables = get_public_callables(cucim_module)
    except ImportError as e:
        print(f"Warning: Could not import cucim.skimage.{submodule_name}: {e}")

    # Get union of all function names
    all_names = sorted(set(skimage_callables.keys()) | set(cucim_callables.keys()))

    for name in all_names:
        func_comp = FunctionComparison(name=name)

        if name in skimage_callables:
            func_comp.in_skimage = True
            func_comp.skimage_sig = FunctionSignature.from_callable(
                skimage_callables[name], name
            )

        if name in cucim_callables:
            func_comp.in_cucim = True
            func_comp.cucim_sig = FunctionSignature.from_callable(
                cucim_callables[name], name
            )

        func_comp.compute_differences()
        result.functions.append(func_comp)

    return result


def generate_markdown_report(comparisons: list[SubmoduleComparison]) -> str:
    """Generate a Markdown report from the comparison results."""
    lines = []

    lines.append("# cuCIM vs. scikit-image API Comparison Report")
    lines.append("")
    lines.append(
        "This report compares the public API of cuCIM with scikit-image, "
        "identifying which functions are available in each library and "
        "any differences in their signatures."
    )
    lines.append("")

    # Summary section
    lines.append("## Executive Summary")
    lines.append("")

    total_skimage = 0
    total_cucim = 0
    total_both = 0
    total_only_skimage = 0
    total_only_cucim = 0
    total_with_diffs = 0

    for comp in comparisons:
        if comp.skimage_available:
            total_skimage += len([f for f in comp.functions if f.in_skimage])
        if comp.cucim_available:
            total_cucim += len([f for f in comp.functions if f.in_cucim])
        total_both += len(comp.functions_in_both)
        total_only_skimage += len(comp.functions_only_in_skimage)
        total_only_cucim += len(comp.functions_only_in_cucim)
        total_with_diffs += len(comp.functions_with_differences)

    lines.append(f"- **Total functions in scikit-image:** {total_skimage}")
    lines.append(f"- **Total functions in cuCIM:** {total_cucim}")
    lines.append(f"- **Functions in both libraries:** {total_both}")
    lines.append(f"- **Functions only in scikit-image:** {total_only_skimage}")
    lines.append(f"- **Functions only in cuCIM:** {total_only_cucim}")
    lines.append(f"- **Functions with signature differences:** {total_with_diffs}")
    lines.append("")

    # Coverage summary table
    lines.append("### Coverage by Submodule")
    lines.append("")
    lines.append(
        "| Submodule | scikit-image | cuCIM | Both | Only skimage | Only cuCIM | Coverage % |"
    )
    lines.append(
        "|-----------|--------------|-------|------|--------------|------------|------------|"
    )

    for comp in comparisons:
        skimage_count = len([f for f in comp.functions if f.in_skimage])
        cucim_count = len([f for f in comp.functions if f.in_cucim])
        both_count = len(comp.functions_in_both)
        only_sk = len(comp.functions_only_in_skimage)
        only_cu = len(comp.functions_only_in_cucim)
        coverage = (both_count / skimage_count * 100) if skimage_count > 0 else 0

        # Skip submodules that don't exist in either library
        if not comp.skimage_available and not comp.cucim_available:
            continue

        # Show availability status for submodules that exist in only one library
        if not comp.skimage_available:
            skimage_count = "N/A"
            coverage = "N/A"
        elif not comp.cucim_available:
            cucim_count = "N/A"
            coverage = f"{0.0:.1f}%"
        else:
            coverage = f"{coverage:.1f}%"

        lines.append(
            f"| {comp.name} | {skimage_count} | {cucim_count} | {both_count} | "
            f"{only_sk} | {only_cu} | {coverage} |"
        )

    lines.append("")

    # Detailed section for each submodule
    lines.append("## Detailed Comparison by Submodule")
    lines.append("")

    for comp in comparisons:
        # Skip submodules that don't exist in either library
        if not comp.skimage_available and not comp.cucim_available:
            continue

        lines.append(f"### {comp.name}")
        lines.append("")

        # Handle submodules that exist only in one library
        if not comp.cucim_available:
            lines.append(
                "*This submodule exists only in scikit-image (not implemented in cuCIM).*"
            )
            lines.append("")
            # Add note if available
            if comp.name in SUBMODULE_NOTES:
                lines.append(f"> **Note:** {SUBMODULE_NOTES[comp.name]}")
                lines.append("")
            all_funcs = [f.name for f in comp.functions if f.in_skimage]
            if all_funcs:
                lines.append(
                    f"**Functions ({len(all_funcs)}):** "
                    + ", ".join(f"`{f}`" for f in all_funcs)
                )
                lines.append("")
            continue

        if not comp.skimage_available:
            lines.append("*This submodule exists only in cuCIM (not in scikit-image).*")
            lines.append("")
            # Add note if available
            if comp.name in SUBMODULE_NOTES:
                lines.append(f"> **Note:** {SUBMODULE_NOTES[comp.name]}")
                lines.append("")
            all_funcs = [f.name for f in comp.functions if f.in_cucim]
            if all_funcs:
                lines.append(
                    f"**Functions ({len(all_funcs)}):** "
                    + ", ".join(f"`{f}`" for f in all_funcs)
                )
                lines.append("")
            continue

        # Add note if available for this submodule
        if comp.name in SUBMODULE_NOTES:
            lines.append(f"> **Note:** {SUBMODULE_NOTES[comp.name]}")
            lines.append("")

        # Functions available in both
        both_funcs = comp.functions_in_both
        if both_funcs:
            lines.append(f"#### Functions Available in Both ({len(both_funcs)})")
            lines.append("")

            # Table of functions in both
            funcs_with_diffs = [f for f in both_funcs if f.has_differences]
            funcs_identical = [f for f in both_funcs if not f.has_differences]

            if funcs_identical:
                lines.append(
                    f"**Identical signatures ({len(funcs_identical)}):** "
                    + ", ".join(f"`{f.name}`" for f in funcs_identical)
                )
                lines.append("")

            if funcs_with_diffs:
                lines.append(f"**With differences ({len(funcs_with_diffs)}):**")
                lines.append("")

                for func in funcs_with_diffs:
                    lines.append(f"##### `{func.name}`")
                    lines.append("")

                    if func.params_only_in_skimage:
                        lines.append(
                            "- **Parameters only in scikit-image:** "
                            + ", ".join(f"`{p}`" for p in func.params_only_in_skimage)
                        )
                    if func.params_only_in_cucim:
                        lines.append(
                            "- **Parameters only in cuCIM:** "
                            + ", ".join(f"`{p}`" for p in func.params_only_in_cucim)
                        )
                    if func.params_with_different_defaults:
                        lines.append("- **Different defaults:**")
                        for (
                            param,
                            sk_def,
                            cu_def,
                        ) in func.params_with_different_defaults:
                            lines.append(
                                f"  - `{param}`: scikit-image={sk_def}, cuCIM={cu_def}"
                            )
                    lines.append("")

        # Functions only in scikit-image
        only_sk = comp.functions_only_in_skimage
        if only_sk:
            lines.append(f"#### Functions Only in scikit-image ({len(only_sk)})")
            lines.append("")
            lines.append(", ".join(f"`{f}`" for f in only_sk))
            lines.append("")

        # Functions only in cuCIM
        only_cu = comp.functions_only_in_cucim
        if only_cu:
            lines.append(f"#### Functions Only in cuCIM ({len(only_cu)})")
            lines.append("")
            lines.append(", ".join(f"`{f}`" for f in only_cu))
            lines.append("")

    # Key differences section
    lines.append("## Key Differences Summary")
    lines.append("")
    lines.append(
        "This section highlights important differences users should be aware of "
        "when migrating from scikit-image to cuCIM."
    )
    lines.append("")

    lines.append("### Missing Functions")
    lines.append("")
    lines.append(
        "The following functions are available in scikit-image but not in cuCIM:"
    )
    lines.append("")

    for comp in comparisons:
        only_sk = comp.functions_only_in_skimage
        if only_sk:
            lines.append(f"**{comp.name}:** " + ", ".join(f"`{f}`" for f in only_sk))

    lines.append("")

    lines.append("### Parameter Differences")
    lines.append("")
    lines.append(
        "The following functions have different parameters between the two libraries:"
    )
    lines.append("")

    for comp in comparisons:
        diffs = comp.functions_with_differences
        if diffs:
            lines.append(f"**{comp.name}:**")
            for func in diffs:
                diff_details = []
                if func.params_only_in_skimage:
                    diff_details.append(
                        f"missing: {', '.join(func.params_only_in_skimage)}"
                    )
                if func.params_only_in_cucim:
                    diff_details.append(
                        f"extra: {', '.join(func.params_only_in_cucim)}"
                    )
                if func.params_with_different_defaults:
                    diff_details.append(
                        f"different defaults: "
                        f"{', '.join(p[0] for p in func.params_with_different_defaults)}"
                    )
                lines.append(f"- `{func.name}`: {'; '.join(diff_details)}")
            lines.append("")

    return "\n".join(lines)


def generate_html_report(comparisons: list[SubmoduleComparison]) -> str:
    """Generate an HTML report from the comparison results."""
    # Convert markdown to simple HTML
    md = generate_markdown_report(comparisons)

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>cuCIM vs scikit-image API Comparison</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background: #f8f9fa;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #76b900; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 40px; }
        h3 { color: #7f8c8d; }
        h4 { color: #95a5a6; }
        h5 { color: #bdc3c7; font-size: 1em; }
        code {
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background: #76b900;
            color: white;
        }
        tr:nth-child(even) { background: #f9f9f9; }
        tr:hover { background: #f5f5f5; }
        blockquote {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px 20px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }
        blockquote strong { color: #856404; }
        .summary-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        ul { margin: 10px 0; }
        li { margin: 5px 0; }
    </style>
</head>
<body>
"""

    # Simple markdown to HTML conversion
    import re

    # Convert headers
    md = re.sub(r"^##### (.+)$", r"<h5>\1</h5>", md, flags=re.MULTILINE)
    md = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", md, flags=re.MULTILINE)
    md = re.sub(r"^### (.+)$", r"<h3>\1</h3>", md, flags=re.MULTILINE)
    md = re.sub(r"^## (.+)$", r"<h2>\1</h2>", md, flags=re.MULTILINE)
    md = re.sub(r"^# (.+)$", r"<h1>\1</h1>", md, flags=re.MULTILINE)

    # Convert code blocks
    md = re.sub(r"`([^`]+)`", r"<code>\1</code>", md)

    # Convert bold
    md = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", md)

    # Convert italic
    md = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", md)

    # Convert blockquotes
    md = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", md, flags=re.MULTILINE)

    # Convert lists
    lines = md.split("\n")
    in_list = False
    result_lines = []
    for line in lines:
        if line.startswith("- "):
            if not in_list:
                result_lines.append("<ul>")
                in_list = True
            result_lines.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            result_lines.append(line)
    if in_list:
        result_lines.append("</ul>")

    # Convert tables
    md = "\n".join(result_lines)
    lines = md.split("\n")
    result_lines = []
    in_table = False
    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            if "---" in line:
                continue  # Skip separator line
            if not in_table:
                result_lines.append("<table>")
                in_table = True
                cells = [c.strip() for c in line.split("|")[1:-1]]
                result_lines.append(
                    "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
                )
            else:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                result_lines.append(
                    "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                )
        else:
            if in_table:
                result_lines.append("</table>")
                in_table = False
            result_lines.append(f"<p>{line}</p>" if line.strip() else "")

    if in_table:
        result_lines.append("</table>")

    html += "\n".join(result_lines)
    html += """
</body>
</html>
"""
    return html


def generate_json_report(comparisons: list[SubmoduleComparison]) -> str:
    """Generate a JSON report from the comparison results."""
    data = {
        "submodules": [],
        "notes": SUBMODULE_NOTES,
    }

    for comp in comparisons:
        submodule_data = {
            "name": comp.name,
            "skimage_available": comp.skimage_available,
            "cucim_available": comp.cucim_available,
            "note": SUBMODULE_NOTES.get(comp.name),
            "functions": [],
        }

        for func in comp.functions:
            func_data = {
                "name": func.name,
                "in_skimage": func.in_skimage,
                "in_cucim": func.in_cucim,
                "params_only_in_skimage": func.params_only_in_skimage,
                "params_only_in_cucim": func.params_only_in_cucim,
                "params_with_different_defaults": [
                    {"param": p[0], "skimage_default": p[1], "cucim_default": p[2]}
                    for p in func.params_with_different_defaults
                ],
            }
            submodule_data["functions"].append(func_data)

        data["submodules"].append(submodule_data)

    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compare cuCIM and scikit-image APIs")
    parser.add_argument(
        "--output-format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format for the report (default: markdown)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--submodules",
        type=str,
        nargs="*",
        default=None,
        help="Specific submodules to compare (default: all)",
    )

    args = parser.parse_args()

    submodules = args.submodules if args.submodules else SUBMODULES

    print("Comparing cuCIM and scikit-image APIs...", file=sys.stderr)
    print(f"Submodules: {', '.join(submodules)}", file=sys.stderr)

    comparisons = []
    for submodule in submodules:
        print(f"  Analyzing {submodule}...", file=sys.stderr)
        comp = compare_submodule(submodule)
        comparisons.append(comp)

    # Generate report
    if args.output_format == "markdown":
        report = generate_markdown_report(comparisons)
    elif args.output_format == "html":
        report = generate_html_report(comparisons)
    else:
        report = generate_json_report(comparisons)

    # Output
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(report)
        print(f"Report written to {args.output_file}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
