"""A lightweight Pine Script executor for a subset of the language."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class PineExecutionError(RuntimeError):
    """Raised when Pine evaluation fails."""


@dataclass
class PlotResult:
    title: str
    series: pd.Series
    color: str = "#ff9800"
    overlay: bool = True
    linewidth: int = 1


@dataclass
class PineResult:
    plots: List[PlotResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    indicator_name: str = "Custom Indicator"
    overlay: bool = True


_COLOR_MAP = {
    "color.red": "#ef5350",
    "color.green": "#66bb6a",
    "color.blue": "#42a5f5",
    "color.yellow": "#ffee58",
    "color.orange": "#ffa726",
    "color.white": "#fafafa",
}


class SafeEvaluator(ast.NodeVisitor):
    """Safely evaluate arithmetic expressions on pandas Series."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def visit(self, node: ast.AST):  # type: ignore[override]
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise PineExecutionError(f"Unsupported syntax: {node.__class__.__name__}")
        return visitor(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name):
        if node.id in self.context:
            return self.context[node.id]
        raise PineExecutionError(f"Unknown identifier: {node.id}")

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise PineExecutionError("Unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
        raise PineExecutionError("Unsupported binary operator")

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise PineExecutionError("Only single comparisons supported")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise PineExecutionError("Unsupported comparison operator")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            full_name = self._resolve_attribute(node.func)
        elif isinstance(node.func, ast.Name):
            full_name = node.func.id
        else:
            raise PineExecutionError("Unsupported call target")

        if full_name not in self.context:
            raise PineExecutionError(f"Unknown function: {full_name}")

        func = self.context[full_name]
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_IfExp(self, node: ast.IfExp):
        condition = self.visit(node.test)
        return self.visit(node.body) if condition else self.visit(node.orelse)

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(value) for value in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for value in values[1:]:
                result = result & value
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for value in values[1:]:
                result = result | value
            return result
        raise PineExecutionError("Unsupported boolean operator")

    def _resolve_attribute(self, node: ast.Attribute) -> str:
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value  # type: ignore[assignment]
        if isinstance(node, ast.Name):
            parts.append(node.id)
        else:
            raise PineExecutionError("Unsupported attribute access")
        return ".".join(reversed(parts))


class PineInterpreter:
    """Execute a small Pine subset against an OHLCV DataFrame."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy().reset_index(drop=True)
        self.context: Dict[str, Any] = {
            "open": self.data["open"],
            "high": self.data["high"],
            "low": self.data["low"],
            "close": self.data["close"],
            "volume": self.data["volume"],
            "math.sqrt": np.sqrt,
            "math.log": np.log,
            "math.exp": np.exp,
            "math.sin": np.sin,
            "math.cos": np.cos,
            "na": np.nan,
        }
        self.context.update(self._function_library())

        self.indicator_name = "Custom Indicator"
        self.overlay = True
        self.plots: List[PlotResult] = []

    # pylint: disable=too-many-return-statements
    def _function_library(self) -> Dict[str, Any]:
        def sma(series: pd.Series, length: int) -> pd.Series:
            return series.rolling(length).mean()

        def ema(series: pd.Series, length: int) -> pd.Series:
            return series.ewm(span=length, adjust=False).mean()

        def rsi(series: pd.Series, length: int = 14) -> pd.Series:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(length).mean()
            avg_loss = loss.rolling(length).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            return 100 - (100 / (1 + rs))

        def highest(series: pd.Series, length: int) -> pd.Series:
            return series.rolling(length).max()

        def lowest(series: pd.Series, length: int) -> pd.Series:
            return series.rolling(length).min()

        def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
            return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))

        def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
            return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))

        class InputNamespace:
            @staticmethod
            def int(value: int, title: str | None = None, minval: int | None = None, maxval: int | None = None) -> int:
                return int(value)

            @staticmethod
            def float(value: float, title: str | None = None, minval: float | None = None, maxval: float | None = None) -> float:
                return float(value)

        return {
            "sma": sma,
            "ema": ema,
            "rsi": rsi,
            "highest": highest,
            "lowest": lowest,
            "crossover": crossover,
            "crossunder": crossunder,
            "input.int": InputNamespace.int,
            "input.float": InputNamespace.float,
        }

    def execute(self, script: str) -> PineResult:
        normalized = self._normalize(script)
        for line in normalized.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.startswith("indicator"):
                self._handle_indicator(stripped)
            elif stripped.startswith("plot"):
                self._handle_plot(stripped)
            else:
                self._handle_assignment(stripped)
        return PineResult(
            plots=self.plots,
            context=self.context,
            indicator_name=self.indicator_name,
            overlay=self.overlay,
        )

    def _normalize(self, script: str) -> str:
        replacements = {"true": "True", "false": "False", "na": "na"}
        normalized = script
        for pine, python in replacements.items():
            normalized = normalized.replace(pine, python)
        return normalized

    def _handle_indicator(self, line: str) -> None:
        call = self._parse_call(line)
        if call[0] != "indicator":
            raise PineExecutionError("Malformed indicator declaration")
        args, kwargs = call[1], call[2]
        if args:
            self.indicator_name = str(args[0])
        if "title" in kwargs:
            self.indicator_name = str(kwargs["title"])
        if "overlay" in kwargs:
            self.overlay = bool(kwargs["overlay"])

    def _handle_plot(self, line: str) -> None:
        call = self._parse_call(line)
        if call[0] != "plot":
            raise PineExecutionError("Malformed plot call")
        args, kwargs = call[1], call[2]
        if not args:
            raise PineExecutionError("plot requires at least one argument")
        series = self._evaluate_expression(args[0])
        if not isinstance(series, pd.Series):
            raise PineExecutionError("plot argument must be a series")
        title = kwargs.get("title") or getattr(series, "name", "plot")
        color_name = kwargs.get("color")
        color = _COLOR_MAP.get(str(color_name), "#ff9800")
        linewidth = int(kwargs.get("linewidth", 1))
        self.plots.append(PlotResult(str(title), series, color=color, overlay=self.overlay, linewidth=linewidth))

    def _handle_assignment(self, line: str) -> None:
        if "=" not in line:
            raise PineExecutionError(f"Unsupported statement: {line}")
        name, expr = line.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        value = self._evaluate_expression(expr)
        self.context[name] = value

    def _evaluate_expression(self, expr: Any) -> Any:
        if isinstance(expr, ast.AST):
            tree = expr
        else:
            tree = ast.parse(expr, mode="eval")
        evaluator = SafeEvaluator(self.context)
        return evaluator.visit(tree)

    def _parse_call(self, line: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        tree = ast.parse(line, mode="eval")
        if not isinstance(tree.body, ast.Call):
            raise PineExecutionError("Expected a function call")
        call = tree.body
        func_name = self._resolve_callable_name(call.func)
        args = [self._convert_argument(arg) for arg in call.args]
        kwargs = {kw.arg: self._convert_argument(kw.value) for kw in call.keywords}
        return func_name, args, kwargs

    def _resolve_callable_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return SafeEvaluator(self.context)._resolve_attribute(node)
        raise PineExecutionError("Unsupported callable")

    def _convert_argument(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        return node
