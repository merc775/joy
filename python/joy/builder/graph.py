"""Joy dialect MLIR graph builder.

Provides a Python API for constructing Joy dialect MLIR programs,
provides a Graph abstraction for building Joy dialect MLIR programs.
"""


class DataType:
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"
    BF16 = "bf16"
    I8 = "i8"
    I32 = "i32"
    I64 = "i64"

    _ALIASES = {
        "fp16": "f16", "float16": "f16", "f16": "f16",
        "fp32": "f32", "float32": "f32", "f32": "f32",
        "fp64": "f64", "float64": "f64", "f64": "f64",
        "bf16": "bf16", "bfloat16": "bf16",
        "i8": "i8", "int8": "i8",
        "i32": "i32", "int32": "i32",
        "i64": "i64", "int64": "i64",
    }

    @staticmethod
    def from_string(s):
        return DataType._ALIASES.get(s.lower(), s)


class Op:
    """Represents an MLIR SSA value produced by a Joy dialect operation."""

    def __init__(self, name, shape, dtype, graph):
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype
        self.graph = graph
        self.debug_name = None

    @property
    def rank(self):
        return len(self.shape)

    def mlir_type(self):
        parts = []
        for dim in self.shape:
            parts.append("?" if dim < 0 else str(dim))
        shape_str = "x".join(parts)
        if shape_str:
            return f"tensor<{shape_str}x{self.dtype}>"
        return f"tensor<{self.dtype}>"


class Graph:
    """Builds an MLIR module containing Joy dialect operations.

    Usage:
        graph = Graph("my_model")
        x = graph.input([1, 64], "i64")
        w = graph.input([151936, 1024], "f16")
        out = ops.embedding(x, w)
        graph.set_outputs([out])
        print(graph.get_ir())
    """

    def __init__(self, name="main"):
        self.name = name
        self._next_id = 0
        self._inputs = []
        self._outputs = []
        self._lines = []

    def _alloc_name(self):
        name = f"%{self._next_id}"
        self._next_id += 1
        return name

    def input(self, shape, dtype="f32", name=None):
        arg_name = f"%arg{len(self._inputs)}"
        dtype = DataType.from_string(dtype)
        op = Op(arg_name, shape, dtype, self)
        if name:
            op.debug_name = name
        self._inputs.append(op)
        return op

    def set_outputs(self, outputs):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        self._outputs = list(outputs)

    def _format_attrs(self, attrs):
        if not attrs:
            return ""
        parts = []
        for k, v in attrs.items():
            if isinstance(v, float):
                parts.append(f"{k} = {v:.6e} : f32")
            elif isinstance(v, int):
                parts.append(f"{k} = {v} : i64")
            elif isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v):
                vals = ", ".join(str(x) for x in v)
                parts.append(f"{k} = dense<[{vals}]> : tensor<{len(v)}xi64>")
        return " {" + ", ".join(parts) + "}"

    def _create_op(self, op_name, inputs, result_shape, result_dtype, attrs=None):
        """Create a single-output Joy dialect operation."""
        name = self._alloc_name()
        result = Op(name, result_shape, result_dtype, self)

        attr_str = self._format_attrs(attrs)
        operands = ", ".join(inp.name for inp in inputs)
        in_types = ", ".join(inp.mlir_type() for inp in inputs)
        out_type = result.mlir_type()

        line = (f'    {name} = "{op_name}"({operands}){attr_str}'
                f" : ({in_types}) -> {out_type}")
        self._lines.append(line)
        return result

    def _create_multi_result_op(self, op_name, inputs, result_shapes,
                                result_dtypes, attrs=None):
        """Create a multi-output Joy dialect operation."""
        results = []
        names = []
        for shape, dtype in zip(result_shapes, result_dtypes):
            n = self._alloc_name()
            results.append(Op(n, shape, dtype, self))
            names.append(n)

        attr_str = self._format_attrs(attrs)
        operands = ", ".join(inp.name for inp in inputs)
        in_types = ", ".join(inp.mlir_type() for inp in inputs)
        out_types = ", ".join(r.mlir_type() for r in results)
        lhs = ", ".join(names)

        line = (f'    {lhs} = "{op_name}"({operands}){attr_str}'
                f" : ({in_types}) -> ({out_types})")
        self._lines.append(line)
        return tuple(results)

    def get_ir(self):
        """Generate MLIR text representation of the graph."""
        lines = ["module {"]

        args = ", ".join(f"{inp.name}: {inp.mlir_type()}" for inp in self._inputs)
        rets = ", ".join(out.mlir_type() for out in self._outputs)
        lines.append(f"  func.func @{self.name}({args}) -> ({rets}) {{")

        lines.extend(self._lines)

        ret_vals = ", ".join(out.name for out in self._outputs)
        ret_types = ", ".join(out.mlir_type() for out in self._outputs)
        lines.append(f"    return {ret_vals} : {ret_types}")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    def get_op_stats(self):
        """Return a dict mapping joy dialect op names to occurrence counts."""
        stats = {}
        for line in self._lines:
            stripped = line.strip()
            eq_idx = stripped.find(" = ")
            if eq_idx < 0:
                continue
            rest = stripped[eq_idx + 3:]
            paren_idx = rest.find("(")
            if paren_idx < 0:
                continue
            op_name = rest[:paren_idx].strip().strip('"')
            stats[op_name] = stats.get(op_name, 0) + 1
        return stats
