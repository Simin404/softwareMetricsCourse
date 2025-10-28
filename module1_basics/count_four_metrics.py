from pathlib import Path
import pandas as pd
from typing import Dict, Any, Iterable, Tuple, Union

# PYSICAL LOC counts all lines in the file, including comments and blank lines
# LOGICAL LOC counts only lines that contain executable code, excluding comments and blank lines
# CYCLOMATIC COMPLEXITY (CC) measures the number of counting decision points. Cyclomatic Complexity = Number of decision points + 1.
# FAN-IN measures how many other modules or functions depend on a given module or function, 
# indicating reusability but also potential impact from changes
# FAN-OUT measures how many other modules a given module depends on, 
# suggesting complexity and interdependency.

WantedExts = {".java", ".c", ".py"}

def _iter_files_from_dir(root: Path, wanted: Iterable[str]) -> Iterable[Path]:
    wanted = {e.lower() for e in wanted}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in wanted:
            yield p.relative_to(root)

def _iter_files_from_dict(tree: Dict[str, Any], prefix: str = "", wanted: Iterable[str] = WantedExts) -> Iterable[Path]:
    wanted = {e.lower() for e in wanted}
    for name, value in tree.items():
        rel = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            # Recurse into sub-dictionaries (folders)
            yield from _iter_files_from_dict(value, rel, wanted)
        else:
            # Treat as a file leaf
            suffix = Path(name).suffix.lower()
            if suffix in wanted:
                yield Path(rel)

def loop_dictionary(
    source: Union[str, Path, Dict[str, Any]],
    *,
    wanted_exts: Iterable[str] = WantedExts,
    print_paths: bool = True
) -> pd.DataFrame:

    results = []
    total_lines =0
    total_logical_lines = 0
    # Choose iterator based on type
    if isinstance(source, (str, Path)):
        root = Path(source)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Directory does not exist or is not a directory: {root}")
        iterator = _iter_files_from_dir(root, wanted_exts)
    elif isinstance(source, dict):
        iterator = _iter_files_from_dict(source, "", wanted_exts)
    else:
        raise TypeError("source must be a directory path (str/Path) or a nested dict.")

    for rel_path in iterator:
        # if print_paths:
        #     print(rel_path.as_posix())

        # Compute your metrics
        loc_physical, loc, cc, fan_in, fan_out = count_four_metrics(source + '/' + str(rel_path))
        total_lines += loc_physical
        total_logical_lines += loc
        
        results.append({
            "filename": rel_path.as_posix(),
            "loc_physical": loc_physical,
            "loc": loc,
            "cc": cc,
            "fan_in": fan_in,
            "fan_out": fan_out
        })
    print(f"Total lines: {total_lines}")
    print(f"Total lines: {total_logical_lines}")
    return pd.DataFrame(results)


def count_four_metrics(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()
    loc, cc, fan_in, fan_out = 0, 0, 0, 0
    loc_physical = len(lines)

    if file_path.endswith('.py'):
        loc = count_python_loc(lines)
        cc = count_python_cc(lines)
        fan_in, fan_out = count_python_fan(lines)

    elif file_path.endswith('.java'):
        loc = count_java_loc(lines)
        cc = count_java_cc(lines)
        fan_in, fan_out = count_java_fan(lines)

    elif file_path.endswith('.c'):
        loc = count_c_loc(lines)
        cc = count_c_cc(lines)
        fan_in, fan_out = count_c_fan(lines)
    else:
        raise ValueError("Unsupported file type. Supported types are: Python, Java and C")
    return loc_physical, loc, cc, fan_in, fan_out
    # print(f"[{file_path}]:")
    # print(f"  Physical LOC: {loc_physical}, \n"
    #       f"  Logical LOC: {loc}, \n"
    #       f"  Cyclomatic Complexity: {cc}, \n"
    #       f"  Fan-In: {fan_in}, \n"
    #       f"  Fan-Out: {fan_out}")

def count_python_loc(lines):
    loc = 0
    in_multiline_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            if in_multiline_comment:
                in_multiline_comment = False
            else:
                in_multiline_comment = True
            continue
        if in_multiline_comment:
            continue
        loc += 1
    return loc

def count_java_loc(lines):
    loc = 0
    in_multiline_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('//'):
            continue
        if stripped_line.startswith('/*'):
            in_multiline_comment = True
            continue
        if in_multiline_comment:
            if stripped_line.endswith('*/'):
                in_multiline_comment = False
            continue
        loc += 1
    return loc

def count_c_loc(lines):
    loc = 0
    in_multiline_comment = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('//'):
            continue
        if stripped_line.startswith('/*'):
            in_multiline_comment = True
            continue
        if in_multiline_comment:
            if stripped_line.endswith('*/'):
                in_multiline_comment = False
            continue
        loc += 1
    return loc




def count_python_fan(lines):
    fan_in, fan_out = 0, 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('def '):
            fan_in += 1
        if 'import ' in stripped_line:
            fan_out += 1    
    return fan_in, fan_out


def count_java_fan(lines):
    fan_in, fan_out = 0, 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('public ') or stripped_line.startswith('private ') or stripped_line.startswith('protected '):
            if ' class ' not in stripped_line and ' interface ' not in stripped_line:
                fan_in += 1
        if 'import ' in stripped_line:
            fan_out += 1    
    return fan_in, fan_out


def count_c_fan(lines):
    fan_in, fan_out = 0, 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('void ') or stripped_line.startswith('int ') or stripped_line.startswith('char '):
            fan_in += 1
        if '#include' in stripped_line:
            fan_out += 1
    return fan_in, fan_out


def count_python_cc(lines):
    cc = 1
    decision_keywords = ['if ', 'for ', 'while ', 'and ', 'or ', 'elif ', 'except ', 'case ', 'catch ']
    for line in lines:
        stripped_line = line.strip()
        if any(keyword in stripped_line for keyword in decision_keywords):
            cc += 1
    return cc

def count_java_cc(lines):
    cc = 1
    decision_keywords = ['if ', 'for ', 'while ', 'case ', 'catch ', '&&', '||']
    for line in lines:
        stripped_line = line.strip()
        if any(keyword in stripped_line for keyword in decision_keywords):
            cc += 1
    return cc

def count_c_cc(lines):
    cc = 1
    decision_keywords = ['if ', 'for ', 'while ', '&&', '||']
    for line in lines:
        stripped_line = line.strip()
        if any(keyword in stripped_line for keyword in decision_keywords):
            cc += 1
    return cc