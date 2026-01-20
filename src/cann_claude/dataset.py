"""
内置算子数据集管理。

提供对 dataset/operator_index.json 的访问，支持：
- 按名称查找算子
- 按类别/复杂度筛选
- 获取算子的 Python 参考文件路径
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OperatorInfo:
    """算子信息。"""
    name: str
    category: str
    complexity: str
    path: str

    @property
    def full_path(self) -> Path:
        """获取完整的 Python 参考文件路径。"""
        return get_dataset_dir() / "py_reference" / self.path


def get_dataset_dir() -> Path:
    """获取 dataset 目录路径。"""
    # dataset 在包的根目录
    package_dir = Path(__file__).parent.parent.parent
    return package_dir / "dataset"


def get_index_path() -> Path:
    """获取 operator_index.json 路径。"""
    return get_dataset_dir() / "operator_index.json"


def load_operator_index() -> dict[str, dict]:
    """加载算子索引。"""
    index_path = get_index_path()
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text())


def get_operator(name: str) -> Optional[OperatorInfo]:
    """根据名称获取算子信息。

    Args:
        name: 算子名称（如 'relu', 'softmax'）

    Returns:
        OperatorInfo 或 None
    """
    index = load_operator_index()
    if name not in index:
        return None

    info = index[name]
    return OperatorInfo(
        name=name,
        category=info["category"],
        complexity=info["complexity"],
        path=info["path"],
    )


def list_operators(
    category: Optional[str] = None,
    complexity: Optional[str] = None,
) -> list[OperatorInfo]:
    """列出算子，支持筛选。

    Args:
        category: 类别筛选（如 'activation', 'matmul'）
        complexity: 复杂度筛选（'low', 'medium', 'high'）

    Returns:
        OperatorInfo 列表
    """
    index = load_operator_index()
    result = []

    for name, info in index.items():
        if category and info["category"] != category:
            continue
        if complexity and info["complexity"] != complexity:
            continue

        result.append(OperatorInfo(
            name=name,
            category=info["category"],
            complexity=info["complexity"],
            path=info["path"],
        ))

    # 按类别和名称排序
    result.sort(key=lambda x: (x.category, x.name))
    return result


def get_categories() -> list[str]:
    """获取所有类别。"""
    index = load_operator_index()
    return sorted(set(info["category"] for info in index.values()))


def get_statistics() -> dict:
    """获取数据集统计信息。"""
    index = load_operator_index()

    by_category: dict[str, int] = {}
    by_complexity: dict[str, int] = {}

    for info in index.values():
        cat = info["category"]
        comp = info["complexity"]
        by_category[cat] = by_category.get(cat, 0) + 1
        by_complexity[comp] = by_complexity.get(comp, 0) + 1

    return {
        "total": len(index),
        "by_category": dict(sorted(by_category.items())),
        "by_complexity": {
            "low": by_complexity.get("low", 0),
            "medium": by_complexity.get("medium", 0),
            "high": by_complexity.get("high", 0),
        },
    }


def is_builtin_operator(name: str) -> bool:
    """检查是否为内置算子。"""
    return get_operator(name) is not None
