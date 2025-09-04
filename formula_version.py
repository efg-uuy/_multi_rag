import datetime
from typing import Dict, List, Optional, Union


class FormulaVersionManager:
    """
    财务公式版本管理器：支持多版本公式管理、按日期自动匹配、TTL过期失效
    核心功能：
    1. 存储多版本财务公式（含调整项）
    2. 根据查询日期自动匹配有效公式
    3. 支持TTL过期机制，过时公式标记为deprecated
    4. 提供公式新增、更新、查询接口
    """

    def __init__(self, default_ttl_days: int = 365):
        """
        初始化公式管理器
        :param default_ttl_days: 默认过期天数（TTL），默认365天
        """
        self.default_ttl_days = default_ttl_days
        # 公式存储结构：{指标名: [{"version": 版本号, "cal": 计算公式, "effective_date": 生效日期, "valid_until": 失效日期, "status": 状态}]}
        self.formulas: Dict[str, List[Dict[str, Union[str, datetime.date, None, str]]]] = {
            # 初始化净利润公式（含2023/2024版本，包含多维度调整项）
            "净利润": [
                {
                    "version": "2023",
                    "cal": "营收 - 营业成本 - 税金及附加 - 销售费用 - 管理费用 - 财务费用 - 资产减值损失",
                    "effective_date": datetime.date(2023, 1, 1),  # 生效日期
                    "valid_until": datetime.date(2023, 12, 31),  # 自然失效日期
                    "status": "active"  # 状态：active（有效）/ deprecated（过期）
                },
                {
                    "version": "2024",
                    "cal": "营收 - 营业成本 - 税金及附加 - 销售费用 - 管理费用 - 财务费用 - 资产减值损失 - 公允价值变动损失 + 投资收益",
                    "effective_date": datetime.date(2024, 1, 1),
                    "valid_until": None,  # None表示长期有效（需通过TTL判断过期）
                    "status": "active"
                }
            ]
        }

    def _check_ttl_expire(self, formula: Dict[str, Union[str, datetime.date, None, str]]) -> bool:
        """
        检查公式是否因TTL过期（私有方法）
        :param formula: 单个公式字典
        :return: 是否过期（True=过期，False=未过期）
        """
        # 若已手动设置失效日期，优先按valid_until判断
        if formula["valid_until"] is not None:
            return datetime.date.today() > formula["valid_until"]

        # 长期有效公式按TTL判断（从生效日期开始计算）
        ttl_expire_date = formula["effective_date"] + datetime.timedelta(days=self.default_ttl_days)
        return datetime.date.today() > ttl_expire_date

    def _update_formula_status(self):
        """更新所有公式的状态（active/deprecated），每日首次查询时自动触发"""
        for indicator, formula_list in self.formulas.items():
            for formula in formula_list:
                if self._check_ttl_expire(formula) and formula["status"] == "active":
                    formula["status"] = "deprecated"
                    print(
                        f"⚠️  公式[{indicator}-v{formula['version']}]已过期（TTL={self.default_ttl_days}天），状态更新为deprecated")

    def get_formula(self, indicator: str, query_date: Optional[Union[str, datetime.date]] = None) -> Dict[str, str]:
        """
        根据指标名和查询日期获取有效公式
        :param indicator: 财务指标名（如“净利润”）
        :param query_date: 查询日期（格式：YYYY-MM-DD 或 datetime.date对象，默认当前日期）
        :return: 包含公式、版本、状态、提示的结果字典
        """
        # 1. 处理输入参数
        if indicator not in self.formulas:
            return {
                "status": "error",
                "message": f"未找到指标「{indicator}」的公式配置",
                "formula": "",
                "version": ""
            }

        # 转换查询日期（默认当前日期）
        if query_date is None:
            query_date = datetime.date.today()
        elif isinstance(query_date, str):
            try:
                query_date = datetime.datetime.strptime(query_date, "%Y-%m-%d").date()
            except ValueError:
                return {
                    "status": "error",
                    "message": "查询日期格式错误，需为YYYY-MM-DD",
                    "formula": "",
                    "version": ""
                }

        # 2. 自动更新公式状态（检查TTL过期）
        self._update_formula_status()

        # 3. 筛选该日期有效的公式（生效日期≤查询日期≤失效日期，且状态未过期）
        valid_formulas = []
        for formula in self.formulas[indicator]:
            # 状态已过期的跳过（除非无其他有效公式）
            if formula["status"] == "deprecated":
                continue
            # 检查日期范围
            effective_ok = formula["effective_date"] <= query_date
            valid_until_ok = formula["valid_until"] is None or query_date <= formula["valid_until"]
            if effective_ok and valid_until_ok:
                valid_formulas.append(formula)

        # 4. 确定返回公式（优先最新版本的有效公式，无有效时返回最新过期公式并提示）
        if valid_formulas:
            # 按版本号降序排序，取最新有效版本
            valid_formulas.sort(key=lambda x: x["version"], reverse=True)
            selected = valid_formulas[0]
            return {
                "status": "success",
                "message": f"匹配「{indicator}」{query_date.strftime('%Y年')}有效公式（v{selected['version']}）",
                "formula": selected["cal"],
                "version": selected["version"],
                "effective_date": selected["effective_date"].strftime("%Y-%m-%d"),
                "valid_until": selected["valid_until"].strftime("%Y-%m-%d") if selected["valid_until"] else "长期有效"
            }
        else:
            # 无有效公式时，返回最新版本的过期公式
            all_formulas = self.formulas[indicator]
            all_formulas.sort(key=lambda x: x["version"], reverse=True)
            latest = all_formulas[0]
            return {
                "status": "warning",
                "message": f"无「{indicator}」{query_date.strftime('%Y年')}有效公式，返回最新过期公式（v{latest['version']}，状态：{latest['status']}），建议更新公式口径",
                "formula": latest["cal"],
                "version": latest["version"],
                "effective_date": latest["effective_date"].strftime("%Y-%m-%d"),
                "valid_until": latest["valid_until"].strftime("%Y-%m-%d") if latest["valid_until"] else "长期有效"
            }

    def add_formula(self, indicator: str, version: str, cal: str, effective_date: str,
                    valid_until: Optional[str] = None) -> Dict[str, str]:
        """
        新增公式版本
        :param indicator: 财务指标名
        :param version: 版本号（建议用年份，如“2025”）
        :param cal: 计算公式（含多维度调整项）
        :param effective_date: 生效日期（YYYY-MM-DD）
        :param valid_until: 失效日期（YYYY-MM-DD，None表示长期有效）
        :return: 操作结果
        """
        # 转换日期格式
        try:
            effective_date = datetime.datetime.strptime(effective_date, "%Y-%m-%d").date()
            valid_until = datetime.datetime.strptime(valid_until, "%Y-%m-%d").date() if valid_until else None
        except ValueError:
            return {
                "status": "error",
                "message": "日期格式错误，需为YYYY-MM-DD"
            }

        # 检查版本是否已存在
        if indicator in self.formulas:
            for formula in self.formulas[indicator]:
                if formula["version"] == version:
                    return {
                        "status": "error",
                        "message": f"指标「{indicator}」已存在v{version}版本，请勿重复添加"
                    }
        else:
            self.formulas[indicator] = []  # 初始化新指标的公式列表

        # 添加新公式（默认状态为active）
        new_formula = {
            "version": version,
            "cal": cal,
            "effective_date": effective_date,
            "valid_until": valid_until,
            "status": "active"
        }
        self.formulas[indicator].append(new_formula)
        # 按生效日期排序（便于后续查询）
        self.formulas[indicator].sort(key=lambda x: x["effective_date"])

        return {
            "status": "success",
            "message": f"成功添加「{indicator}」v{version}版本公式，生效日期：{effective_date.strftime('%Y-%m-%d')}",
            "formula": cal
        }

    def list_formulas(self, indicator: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        列出指定指标或所有指标的公式版本
        :param indicator: 指标名（None表示列出所有）
        :return: 公式列表
        """
        self._update_formula_status()  # 先更新状态
        result = {"formulas": []}

        if indicator:
            if indicator not in self.formulas:
                return {"status": "error", "message": f"未找到指标「{indicator}」的公式"}
            target_formulas = self.formulas[indicator]
        else:
            # 合并所有指标的公式
            target_formulas = []
            for ind, formulas in self.formulas.items():
                for f in formulas:
                    f["indicator"] = ind
                    target_formulas.append(f)

        # 格式化日期输出
        for f in target_formulas:
            result["formulas"].append({
                "indicator": f.get("indicator", indicator),
                "version": f["version"],
                "formula": f["cal"],
                "effective_date": f["effective_date"].strftime("%Y-%m-%d"),
                "valid_until": f["valid_until"].strftime("%Y-%m-%d") if f["valid_until"] else "长期有效",
                "status": f["status"]
            })

        result["status"] = "success"
        result["count"] = len(result["formulas"])
        return result


# ===================== 示例：公式管理器使用演示 =====================
if __name__ == "__main__":
    # 1. 初始化公式管理器（默认TTL=365天）
    formula_mgr = FormulaVersionManager(default_ttl_days=365)
    print("=== 初始化完成，当前公式列表 ===")
    print(json.dumps(formula_mgr.list_formulas(), ensure_ascii=False, indent=2))
    print("-" * 80)

    # 2. 查询不同日期的净利润公式
    print("=== 场景1：查询2023年10月净利润公式（匹配2023版） ===")
    res_2023 = formula_mgr.get_formula(indicator="净利润", query_date="2023-10-15")
    print(json.dumps(res_2023, ensure_ascii=False, indent=2))
    print("-" * 80)

    print("=== 场景2：查询2024年5月净利润公式（匹配2024版，长期有效） ===")
    res_2024 = formula_mgr.get_formula(indicator="净利润", query_date="2024-05-20")
    print(json.dumps(res_2024, ensure_ascii=False, indent=2))
    print("-" * 80)

    # 3. 模拟查询过期公式（假设2025年查询2023版，已过TTL）
    print("=== 场景3：查询2025年1月净利润公式（2024版未过期，正常返回） ===")
    res_2025 = formula_mgr.get_formula(indicator="净利润", query_date="2025-01-10")
    print(json.dumps(res_2025, ensure_ascii=False, indent=2))
    print("-" * 80)

    # 4. 新增2025版净利润公式（包含更多调整项）
    print("=== 场景4：新增2025版净利润公式 ===")
    add_res = formula_mgr.add_formula(
        indicator="净利润",
        version="2025",
        cal="营收 - 营业成本 - 税金及附加 - 销售费用 - 管理费用 - 财务费用 - 资产减值损失 - 公允价值变动损失 + 投资收益 + 其他收益",
        effective_date="2025-01-01",
        valid_until=None  # 长期有效
    )
    print(json.dumps(add_res, ensure_ascii=False, indent=2))
    print("新增后公式列表：")
    print(json.dumps(formula_mgr.list_formulas("净利润"), ensure_ascii=False, indent=2))
    print("-" * 80)

    # 5. 查询新增的2025版公式
    print("=== 场景5：查询2025年6月净利润公式（匹配新增的2025版） ===")
    res_2025_new = formula_mgr.get_formula(indicator="净利润", query_date="2025-06-30")
    print(json.dumps(res_2025_new, ensure_ascii=False, indent=2))