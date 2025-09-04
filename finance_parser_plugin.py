import os
import json
import re
from typing import List, Dict, Any
import pandas as pd


class FinanceParserPlugin:
    """金融解析插件，提供金融文档专属处理功能"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化插件，加载配置"""
        self.config = config or self._load_default_config()
        self.finance_table_titles = self.config.get('finance_table_titles', {})
        self.finance_indicators = self.config.get('finance_indicators', [])

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            'finance_table_titles': {
                'income_statement': ['利润表', '损益表', '收益表'],
                'balance_sheet': ['资产负债表', '财务状况表'],
                'cash_flow': ['现金流量表', '现金流动表'],
                'financial_ratios': ['财务比率', '财务指标']
            },
            'finance_indicators': [
                '营业收入', '净利润', '毛利率', '净利率', '资产负债率',
                '现金及等价物', '经营活动现金流', '投资活动现金流',
                '筹资活动现金流', '每股收益', '市盈率'
            ],
            'table_column_counts': {
                'quarterly': 4,  # 季度数据通常4列
                'yearly': 1,  # 年度数据通常1列
                'comparative': 2  # 对比数据通常2列
            }
        }

    def fix_finance_cross_page_tables(self, tables: List[pd.DataFrame],
                                      page_num: int, pdf_path: str) -> List[pd.DataFrame]:
        """
        修复跨页金融表格（如利润表、现金流量表）

        Args:
            tables: 当前页提取的表格列表
            page_num: 当前页码
            pdf_path: PDF文件路径

        Returns:
            合并后的表格列表
        """
        merged_tables = []
        prev_tables = self._get_previous_page_tables(pdf_path, page_num)

        for table in tables:
            # 尝试获取表格标题
            table_title = self._extract_table_title(table)
            matched = False

            # 检查是否与前页表格匹配
            for prev_table, prev_title, prev_page in prev_tables:
                if self._is_related_table(table_title, prev_title, table, prev_table):
                    # 合并表格
                    merged_table = self._merge_tables(prev_table, table)
                    merged_tables.append(merged_table)
                    # 从prev_tables中移除已合并的表格
                    prev_tables = [(t, tt, p) for t, tt, p in prev_tables if (t is not prev_table)]
                    matched = True
                    break

            if not matched:
                merged_tables.append(table)

        # 保存当前页表格供后续页面参考
        self._save_current_page_tables(pdf_path, page_num, merged_tables)

        return merged_tables

    def analyze_finance_chart(self, image_path: str, analyzer) -> Dict[str, Any]:
        """
        解析金融图表（如趋势图→数据点提取）

        Args:
            image_path: 图表图片路径
            analyzer: 图像分析器实例，需实现analyze_image方法

        Returns:
            解析后的图表数据
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图表图片不存在: {image_path}")

        prompt = """你是专业金融图表分析师，需要从图表中提取关键数据。
        请识别图表类型、指标名称和对应数据。
        输出格式为JSON，包含"chart_type"（图表类型）、"indicator"（指标名称）、
        "period_type"（周期类型，如季度、年度）和"data"（数据字典，键为周期，值为数据）。
        示例: {"chart_type": "折线图", "indicator": "毛利率", "period_type": "季度", "data": {"Q1": "30%", "Q2": "32%"}}
        """

        try:
            result = analyzer.analyze_image(local_image_path=image_path, prompt=prompt)
            # 解析结果为JSON
            return json.loads(result)
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取数据
            return self._extract_data_from_text(result)
        except Exception as e:
            print(f"图表解析错误: {str(e)}")
            return {"error": str(e)}

    def extract_finance_indicators(self, text: str) -> Dict[str, str]:
        """
        从文本中提取金融指标

        Args:
            text: 待提取的文本

        Returns:
            提取的金融指标字典
        """
        indicators = {}
        # 正则模式匹配常见金融指标格式，如"毛利率：30%"或"净利润为5000万元"
        pattern = r'(' + '|'.join(self.finance_indicators) + r')[:：为]?\s*([\d,.%]+[万千百亿]*元?|[\d.]+%)'

        matches = re.findall(pattern, text)
        for indicator, value in matches:
            indicators[indicator] = value.strip()

        return indicators

    def standardize_finance_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        标准化金融数据格式

        Args:
            data: 原始金融数据
            data_type: 数据类型，如"income_statement"、"balance_sheet"等

        Returns:
            标准化后的金融数据
        """
        if data_type not in self.finance_table_titles:
            return data

        standardized = {}
        # 映射到标准字段名
        field_mappings = self.config.get('field_mappings', {}).get(data_type, {})

        for key, value in data.items():
            # 查找标准字段名，如果没有则使用原字段名
            std_key = field_mappings.get(key, key)
            # 标准化数值格式
            std_value = self._standardize_value(value)
            standardized[std_key] = std_value

        return standardized

    def _extract_table_title(self, table: pd.DataFrame) -> str:
        """从表格中提取标题"""
        if table.empty:
            return ""

        # 尝试从表格第一行提取标题
        first_row = ' '.join([str(cell) for cell in table.iloc[0].tolist() if pd.notna(cell)])
        return first_row

    def _is_related_table(self, curr_title: str, prev_title: str,
                          curr_table: pd.DataFrame, prev_table: pd.DataFrame) -> bool:
        """判断当前表格是否与前页表格相关（需要合并）"""
        # 1. 检查标题是否相关（如包含"续"或相同关键词）
        if any(title in curr_title and title in prev_title for title in self.finance_indicators):
            return True

        # 2. 检查是否有"续"字表示续表
        if "续" in curr_title and any(title in prev_title for title in curr_title.replace("续", "")):
            return True

        # 3. 检查列数是否一致（金融表格列数通常固定）
        if curr_table.shape[1] == prev_table.shape[1] and curr_table.shape[1] in self.config[
            'table_column_counts'].values():
            return True

        return False

    def _merge_tables(self, prev_table: pd.DataFrame, curr_table: pd.DataFrame) -> pd.DataFrame:
        """合并两个相关表格"""
        # 移除可能的标题行重复
        merged = pd.concat([prev_table, curr_table], ignore_index=True)
        # 去重但保留顺序
        merged = merged.drop_duplicates(keep='first').reset_index(drop=True)
        return merged

    def _get_previous_page_tables(self, pdf_path: str, page_num: int) -> List[tuple]:
        """获取前一页的表格数据"""
        # 实际实现中可以从缓存或文件中读取
        cache_dir = self._get_cache_dir(pdf_path)
        prev_page = page_num - 1
        prev_cache_file = os.path.join(cache_dir, f"page_{prev_page}.json")

        if os.path.exists(prev_cache_file):
            with open(prev_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [(pd.DataFrame(table_data), title, prev_page)
                        for table_data, title in zip(data['tables'], data['titles'])]

        return []

    def _save_current_page_tables(self, pdf_path: str, page_num: int, tables: List[pd.DataFrame]):
        """保存当前页的表格数据供后续页面使用"""
        cache_dir = self._get_cache_dir(pdf_path)
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f"page_{page_num}.json")
        data = {
            'tables': [table.to_dict(orient='records') for table in tables],
            'titles': [self._extract_table_title(table) for table in tables],
            'page_num': page_num
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_cache_dir(self, pdf_path: str) -> str:
        """获取缓存目录"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(os.path.dirname(pdf_path), f".{pdf_name}_table_cache")

    def _standardize_value(self, value: str) -> str:
        """标准化数值格式"""
        if not value:
            return ""

        # 移除多余空格
        value = value.strip()
        # 统一百分比格式
        value = re.sub(r'(\d+)%', r'\1%', value)
        # 统一数字格式，移除千位分隔符
        value = re.sub(r'(\d+),(\d+)', r'\1\2', value)
        return value

    def _extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取数据（当JSON解析失败时使用）"""
        result = {"chart_type": "", "indicator": "", "period_type": "", "data": {}}

        # 简单提取逻辑示例
        if "折线图" in text:
            result["chart_type"] = "折线图"
        elif "柱状图" in text:
            result["chart_type"] = "柱状图"

        for indicator in self.finance_indicators:
            if indicator in text:
                result["indicator"] = indicator
                break

        # 提取季度数据
        quarter_pattern = r'(Q[1-4])[:：]\s*([\d.]+%)'
        quarters = re.findall(quarter_pattern, text)
        if quarters:
            result["period_type"] = "季度"
            result["data"] = dict(quarters)

        # 提取年度数据
        if not result["data"]:
            year_pattern = r'(\d{4}年?)[:：]\s*([\d.]+%|[\d,.]+元)'
            years = re.findall(year_pattern, text)
            if years:
                result["period_type"] = "年度"
                result["data"] = dict(years)

        return result
