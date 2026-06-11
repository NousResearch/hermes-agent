import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)

# ===================== 获取EVA体检模型信息 数据模型 =====================
class EvaBatchGetInput(BaseModel):
    """EVA体检模型输入参数"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    year: str = Field(..., description="年份（必填）")
    month: str = Field(..., description="月份（必填）")


# ===================== 获取商业盈利状况评价信息 数据模型 =====================
class ProductProfitabilityBatchGetInput(BaseModel):
    """商业盈利状况评价输入参数"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    year: Optional[str] = Field(None, description="年份")
    month: Optional[str] = Field(None, description="月份")


# ===================== 获取资金体检模型信息 数据模型 =====================
class FundBatchGetInput(BaseModel):
    """资金体检模型输入参数"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    year: str = Field(..., description="年份（必填）")
    month: str = Field(..., description="月份（必填）")


# ===================== 获取EVA体检模型信息 Tool函数 =====================
def u8_eva_batch_get_tool(input_data: EvaBatchGetInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取EVA体检模型信息，支持分页查询。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eva/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "EVA体检模型信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "EVA体检模型信息获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "eva": result.get("eva", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 获取商业盈利状况评价信息 Tool函数 =====================
def u8_productprofitability_batch_get_tool(input_data: ProductProfitabilityBatchGetInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取商业盈利状况评价信息，支持分页查询。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/productprofitability/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "商业盈利状况评价信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "商业盈利状况评价信息获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "productprofitability": result.get("productprofitability", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 获取资金体检模型信息 Tool函数 =====================
def u8_fund_batch_get_tool(input_data: FundBatchGetInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取资金体检模型信息，支持分页查询。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/fund/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "资金体检模型信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "资金体检模型信息获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "fund": result.get("fund", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 获取EVA体检模型信息 Schema定义 =====================
U8_EVA_BATCH_GET_SCHEMA = {
    "name": "u8_eva_batch_get",
    "description": "在用友U8 OpenAPI中获取EVA体检模型信息，返回指标名称、当期、上期、增长额、增长率等数据",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "year": {"type": "string", "description": "年份（必填）"},
            "month": {"type": "string", "description": "月份（必填）"}
        },
        "required": ["year", "month"]
    }
}


# ===================== 获取商业盈利状况评价信息 Schema定义 =====================
U8_PRODUCTPROFITABILITY_BATCH_GET_SCHEMA = {
    "name": "u8_productprofitability_batch_get",
    "description": "在用友U8 OpenAPI中获取商业盈利状况评价信息，返回主要指标、次要指标及商品明细数据",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "year": {"type": "string", "description": "年份"},
            "month": {"type": "string", "description": "月份"}
        },
        "required": []
    }
}


# ===================== 获取资金体检模型信息 Schema定义 =====================
U8_FUND_BATCH_GET_SCHEMA = {
    "name": "u8_fund_batch_get",
    "description": "在用友U8 OpenAPI中获取资金体检模型信息，返回指标名称、当期、上期、增长额、增长率等数据",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "year": {"type": "string", "description": "年份（必填）"},
            "month": {"type": "string", "description": "月份（必填）"}
        },
        "required": ["year", "month"]
    }
}
