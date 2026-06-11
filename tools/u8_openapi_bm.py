import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 获取单个预算项目 数据模型 =====================
class GetBgitemInput(BaseModel):
    """获取单个预算项目输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="预算项目编码（必填）")


# ===================== 批量获取预算项目 数据模型 =====================
class GetBgitemBatchInput(BaseModel):
    """批量获取预算项目输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    year: Optional[int] = Field(None, description="编制年度")
    imakeyear: Optional[int] = Field(None, description="编制年度")
    citemcode: Optional[str] = Field(None, description="预算项目编码")
    citemname: Optional[str] = Field(None, description="预算项目名称关键字")
    citemgroupcode: Optional[str] = Field(None, description="项目组编码")
    citemtypecode: Optional[str] = Field(None, description="项目类型编码")


# ===================== 获取单个预算项目 Tool函数 =====================
def u8_bgitem_get_tool(input_data: GetBgitemInput, client: U8OpenAPIClient) -> str:
    """
    通过预算项目编码获取用友U8中的预算项目信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/bgitem/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "预算项目信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "预算项目信息获取成功",
            "data": {
                "bgitem": result.get("bgitem", {})
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


# ===================== 批量获取预算项目 Tool函数 =====================
def u8_bgitem_batch_get_tool(input_data: GetBgitemBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取预算项目信息，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/bgitem/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "预算项目列表获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "预算项目列表获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "bgitem": result.get("bgitem", {})
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


# ===================== 获取单个预算项目 Schema定义 =====================
U8_BGITEM_GET_SCHEMA = {
    "name": "u8_bgitem_get",
    "description": "在用友U8 OpenAPI中通过预算项目编码获取预算项目信息，返回编制年度、预算项目编码、名称、项目组编码、项目类型编码等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "预算项目编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取预算项目 Schema定义 =====================
U8_BGITEM_BATCH_GET_SCHEMA = {
    "name": "u8_bgitem_batch_get",
    "description": "在用友U8 OpenAPI中批量获取预算项目信息，支持按编制年度、预算项目编码、名称关键字、项目组编码、项目类型编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "year": {"type": "integer", "description": "编制年度"},
            "imakeyear": {"type": "integer", "description": "编制年度"},
            "citemcode": {"type": "string", "description": "预算项目编码"},
            "citemname": {"type": "string", "description": "预算项目名称关键字"},
            "citemgroupcode": {"type": "string", "description": "项目组编码"},
            "citemtypecode": {"type": "string", "description": "项目类型编码"}
        },
        "required": []
    }
}

