import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)



# ===================== 获取单个客户资质审批 数据模型 =====================
class GetCustomerlicenceInput(BaseModel):
    """获取单个客户资质审批输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="客户编码（必填）")


# ===================== 批量获取客户资质审批 数据模型 =====================
class GetCustomerlicenceBatchInput(BaseModel):
    """批量获取客户资质审批输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    ccuscode_begin: Optional[str] = Field(None, description="起始客户编码")
    ccuscode_end: Optional[str] = Field(None, description="结束客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称关键字")
    ccusabbname: Optional[str] = Field(None, description="客户简称关键字")
    cmaker: Optional[str] = Field(None, description="制单人关键字")
    cverifier: Optional[str] = Field(None, description="审核人关键字")
    dveridate: Optional[str] = Field(None, description="审核日期")
    bccode_begin: Optional[str] = Field(None, description="起始资质类型编码")
    bccode_end: Optional[str] = Field(None, description="结束资质类型编码")
    cname: Optional[str] = Field(None, description="资质类型名称关键字")
    clccode_begin: Optional[str] = Field(None, description="起始资质编码")
    clccode_end: Optional[str] = Field(None, description="结束资质编码")
    clcname: Optional[str] = Field(None, description="资质名称关键字")
    qcno: Optional[str] = Field(None, description="资质证号")


# ===================== 获取单个客户资质经营范围审批 数据模型 =====================
class GetCustomerlicencebizscopeInput(BaseModel):
    """获取单个客户资质经营范围审批输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="客户编码（必填）")


# ===================== 批量获取客户资质经营范围审批 数据模型 =====================
class GetCustomerlicencebizscopeBatchInput(BaseModel):
    """批量获取客户资质经营范围审批输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    clicenceclasscode_begin: Optional[str] = Field(None, description="起始资质类型编号")
    clicenceclasscode_end: Optional[str] = Field(None, description="结束资质类型编号")
    clicenceclassname: Optional[str] = Field(None, description="资质类型名称关键字")
    clicencecode_begin: Optional[str] = Field(None, description="起始资质编号")
    clicencecode_end: Optional[str] = Field(None, description="结束资质编号")
    clicencename: Optional[str] = Field(None, description="资质名称关键字")
    cmaker: Optional[str] = Field(None, description="制单人关键字")
    ccuscode_begin: Optional[str] = Field(None, description="起始客户编码")
    ccuscode_end: Optional[str] = Field(None, description="结束客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称关键字")
    ccusabbname: Optional[str] = Field(None, description="客户简称关键字")
    ddate: Optional[str] = Field(None, description="单据日期")
    bccdoe_begin: Optional[str] = Field(None, description="起始药品分类编号")
    bccdoe_end: Optional[str] = Field(None, description="结束药品分类编号")
    cname: Optional[str] = Field(None, description="药品分类名称关键字")
    cinvstd: Optional[str] = Field(None, description="存货规格")
    clccode_begin: Optional[str] = Field(None, description="起始存货编码")
    clccode_end: Optional[str] = Field(None, description="结束存货编码")
    clcname: Optional[str] = Field(None, description="存货名称关键字")

# ===================== 获取单个客户资质审批 Tool函数 =====================
def u8_customerlicence_get_tool(input_data: GetCustomerlicenceInput, client: U8OpenAPIClient) -> str:
    """
    通过客户编码获取用友U8中的客户资质审批信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/customerlicence/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "客户资质审批获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "客户资质审批获取成功",
            "data": {
                "customerlicence": result.get("customerlicence", {})
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


# ===================== 批量获取客户资质审批 Tool函数 =====================
def u8_customerlicence_batch_get_tool(input_data: GetCustomerlicenceBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取客户资质审批信息，支持按客户编码、资质类型、资质名称等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/customerlicence/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "客户资质审批批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "客户资质审批批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customerlicence": result.get("customerlicence", {})
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


# ===================== 获取单个客户资质经营范围审批 Tool函数 =====================
def u8_customerlicencebizscope_get_tool(input_data: GetCustomerlicencebizscopeInput, client: U8OpenAPIClient) -> str:
    """
    通过客户编码获取用友U8中的客户资质经营范围审批信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/customerlicencebizscope/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "客户资质经营范围审批获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "客户资质经营范围审批获取成功",
            "data": {
                "customerlicencebizscope": result.get("customerlicencebizscope", {})
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


# ===================== 批量获取客户资质经营范围审批 Tool函数 =====================
def u8_customerlicencebizscope_batch_get_tool(input_data: GetCustomerlicencebizscopeBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取客户资质经营范围审批信息，支持按资质类型、资质编号、客户编码、药品分类等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/customerlicencebizscope/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "客户资质经营范围审批批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "客户资质经营范围审批批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customerlicencebizscope": result.get("customerlicencebizscope", {})
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


# ===================== 获取单个客户资质审批 Schema定义 =====================
U8_CUSTOMERLICENCE_GET_SCHEMA = {
    "name": "u8_customerlicence_get",
    "description": "在用友U8资质管理系统中通过客户编码获取单个客户资质审批信息，包含资质类型、资质名称、生效日、到期日、审核信息等",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "客户编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取客户资质审批 Schema定义 =====================
U8_CUSTOMERLICENCE_BATCH_GET_SCHEMA = {
    "name": "u8_customerlicence_batch_get",
    "description": "在用友U8资质管理系统中批量获取客户资质审批信息，支持按客户编码、客户名称、资质类型、资质名称、资质证号等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "ccuscode_begin": {"type": "string", "description": "起始客户编码"},
            "ccuscode_end": {"type": "string", "description": "结束客户编码"},
            "ccusname": {"type": "string", "description": "客户名称关键字"},
            "ccusabbname": {"type": "string", "description": "客户简称关键字"},
            "cmaker": {"type": "string", "description": "制单人关键字"},
            "cverifier": {"type": "string", "description": "审核人关键字"},
            "dveridate": {"type": "string", "description": "审核日期"},
            "bccode_begin": {"type": "string", "description": "起始资质类型编码"},
            "bccode_end": {"type": "string", "description": "结束资质类型编码"},
            "cname": {"type": "string", "description": "资质类型名称关键字"},
            "clccode_begin": {"type": "string", "description": "起始资质编码"},
            "clccode_end": {"type": "string", "description": "结束资质编码"},
            "clcname": {"type": "string", "description": "资质名称关键字"},
            "qcno": {"type": "string", "description": "资质证号"}
        },
        "required": []
    }
}


# ===================== 获取单个客户资质经营范围审批 Schema定义 =====================
U8_CUSTOMERLICENCEBIZSCOPE_GET_SCHEMA = {
    "name": "u8_customerlicencebizscope_get",
    "description": "在用友U8资质管理系统中通过客户编码获取单个客户资质经营范围审批信息，包含资质类型、药品分类、存货信息等",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "客户编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取客户资质经营范围审批 Schema定义 =====================
U8_CUSTOMERLICENCEBIZSCOPE_BATCH_GET_SCHEMA = {
    "name": "u8_customerlicencebizscope_batch_get",
    "description": "在用友U8资质管理系统中批量获取客户资质经营范围审批信息，支持按资质类型、资质编号、客户编码、药品分类、存货编码等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "clicenceclasscode_begin": {"type": "string", "description": "起始资质类型编号"},
            "clicenceclasscode_end": {"type": "string", "description": "结束资质类型编号"},
            "clicenceclassname": {"type": "string", "description": "资质类型名称关键字"},
            "clicencecode_begin": {"type": "string", "description": "起始资质编号"},
            "clicencecode_end": {"type": "string", "description": "结束资质编号"},
            "clicencename": {"type": "string", "description": "资质名称关键字"},
            "cmaker": {"type": "string", "description": "制单人关键字"},
            "ccuscode_begin": {"type": "string", "description": "起始客户编码"},
            "ccuscode_end": {"type": "string", "description": "结束客户编码"},
            "ccusname": {"type": "string", "description": "客户名称关键字"},
            "ccusabbname": {"type": "string", "description": "客户简称关键字"},
            "ddate": {"type": "string", "description": "单据日期"},
            "bccdoe_begin": {"type": "string", "description": "起始药品分类编号"},
            "bccdoe_end": {"type": "string", "description": "结束药品分类编号"},
            "cname": {"type": "string", "description": "药品分类名称关键字"},
            "cinvstd": {"type": "string", "description": "存货规格"},
            "clccode_begin": {"type": "string", "description": "起始存货编码"},
            "clccode_end": {"type": "string", "description": "结束存货编码"},
            "clcname": {"type": "string", "description": "存货名称关键字"}
        },
        "required": []
    }
}
