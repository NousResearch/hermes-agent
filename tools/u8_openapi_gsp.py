import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)



# ===================== 获取单个药品停售通知单 数据模型 =====================
class GetStopsalenoticeInput(BaseModel):
    """获取单个药品停售通知单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="单据编号（必填）")


# ===================== 批量获取药品停售通知单 数据模型 =====================
class GetStopsalenoticeBatchInput(BaseModel):
    """批量获取药品停售通知单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cinid_begin: Optional[str] = Field(None, description="起始单据编号")
    cinid_end: Optional[str] = Field(None, description="结束单据编号")
    cinvcode_begin: Optional[str] = Field(None, description="起始商品编码")
    cinvcode_end: Optional[str] = Field(None, description="结束商品编码")
    cinvname: Optional[str] = Field(None, description="商品名称关键字")
    cbatch: Optional[str] = Field(None, description="批号")
    cdefine8: Optional[str] = Field(None, description="停售方式")
    cprocess: Optional[str] = Field(None, description="后续处理")
    dverifydate_begin: Optional[str] = Field(None, description="起始审核日期")
    dverifydate_end: Optional[str] = Field(None, description="结束审核日期")
    ddate_begin: Optional[str] = Field(None, description="起始通知日期")
    ddate_end: Optional[str] = Field(None, description="结束通知日期")
    cdefine1: Optional[str] = Field(None, description="表头自定义项1")
    cdefine2: Optional[str] = Field(None, description="表头自定义项2")
    cdefine3: Optional[str] = Field(None, description="表头自定义项3")
    cdefine4: Optional[str] = Field(None, description="表头自定义项4")
    cdefine5: Optional[str] = Field(None, description="表头自定义项5")
    cdefine6: Optional[str] = Field(None, description="表头自定义项6")
    cdefine7: Optional[str] = Field(None, description="表头自定义项7")
    cdefine8: Optional[str] = Field(None, description="表头自定义项8")
    cdefine9: Optional[str] = Field(None, description="表头自定义项9")
    cdefine10: Optional[str] = Field(None, description="表头自定义项10")
    cdefine11: Optional[str] = Field(None, description="表头自定义项11")
    cdefine12: Optional[str] = Field(None, description="表头自定义项12")
    cdefine13: Optional[str] = Field(None, description="表头自定义项13")
    cdefine14: Optional[str] = Field(None, description="表头自定义项14")
    cdefine15: Optional[str] = Field(None, description="表头自定义项15")
    cdefine16: Optional[str] = Field(None, description="表头自定义项16")
    cbatchproperty1: Optional[str] = Field(None, description="批次属性1")
    cbatchproperty2: Optional[str] = Field(None, description="批次属性2")
    cbatchproperty3: Optional[str] = Field(None, description="批次属性3")
    cbatchproperty4: Optional[str] = Field(None, description="批次属性4")
    cbatchproperty5: Optional[str] = Field(None, description="批次属性5")
    cbatchproperty6: Optional[str] = Field(None, description="批次属性6")
    cbatchproperty7: Optional[str] = Field(None, description="批次属性7")
    cbatchproperty8: Optional[str] = Field(None, description="批次属性8")
    cbatchproperty9: Optional[str] = Field(None, description="批次属性9")
    cbatchproperty10: Optional[str] = Field(None, description="批次属性10")
    cfree1: Optional[str] = Field(None, description="商品自由项1")
    cfree2: Optional[str] = Field(None, description="商品自由项2")
    cfree3: Optional[str] = Field(None, description="商品自由项3")
    cfree4: Optional[str] = Field(None, description="商品自由项4")
    cfree5: Optional[str] = Field(None, description="商品自由项5")
    cfree6: Optional[str] = Field(None, description="商品自由项6")
    cfree7: Optional[str] = Field(None, description="商品自由项7")
    cfree8: Optional[str] = Field(None, description="商品自由项8")
    cfree9: Optional[str] = Field(None, description="商品自由项9")
    cfree10: Optional[str] = Field(None, description="商品自由项10")


# ===================== 获取单个药品停售通知单 Tool函数 =====================
def u8_stopsalenotice_get_tool(input_data: GetStopsalenoticeInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编号获取用友U8中的药品停售通知单信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/stopsalenotice/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "药品停售通知单获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "药品停售通知单获取成功",
            "data": {
                "stopsalenotice": result.get("stopsalenotice", {})
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


# ===================== 批量获取药品停售通知单 Tool函数 =====================
def u8_stopsalenotice_batch_get_tool(input_data: GetStopsalenoticeBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取药品停售通知单信息，支持按单据编号、商品编码、批号、停售方式、通知日期等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/stopsalenotice/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "药品停售通知单批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "药品停售通知单批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "stopsalenotice": result.get("stopsalenotice", [])
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


# ===================== 获取单个药品停售通知单 Schema定义 =====================
U8_STOPSALENOTICE_GET_SCHEMA = {
    "name": "u8_stopsalenotice_get",
    "description": "在用友U8 GSP管理系统中通过单据编号获取单个药品停售通知单信息，包含商品编码、商品名称、规格、批号、数量、停售方式、停售原因、通知日期、审核日期、仓库、生产企业、生产日期、失效日期等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取药品停售通知单 Schema定义 =====================
U8_STOPSALENOTICE_BATCH_GET_SCHEMA = {
    "name": "u8_stopsalenotice_batch_get",
    "description": "在用友U8 GSP管理系统中批量获取药品停售通知单信息，支持按单据编号、商品编码、商品名称、批号、停售方式、通知日期、审核日期等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cinid_begin": {"type": "string", "description": "起始单据编号"},
            "cinid_end": {"type": "string", "description": "结束单据编号"},
            "cinvcode_begin": {"type": "string", "description": "起始商品编码"},
            "cinvcode_end": {"type": "string", "description": "结束商品编码"},
            "cinvname": {"type": "string", "description": "商品名称关键字"},
            "cbatch": {"type": "string", "description": "批号"},
            "cdefine8": {"type": "string", "description": "停售方式"},
            "cprocess": {"type": "string", "description": "后续处理"},
            "dverifydate_begin": {"type": "string", "description": "起始审核日期"},
            "dverifydate_end": {"type": "string", "description": "结束审核日期"},
            "ddate_begin": {"type": "string", "description": "起始通知日期"},
            "ddate_end": {"type": "string", "description": "结束通知日期"},
            "cdefine1": {"type": "string", "description": "表头自定义项1"},
            "cdefine2": {"type": "string", "description": "表头自定义项2"},
            "cdefine3": {"type": "string", "description": "表头自定义项3"},
            "cdefine4": {"type": "string", "description": "表头自定义项4"},
            "cdefine5": {"type": "string", "description": "表头自定义项5"},
            "cdefine6": {"type": "string", "description": "表头自定义项6"},
            "cdefine7": {"type": "string", "description": "表头自定义项7"},
            "cdefine8": {"type": "string", "description": "表头自定义项8"},
            "cdefine9": {"type": "string", "description": "表头自定义项9"},
            "cdefine10": {"type": "string", "description": "表头自定义项10"},
            "cdefine11": {"type": "string", "description": "表头自定义项11"},
            "cdefine12": {"type": "string", "description": "表头自定义项12"},
            "cdefine13": {"type": "string", "description": "表头自定义项13"},
            "cdefine14": {"type": "string", "description": "表头自定义项14"},
            "cdefine15": {"type": "string", "description": "表头自定义项15"},
            "cdefine16": {"type": "string", "description": "表头自定义项16"},
            "cbatchproperty1": {"type": "string", "description": "批次属性1"},
            "cbatchproperty2": {"type": "string", "description": "批次属性2"},
            "cbatchproperty3": {"type": "string", "description": "批次属性3"},
            "cbatchproperty4": {"type": "string", "description": "批次属性4"},
            "cbatchproperty5": {"type": "string", "description": "批次属性5"},
            "cbatchproperty6": {"type": "string", "description": "批次属性6"},
            "cbatchproperty7": {"type": "string", "description": "批次属性7"},
            "cbatchproperty8": {"type": "string", "description": "批次属性8"},
            "cbatchproperty9": {"type": "string", "description": "批次属性9"},
            "cbatchproperty10": {"type": "string", "description": "批次属性10"},
            "cfree1": {"type": "string", "description": "商品自由项1"},
            "cfree2": {"type": "string", "description": "商品自由项2"},
            "cfree3": {"type": "string", "description": "商品自由项3"},
            "cfree4": {"type": "string", "description": "商品自由项4"},
            "cfree5": {"type": "string", "description": "商品自由项5"},
            "cfree6": {"type": "string", "description": "商品自由项6"},
            "cfree7": {"type": "string", "description": "商品自由项7"},
            "cfree8": {"type": "string", "description": "商品自由项8"},
            "cfree9": {"type": "string", "description": "商品自由项9"},
            "cfree10": {"type": "string", "description": "商品自由项10"}
        },
        "required": []
    }
}

