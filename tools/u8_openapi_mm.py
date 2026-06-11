import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)



# ===================== 获取单个物料清单 数据模型 =====================
class GetBomInput(BaseModel):
    """获取单个物料清单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: int = Field(..., description="母件物料Id（必填）")


# ===================== 批量获取物料清单 数据模型 =====================
class GetBomBatchInput(BaseModel):
    """批量获取物料清单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    bomtype: Optional[int] = Field(None, description="BOM类型（主要/替代）")
    version: Optional[int] = Field(None, description="版本号")
    versiondesc: Optional[str] = Field(None, description="版本说明关键字")
    versioneffdate: Optional[str] = Field(None, description="版本生效日")
    identcode: Optional[str] = Field(None, description="替代标识")
    identdesc: Optional[str] = Field(None, description="替代说明关键字")
    cinvcode: Optional[str] = Field(None, description="母件编码")
    cinvname: Optional[str] = Field(None, description="母件名称关键字")
    cinvstd: Optional[str] = Field(None, description="规格型号")
    cinvccode: Optional[str] = Field(None, description="存货大类编码")
    cinvcname: Optional[str] = Field(None, description="存货大类关键字")
    free1: Optional[str] = Field(None, description="自由项1")
    free2: Optional[str] = Field(None, description="自由项2")
    free3: Optional[str] = Field(None, description="自由项3")
    free4: Optional[str] = Field(None, description="自由项4")
    free5: Optional[str] = Field(None, description="自由项5")
    free6: Optional[str] = Field(None, description="自由项6")
    free7: Optional[str] = Field(None, description="自由项7")
    free8: Optional[str] = Field(None, description="自由项8")
    free9: Optional[str] = Field(None, description="自由项9")
    free10: Optional[str] = Field(None, description="自由项10")
    status: Optional[int] = Field(None, description="状态（1:新建/3:审核/4:停用）")
    createuser: Optional[str] = Field(None, description="创建人关键字")
    closeuser: Optional[str] = Field(None, description="关闭人关键字")


# ===================== 获取单个物料清单 Tool函数 =====================
def u8_bom_get_tool(input_data: GetBomInput, client: U8OpenAPIClient) -> str:
    """
    通过母件物料Id获取用友U8中的物料清单(BOM)信息，包含表头及表体子件明细(entry)。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/bom/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "物料清单获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "物料清单获取成功",
            "data": {
                "bom": result.get("bom", {}),
                "entry": result.get("entry", [])
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


# ===================== 批量获取物料清单 Tool函数 =====================
def u8_bom_batch_get_tool(input_data: GetBomBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取物料清单(BOM)信息，支持按BOM类型、版本号、母件编码、存货大类、状态等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/bom/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "物料清单批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "物料清单批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "bom": result.get("bom", [])
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


# ===================== 获取单个物料清单 Schema定义 =====================
U8_BOM_GET_SCHEMA = {
    "name": "u8_bom_get",
    "description": "在用友U8物料清单系统中通过母件物料Id获取单个物料清单(BOM)信息，包含表头（BOM类型、版本号、版本说明、生效日、存货编码等）及表体子件明细(entry)",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "integer", "description": "母件物料Id（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取物料清单 Schema定义 =====================
U8_BOM_BATCH_GET_SCHEMA = {
    "name": "u8_bom_batch_get",
    "description": "在用友U8物料清单系统中批量获取物料清单(BOM)信息，支持按BOM类型、版本号、版本说明、替代标识、母件编码、母件名称、存货大类、状态、创建人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "bomtype": {"type": "integer", "description": "BOM类型（主要/替代）"},
            "version": {"type": "integer", "description": "版本号"},
            "versiondesc": {"type": "string", "description": "版本说明关键字"},
            "versioneffdate": {"type": "string", "description": "版本生效日"},
            "identcode": {"type": "string", "description": "替代标识"},
            "identdesc": {"type": "string", "description": "替代说明关键字"},
            "cinvcode": {"type": "string", "description": "母件编码"},
            "cinvname": {"type": "string", "description": "母件名称关键字"},
            "cinvstd": {"type": "string", "description": "规格型号"},
            "cinvccode": {"type": "string", "description": "存货大类编码"},
            "cinvcname": {"type": "string", "description": "存货大类关键字"},
            "free1": {"type": "string", "description": "自由项1"},
            "free2": {"type": "string", "description": "自由项2"},
            "free3": {"type": "string", "description": "自由项3"},
            "free4": {"type": "string", "description": "自由项4"},
            "free5": {"type": "string", "description": "自由项5"},
            "free6": {"type": "string", "description": "自由项6"},
            "free7": {"type": "string", "description": "自由项7"},
            "free8": {"type": "string", "description": "自由项8"},
            "free9": {"type": "string", "description": "自由项9"},
            "free10": {"type": "string", "description": "自由项10"},
            "status": {"type": "integer", "description": "状态（1:新建/3:审核/4:停用）"},
            "createuser": {"type": "string", "description": "创建人关键字"},
            "closeuser": {"type": "string", "description": "关闭人关键字"}
        },
        "required": []
    }
}
