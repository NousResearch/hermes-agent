import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)

# ===================== 获取单个条码档案 数据模型 =====================
class GetBarcodeInput(BaseModel):
    """获取单个条码档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="条码（必填）")


# ===================== 批量获取条码档案 数据模型 =====================
class GetBarcodeBatchInput(BaseModel):
    """批量获取条码档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    barcode_begin: Optional[str] = Field(None, description="起始条码")
    barcode_end: Optional[str] = Field(None, description="结束条码")
    cinvcode: Optional[str] = Field(None, description="存货编码")
    cinvccode: Optional[str] = Field(None, description="存货分类编码")
    cinvcname: Optional[str] = Field(None, description="存货分类名称关键字")
    cinvstd: Optional[str] = Field(None, description="规格型号关键字")
    cposition: Optional[str] = Field(None, description="货位编码")
    cdefwarehouse: Optional[str] = Field(None, description="仓库编码")
    cvencode: Optional[str] = Field(None, description="供应商编码")
    cvenname: Optional[str] = Field(None, description="供应商名称关键字")
    dmdate: Optional[str] = Field(None, description="生产日期")
    dvdate: Optional[str] = Field(None, description="失效日期")
    cwhcode: Optional[str] = Field(None, description="仓库编码")
    cwhname: Optional[str] = Field(None, description="仓库名称关键字")
    cposcode: Optional[str] = Field(None, description="货位编码")
    cposname: Optional[str] = Field(None, description="货位名称关键字")
    plot: Optional[str] = Field(None, description="批次")
    createdate: Optional[str] = Field(None, description="生成日期")
    createtime: Optional[str] = Field(None, description="生成时间")
    csrccode: Optional[str] = Field(None, description="来源单code")
    csrcvouchtype: Optional[str] = Field(None, description="来源单据类型")
    csrcsubid: Optional[int] = Field(None, description="来源单")
    cmaker: Optional[str] = Field(None, description="制单人关键字")
    cinvsn: Optional[str] = Field(None, description="序列号")
    iprtperson: Optional[str] = Field(None, description="打印人关键字")
    ibarcodestate: Optional[str] = Field(None, description="条码状态")
    dbusdate: Optional[str] = Field(None, description="业务日期")
    busels: Optional[int] = Field(None, description="是否零售")
    ccusabbname: Optional[str] = Field(None, description="客户简称关键字")
    ccusinvname: Optional[str] = Field(None, description="客户存货名称关键字")
    ccusname: Optional[str] = Field(None, description="客户名称关键字")
    ccusinvcode: Optional[str] = Field(None, description="客户存货编码")
    cinvname: Optional[str] = Field(None, description="存货名称关键字")
    cfree1: Optional[str] = Field(None, description="存货自由项1")
    cfree2: Optional[str] = Field(None, description="存货自由项2")
    cfree3: Optional[str] = Field(None, description="存货自由项3")
    cfree4: Optional[str] = Field(None, description="存货自由项4")
    cfree5: Optional[str] = Field(None, description="存货自由项5")
    cfree6: Optional[str] = Field(None, description="存货自由项6")
    cfree7: Optional[str] = Field(None, description="存货自由项7")
    cfree8: Optional[str] = Field(None, description="存货自由项8")
    cfree9: Optional[str] = Field(None, description="存货自由项9")
    cfree10: Optional[str] = Field(None, description="存货自由项10")


# ===================== 获取单个条码档案 Tool函数 =====================
def u8_barcode_get_tool(input_data: GetBarcodeInput, client: U8OpenAPIClient) -> str:
    """
    通过条码获取用友U8中的条码档案信息，包含存货信息、仓库信息、供应商信息等。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/barcode/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "条码档案获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "条码档案获取成功",
            "data": {
                "barcode": result.get("barcode", {})
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


# ===================== 批量获取条码档案 Tool函数 =====================
def u8_barcode_batch_get_tool(input_data: GetBarcodeBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取条码档案信息，支持按条码、存货编码、仓库、供应商等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/barcode/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "条码档案批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "条码档案批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "barcode": result.get("barcode", [])
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


# ===================== 获取单个条码档案 Schema定义 =====================
U8_BARCODE_GET_SCHEMA = {
    "name": "u8_barcode_get",
    "description": "在用友U8 OpenAPI中通过条码获取条码档案信息，包含存货编码、存货名称、规格型号、仓库、货位、数量、批次、生产日期、失效日期、供应商等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "条码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取条码档案 Schema定义 =====================
U8_BARCODE_BATCH_GET_SCHEMA = {
    "name": "u8_barcode_batch_get",
    "description": "在用友U8 OpenAPI中批量获取条码档案信息，支持按条码范围、存货编码、存货分类、仓库、货位、供应商、生产日期、批次等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "barcode_begin": {"type": "string", "description": "起始条码"},
            "barcode_end": {"type": "string", "description": "结束条码"},
            "cinvcode": {"type": "string", "description": "存货编码"},
            "cinvccode": {"type": "string", "description": "存货分类编码"},
            "cinvcname": {"type": "string", "description": "存货分类名称关键字"},
            "cinvstd": {"type": "string", "description": "规格型号关键字"},
            "cposition": {"type": "string", "description": "货位编码"},
            "cdefwarehouse": {"type": "string", "description": "仓库编码"},
            "cvencode": {"type": "string", "description": "供应商编码"},
            "cvenname": {"type": "string", "description": "供应商名称关键字"},
            "dmdate": {"type": "string", "description": "生产日期"},
            "dvdate": {"type": "string", "description": "失效日期"},
            "cwhcode": {"type": "string", "description": "仓库编码"},
            "cwhname": {"type": "string", "description": "仓库名称关键字"},
            "cposcode": {"type": "string", "description": "货位编码"},
            "cposname": {"type": "string", "description": "货位名称关键字"},
            "plot": {"type": "string", "description": "批次"},
            "createdate": {"type": "string", "description": "生成日期"},
            "createtime": {"type": "string", "description": "生成时间"},
            "csrccode": {"type": "string", "description": "来源单code"},
            "csrcvouchtype": {"type": "string", "description": "来源单据类型"},
            "csrcsubid": {"type": "integer", "description": "来源单"},
            "cmaker": {"type": "string", "description": "制单人关键字"},
            "cinvsn": {"type": "string", "description": "序列号"},
            "iprtperson": {"type": "string", "description": "打印人关键字"},
            "ibarcodestate": {"type": "string", "description": "条码状态"},
            "dbusdate": {"type": "string", "description": "业务日期"},
            "busels": {"type": "integer", "description": "是否零售"},
            "ccusabbname": {"type": "string", "description": "客户简称关键字"},
            "ccusinvname": {"type": "string", "description": "客户存货名称关键字"},
            "ccusname": {"type": "string", "description": "客户名称关键字"},
            "ccusinvcode": {"type": "string", "description": "客户存货编码"},
            "cinvname": {"type": "string", "description": "存货名称关键字"},
            "cfree1": {"type": "string", "description": "存货自由项1"},
            "cfree2": {"type": "string", "description": "存货自由项2"},
            "cfree3": {"type": "string", "description": "存货自由项3"},
            "cfree4": {"type": "string", "description": "存货自由项4"},
            "cfree5": {"type": "string", "description": "存货自由项5"},
            "cfree6": {"type": "string", "description": "存货自由项6"},
            "cfree7": {"type": "string", "description": "存货自由项7"},
            "cfree8": {"type": "string", "description": "存货自由项8"},
            "cfree9": {"type": "string", "description": "存货自由项9"},
            "cfree10": {"type": "string", "description": "存货自由项10"}
        },
        "required": []
    }
}
