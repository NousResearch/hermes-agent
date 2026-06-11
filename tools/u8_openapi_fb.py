import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 费用预算查看 数据模型 =====================
class BudgetQueryInput(BaseModel):
    """费用预算查看输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    SysID: str = Field(..., description="来源系统ID（必填）")
    SysName: str = Field(..., description="来源系统名称（必填）")
    VoucherID: str = Field(..., description="单据编号号（必填）")
    VoucherType: str = Field(..., description="单据类型（必填）")
    VoucherCode: str = Field(..., description="单据编号（必填）")
    VoucherDate: str = Field(..., description="单据日期，格式：yyyy-MM-dd（必填）")
    BudgetFactAddDate: Optional[str] = Field(None, description="预算扣减日期，如果不填值，默认使用VoucherDate作为扣减日期")
    VoucherUpdateDate: str = Field(..., description="单据修改日期，格式：yyyy-MM-dd（必填）")
    BudgetFactAdd: str = Field(..., description="同一张单据预算扣减标志 (True:扣减预算 False:回冲预算)（必填）")
    VoucherEntrys: Optional[List[Dict[str, Any]]] = Field(None, description="单据分录列表")


# ===================== 批量获取预算信息 数据模型 =====================
class GetBudgetBatchInput(BaseModel):
    """批量获取预算信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cformcode: Optional[str] = Field(None, description="预算表")
    ccalibercode1: Optional[str] = Field(None, description="口径1")
    cversioncode: Optional[str] = Field(None, description="版本号")
    ctargetcode: Optional[str] = Field(None, description="预算指标")
    ctargetcode_ctl: Optional[str] = Field(None, description="预算控制指标")
    citemcode: Optional[str] = Field(None, description="预算项目编码")
    citemname: Optional[str] = Field(None, description="预算项目名称关键字")
    fperiod13: Optional[float] = Field(None, description="预算数")
    fperiod12: Optional[float] = Field(None, description="实际发生")
    freserve12: Optional[float] = Field(None, description="实际占用")
    pk: Optional[int] = Field(None, description="唯一标识")


# ===================== 获取单个预算项目 数据模型 =====================
class GetBudgetitemInput(BaseModel):
    """获取单个预算项目输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取预算项目 数据模型 =====================
class GetBudgetitemBatchInput(BaseModel):
    """批量获取预算项目输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    iyear: Optional[int] = Field(None, description="年度")
    csystem: Optional[str] = Field(None, description="控制系统编号")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")


# ===================== 费用预算查看 Tool函数 =====================
def u8_budget_query_tool(input_data: BudgetQueryInput, client: U8OpenAPIClient) -> str:
    """
    费用预算查看，用于预算扣减/回冲操作，返回预算结果及预算数据明细。
    """
    # 构造接口要求的标准 JSON 结构
    request_body: dict = input_data.model_dump(exclude_none=True)
    # 移除 ds_sequence 从 body，它应该放在 URL 参数中
    ds_sequence = request_body.pop("ds_sequence", None)
    inparams = {}
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    api_path = "/api/budget/query"

    try:
        result = client.request_api("POST", api_path, inparams=inparams if inparams else None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "费用预算查看失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "费用预算查看成功",
            "data": {
                "tradeid": result.get("tradeid"),
                "Result": result.get("Result"),
                "ResultInfo": result.get("ResultInfo"),
                "BudgetDataRows": result.get("BudgetDataRows", [])
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


# ===================== 批量获取预算信息 Tool函数 =====================
def u8_budget_batch_get_tool(input_data: GetBudgetBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取预算信息，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/budget/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "预算信息列表获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "预算信息列表获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "budget": result.get("budget", {})
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


# ===================== 获取单个预算项目 Tool函数 =====================
def u8_budgetitem_get_tool(input_data: GetBudgetitemInput, client: U8OpenAPIClient) -> str:
    """
    获取用友U8中的预算项目信息，返回年度、预算项目编码、名称、控制系统等信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/budgetitem/get"

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
                "budgetitem": result.get("budgetitem", {})
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
def u8_budgetitem_batch_get_tool(input_data: GetBudgetitemBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取预算项目信息，支持分页查询和多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/budgetitem/batch_get"

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
                "budgetitem": result.get("budgetitem", [])
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


# ===================== 费用预算查看 Schema定义 =====================
U8_BUDGET_QUERY_SCHEMA = {
    "name": "u8_budget_query",
    "description": "在用友U8 OpenAPI中进行费用预算查看，用于预算扣减/回冲操作，返回预算结果(-1:通过 0:严格控制未通过 2:提示超过允许继续)及预算数据明细",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "SysID": {"type": "string", "description": "来源系统ID（必填）"},
            "SysName": {"type": "string", "description": "来源系统名称（必填）"},
            "VoucherID": {"type": "string", "description": "单据编号号（必填）"},
            "VoucherType": {"type": "string", "description": "单据类型（必填）"},
            "VoucherCode": {"type": "string", "description": "单据编号（必填）"},
            "VoucherDate": {"type": "string", "description": "单据日期，格式：yyyy-MM-dd（必填）"},
            "BudgetFactAddDate": {"type": "string", "description": "预算扣减日期，如果不填值，默认使用VoucherDate作为扣减日期"},
            "VoucherUpdateDate": {"type": "string", "description": "单据修改日期，格式：yyyy-MM-dd（必填）"},
            "BudgetFactAdd": {"type": "string", "description": "同一张单据预算扣减标志 (True:扣减预算 False:回冲预算)（必填）"},
            "VoucherEntrys": {
                "type": "array",
                "description": "单据分录列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "RowNum": {"type": "number", "description": "行号"},
                        "cDeptID": {"type": "string", "description": "部门编码"},
                        "cDeptName": {"type": "string", "description": "部门名称"},
                        "cItemID": {"type": "string", "description": "项目编码"},
                        "cItemName": {"type": "string", "description": "项目名称"},
                        "Digest": {"type": "string", "description": "摘要（必填）"},
                        "fMoney": {"type": "string", "description": "单据金额（必填）"},
                        "cVchMaker": {"type": "string", "description": "制单人（必填）"},
                        "IsCtrl": {"type": "number", "description": "是否需要预算控制 1:控制 0:不控制（必填）"},
                        "BudgetCode": {"type": "string", "description": "预算编码"},
                        "DepCode": {"type": "string", "description": "部门编码"},
                        "ItemClass": {"type": "string", "description": "项目大类"},
                        "ItemCode": {"type": "string", "description": "项目编码"},
                        "fBillFactValue": {"type": "number", "description": "实际数（可为0）（必填）"},
                        "fBillReserveValue": {"type": "number", "description": "占用数（可为0）（必填）"}
                    }
                }
            }
        },
        "required": ["SysID", "SysName", "VoucherID", "VoucherType", "VoucherCode", "VoucherDate", "VoucherUpdateDate", "BudgetFactAdd"]
    }
}


# ===================== 批量获取预算信息 Schema定义 =====================
U8_BUDGET_BATCH_GET_SCHEMA = {
    "name": "u8_budget_batch_get",
    "description": "在用友U8 OpenAPI中批量获取预算信息，支持按预算表、口径、版本号、预算指标、预算项目编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cformcode": {"type": "string", "description": "预算表"},
            "ccalibercode1": {"type": "string", "description": "口径1"},
            "cversioncode": {"type": "string", "description": "版本号"},
            "ctargetcode": {"type": "string", "description": "预算指标"},
            "ctargetcode_ctl": {"type": "string", "description": "预算控制指标"},
            "citemcode": {"type": "string", "description": "预算项目编码"},
            "citemname": {"type": "string", "description": "预算项目名称关键字"},
            "fperiod13": {"type": "number", "description": "预算数"},
            "fperiod12": {"type": "number", "description": "实际发生"},
            "freserve12": {"type": "number", "description": "实际占用"},
            "pk": {"type": "integer", "description": "唯一标识"}
        },
        "required": []
    }
}


# ===================== 获取单个预算项目 Schema定义 =====================
U8_BUDGETITEM_GET_SCHEMA = {
    "name": "u8_budgetitem_get",
    "description": "在用友U8 OpenAPI中获取预算项目信息，返回年度、预算项目编码、名称、控制系统编号、项目编制、部门编制、级次、是否末级、备注等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": []
    }
}


# ===================== 批量获取预算项目 Schema定义 =====================
U8_BUDGETITEM_BATCH_GET_SCHEMA = {
    "name": "u8_budgetitem_batch_get",
    "description": "在用友U8 OpenAPI中批量获取预算项目信息，支持分页查询和多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "iyear": {"type": "integer", "description": "年度"},
            "csystem": {"type": "string", "description": "控制系统编号"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"}
        },
        "required": []
    }
}

