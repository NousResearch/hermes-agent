import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 对私支付单 entry 子数据模型 =====================
class PrivatepaymentEntry(BaseModel):
    """对私支付单 entry 子表数据模型"""
    cAccountTo: Optional[str] = Field(None, description="收方帐号")
    cAccountToName: Optional[str] = Field(None, description="收方名称")
    mOriginMoney: Optional[float] = Field(None, description="原币金额")
    cExpCode: Optional[str] = Field(None, description="费用项目")
    cDepCode: Optional[str] = Field(None, description="部门编码")
    cPerson: Optional[str] = Field(None, description="职员编码")
    cItemClass: Optional[str] = Field(None, description="项目大类编号")
    cItemCode: Optional[str] = Field(None, description="项目编码")
    cItemCodeName: Optional[str] = Field(None, description="项目")
    cPurpose: Optional[str] = Field(None, description="用途")
    cRemark: Optional[str] = Field(None, description="备注")
    define22: Optional[str] = Field(None, description="表体自定义项22")
    define23: Optional[str] = Field(None, description="表体自定义项23")
    define24: Optional[str] = Field(None, description="表体自定义项24")
    define25: Optional[str] = Field(None, description="表体自定义项25")
    define26: Optional[float] = Field(None, description="表体自定义项26")
    define27: Optional[float] = Field(None, description="表体自定义项27")
    define28: Optional[str] = Field(None, description="表体自定义项28")
    define29: Optional[str] = Field(None, description="表体自定义项29")
    define30: Optional[str] = Field(None, description="表体自定义项30")
    define31: Optional[str] = Field(None, description="表体自定义项31")
    define32: Optional[str] = Field(None, description="表体自定义项32")
    define33: Optional[str] = Field(None, description="表体自定义项33")
    define34: Optional[int] = Field(None, description="表体自定义项34")
    define35: Optional[int] = Field(None, description="表体自定义项35")
    define36: Optional[str] = Field(None, description="表体自定义项36")
    define37: Optional[str] = Field(None, description="表体自定义项37")


# ===================== 获取单个对私支付单 数据模型 =====================
class GetPrivatepaymentInput(BaseModel):
    """获取单个对私支付单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="单据编号（必填）")


# ===================== 批量获取对私支付单 数据模型 =====================
class GetPrivatepaymentBatchInput(BaseModel):
    """批量获取对私支付单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    cBillID_begin: Optional[str] = Field(None, description="起始单据编号")
    cBillID_end: Optional[str] = Field(None, description="结束单据编号")
    dBillDate_begin: Optional[str] = Field(None, description="起始单据日期")
    dBillDate_end: Optional[str] = Field(None, description="结束单据日期")
    cAccountFrom: Optional[str] = Field(None, description="支付帐号")
    cSettle: Optional[str] = Field(None, description="结算方式")
    cOperator: Optional[str] = Field(None, description="制单人关键字")
    cExaminer: Optional[str] = Field(None, description="复核人关键字")
    cApprover: Optional[str] = Field(None, description="审批人关键字")
    cPayer: Optional[str] = Field(None, description="支付人关键字")
    iBuscod: Optional[int] = Field(None, description="业务类型")


# ===================== 新增对私支付单 数据模型 =====================
class AddPrivatepaymentInput(BaseModel):
    """新增对私支付单输入模型"""
    cBillID: Optional[str] = Field(None, description="单据编号")
    dBillDate: Optional[str] = Field(None, description="单据日期")
    cAccountFrom: Optional[str] = Field(None, description="支付帐号")
    mNativeMoney: Optional[float] = Field(None, description="本币金额")
    mOriginMoney: Optional[float] = Field(None, description="原币金额")
    cMoneyType: Optional[str] = Field(None, description="币种")
    sExchRate: Optional[float] = Field(None, description="汇率")
    cSettle: Optional[str] = Field(None, description="结算方式")
    cPaySpeed: Optional[str] = Field(None, description="汇款速度")
    cOperator: Optional[str] = Field(None, description="制单人")
    cExaminer: Optional[str] = Field(None, description="复核人")
    cApprover: Optional[str] = Field(None, description="审批人")
    cPayer: Optional[str] = Field(None, description="支付人")
    dOperateDate: Optional[str] = Field(None, description="支付日期")
    cNoteNo: Optional[str] = Field(None, description="票据号")
    cAppend: Optional[str] = Field(None, description="用途")
    iBuscod: Optional[int] = Field(None, description="业务类型")
    iCrdttyp: Optional[int] = Field(None, description="代发卡类型")
    define1: Optional[str] = Field(None, description="自定义项1")
    define2: Optional[str] = Field(None, description="自定义项2")
    define3: Optional[str] = Field(None, description="自定义项3")
    define4: Optional[str] = Field(None, description="自定义项4")
    define5: Optional[str] = Field(None, description="自定义项5")
    define6: Optional[str] = Field(None, description="自定义项6")
    define7: Optional[str] = Field(None, description="自定义项7")
    define8: Optional[str] = Field(None, description="自定义项8")
    define9: Optional[str] = Field(None, description="自定义项9")
    define10: Optional[str] = Field(None, description="自定义项10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[str] = Field(None, description="自定义项15")
    define16: Optional[str] = Field(None, description="自定义项16")
    entry: Optional[List[PrivatepaymentEntry]] = Field(None, description="对私支付单明细列表")


# ===================== 获取单个普通支付单 数据模型 =====================
class GetPaymentInput(BaseModel):
    """获取单个普通支付单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="单据编号（必填）")


# ===================== 批量获取普通支付单 数据模型 =====================
class GetPaymentBatchInput(BaseModel):
    """批量获取普通支付单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    cbillid: Optional[str] = Field(None, description="单据编号")
    dbilldate_begin: Optional[str] = Field(None, description="起始单据日期")
    dbilldate_end: Optional[str] = Field(None, description="结束单据日期")
    cdepcode: Optional[str] = Field(None, description="部门编码")
    cperson: Optional[str] = Field(None, description="职员编码")
    coperator: Optional[str] = Field(None, description="制单人关键字")
    cexaminer: Optional[str] = Field(None, description="复核人关键字")
    capprover: Optional[str] = Field(None, description="审批人关键字")
    cpayer: Optional[str] = Field(None, description="支付人关键字")
    doperatedate_begin: Optional[str] = Field(None, description="起始支付日期")
    doperatedate_end: Optional[str] = Field(None, description="结束支付日期")


# ===================== 新增普通支付单 数据模型 =====================
class AddPaymentInput(BaseModel):
    """新增普通支付单输入模型"""
    cBillID: Optional[str] = Field(None, description="单据编号")
    dBillDate: Optional[str] = Field(None, description="单据日期")
    cAccountFrom: Optional[str] = Field(None, description="支付帐号")
    cAccountTo: Optional[str] = Field(None, description="收方帐号")
    mNativeMoney: Optional[float] = Field(None, description="本币金额")
    mOriginMoney: Optional[float] = Field(None, description="原币金额")
    cMoneyType: Optional[str] = Field(None, description="币种")
    sExchRate: Optional[float] = Field(None, description="汇率")
    cSettle: Optional[str] = Field(None, description="结算方式")
    cDepCode: Optional[str] = Field(None, description="部门编码")
    cPerson: Optional[str] = Field(None, description="职员编码")
    cItemClass: Optional[str] = Field(None, description="项目大类编号")
    cItemCode: Optional[str] = Field(None, description="项目编码")
    cPaySpeed: Optional[str] = Field(None, description="汇款速度")
    cOperator: Optional[str] = Field(None, description="制单人")
    cExaminer: Optional[str] = Field(None, description="复核人")
    cApprover: Optional[str] = Field(None, description="审批人")
    cPayer: Optional[str] = Field(None, description="支付人")
    dOperateDate: Optional[str] = Field(None, description="支付日期")
    bInner: Optional[bool] = Field(None, description="是否同城")
    cNoteNo: Optional[str] = Field(None, description="票据号")
    cExpCode: Optional[str] = Field(None, description="费用项目")
    cPurpose: Optional[str] = Field(None, description="用途")
    cAppend: Optional[str] = Field(None, description="附言")
    cRemark: Optional[str] = Field(None, description="备注")
    define1: Optional[str] = Field(None, description="自定义项1")
    define2: Optional[str] = Field(None, description="自定义项2")
    define3: Optional[str] = Field(None, description="自定义项3")
    define4: Optional[str] = Field(None, description="自定义项4")
    define5: Optional[str] = Field(None, description="自定义项5")
    define6: Optional[str] = Field(None, description="自定义项6")
    define7: Optional[str] = Field(None, description="自定义项7")
    define8: Optional[str] = Field(None, description="自定义项8")
    define9: Optional[str] = Field(None, description="自定义项9")
    define10: Optional[str] = Field(None, description="自定义项10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[str] = Field(None, description="自定义项15")
    define16: Optional[str] = Field(None, description="自定义项16")


# ===================== 获取单个对私支付单 Tool函数 =====================
def u8_privatepayment_get_tool(input_data: GetPrivatepaymentInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编号获取用友U8中的对私支付单信息，包含表头及表体明细(entry)。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/privatepayment/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "对私支付单获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "对私支付单获取成功",
            "data": {
                "privatepayment": result.get("privatepayment", {}),
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


# ===================== 批量获取对私支付单 Tool函数 =====================
def u8_privatepayment_batch_get_tool(input_data: GetPrivatepaymentBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取对私支付单信息，支持按单据编号、单据日期、支付帐号、结算方式、制单人等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/privatepayment/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "对私支付单批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "对私支付单批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "privatepayment": result.get("privatepayment", [])
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


# ===================== 新增对私支付单 Tool函数 =====================
def u8_privatepayment_add_tool(input_data: AddPrivatepaymentInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增对私支付单信息。
    """
    request_body: Dict[str, Any] = {
        "privatepayment": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/privatepayment/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "对私支付单新增失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "对私支付单新增成功",
            "data": {
                "id": result.get("id"),
                "tradeid": result.get("tradeid")
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


# ===================== 获取单个普通支付单 Tool函数 =====================
def u8_payment_get_tool(input_data: GetPaymentInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编号获取用友U8中的普通支付单信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/payment/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "普通支付单获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "普通支付单获取成功",
            "data": {
                "payment": result.get("payment", {})
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


# ===================== 批量获取普通支付单 Tool函数 =====================
def u8_payment_batch_get_tool(input_data: GetPaymentBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取普通支付单信息，支持按单据编号、单据日期、部门编码、职员编码、制单人、支付日期等多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/payment/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "普通支付单批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "普通支付单批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "payment": result.get("payment", [])
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


# ===================== 新增普通支付单 Tool函数 =====================
def u8_payment_add_tool(input_data: AddPaymentInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增普通支付单信息。
    """
    request_body: Dict[str, Any] = {
        "payment": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/payment/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "普通支付单新增失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "普通支付单新增成功",
            "data": {
                "id": result.get("id"),
                "tradeid": result.get("tradeid")
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


# ===================== 获取单个对私支付单 Schema定义 =====================
U8_PRIVATEPAYMENT_GET_SCHEMA = {
    "name": "u8_privatepayment_get",
    "description": "在用友U8网上银行系统中通过单据编号获取单个对私支付单信息，包含单据日期、支付帐号、本币金额、原币金额、币种、汇率、结算方式、汇款速度、制单人、复核人、审批人、支付人、支付日期、票据号、用途及表体明细(entry)",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取对私支付单 Schema定义 =====================
U8_PRIVATEPAYMENT_BATCH_GET_SCHEMA = {
    "name": "u8_privatepayment_batch_get",
    "description": "在用友U8网上银行系统中批量获取对私支付单信息，支持按单据编号、单据日期、支付帐号、结算方式、制单人、复核人、审批人、支付人、业务类型等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "cBillID_begin": {"type": "string", "description": "起始单据编号"},
            "cBillID_end": {"type": "string", "description": "结束单据编号"},
            "dBillDate_begin": {"type": "string", "description": "起始单据日期"},
            "dBillDate_end": {"type": "string", "description": "结束单据日期"},
            "cAccountFrom": {"type": "string", "description": "支付帐号"},
            "cSettle": {"type": "string", "description": "结算方式"},
            "cOperator": {"type": "string", "description": "制单人关键字"},
            "cExaminer": {"type": "string", "description": "复核人关键字"},
            "cApprover": {"type": "string", "description": "审批人关键字"},
            "cPayer": {"type": "string", "description": "支付人关键字"},
            "iBuscod": {"type": "integer", "description": "业务类型"}
        },
        "required": []
    }
}


# ===================== 新增对私支付单 Schema定义 =====================
U8_PRIVATEPAYMENT_ADD_SCHEMA = {
    "name": "u8_privatepayment_add",
    "description": "在用友U8网上银行系统中新增对私支付单，支持录入单据表头信息（单据编号、单据日期、支付帐号、金额、币种、结算方式、汇款速度等）及表体明细（收方帐号、收方名称、原币金额、费用项目、部门、职员等）",
    "parameters": {
        "type": "object",
        "properties": {
            "cBillID": {"type": "string", "description": "单据编号"},
            "dBillDate": {"type": "string", "description": "单据日期"},
            "cAccountFrom": {"type": "string", "description": "支付帐号"},
            "mNativeMoney": {"type": "number", "description": "本币金额"},
            "mOriginMoney": {"type": "number", "description": "原币金额"},
            "cMoneyType": {"type": "string", "description": "币种"},
            "sExchRate": {"type": "number", "description": "汇率"},
            "cSettle": {"type": "string", "description": "结算方式"},
            "cPaySpeed": {"type": "string", "description": "汇款速度"},
            "cOperator": {"type": "string", "description": "制单人"},
            "cExaminer": {"type": "string", "description": "复核人"},
            "cApprover": {"type": "string", "description": "审批人"},
            "cPayer": {"type": "string", "description": "支付人"},
            "dOperateDate": {"type": "string", "description": "支付日期"},
            "cNoteNo": {"type": "string", "description": "票据号"},
            "cAppend": {"type": "string", "description": "用途"},
            "iBuscod": {"type": "integer", "description": "业务类型"},
            "iCrdttyp": {"type": "integer", "description": "代发卡类型"},
            "define1": {"type": "string", "description": "自定义项1"},
            "define2": {"type": "string", "description": "自定义项2"},
            "define3": {"type": "string", "description": "自定义项3"},
            "define4": {"type": "string", "description": "自定义项4"},
            "define5": {"type": "string", "description": "自定义项5"},
            "define6": {"type": "string", "description": "自定义项6"},
            "define7": {"type": "string", "description": "自定义项7"},
            "define8": {"type": "string", "description": "自定义项8"},
            "define9": {"type": "string", "description": "自定义项9"},
            "define10": {"type": "string", "description": "自定义项10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "string", "description": "自定义项15"},
            "define16": {"type": "string", "description": "自定义项16"},
            "entry": {
                "type": "array",
                "description": "对私支付单明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "cAccountTo": {"type": "string", "description": "收方帐号"},
                        "cAccountToName": {"type": "string", "description": "收方名称"},
                        "mOriginMoney": {"type": "number", "description": "原币金额"},
                        "cExpCode": {"type": "string", "description": "费用项目"},
                        "cDepCode": {"type": "string", "description": "部门编码"},
                        "cPerson": {"type": "string", "description": "职员编码"},
                        "cItemClass": {"type": "string", "description": "项目大类编号"},
                        "cItemCode": {"type": "string", "description": "项目编码"},
                        "cItemCodeName": {"type": "string", "description": "项目"},
                        "cPurpose": {"type": "string", "description": "用途"},
                        "cRemark": {"type": "string", "description": "备注"},
                        "define22": {"type": "string", "description": "表体自定义项22"},
                        "define23": {"type": "string", "description": "表体自定义项23"},
                        "define24": {"type": "string", "description": "表体自定义项24"},
                        "define25": {"type": "string", "description": "表体自定义项25"},
                        "define26": {"type": "number", "description": "表体自定义项26"},
                        "define27": {"type": "number", "description": "表体自定义项27"},
                        "define28": {"type": "string", "description": "表体自定义项28"},
                        "define29": {"type": "string", "description": "表体自定义项29"},
                        "define30": {"type": "string", "description": "表体自定义项30"},
                        "define31": {"type": "string", "description": "表体自定义项31"},
                        "define32": {"type": "string", "description": "表体自定义项32"},
                        "define33": {"type": "string", "description": "表体自定义项33"},
                        "define34": {"type": "integer", "description": "表体自定义项34"},
                        "define35": {"type": "integer", "description": "表体自定义项35"},
                        "define36": {"type": "string", "description": "表体自定义项36"},
                        "define37": {"type": "string", "description": "表体自定义项37"}
                    }
                }
            }
        },
        "required": []
    }
}


# ===================== 获取单个普通支付单 Schema定义 =====================
U8_PAYMENT_GET_SCHEMA = {
    "name": "u8_payment_get",
    "description": "在用友U8网上银行系统中通过单据编号获取单个普通支付单信息，包含单据日期、支付帐号、收方帐号、本币金额、原币金额、币种、汇率、结算方式、部门编码、职员编码、项目编码、汇款速度、制单人、复核人、审批人、支付人、支付日期、是否同城、票据号、费用项目、用途、附言、备注等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取普通支付单 Schema定义 =====================
U8_PAYMENT_BATCH_GET_SCHEMA = {
    "name": "u8_payment_batch_get",
    "description": "在用友U8网上银行系统中批量获取普通支付单信息，支持按单据编号、单据日期、部门编码、职员编码、制单人、复核人、审批人、支付人、支付日期等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "cbillid": {"type": "string", "description": "单据编号"},
            "dbilldate_begin": {"type": "string", "description": "起始单据日期"},
            "dbilldate_end": {"type": "string", "description": "结束单据日期"},
            "cdepcode": {"type": "string", "description": "部门编码"},
            "cperson": {"type": "string", "description": "职员编码"},
            "coperator": {"type": "string", "description": "制单人关键字"},
            "cexaminer": {"type": "string", "description": "复核人关键字"},
            "capprover": {"type": "string", "description": "审批人关键字"},
            "cpayer": {"type": "string", "description": "支付人关键字"},
            "doperatedate_begin": {"type": "string", "description": "起始支付日期"},
            "doperatedate_end": {"type": "string", "description": "结束支付日期"}
        },
        "required": []
    }
}


# ===================== 新增普通支付单 Schema定义 =====================
U8_PAYMENT_ADD_SCHEMA = {
    "name": "u8_payment_add",
    "description": "在用友U8网上银行系统中新增普通支付单，支持录入单据信息（单据编号、单据日期、支付帐号、收方帐号、金额、币种、汇率、结算方式、部门、职员、项目、汇款速度、是否同城、票据号、费用项目、用途、附言、备注等）",
    "parameters": {
        "type": "object",
        "properties": {
            "cBillID": {"type": "string", "description": "单据编号"},
            "dBillDate": {"type": "string", "description": "单据日期"},
            "cAccountFrom": {"type": "string", "description": "支付帐号"},
            "cAccountTo": {"type": "string", "description": "收方帐号"},
            "mNativeMoney": {"type": "number", "description": "本币金额"},
            "mOriginMoney": {"type": "number", "description": "原币金额"},
            "cMoneyType": {"type": "string", "description": "币种"},
            "sExchRate": {"type": "number", "description": "汇率"},
            "cSettle": {"type": "string", "description": "结算方式"},
            "cDepCode": {"type": "string", "description": "部门编码"},
            "cPerson": {"type": "string", "description": "职员编码"},
            "cItemClass": {"type": "string", "description": "项目大类编号"},
            "cItemCode": {"type": "string", "description": "项目编码"},
            "cPaySpeed": {"type": "string", "description": "汇款速度"},
            "cOperator": {"type": "string", "description": "制单人"},
            "cExaminer": {"type": "string", "description": "复核人"},
            "cApprover": {"type": "string", "description": "审批人"},
            "cPayer": {"type": "string", "description": "支付人"},
            "dOperateDate": {"type": "string", "description": "支付日期"},
            "bInner": {"type": "boolean", "description": "是否同城"},
            "cNoteNo": {"type": "string", "description": "票据号"},
            "cExpCode": {"type": "string", "description": "费用项目"},
            "cPurpose": {"type": "string", "description": "用途"},
            "cAppend": {"type": "string", "description": "附言"},
            "cRemark": {"type": "string", "description": "备注"},
            "define1": {"type": "string", "description": "自定义项1"},
            "define2": {"type": "string", "description": "自定义项2"},
            "define3": {"type": "string", "description": "自定义项3"},
            "define4": {"type": "string", "description": "自定义项4"},
            "define5": {"type": "string", "description": "自定义项5"},
            "define6": {"type": "string", "description": "自定义项6"},
            "define7": {"type": "string", "description": "自定义项7"},
            "define8": {"type": "string", "description": "自定义项8"},
            "define9": {"type": "string", "description": "自定义项9"},
            "define10": {"type": "string", "description": "自定义项10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "string", "description": "自定义项15"},
            "define16": {"type": "string", "description": "自定义项16"}
        },
        "required": []
    }
}
