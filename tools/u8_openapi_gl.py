import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)

# ==============================================================================
#                                    数据模型
# ==============================================================================

# ===================== 凭证列表批量查询 数据模型 =====================
class GetVoucherlistBatchInput(BaseModel):
    """获取凭证列表信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    bill_date_from: Optional[str] = Field(None, description="凭证日期(from)")
    bill_date_to: Optional[str] = Field(None, description="凭证日期(to)")
    bill_code_from: Optional[str] = Field(None, description="科目编码(from)")
    bill_code_to: Optional[str] = Field(None, description="科目编码(to)")
    coutno_id: Optional[str] = Field(None, description="外部系统编码")
    cno_id: Optional[str] = Field(None, description="凭证编号")
    csign: Optional[str] = Field(None, description="凭证类别字")
    cbill: Optional[str] = Field(None, description="制单人")


# ===================== 凭证详情列表批量查询 数据模型 =====================
class GetVoucherdetaillistBatchInput(BaseModel):
    """获取凭证详情列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    bill_date_from: Optional[str] = Field(None, description="凭证日期(from)")
    bill_date_to: Optional[str] = Field(None, description="凭证日期(to)")
    bill_code_from: Optional[str] = Field(None, description="科目编码(from)")
    bill_code_to: Optional[str] = Field(None, description="科目编码(to)")
    cno_id: Optional[str] = Field(None, description="凭证编号")
    csign: Optional[str] = Field(None, description="凭证类别")
    cbill: Optional[str] = Field(None, description="制单人")
    coutno_id: Optional[str] = Field(None, description="外部系统编号")


# ===================== 凭证作废 数据模型 =====================
class VoucherCancelInput(BaseModel):
    """凭证作废处理输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="外部系统号（传入时请保证外部系统唯一性并加上外部系统标识前缀）")


# ===================== 凭证新增(扩展) 数据模型 =====================
class VoucherExAuxiliaryInput(BaseModel):
    """凭证分录辅助核算输入模型"""
    dept_id: Optional[str] = Field(None, description="部门")
    personnel_id: Optional[str] = Field(None, description="人员")
    cust_id: Optional[str] = Field(None, description="客户")
    supplier_id: Optional[str] = Field(None, description="供应商")
    item_class: Optional[str] = Field(None, description="项目大类")
    item_id: Optional[str] = Field(None, description="项目档案")
    operator: Optional[str] = Field(None, description="业务员")
    self_define1: Optional[str] = Field(None, description="自定义字段1")
    self_define2: Optional[str] = Field(None, description="自定义字段2")
    self_define3: Optional[str] = Field(None, description="自定义字段3")
    self_define4: Optional[str] = Field(None, description="自定义字段4")
    self_define5: Optional[str] = Field(None, description="自定义字段5")
    self_define6: Optional[str] = Field(None, description="自定义字段6")
    self_define7: Optional[str] = Field(None, description="自定义字段7")
    self_define8: Optional[str] = Field(None, description="自定义字段8")
    self_define9: Optional[str] = Field(None, description="自定义字段9")
    self_define10: Optional[str] = Field(None, description="自定义字段10")
    self_define11: Optional[str] = Field(None, description="自定义字段11")
    self_define12: Optional[str] = Field(None, description="自定义字段12")
    self_define13: Optional[str] = Field(None, description="自定义字段13")
    self_define14: Optional[str] = Field(None, description="自定义字段14")
    self_define15: Optional[str] = Field(None, description="自定义字段15")
    self_define16: Optional[str] = Field(None, description="自定义字段16")


class VoucherExCashFlowInput(BaseModel):
    """凭证分录现金流量输入模型"""
    cexch_name: Optional[str] = Field(None, description="币种")
    RowGuid: Optional[str] = Field(None, description="行标识")
    iYPeriod: Optional[str] = Field(None, description="年期间")
    iyear: Optional[str] = Field(None, description="年")
    csign: Optional[str] = Field(None, description="凭证类别字")
    nd_s: Optional[float] = Field(None, description="数量借方")
    md_f: Optional[float] = Field(None, description="外币借方")
    ccode: Optional[str] = Field(None, description="科目编码")
    md: Optional[float] = Field(None, description="借方金额")
    cCashItem: Optional[str] = Field(None, description="现金项目")
    cash_item: Optional[str] = Field(None, description="现金项目")
    natural_currency: Optional[float] = Field(None, description="本币借方发生额")
    cdept_id: Optional[str] = Field(None, description="部门")
    cperson_id: Optional[str] = Field(None, description="人员")
    ccus_id: Optional[str] = Field(None, description="客户")
    csup_id: Optional[str] = Field(None, description="供应商")
    citem_class: Optional[str] = Field(None, description="项目大类")
    citem_id: Optional[str] = Field(None, description="项目档案")
    cDefine1: Optional[str] = Field(None, description="自定义字段1")
    cDefine2: Optional[str] = Field(None, description="自定义字段2")
    cDefine3: Optional[str] = Field(None, description="自定义字段3")
    cDefine4: Optional[str] = Field(None, description="自定义字段4")
    cDefine5: Optional[str] = Field(None, description="自定义字段5")
    cDefine6: Optional[str] = Field(None, description="自定义字段6")
    cDefine7: Optional[str] = Field(None, description="自定义字段7")
    cDefine8: Optional[str] = Field(None, description="自定义字段8")
    cDefine9: Optional[str] = Field(None, description="自定义字段9")
    cDefine10: Optional[str] = Field(None, description="自定义字段10")
    cDefine11: Optional[str] = Field(None, description="自定义字段11")
    cDefine12: Optional[str] = Field(None, description="自定义字段12")
    cDefine13: Optional[str] = Field(None, description="自定义字段13")
    cDefine14: Optional[str] = Field(None, description="自定义字段14")
    cDefine15: Optional[str] = Field(None, description="自定义字段15")
    cDefine16: Optional[str] = Field(None, description="自定义字段16")


class VoucherExEntryInput(BaseModel):
    """凭证分录输入模型(扩展版)"""
    account_code: Optional[str] = Field(None, description="科目编码")
    abstract: Optional[str] = Field(None, description="摘要")
    currency: Optional[str] = Field(None, description="币种，默认人民币")
    unit_price: Optional[float] = Field(None, description="单价,在科目有数量核算时，填写此项")
    exchange_rate1: Optional[float] = Field(None, description="汇率1，主辅币核算时使用，原币*汇率1=辅币，NC用户使用")
    exchange_rate2: Optional[float] = Field(None, description="汇率2，折本汇率，本币*汇率2=主币，U8用户使用")
    quantity: Optional[float] = Field(None, description="借方数量,在科目有数量核算时，填写此项")
    primary_amount: Optional[float] = Field(None, description="原币借方发生额")
    secondary_amount: Optional[float] = Field(None, description="辅币借方发生额")
    natural_currency: float = Field(..., description="本币借方发生额")
    auxiliary: Optional[VoucherExAuxiliaryInput] = Field(None, description="辅助核算")
    cash_flow: Optional[List[VoucherExCashFlowInput]] = Field(None, description="现金流量")


class AddVoucherExInput(BaseModel):
    """新增凭证(扩展版)输入模型"""
    voucher_type: str = Field(..., description="凭证类别")
    fiscal_year: Optional[int] = Field(None, description="凭证所属的会计年度，不填写取当前年")
    accounting_period: Optional[int] = Field(None, description="所属的会计期间，不填写取当前月份")
    voucher_id: Optional[str] = Field(None, description="凭证号")
    date: Optional[str] = Field(None, description="制单日期")
    enter: Optional[str] = Field(None, description="制单人名称")
    cashier: Optional[str] = Field(None, description="出纳名称")
    attachment_number: Optional[int] = Field(None, description="附单据数")
    voucher_making_system: Optional[str] = Field(None, description="外部系统类型，如果需要传入外部凭证业务号，此属性必须传入固定值 CDM")
    reserve2: Optional[str] = Field(None, description="外部凭证业务号")
    entry: List[VoucherExEntryInput] = Field(..., description="凭证分录列表")
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    biz_id: Optional[str] = Field(None, description="上游id，需要保证biz_id与ERP主键唯一对应关系")
    sync: Optional[int] = Field(None, description="0=异步新增（默认）;1=同步新增")


# ===================== 凭证新增 数据模型 =====================
class VoucherAuxiliaryInput(BaseModel):
    """凭证分录辅助核算输入模型"""
    dept_id: Optional[str] = Field(None, description="部门")
    personnel_id: Optional[str] = Field(None, description="人员")
    cust_id: Optional[str] = Field(None, description="客户")
    supplier_id: Optional[str] = Field(None, description="供应商")
    item_class: Optional[str] = Field(None, description="项目大类")
    item_id: Optional[str] = Field(None, description="项目档案")
    operator: Optional[str] = Field(None, description="业务员")
    self_define1: Optional[str] = Field(None, description="自定义字段1")
    self_define2: Optional[str] = Field(None, description="自定义字段2")
    self_define3: Optional[str] = Field(None, description="自定义字段3")
    self_define4: Optional[str] = Field(None, description="自定义字段4")
    self_define5: Optional[str] = Field(None, description="自定义字段5")
    self_define6: Optional[str] = Field(None, description="自定义字段6")
    self_define7: Optional[str] = Field(None, description="自定义字段7")
    self_define8: Optional[str] = Field(None, description="自定义字段8")
    self_define9: Optional[str] = Field(None, description="自定义字段9")
    self_define10: Optional[str] = Field(None, description="自定义字段10")
    self_define11: Optional[str] = Field(None, description="自定义字段11")
    self_define12: Optional[str] = Field(None, description="自定义字段12")
    self_define13: Optional[str] = Field(None, description="自定义字段13")
    self_define14: Optional[str] = Field(None, description="自定义字段14")
    self_define15: Optional[str] = Field(None, description="自定义字段15")
    self_define16: Optional[str] = Field(None, description="自定义字段16")


class VoucherCashFlowInput(BaseModel):
    """凭证分录现金流量输入模型"""
    cexch_name: Optional[str] = Field(None, description="币种")
    RowGuid: Optional[str] = Field(None, description="行标识")
    iYPeriod: Optional[str] = Field(None, description="年期间")
    iyear: Optional[str] = Field(None, description="年")
    csign: Optional[str] = Field(None, description="凭证类别字")
    nd_s: Optional[float] = Field(None, description="数量借方")
    nc_s: Optional[float] = Field(None, description="数量贷方")
    md_f: Optional[float] = Field(None, description="外币借方")
    mc_f: Optional[float] = Field(None, description="外币贷方")
    ccode: Optional[str] = Field(None, description="科目编码")
    md: Optional[float] = Field(None, description="借方金额")
    mc: Optional[float] = Field(None, description="贷方金额")
    cCashItem: Optional[str] = Field(None, description="现金项目")
    cash_item: Optional[str] = Field(None, description="现金项目")
    natural_debit_currency: Optional[float] = Field(None, description="本币借方发生额")
    natural_credit_currency: Optional[float] = Field(None, description="本币贷方发生额")
    cdept_id: Optional[str] = Field(None, description="部门")
    cperson_id: Optional[str] = Field(None, description="人员")
    ccus_id: Optional[str] = Field(None, description="客户")
    csup_id: Optional[str] = Field(None, description="供应商")
    citem_class: Optional[str] = Field(None, description="项目大类")
    citem_id: Optional[str] = Field(None, description="项目档案")
    cDefine1: Optional[str] = Field(None, description="自定义字段1")
    cDefine2: Optional[str] = Field(None, description="自定义字段2")
    cDefine3: Optional[str] = Field(None, description="自定义字段3")
    cDefine4: Optional[str] = Field(None, description="自定义字段4")
    cDefine5: Optional[str] = Field(None, description="自定义字段5")
    cDefine6: Optional[str] = Field(None, description="自定义字段6")
    cDefine7: Optional[str] = Field(None, description="自定义字段7")
    cDefine8: Optional[str] = Field(None, description="自定义字段8")
    cDefine9: Optional[str] = Field(None, description="自定义字段9")
    cDefine10: Optional[str] = Field(None, description="自定义字段10")
    cDefine11: Optional[str] = Field(None, description="自定义字段11")
    cDefine12: Optional[str] = Field(None, description="自定义字段12")
    cDefine13: Optional[str] = Field(None, description="自定义字段13")
    cDefine14: Optional[str] = Field(None, description="自定义字段14")
    cDefine15: Optional[str] = Field(None, description="自定义字段15")
    cDefine16: Optional[str] = Field(None, description="自定义字段16")


class VoucherDebitEntryInput(BaseModel):
    """凭证借方分录输入模型"""
    entry_id: Optional[int] = Field(None, description="分录号")
    account_code: str = Field(..., description="科目编码")
    abstract: str = Field(..., description="摘要")
    currency: Optional[str] = Field(None, description="币种，默认人民币")
    unit_price: Optional[float] = Field(None, description="单价,在科目有数量核算时，填写此项")
    exchange_rate1: Optional[float] = Field(None, description="汇率1，主辅币核算时使用")
    exchange_rate2: Optional[float] = Field(None, description="汇率2，折本汇率")
    debit_quantity: Optional[float] = Field(None, description="借方数量")
    primary_debit_amount: Optional[float] = Field(None, description="原币借方发生额")
    secondary_debit_amount: Optional[float] = Field(None, description="辅币借方发生额")
    natural_debit_currency: float = Field(..., description="本币借方发生额")
    auxiliary: Optional[VoucherAuxiliaryInput] = Field(None, description="辅助核算")
    cash_flow: Optional[List[VoucherCashFlowInput]] = Field(None, description="现金流量")


class VoucherCreditEntryInput(BaseModel):
    """凭证贷方分录输入模型"""
    entry_id: Optional[int] = Field(None, description="分录号")
    account_code: str = Field(..., description="科目编码")
    abstract: str = Field(..., description="摘要")
    currency: Optional[str] = Field(None, description="币种，默认人民币")
    unit_price: Optional[float] = Field(None, description="单价")
    exchange_rate1: Optional[float] = Field(None, description="汇率1")
    exchange_rate2: Optional[float] = Field(None, description="汇率2")
    credit_quantity: Optional[float] = Field(None, description="贷方数量")
    primary_credit_amount: Optional[float] = Field(None, description="原币贷方发生额")
    secondary_credit_amount: Optional[float] = Field(None, description="辅币贷方发生额")
    natural_credit_currency: float = Field(..., description="本币贷方发生额")
    auxiliary: Optional[VoucherAuxiliaryInput] = Field(None, description="辅助核算")
    cash_flow: Optional[List[VoucherCashFlowInput]] = Field(None, description="现金流量")


class AddVoucherInput(BaseModel):
    """新增凭证输入模型"""
    voucher_type: str = Field(..., description="凭证类别")
    fiscal_year: Optional[int] = Field(None, description="凭证所属的会计年度")
    accounting_period: Optional[int] = Field(None, description="所属的会计期间")
    voucher_id: Optional[str] = Field(None, description="凭证号")
    date: Optional[str] = Field(None, description="制单日期")
    enter: Optional[str] = Field(None, description="制单人名称")
    cashier: Optional[str] = Field(None, description="出纳名称")
    attachment_number: Optional[int] = Field(None, description="附单据数")
    voucher_making_system: Optional[str] = Field(None, description="外部系统类型")
    reserve2: Optional[str] = Field(None, description="外部凭证业务号")
    debit: Optional[dict] = Field(None, description="借方分录")
    credit: Optional[dict] = Field(None, description="贷方分录")
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    biz_id: Optional[str] = Field(None, description="上游id")
    sync: Optional[int] = Field(None, description="0=异步新增;1=同步新增")


# ===================== 启用期间批量查询 数据模型 =====================
class GetStartperiodBatchInput(BaseModel):
    """获取启用期间输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cAcc_Id: Optional[str] = Field(None, description="帐套名称")


# ===================== 总账结账状态批量查询 数据模型 =====================
class GetMendglgzBatchInput(BaseModel):
    """批量获取总账结账状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    iyear: Optional[int] = Field(None, description="会计年度")
    iperiod_begin: Optional[int] = Field(None, description="起始会计期间")
    iperiod_end: Optional[int] = Field(None, description="结束会计期间")


# ===================== 科目总账批量查询 数据模型 =====================
class GetAccountsumBatchInput(BaseModel):
    """批量获取科目总账信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="科目编码")
    iperiod: Optional[int] = Field(None, description="会计期间")
    cbegind_c: Optional[str] = Field(None, description="金额期初方向")
    cendd_c: Optional[str] = Field(None, description="金额期末方向")
    iyear: Optional[int] = Field(None, description="会计年度")


# ===================== 辅助总账批量查询 数据模型 =====================
class GetAccountassBatchInput(BaseModel):
    """批量获取辅助总账信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode_begin: Optional[str] = Field(None, description="起始科目编码")
    ccode_end: Optional[str] = Field(None, description="结束科目编码")
    iperiod: Optional[int] = Field(None, description="会计期间")
    cbegind_c: Optional[str] = Field(None, description="金额期初方向")
    cendd_c: Optional[str] = Field(None, description="金额期末方向")
    iyear: Optional[int] = Field(None, description="会计年度")
    cdept_id: Optional[str] = Field(None, description="部门编码")
    cperson_id: Optional[str] = Field(None, description="职员编码")
    ccus_id: Optional[str] = Field(None, description="客户编码")
    csup_id: Optional[str] = Field(None, description="供应商编码")
    citem_class: Optional[str] = Field(None, description="项目大类编码")
    citem_id: Optional[str] = Field(None, description="项目编码")


# ==============================================================================
#                                    Tool函数
# ==============================================================================

# ===================== 凭证列表批量查询 Tool函数 =====================
def u8_voucherlist_batch_get_tool(input_data: GetVoucherlistBatchInput, client: U8OpenAPIClient) -> str:
    """
    获取凭证列表信息，支持分页和多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/voucherlist/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取凭证列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取凭证列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "voucherlist": result.get("voucherlist", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 凭证详情列表批量查询 Tool函数 =====================
def u8_voucherdetaillist_batch_get_tool(input_data: GetVoucherdetaillistBatchInput, client: U8OpenAPIClient) -> str:
    """
    获取凭证详情列表，用于凭证及关联辅助核算项相关信息查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/voucherdetaillist/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取凭证详情列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取凭证详情列表成功",
            "data": {
                "voucherdetaillist": result.get("voucherdetaillist", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 凭证作废 Tool函数 =====================
def u8_voucher_cancel_tool(input_data: VoucherCancelInput, client: U8OpenAPIClient) -> str:
    """
    凭证作废处理，通过外部系统号作废凭证。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/voucher/cancel"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "凭证作废失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "凭证作废成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 凭证新增(扩展版) Tool函数 =====================
def u8_voucher_ex_add_tool(input_data: AddVoucherExInput, client: U8OpenAPIClient) -> str:
    """
    新增一张凭证，凭证分录数据按照自然顺序保存。
    """
    # 构造请求体
    voucher_data = {
        "voucher_type": input_data.voucher_type,
        "fiscal_year": input_data.fiscal_year,
        "accounting_period": input_data.accounting_period,
        "voucher_id": input_data.voucher_id,
        "date": input_data.date,
        "enter": input_data.enter,
        "cashier": input_data.cashier,
        "attachment_number": input_data.attachment_number,
        "voucher_making_system": input_data.voucher_making_system,
        "reserve2": input_data.reserve2,
        "entry": [e.model_dump(exclude_none=True) for e in input_data.entry]
    }
    
    request_body = {"voucher_ex": voucher_data}
    
    # URL参数
    url_params = {}
    if input_data.ds_sequence is not None:
        url_params["ds_sequence"] = input_data.ds_sequence
    if input_data.biz_id is not None:
        url_params["biz_id"] = input_data.biz_id
    if input_data.sync is not None:
        url_params["sync"] = input_data.sync
    
    api_path = "/api/voucher_ex/add"
    
    try:
        result = client.request_api("POST", api_path, inparams=url_params, json_body=request_body, is_tradeid=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增凭证失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "新增凭证成功",
            "data": {
                "id": result.get("id"),
                "tradeid": result.get("tradeid")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 凭证新增 Tool函数 =====================
def u8_voucher_add_tool(input_data: AddVoucherInput, client: U8OpenAPIClient) -> str:
    """
    新增一张凭证，借方和贷方分录分开录入。
    """
    # 构造请求体
    voucher_data = {
        "voucher_type": input_data.voucher_type,
        "fiscal_year": input_data.fiscal_year,
        "accounting_period": input_data.accounting_period,
        "voucher_id": input_data.voucher_id,
        "date": input_data.date,
        "enter": input_data.enter,
        "cashier": input_data.cashier,
        "attachment_number": input_data.attachment_number,
        "voucher_making_system": input_data.voucher_making_system,
        "reserve2": input_data.reserve2,
    }
    
    if input_data.debit:
        voucher_data["debit"] = input_data.debit
    if input_data.credit:
        voucher_data["credit"] = input_data.credit
    
    request_body = {"voucher": voucher_data}
    
    # URL参数
    url_params = {}
    if input_data.ds_sequence is not None:
        url_params["ds_sequence"] = input_data.ds_sequence
    if input_data.biz_id is not None:
        url_params["biz_id"] = input_data.biz_id
    if input_data.sync is not None:
        url_params["sync"] = input_data.sync
    
    api_path = "/api/voucher/add"
    
    try:
        result = client.request_api("POST", api_path, inparams=url_params, json_body=request_body, is_tradeid=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增凭证失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "新增凭证成功",
            "data": {
                "id": result.get("id"),
                "tradeid": result.get("tradeid")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 启用期间批量查询 Tool函数 =====================
def u8_startperiod_batch_get_tool(input_data: GetStartperiodBatchInput, client: U8OpenAPIClient) -> str:
    """
    获取启用期间信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/startperiod/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取启用期间失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取启用期间成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "startperiod": result.get("startperiod", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 总账结账状态批量查询 Tool函数 =====================
def u8_mendglgz_batch_get_tool(input_data: GetMendglgzBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取总账结账状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/mendglgz/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取总账结账状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取总账结账状态成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "mendglgz": result.get("mendglgz", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 科目总账批量查询 Tool函数 =====================
def u8_accountsum_batch_get_tool(input_data: GetAccountsumBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取科目总账信息，用于基础数据同步或前端展示。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/accountsum/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取科目总账信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取科目总账信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "accountsum": result.get("accoutsum", [])  # 注意API返回字段名是accoutsum
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 辅助总账批量查询 Tool函数 =====================
def u8_accountass_batch_get_tool(input_data: GetAccountassBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取辅助总账信息，用于基础数据同步或前端展示。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/accountass/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取辅助总账信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取辅助总账信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "accountass": result.get("accountass", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ==============================================================================
#                                   Schema定义
# ==============================================================================

# ===================== 凭证列表批量查询 Schema定义 =====================
U8_VOUCHERLIST_BATCH_GET_SCHEMA = {
    "name": "u8_voucherlist_batch_get",
    "description": "获取凭证列表信息，支持分页、日期、科目编码、凭证编号等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "bill_date_from": {"type": "string", "description": "凭证日期(from)"},
            "bill_date_to": {"type": "string", "description": "凭证日期(to)"},
            "bill_code_from": {"type": "string", "description": "科目编码(from)"},
            "bill_code_to": {"type": "string", "description": "科目编码(to)"},
            "coutno_id": {"type": "string", "description": "外部系统编码"},
            "cno_id": {"type": "string", "description": "凭证编号"},
            "csign": {"type": "string", "description": "凭证类别字"},
            "cbill": {"type": "string", "description": "制单人"}
        },
        "required": []
    }
}

# ===================== 凭证详情列表批量查询 Schema定义 =====================
U8_VOUCHERDETAILLIST_BATCH_GET_SCHEMA = {
    "name": "u8_voucherdetaillist_batch_get",
    "description": "获取凭证详情列表，用于凭证及关联辅助核算项相关信息查询",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "bill_date_from": {"type": "string", "description": "凭证日期(from)"},
            "bill_date_to": {"type": "string", "description": "凭证日期(to)"},
            "bill_code_from": {"type": "string", "description": "科目编码(from)"},
            "bill_code_to": {"type": "string", "description": "科目编码(to)"},
            "cno_id": {"type": "string", "description": "凭证编号"},
            "csign": {"type": "string", "description": "凭证类别"},
            "cbill": {"type": "string", "description": "制单人"},
            "coutno_id": {"type": "string", "description": "外部系统编号"}
        },
        "required": []
    }
}

# ===================== 凭证作废 Schema定义 =====================
U8_VOUCHER_CANCEL_SCHEMA = {
    "name": "u8_voucher_cancel",
    "description": "凭证作废处理，通过外部系统号作废凭证",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "外部系统号（传入时请保证外部系统唯一性并加上外部系统标识前缀）"}
        },
        "required": ["id"]
    }
}

# ===================== 凭证新增(扩展版) Schema定义 =====================
U8_VOUCHER_EX_ADD_SCHEMA = {
    "name": "u8_voucher_ex_add",
    "description": "新增一张凭证，凭证分录数据按照自然顺序保存。适用于报销应用、财务管控应用。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_type": {"type": "string", "description": "凭证类别（必填）"},
            "fiscal_year": {"type": "integer", "description": "凭证所属的会计年度，不填写取当前年"},
            "accounting_period": {"type": "integer", "description": "所属的会计期间，不填写取当前月份"},
            "voucher_id": {"type": "string", "description": "凭证号"},
            "date": {"type": "string", "description": "制单日期"},
            "enter": {"type": "string", "description": "制单人名称"},
            "cashier": {"type": "string", "description": "出纳名称"},
            "attachment_number": {"type": "integer", "description": "附单据数"},
            "voucher_making_system": {"type": "string", "description": "外部系统类型，如果需要传入外部凭证业务号，此属性必须传入固定值 CDM"},
            "reserve2": {"type": "string", "description": "外部凭证业务号"},
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "biz_id": {"type": "string", "description": "上游id，需要保证biz_id与ERP主键唯一对应关系"},
            "sync": {"type": "integer", "description": "0=异步新增（默认）;1=同步新增"},
            "entry": {
                "type": "array",
                "description": "凭证分录列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "account_code": {"type": "string", "description": "科目编码"},
                        "abstract": {"type": "string", "description": "摘要"},
                        "currency": {"type": "string", "description": "币种，默认人民币"},
                        "unit_price": {"type": "number", "description": "单价"},
                        "exchange_rate1": {"type": "number", "description": "汇率1"},
                        "exchange_rate2": {"type": "number", "description": "汇率2"},
                        "quantity": {"type": "number", "description": "借方数量"},
                        "primary_amount": {"type": "number", "description": "原币借方发生额"},
                        "secondary_amount": {"type": "number", "description": "辅币借方发生额"},
                        "natural_currency": {"type": "number", "description": "本币借方发生额（必填）"}
                    }
                }
            }
        },
        "required": ["voucher_type", "entry"]
    }
}

# ===================== 凭证新增 Schema定义 =====================
U8_VOUCHER_ADD_SCHEMA = {
    "name": "u8_voucher_add",
    "description": "新增一张凭证，借方和贷方分录分开录入。适用于报销应用、财务管控应用。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_type": {"type": "string", "description": "凭证类别（必填）"},
            "fiscal_year": {"type": "integer", "description": "凭证所属的会计年度"},
            "accounting_period": {"type": "integer", "description": "所属的会计期间"},
            "voucher_id": {"type": "string", "description": "凭证号"},
            "date": {"type": "string", "description": "制单日期"},
            "enter": {"type": "string", "description": "制单人名称"},
            "cashier": {"type": "string", "description": "出纳名称"},
            "attachment_number": {"type": "integer", "description": "附单据数"},
            "voucher_making_system": {"type": "string", "description": "外部系统类型"},
            "reserve2": {"type": "string", "description": "外部凭证业务号"},
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "biz_id": {"type": "string", "description": "上游id"},
            "sync": {"type": "integer", "description": "0=异步新增;1=同步新增"},
            "debit": {
                "type": "object",
                "description": "借方分录",
                "properties": {
                    "entry": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entry_id": {"type": "integer", "description": "分录号"},
                                "account_code": {"type": "string", "description": "科目编码（必填）"},
                                "abstract": {"type": "string", "description": "摘要（必填）"},
                                "natural_debit_currency": {"type": "number", "description": "本币借方发生额（必填）"}
                            }
                        }
                    }
                }
            },
            "credit": {
                "type": "object",
                "description": "贷方分录",
                "properties": {
                    "entry": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entry_id": {"type": "integer", "description": "分录号"},
                                "account_code": {"type": "string", "description": "科目编码（必填）"},
                                "abstract": {"type": "string", "description": "摘要（必填）"},
                                "natural_credit_currency": {"type": "number", "description": "本币贷方发生额（必填）"}
                            }
                        }
                    }
                }
            }
        },
        "required": ["voucher_type"]
    }
}

# ===================== 启用期间批量查询 Schema定义 =====================
U8_STARTPERIOD_BATCH_GET_SCHEMA = {
    "name": "u8_startperiod_batch_get",
    "description": "获取启用期间信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cAcc_Id": {"type": "string", "description": "帐套名称"}
        },
        "required": []
    }
}

# ===================== 总账结账状态批量查询 Schema定义 =====================
U8_MENDGLGZ_BATCH_GET_SCHEMA = {
    "name": "u8_mendglgz_batch_get",
    "description": "批量获取总账结账状态",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "iyear": {"type": "integer", "description": "会计年度"},
            "iperiod_begin": {"type": "integer", "description": "起始会计期间"},
            "iperiod_end": {"type": "integer", "description": "结束会计期间"}
        },
        "required": []
    }
}

# ===================== 科目总账批量查询 Schema定义 =====================
U8_ACCOUNTSUM_BATCH_GET_SCHEMA = {
    "name": "u8_accountsum_batch_get",
    "description": "批量获取科目总账信息，用于基础数据同步或前端展示",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "科目编码"},
            "iperiod": {"type": "integer", "description": "会计期间"},
            "cbegind_c": {"type": "string", "description": "金额期初方向"},
            "cendd_c": {"type": "string", "description": "金额期末方向"},
            "iyear": {"type": "integer", "description": "会计年度"}
        },
        "required": []
    }
}

# ===================== 辅助总账批量查询 Schema定义 =====================
U8_ACCOUNTASS_BATCH_GET_SCHEMA = {
    "name": "u8_accountass_batch_get",
    "description": "批量获取辅助总账信息，用于基础数据同步或前端展示",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode_begin": {"type": "string", "description": "起始科目编码"},
            "ccode_end": {"type": "string", "description": "结束科目编码"},
            "iperiod": {"type": "integer", "description": "会计期间"},
            "cbegind_c": {"type": "string", "description": "金额期初方向"},
            "cendd_c": {"type": "string", "description": "金额期末方向"},
            "iyear": {"type": "integer", "description": "会计年度"},
            "cdept_id": {"type": "string", "description": "部门编码"},
            "cperson_id": {"type": "string", "description": "职员编码"},
            "ccus_id": {"type": "string", "description": "客户编码"},
            "csup_id": {"type": "string", "description": "供应商编码"},
            "citem_class": {"type": "string", "description": "项目大类编码"},
            "citem_id": {"type": "string", "description": "项目编码"}
        },
        "required": []
    }
}

