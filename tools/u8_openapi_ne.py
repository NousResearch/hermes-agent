import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)

# =============================================================================
# 数据模型 - Data Models
# =============================================================================

# ===================== 商旅订单体模型 =====================
class BusinessTravelOrderEntry(BaseModel):
    """商旅订单体模型"""
    rowno: Optional[int] = Field(None, description="行号")
    citemcode: Optional[str] = Field(None, description="项目编码")
    citemname: Optional[str] = Field(None, description="项目名称")
    citem_class: Optional[str] = Field(None, description="项目大类编码")
    citem_cname: Optional[str] = Field(None, description="项目大类名称")
    ccode: Optional[str] = Field(None, description="科目编码")
    ccode_name: Optional[str] = Field(None, description="科目名称")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cdeptname: Optional[str] = Field(None, description="部门名称")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    cpersonname: Optional[str] = Field(None, description="人员名称")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    csupname: Optional[str] = Field(None, description="供应商名称")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cbustype_name: Optional[str] = Field(None, description="业务类型名称")
    iamount: Optional[float] = Field(None, description="金额")
    itax: Optional[float] = Field(None, description="税额")
    isum: Optional[float] = Field(None, description="含税金额")
    cmemo: Optional[str] = Field(None, description="备注")
    # 单据体自定义项
    define22: Optional[str] = Field(None, description="单据体自定义项1")
    define23: Optional[str] = Field(None, description="单据体自定义项2")
    define24: Optional[str] = Field(None, description="单据体自定义项3")
    define25: Optional[str] = Field(None, description="单据体自定义项4")
    define26: Optional[float] = Field(None, description="单据体自定义项5")
    define27: Optional[float] = Field(None, description="单据体自定义项6")
    define28: Optional[str] = Field(None, description="单据体自定义项7")
    define29: Optional[str] = Field(None, description="单据体自定义项8")
    define30: Optional[str] = Field(None, description="单据体自定义项9")
    define31: Optional[str] = Field(None, description="单据体自定义项10")
    define32: Optional[str] = Field(None, description="单据体自定义项11")
    define33: Optional[str] = Field(None, description="单据体自定义项12")
    define34: Optional[float] = Field(None, description="单据体自定义项13")
    define35: Optional[float] = Field(None, description="单据体自定义项14")
    define36: Optional[str] = Field(None, description="单据体自定义项15")
    define37: Optional[str] = Field(None, description="单据体自定义项16")


# ===================== 商旅订单主表模型 =====================
class BusinessTravelOrderInfo(BaseModel):
    """商旅订单主表模型"""
    ccode: Optional[str] = Field(None, description="单据编号")
    cvouchtype: Optional[str] = Field(None, description="单据类型")
    cdate: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cdeptname: Optional[str] = Field(None, description="部门名称")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    cpersonname: Optional[str] = Field(None, description="人员名称")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    csupname: Optional[str] = Field(None, description="供应商名称")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cbustype_name: Optional[str] = Field(None, description="业务类型名称")
    iamount: Optional[float] = Field(None, description="金额")
    itax: Optional[float] = Field(None, description="税额")
    isum: Optional[float] = Field(None, description="含税金额")
    cmemo: Optional[str] = Field(None, description="备注")
    cmaker: Optional[str] = Field(None, description="制单人")
    dcreatetime: Optional[str] = Field(None, description="制单时间")
    cchecker: Optional[str] = Field(None, description="审核人")
    dchecktime: Optional[str] = Field(None, description="审核时间")
    istatus: Optional[int] = Field(None, description="单据状态")
    # 单据头自定义项
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6")
    define7: Optional[float] = Field(None, description="单据头自定义项7")
    define8: Optional[str] = Field(None, description="单据头自定义项8")
    define9: Optional[str] = Field(None, description="单据头自定义项9")
    define10: Optional[str] = Field(None, description="单据头自定义项10")
    define11: Optional[str] = Field(None, description="单据头自定义项11")
    define12: Optional[str] = Field(None, description="单据头自定义项12")
    define13: Optional[str] = Field(None, description="单据头自定义项13")
    define14: Optional[str] = Field(None, description="单据头自定义项14")
    define15: Optional[float] = Field(None, description="单据头自定义项15")
    define16: Optional[float] = Field(None, description="单据头自定义项16")
    # 单据体列表
    entry: Optional[List[BusinessTravelOrderEntry]] = Field(None, description="商旅订单体列表")


# ===================== 获取单张商旅订单 数据模型 =====================
class GetBusinessTravelOrderInput(BaseModel):
    """获取单张商旅订单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="商旅订单编号（必填）")


# ===================== 新增商旅订单 数据模型 =====================
class AddBusinessTravelOrderInput(BaseModel):
    """新增商旅订单输入模型"""
    cdate: str = Field(..., description="单据日期（格式：yyyy-MM-dd）（必填）")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cmemo: Optional[str] = Field(None, description="备注")
    # 单据头自定义项
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6")
    define7: Optional[float] = Field(None, description="单据头自定义项7")
    define8: Optional[str] = Field(None, description="单据头自定义项8")
    define9: Optional[str] = Field(None, description="单据头自定义项9")
    define10: Optional[str] = Field(None, description="单据头自定义项10")
    define11: Optional[str] = Field(None, description="单据头自定义项11")
    define12: Optional[str] = Field(None, description="单据头自定义项12")
    define13: Optional[str] = Field(None, description="单据头自定义项13")
    define14: Optional[str] = Field(None, description="单据头自定义项14")
    define15: Optional[float] = Field(None, description="单据头自定义项15")
    define16: Optional[float] = Field(None, description="单据头自定义项16")
    # 单据体列表
    entry: List[BusinessTravelOrderEntry] = Field(..., description="商旅订单体列表（必填）")


# ===================== 费用报销单体模型 =====================
class ExpensesClaimEntry(BaseModel):
    """费用报销单体模型"""
    rowno: Optional[int] = Field(None, description="行号")
    citemcode: Optional[str] = Field(None, description="项目编码")
    citemname: Optional[str] = Field(None, description="项目名称")
    citem_class: Optional[str] = Field(None, description="项目大类编码")
    citem_cname: Optional[str] = Field(None, description="项目大类名称")
    ccode: Optional[str] = Field(None, description="科目编码")
    ccode_name: Optional[str] = Field(None, description="科目名称")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cdeptname: Optional[str] = Field(None, description="部门名称")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    cpersonname: Optional[str] = Field(None, description="人员名称")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    csupname: Optional[str] = Field(None, description="供应商名称")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cbustype_name: Optional[str] = Field(None, description="业务类型名称")
    iamount: Optional[float] = Field(None, description="金额")
    itax: Optional[float] = Field(None, description="税额")
    isum: Optional[float] = Field(None, description="含税金额")
    cmemo: Optional[str] = Field(None, description="备注")
    # 单据体自定义项
    define22: Optional[str] = Field(None, description="单据体自定义项1")
    define23: Optional[str] = Field(None, description="单据体自定义项2")
    define24: Optional[str] = Field(None, description="单据体自定义项3")
    define25: Optional[str] = Field(None, description="单据体自定义项4")
    define26: Optional[float] = Field(None, description="单据体自定义项5")
    define27: Optional[float] = Field(None, description="单据体自定义项6")
    define28: Optional[str] = Field(None, description="单据体自定义项7")
    define29: Optional[str] = Field(None, description="单据体自定义项8")
    define30: Optional[str] = Field(None, description="单据体自定义项9")
    define31: Optional[str] = Field(None, description="单据体自定义项10")
    define32: Optional[str] = Field(None, description="单据体自定义项11")
    define33: Optional[str] = Field(None, description="单据体自定义项12")
    define34: Optional[float] = Field(None, description="单据体自定义项13")
    define35: Optional[float] = Field(None, description="单据体自定义项14")
    define36: Optional[str] = Field(None, description="单据体自定义项15")
    define37: Optional[str] = Field(None, description="单据体自定义项16")


# ===================== 费用报销单主表模型 =====================
class ExpensesClaimInfo(BaseModel):
    """费用报销单主表模型"""
    ccode: Optional[str] = Field(None, description="单据编号")
    cvouchtype: Optional[str] = Field(None, description="单据类型")
    cdate: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cdeptname: Optional[str] = Field(None, description="部门名称")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    cpersonname: Optional[str] = Field(None, description="人员名称")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    csupname: Optional[str] = Field(None, description="供应商名称")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cbustype_name: Optional[str] = Field(None, description="业务类型名称")
    iamount: Optional[float] = Field(None, description="金额")
    itax: Optional[float] = Field(None, description="税额")
    isum: Optional[float] = Field(None, description="含税金额")
    cmemo: Optional[str] = Field(None, description="备注")
    cmaker: Optional[str] = Field(None, description="制单人")
    dcreatetime: Optional[str] = Field(None, description="制单时间")
    cchecker: Optional[str] = Field(None, description="审核人")
    dchecktime: Optional[str] = Field(None, description="审核时间")
    istatus: Optional[int] = Field(None, description="单据状态")
    # 单据头自定义项
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6")
    define7: Optional[float] = Field(None, description="单据头自定义项7")
    define8: Optional[str] = Field(None, description="单据头自定义项8")
    define9: Optional[str] = Field(None, description="单据头自定义项9")
    define10: Optional[str] = Field(None, description="单据头自定义项10")
    define11: Optional[str] = Field(None, description="单据头自定义项11")
    define12: Optional[str] = Field(None, description="单据头自定义项12")
    define13: Optional[str] = Field(None, description="单据头自定义项13")
    define14: Optional[str] = Field(None, description="单据头自定义项14")
    define15: Optional[float] = Field(None, description="单据头自定义项15")
    define16: Optional[float] = Field(None, description="单据头自定义项16")
    # 单据体列表
    entry: Optional[List[ExpensesClaimEntry]] = Field(None, description="费用报销单体列表")


# ===================== 获取费用报销单列表 数据模型 =====================
class GetExpensesClaimListInput(BaseModel):
    """获取费用报销单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始单据日期（格式：yyyy-MM-dd）")
    date_end: Optional[str] = Field(None, description="结束单据日期（格式：yyyy-MM-dd）")
    cvouchtype: Optional[str] = Field(None, description="单据类型")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cdeptname: Optional[str] = Field(None, description="部门名称关键字")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    cpersonname: Optional[str] = Field(None, description="人员名称关键字")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    csupname: Optional[str] = Field(None, description="供应商名称关键字")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称关键字")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cbustype_name: Optional[str] = Field(None, description="业务类型名称关键字")
    cmemo: Optional[str] = Field(None, description="备注关键字")
    cmaker: Optional[str] = Field(None, description="制单人")
    cchecker: Optional[str] = Field(None, description="审核人")
    istatus: Optional[int] = Field(None, description="单据状态")


# ===================== 获取费用报销单待办任务 数据模型 =====================
class GetExpensesClaimTasksInput(BaseModel):
    """获取费用报销单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[int] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[int] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[int] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[int] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[int] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 查看费用报销单审批进程 数据模型 =====================
class GetExpensesClaimHistoryInput(BaseModel):
    """查看费用报销单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单张费用报销单 数据模型 =====================
class GetExpensesClaimInput(BaseModel):
    """获取单张费用报销单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="费用报销单编号（必填）")


# ===================== 获取费用报销单工作流按钮状态 数据模型 =====================
class GetExpensesClaimButtonstateInput(BaseModel):
    """获取费用报销单工作流按钮状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审核费用报销单 数据模型 =====================
class AuditExpensesClaimInput(BaseModel):
    """审核费用报销单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 新增费用报销单 数据模型 =====================
class AddExpensesClaimInput(BaseModel):
    """新增费用报销单输入模型"""
    cdate: str = Field(..., description="单据日期（格式：yyyy-MM-dd）（必填）")
    cdeptcode: Optional[str] = Field(None, description="部门编码")
    cpersoncode: Optional[str] = Field(None, description="人员编码")
    csupcode: Optional[str] = Field(None, description="供应商编码")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    cbustype: Optional[str] = Field(None, description="业务类型编码")
    cmemo: Optional[str] = Field(None, description="备注")
    # 单据头自定义项
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6")
    define7: Optional[float] = Field(None, description="单据头自定义项7")
    define8: Optional[str] = Field(None, description="单据头自定义项8")
    define9: Optional[str] = Field(None, description="单据头自定义项9")
    define10: Optional[str] = Field(None, description="单据头自定义项10")
    define11: Optional[str] = Field(None, description="单据头自定义项11")
    define12: Optional[str] = Field(None, description="单据头自定义项12")
    define13: Optional[str] = Field(None, description="单据头自定义项13")
    define14: Optional[str] = Field(None, description="单据头自定义项14")
    define15: Optional[float] = Field(None, description="单据头自定义项15")
    define16: Optional[float] = Field(None, description="单据头自定义项16")
    # 单据体列表
    entry: List[ExpensesClaimEntry] = Field(..., description="费用报销单体列表（必填）")


# ===================== 弃审费用报销单 数据模型 =====================
class AbandonExpensesClaimInput(BaseModel):
    """弃审费用报销单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")


# =============================================================================
# Tool函数 - Tool Functions
# =============================================================================

# ===================== 获取单张商旅订单 Tool函数 =====================
def u8_businesstravelorder_get_tool(input_data: GetBusinessTravelOrderInput, client: U8OpenAPIClient) -> str:
    """
    通过商旅订单编号获取用友U8中的商旅订单单据信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/businesstravelorder/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===================== 新增商旅订单 Tool函数 =====================
def u8_businesstravelorder_add_tool(input_data: AddBusinessTravelOrderInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增商旅订单，包含单据头和单据体（entry）完整信息。
    """
    request_body: dict = {
        "businesstravelorder": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/businesstravelorder/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "商旅订单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "商旅订单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取费用报销单列表 Tool函数 =====================
def u8_expensesclaimlist_batch_get_tool(input_data: GetExpensesClaimListInput, client: U8OpenAPIClient) -> str:
    """
    获取费用报销单列表信息，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/expensesclaimlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取费用报销单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取费用报销单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "expensesclaimlist": result.get("expensesclaimlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取费用报销单待办任务 Tool函数 =====================
def u8_expensesclaim_tasks_tool(input_data: GetExpensesClaimTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取费用报销单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/expensesclaim/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取费用报销单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取费用报销单待办任务成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "tasks": result.get("tasks")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 查看费用报销单审批进程 Tool函数 =====================
def u8_expensesclaim_history_tool(input_data: GetExpensesClaimHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看费用报销单审批进程，获取单据的审批历史记录。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/expensesclaim/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取费用报销单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取费用报销单审批进程成功",
            "data": {
                "history": result.get("history", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单张费用报销单 Tool函数 =====================
def u8_expensesclaim_get_tool(input_data: GetExpensesClaimInput, client: U8OpenAPIClient) -> str:
    """
    通过费用报销单编号获取用友U8中的费用报销单单据信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/expensesclaim/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===================== 获取费用报销单工作流按钮状态 Tool函数 =====================
def u8_expensesclaim_buttonstate_tool(input_data: GetExpensesClaimButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取费用报销单工作流按钮是否可用状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/expensesclaim/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取费用报销单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取费用报销单工作流按钮状态成功",
            "data": {
                "buttonstate": result.get("buttonstate")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核费用报销单 Tool函数 =====================
def u8_expensesclaim_audit_tool(input_data: AuditExpensesClaimInput, client: U8OpenAPIClient) -> str:
    """
    审核费用报销单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "expensesclaim": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/expensesclaim/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "费用报销单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "费用报销单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增费用报销单 Tool函数 =====================
def u8_expensesclaim_add_tool(input_data: AddExpensesClaimInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增费用报销单，包含单据头和单据体（entry）完整信息。
    """
    request_body: dict = {
        "expensesclaim": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/expensesclaim/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "费用报销单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "费用报销单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审费用报销单 Tool函数 =====================
def u8_expensesclaim_abandon_tool(input_data: AbandonExpensesClaimInput, client: U8OpenAPIClient) -> str:
    """
    弃审费用报销单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "expensesclaim": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/expensesclaim/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "费用报销单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "费用报销单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# =============================================================================
# Schema定义 - Schema Definitions
# =============================================================================

# ===================== 获取单张商旅订单 Schema定义 =====================
U8_BUSINESSTRAVELORDER_GET_SCHEMA = {
    "name": "u8_businesstravelorder_get",
    "description": "通过商旅订单编号获取单张商旅订单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {
                "type": "string",
                "description": "商旅订单编号（必填）"
            }
        },
        "required": ["id"]
    }
}

# ===================== 新增商旅订单 Schema定义 =====================
U8_BUSINESSTRAVELORDER_ADD_SCHEMA = {
    "name": "u8_businesstravelorder_add",
    "description": "在用友U8 OpenAPI中新增商旅订单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            "cdate": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）（必填）"},
            "cdeptcode": {"type": "string", "description": "部门编码"},
            "cpersoncode": {"type": "string", "description": "人员编码"},
            "csupcode": {"type": "string", "description": "供应商编码"},
            "ccuscode": {"type": "string", "description": "客户编码"},
            "cbustype": {"type": "string", "description": "业务类型编码"},
            "cmemo": {"type": "string", "description": "备注"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6"},
            "define7": {"type": "number", "description": "单据头自定义项7"},
            "define8": {"type": "string", "description": "单据头自定义项8"},
            "define9": {"type": "string", "description": "单据头自定义项9"},
            "define10": {"type": "string", "description": "单据头自定义项10"},
            "define11": {"type": "string", "description": "单据头自定义项11"},
            "define12": {"type": "string", "description": "单据头自定义项12"},
            "define13": {"type": "string", "description": "单据头自定义项13"},
            "define14": {"type": "string", "description": "单据头自定义项14"},
            "define15": {"type": "number", "description": "单据头自定义项15"},
            "define16": {"type": "number", "description": "单据头自定义项16"},
            # 单据体（entry）列表
            "entry": {
                "type": "array",
                "description": "商旅订单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "rowno": {"type": "integer", "description": "行号"},
                        "citemcode": {"type": "string", "description": "项目编码"},
                        "citemname": {"type": "string", "description": "项目名称"},
                        "citem_class": {"type": "string", "description": "项目大类编码"},
                        "citem_cname": {"type": "string", "description": "项目大类名称"},
                        "ccode": {"type": "string", "description": "科目编码"},
                        "ccode_name": {"type": "string", "description": "科目名称"},
                        "cdeptcode": {"type": "string", "description": "部门编码"},
                        "cdeptname": {"type": "string", "description": "部门名称"},
                        "cpersoncode": {"type": "string", "description": "人员编码"},
                        "cpersonname": {"type": "string", "description": "人员名称"},
                        "csupcode": {"type": "string", "description": "供应商编码"},
                        "csupname": {"type": "string", "description": "供应商名称"},
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "ccusname": {"type": "string", "description": "客户名称"},
                        "cbustype": {"type": "string", "description": "业务类型编码"},
                        "cbustype_name": {"type": "string", "description": "业务类型名称"},
                        "iamount": {"type": "number", "description": "金额"},
                        "itax": {"type": "number", "description": "税额"},
                        "isum": {"type": "number", "description": "含税金额"},
                        "cmemo": {"type": "string", "description": "备注"},
                        # 单据体自定义项 1-16
                        "define22": {"type": "string", "description": "单据体自定义项1"},
                        "define23": {"type": "string", "description": "单据体自定义项2"},
                        "define24": {"type": "string", "description": "单据体自定义项3"},
                        "define25": {"type": "string", "description": "单据体自定义项4"},
                        "define26": {"type": "number", "description": "单据体自定义项5"},
                        "define27": {"type": "number", "description": "单据体自定义项6"},
                        "define28": {"type": "string", "description": "单据体自定义项7"},
                        "define29": {"type": "string", "description": "单据体自定义项8"},
                        "define30": {"type": "string", "description": "单据体自定义项9"},
                        "define31": {"type": "string", "description": "单据体自定义项10"},
                        "define32": {"type": "string", "description": "单据体自定义项11"},
                        "define33": {"type": "string", "description": "单据体自定义项12"},
                        "define34": {"type": "number", "description": "单据体自定义项13"},
                        "define35": {"type": "number", "description": "单据体自定义项14"},
                        "define36": {"type": "string", "description": "单据体自定义项15"},
                        "define37": {"type": "string", "description": "单据体自定义项16"}
                    }
                }
            }
        },
        "required": ["cdate", "entry"]
    }
}

# ===================== 获取费用报销单列表 Schema定义 =====================
U8_EXPENSESCLAIMLIST_BATCH_GET_SCHEMA = {
    "name": "u8_expensesclaimlist_batch_get",
    "description": "在用友U8 OpenAPI中查询费用报销单列表信息，支持分页、单据号/日期范围、部门/人员/供应商/客户/业务类型等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始单据日期（格式：yyyy-MM-dd）"},
            "date_end": {"type": "string", "description": "结束单据日期（格式：yyyy-MM-dd）"},
            "cvouchtype": {"type": "string", "description": "单据类型"},
            "cdeptcode": {"type": "string", "description": "部门编码"},
            "cdeptname": {"type": "string", "description": "部门名称关键字"},
            "cpersoncode": {"type": "string", "description": "人员编码"},
            "cpersonname": {"type": "string", "description": "人员名称关键字"},
            "csupcode": {"type": "string", "description": "供应商编码"},
            "csupname": {"type": "string", "description": "供应商名称关键字"},
            "ccuscode": {"type": "string", "description": "客户编码"},
            "ccusname": {"type": "string", "description": "客户名称关键字"},
            "cbustype": {"type": "string", "description": "业务类型编码"},
            "cbustype_name": {"type": "string", "description": "业务类型名称关键字"},
            "cmemo": {"type": "string", "description": "备注关键字"},
            "cmaker": {"type": "string", "description": "制单人"},
            "cchecker": {"type": "string", "description": "审核人"},
            "istatus": {"type": "number", "description": "单据状态"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 获取费用报销单待办任务 Schema定义 =====================
U8_EXPENSESCLAIM_TASKS_SCHEMA = {
    "name": "u8_expensesclaim_tasks",
    "description": "获取费用报销单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "number", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "number", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "number", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "number", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "number", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 查看费用报销单审批进程 Schema定义 =====================
U8_EXPENSESCLAIM_HISTORY_SCHEMA = {
    "name": "u8_expensesclaim_history",
    "description": "查看费用报销单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "user_id": {"type": "string", "description": "审批人(用户编码)，user_id与person_code输入一个参数即可"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可"},
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            }
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单张费用报销单 Schema定义 =====================
U8_EXPENSESCLAIM_GET_SCHEMA = {
    "name": "u8_expensesclaim_get",
    "description": "通过费用报销单编号获取单张费用报销单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {
                "type": "string",
                "description": "费用报销单编号（必填）"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取费用报销单工作流按钮状态 Schema定义 =====================
U8_EXPENSESCLAIM_BUTTONSTATE_SCHEMA = {
    "name": "u8_expensesclaim_buttonstate",
    "description": "获取费用报销单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            }
        },
        "required": ["voucher_code"]
    }
}

# ===================== 审核费用报销单 Schema定义 =====================
U8_EXPENSESCLAIM_AUDIT_SCHEMA = {
    "name": "u8_expensesclaim_audit",
    "description": "审核费用报销单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "user_id": {
                "type": "string",
                "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"
            },
            "person_code": {
                "type": "string",
                "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"
            },
            "opinion": {
                "type": "string",
                "description": "审批意见"
            },
            "agree": {
                "type": "number",
                "description": "是否同意(1=同意;0=不同意)（必填）"
            }
        },
        "required": [
            "voucher_code",
            "agree"
        ]
    }
}

# ===================== 新增费用报销单 Schema定义 =====================
U8_EXPENSESCLAIM_ADD_SCHEMA = {
    "name": "u8_expensesclaim_add",
    "description": "在用友U8 OpenAPI中新增费用报销单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            "cdate": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）（必填）"},
            "cdeptcode": {"type": "string", "description": "部门编码"},
            "cpersoncode": {"type": "string", "description": "人员编码"},
            "csupcode": {"type": "string", "description": "供应商编码"},
            "ccuscode": {"type": "string", "description": "客户编码"},
            "cbustype": {"type": "string", "description": "业务类型编码"},
            "cmemo": {"type": "string", "description": "备注"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6"},
            "define7": {"type": "number", "description": "单据头自定义项7"},
            "define8": {"type": "string", "description": "单据头自定义项8"},
            "define9": {"type": "string", "description": "单据头自定义项9"},
            "define10": {"type": "string", "description": "单据头自定义项10"},
            "define11": {"type": "string", "description": "单据头自定义项11"},
            "define12": {"type": "string", "description": "单据头自定义项12"},
            "define13": {"type": "string", "description": "单据头自定义项13"},
            "define14": {"type": "string", "description": "单据头自定义项14"},
            "define15": {"type": "number", "description": "单据头自定义项15"},
            "define16": {"type": "number", "description": "单据头自定义项16"},
            # 单据体（entry）列表
            "entry": {
                "type": "array",
                "description": "费用报销单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "rowno": {"type": "integer", "description": "行号"},
                        "citemcode": {"type": "string", "description": "项目编码"},
                        "citemname": {"type": "string", "description": "项目名称"},
                        "citem_class": {"type": "string", "description": "项目大类编码"},
                        "citem_cname": {"type": "string", "description": "项目大类名称"},
                        "ccode": {"type": "string", "description": "科目编码"},
                        "ccode_name": {"type": "string", "description": "科目名称"},
                        "cdeptcode": {"type": "string", "description": "部门编码"},
                        "cdeptname": {"type": "string", "description": "部门名称"},
                        "cpersoncode": {"type": "string", "description": "人员编码"},
                        "cpersonname": {"type": "string", "description": "人员名称"},
                        "csupcode": {"type": "string", "description": "供应商编码"},
                        "csupname": {"type": "string", "description": "供应商名称"},
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "ccusname": {"type": "string", "description": "客户名称"},
                        "cbustype": {"type": "string", "description": "业务类型编码"},
                        "cbustype_name": {"type": "string", "description": "业务类型名称"},
                        "iamount": {"type": "number", "description": "金额"},
                        "itax": {"type": "number", "description": "税额"},
                        "isum": {"type": "number", "description": "含税金额"},
                        "cmemo": {"type": "string", "description": "备注"},
                        # 单据体自定义项 1-16
                        "define22": {"type": "string", "description": "单据体自定义项1"},
                        "define23": {"type": "string", "description": "单据体自定义项2"},
                        "define24": {"type": "string", "description": "单据体自定义项3"},
                        "define25": {"type": "string", "description": "单据体自定义项4"},
                        "define26": {"type": "number", "description": "单据体自定义项5"},
                        "define27": {"type": "number", "description": "单据体自定义项6"},
                        "define28": {"type": "string", "description": "单据体自定义项7"},
                        "define29": {"type": "string", "description": "单据体自定义项8"},
                        "define30": {"type": "string", "description": "单据体自定义项9"},
                        "define31": {"type": "string", "description": "单据体自定义项10"},
                        "define32": {"type": "string", "description": "单据体自定义项11"},
                        "define33": {"type": "string", "description": "单据体自定义项12"},
                        "define34": {"type": "number", "description": "单据体自定义项13"},
                        "define35": {"type": "number", "description": "单据体自定义项14"},
                        "define36": {"type": "string", "description": "单据体自定义项15"},
                        "define37": {"type": "string", "description": "单据体自定义项16"}
                    }
                }
            }
        },
        "required": ["cdate", "entry"]
    }
}

# ===================== 弃审费用报销单 Schema定义 =====================
U8_EXPENSESCLAIM_ABANDON_SCHEMA = {
    "name": "u8_expensesclaim_abandon",
    "description": "弃审费用报销单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "user_id": {
                "type": "string",
                "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"
            },
            "person_code": {
                "type": "string",
                "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"
            },
            "opinion": {
                "type": "string",
                "description": "审批意见"
            }
        },
        "required": [
            "voucher_code"
        ]
    }
}
