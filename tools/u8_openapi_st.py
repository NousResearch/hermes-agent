import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 产成品入库单列表查询 数据模型 =====================
class GetProductinListInput(BaseModel):
    """获取产成品入库单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始单据号")
    code_end: Optional[str] = Field(None, description="结束单据号")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    warehousename: Optional[str] = Field(None, description="仓库名称关键字")
    receivecode: Optional[str] = Field(None, description="收发类别编码")
    receivename: Optional[str] = Field(None, description="收发类别名称关键字")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    handler: Optional[str] = Field(None, description="审核人关键字")
    memory: Optional[str] = Field(None, description="备注关键字")
    maker: Optional[str] = Field(None, description="制单人关键字")


# ===================== 审核产成品入库单 数据模型 =====================
class VerifyProductinInput(BaseModel):
    """审核产成品入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")


# ===================== 弃审产成品入库单 数据模型 =====================
class UnverifyProductinInput(BaseModel):
    """弃审产成品入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个产成品入库单 数据模型 =====================
class GetProductinInput(BaseModel):
    """获取单个产成品入库单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="产成品入库单编码（必填）")


# ===================== 新增产成品入库单 数据模型 =====================
class ProductinEntry(BaseModel):
    """产成品入库单表体数据模型"""
    inventorycode: str = Field(..., description="存货编码（必填）")
    quantity: Optional[float] = Field(None, description="数量")
    assitantunit: Optional[str] = Field(None, description="辅记量单位编码")
    irate: Optional[float] = Field(None, description="换算率")
    number: Optional[float] = Field(None, description="件数")
    price: Optional[float] = Field(None, description="单价")
    cost: Optional[float] = Field(None, description="金额")
    serial: Optional[str] = Field(None, description="批号")
    makedate: Optional[str] = Field(None, description="生产日期")
    validdate: Optional[str] = Field(None, description="失效日期")
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
    rowno: Optional[float] = Field(None, description="行号")


class AddProductinInput(BaseModel):
    """新增产成品入库单输入模型"""
    code: Optional[str] = Field(None, description="单据编号")
    date: Optional[str] = Field(None, description="制单日期")
    maker: Optional[str] = Field(None, description="制单人名称")
    warehousecode: str = Field(..., description="仓库编码（必填）")
    memory: Optional[str] = Field(None, description="备注")
    receivecode: str = Field(..., description="收发类型编码（必填）")
    departmentcode: str = Field(..., description="部门编码（必填）")
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
    entry: Optional[List[ProductinEntry]] = Field(None, description="表体信息列表")


# ===================== 其他入库单列表查询 数据模型 =====================
class GetOtherinListInput(BaseModel):
    """获取其它入库单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始制单日期")
    date_end: Optional[str] = Field(None, description="结束制单日期")
    maker: Optional[str] = Field(None, description="制单人名称关键字")
    handler: Optional[str] = Field(None, description="审核人关键字")
    businesstype: Optional[str] = Field(None, description="业务类型")
    businesscode: Optional[str] = Field(None, description="业务编码")
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    warehousename: Optional[str] = Field(None, description="仓库名称关键字")
    memory: Optional[str] = Field(None, description="备注关键字")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    auditdate_begin: Optional[str] = Field(None, description="起始审核日期")
    auditdate_end: Optional[str] = Field(None, description="结束审核日期")
    cvoucherstate: Optional[str] = Field(None, description="状态")


# ===================== 审核其他入库单 数据模型 =====================
class VerifyOtherinInput(BaseModel):
    """审核其他入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")


# ===================== 弃审其他入库单 数据模型 =====================
class UnverifyOtherinInput(BaseModel):
    """弃审其他入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取其他入库单待办任务 数据模型 =====================
class GetOtherinTasksInput(BaseModel):
    """获取其他入库单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[str] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[int] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[int] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[int] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[int] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 查看其他入库单审批进程 数据模型 =====================
class GetOtherinHistoryInput(BaseModel):
    """查看其他入库单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个其它入库单 数据模型 =====================
class GetOtherinInput(BaseModel):
    """获取单张其它入库单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: Optional[str] = Field(None, description="其他入库单编码")


# ===================== 获取其他入库单是否启用工作流 数据模型 =====================
class GetOtherinFlowenabledInput(BaseModel):
    """获取其他入库单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")


# ===================== 获取其他入库单工作流按钮是否可用状态 数据模型 =====================
class GetOtherinButtonstateInput(BaseModel):
    """获取其他入库单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审核其他入库单(工作流) 数据模型 =====================
class AuditOtherinInput(BaseModel):
    """审核其他入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 弃审其他入库单(工作流) 数据模型 =====================
class AbandonOtherinInput(BaseModel):
    """弃审其他入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")


# ===================== 新增其它入库单 数据模型 =====================
class OtherinEntry(BaseModel):
    """其他入库单表体数据模型"""
    inventorycode: str = Field(..., description="存货编码（必填）")
    inventoryname: Optional[str] = Field(None, description="存货")
    inventorystd: Optional[str] = Field(None, description="规格型号")
    quantity: Optional[float] = Field(None, description="数量")
    price: Optional[float] = Field(None, description="单价")
    cost: Optional[float] = Field(None, description="金额（必填）")
    cmassunitname: Optional[str] = Field(None, description="主计量单位名称")
    serial: Optional[str] = Field(None, description="批号")
    makedate: Optional[str] = Field(None, description="生产日期")
    validdate: Optional[str] = Field(None, description="失效日期")
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
    rowno: Optional[float] = Field(None, description="行号")


class AddOtherinInput(BaseModel):
    """新增其它入库单输入模型"""
    code: Optional[str] = Field(None, description="单据编号")
    date: Optional[str] = Field(None, description="制单日期")
    maker: Optional[str] = Field(None, description="制单人名称")
    warehousecode: str = Field(..., description="仓库编码（必填）")
    warehousename: Optional[str] = Field(None, description="仓库名称")
    memory: Optional[str] = Field(None, description="备注")
    receivecode: str = Field(..., description="收发类型编码（必填）")
    receivename: Optional[str] = Field(None, description="收发类型")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
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
    entry: Optional[List[OtherinEntry]] = Field(None, description="表体信息列表")


# ===================== 其他出库单列表查询 数据模型 =====================
class GetOtheroutListInput(BaseModel):
    """获取其它出库单列表信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始制单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束制单日期，格式:yyyy-MM-dd")
    auditdate_begin: Optional[str] = Field(None, description="起始审核日期，格式:yyyy-MM-dd")
    auditdate_end: Optional[str] = Field(None, description="结束审核日期，格式:yyyy-MM-dd")
    state: Optional[str] = Field(None, description="单据状态")
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    warehousename: Optional[str] = Field(None, description="仓库名称")
    maker: Optional[str] = Field(None, description="制单人")
    departmentcode: Optional[str] = Field(None, description="部门编码，可以通过api/department获取")
    departmentname: Optional[str] = Field(None, description="部门名称关键字，可以通过api/department获取")
    memory: Optional[str] = Field(None, description="备注关键字")


# ===================== 审核其他出库单 数据模型 =====================
class VerifyOtheroutInput(BaseModel):
    """审核其他出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")


# ===================== 弃审其他出库单 数据模型 =====================
class UnverifyOtheroutInput(BaseModel):
    """弃审其他出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取其他出库单待办任务 数据模型 =====================
class GetOtheroutTasksInput(BaseModel):
    """获取其他出库单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[str] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[int] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[int] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[int] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[int] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 查看其他出库单审批进程 数据模型 =====================
class GetOtheroutHistoryInput(BaseModel):
    """查看其他出库单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个其它出库单 数据模型 =====================
class GetOtheroutInput(BaseModel):
    """获取单张其它出库单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="单据编号（必填）")


# ===================== 获取其他出库单是否启用工作流 数据模型 =====================
class GetOtheroutFlowenabledInput(BaseModel):
    """获取其他出库单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")


# ===================== 获取其他出库单工作流按钮是否可用状态 数据模型 =====================
class GetOtheroutButtonstateInput(BaseModel):
    """获取其他出库单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审核其他出库单(工作流) 数据模型 =====================
class AuditOtheroutInput(BaseModel):
    """审核其他出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 弃审其他出库单(工作流) 数据模型 =====================
class AbandonOtheroutInput(BaseModel):
    """弃审其他出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")


# ===================== 新增其它出库单 数据模型 =====================
class OtheroutEntry(BaseModel):
    """其他出库单表体数据模型"""
    inventorycode: str = Field(..., description="存货编码（必填）")
    inventoryname: Optional[str] = Field(None, description="存货")
    inventorystd: Optional[str] = Field(None, description="规格型号")
    quantity: Optional[float] = Field(None, description="数量")
    price: Optional[float] = Field(None, description="单价")
    cost: Optional[float] = Field(None, description="金额（必填）")
    cmassunitname: Optional[str] = Field(None, description="主计量单位名称")
    serial: Optional[str] = Field(None, description="批号")
    makedate: Optional[str] = Field(None, description="生产日期")
    validdate: Optional[str] = Field(None, description="失效日期")
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
    rowno: Optional[float] = Field(None, description="行号")

class AddOtheroutInput(BaseModel):
    """新增其它出库单输入模型"""
    code: Optional[str] = Field(None, description="单据编号")
    date: Optional[str] = Field(None, description="制单日期")
    maker: Optional[str] = Field(None, description="制单人名称")
    warehousecode: str = Field(..., description="仓库编码（必填）")
    warehousename: Optional[str] = Field(None, description="仓库名称")
    memory: Optional[str] = Field(None, description="备注")
    receivecode: str = Field(..., description="收发类型编码（必填）")
    receivename: Optional[str] = Field(None, description="收发类型")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
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
    entry: Optional[List[OtheroutEntry]] = Field(None, description="表体信息列表")

# ===================== 批量获取库存结账状态 数据模型 =====================
class GetMendstListInput(BaseModel):
    """批量获取库存结账状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    iyear: Optional[int] = Field(None, description="会计年度")
    iperiod_begin: Optional[int] = Field(None, description="起始会计期间")
    iperiod_end: Optional[int] = Field(None, description="结束会计期间")


# ===================== 材料出库单 - 子数据模型 (entry) 数据模型 =====================
class MaterialoutEntry(BaseModel):
    """材料出库单表体明细"""
    inventorycode: str = Field(..., description="存货编码")
    quantity: Optional[float] = Field(None, description="数量")
    assitantunit: Optional[str] = Field(None, description="辅计量单位编码")
    irate: Optional[float] = Field(None, description="换算率")
    number: Optional[float] = Field(None, description="件数")
    price: Optional[float] = Field(None, description="单价")
    cost: Optional[float] = Field(None, description="金额")
    serial: Optional[str] = Field(None, description="批号")
    makedate: Optional[str] = Field(None, description="生产日期")
    validdate: Optional[str] = Field(None, description="失效日期")
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
    memory: Optional[str] = Field(None, description="表体备注")
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
    rowno: Optional[float] = Field(None, description="行号")

# ===================== 获取材料出库单列表 数据模型 =====================
class GetMaterialoutListInput(BaseModel):
    """获取材料出库单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    warehousename: Optional[str] = Field(None, description="仓库名称关键字")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    auditdate_begin: Optional[str] = Field(None, description="起始审核日期，格式:yyyy-MM-dd")
    auditdate_end: Optional[str] = Field(None, description="结束审核日期，格式:yyyy-MM-dd")
    code_begin: Optional[str] = Field(None, description="起始出库单号")
    code_end: Optional[str] = Field(None, description="结束出库单号")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    memory: Optional[str] = Field(None, description="备注关键字")
    maker: Optional[str] = Field(None, description="制单人关键字")
    businesstype: Optional[str] = Field(None, description="业务类型")
    source: Optional[str] = Field(None, description="单据来源")
    cmpocode: Optional[str] = Field(None, description="生产订单号")
    serial: Optional[str] = Field(None, description="生产批号")
    businesscode: Optional[str] = Field(None, description="对应业务单号")
    receivecode: Optional[str] = Field(None, description="收发类别编码")
    receivename: Optional[str] = Field(None, description="收发类别名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    vendorabbname: Optional[str] = Field(None, description="供应商简称关键字")
    handler: Optional[str] = Field(None, description="审核人关键字")
    define1: Optional[str] = Field(None, description="自定义项1")
    define2: Optional[str] = Field(None, description="自定义项2")
    define3: Optional[str] = Field(None, description="自定义项3")
    define4_begin: Optional[str] = Field(None, description="起始自定义项4")
    define4_end: Optional[str] = Field(None, description="结束自定义项4")
    define5: Optional[float] = Field(None, description="自定义项5")
    define6_begin: Optional[str] = Field(None, description="起始自定义项6")
    define6_end: Optional[str] = Field(None, description="结束自定义项6")
    define7: Optional[float] = Field(None, description="自定义项7")
    define8: Optional[str] = Field(None, description="自定义项8")
    define9: Optional[str] = Field(None, description="自定义项9")
    define10: Optional[str] = Field(None, description="自定义项10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[float] = Field(None, description="自定义项15")
    define16: Optional[float] = Field(None, description="自定义项16")


# ===================== 审核材料出库单 数据模型 =====================
class VerifyMaterialoutInput(BaseModel):
    """审核材料出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code二选一传入")


# ===================== 弃审材料出库单 数据模型 =====================
class UnverifyMaterialoutInput(BaseModel):
    """弃审材料出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code二选一传入")


# ===================== 获取材料出库单待办任务 数据模型 =====================
class GetMaterialoutTasksInput(BaseModel):
    """获取材料出库单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[int] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[int] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[int] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[int] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[int] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 获取材料出库单审批进程 数据模型 =====================
class GetMaterialoutHistoryInput(BaseModel):
    """获取材料出库单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个材料出库单 数据模型 =====================
class GetMaterialoutInput(BaseModel):
    """获取单个材料出库单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="收发记录主表标识（必填）")


# ===================== 获取材料出库单是否启用工作流 数据模型 =====================
class GetMaterialoutFlowenabledInput(BaseModel):
    """获取材料出库单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")


# ===================== 获取材料出库单工作流按钮是否可用状态 数据模型 =====================
class GetMaterialoutButtonstateInput(BaseModel):
    """获取材料出库单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审核材料出库单（工作流） 数据模型 =====================
class AuditMaterialoutInput(BaseModel):
    """审核材料出库单（工作流）输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 新增材料出库单 数据模型 =====================
class AddMaterialoutInput(BaseModel):
    """新增材料出库单输入模型"""
    code: Optional[str] = Field(None, description="单据编号")
    date: Optional[str] = Field(None, description="制单日期")
    maker: Optional[str] = Field(None, description="制单人名称")
    warehousecode: str = Field(..., description="仓库编码（必填）")
    memory: Optional[str] = Field(None, description="备注")
    receivecode: str = Field(..., description="收发类型编码（必填）")
    departmentcode: Optional[str] = Field(None, description="部门编码")
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
    entry: Optional[List[MaterialoutEntry]] = Field(None, description="表体明细列表")


# ===================== 弃审材料出库单（工作流） 数据模型 =====================
class AbandonMaterialoutInput(BaseModel):
    """弃审材料出库单（工作流）输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")

# ===================== 现存量查询 数据模型 =====================
class GetCurrentstockInput(BaseModel):
    """现存量查询输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号(默认取应用的第一个数据源)")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    whcode_begin: Optional[str] = Field(None, description="起始仓库编码")
    whcode_end: Optional[str] = Field(None, description="结束仓库编码")
    whname: Optional[str] = Field(None, description="仓库名称关键字")
    invcode_begin: Optional[str] = Field(None, description="起始存货编码")
    invcode_end: Optional[str] = Field(None, description="结束存货编码")
    invname: Optional[str] = Field(None, description="存货名称关键字")
    batch: Optional[str] = Field(None, description="批号")

# ===================== 调拨单 - 子数据模型 (entry) 数据模型 =====================
class TransvouchEntry(BaseModel):
    """调拨单表体明细"""
    barcode: Optional[str] = Field(None, description="条形码")
    inventorycode: Optional[str] = Field(None, description="存货编码")
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
    quantity: Optional[float] = Field(None, description="数量（主记量数量）")
    cmassunitname: Optional[str] = Field(None, description="主计量单位名称")
    assitantunit: Optional[str] = Field(None, description="辅记量单位")
    assitantunitname: Optional[str] = Field(None, description="辅计量单位名称")
    irate: Optional[float] = Field(None, description="换算率")
    number: Optional[float] = Field(None, description="件数")
    actualcost: Optional[float] = Field(None, description="实际价格")
    actualprice: Optional[float] = Field(None, description="实际金额")
    inposcode: Optional[str] = Field(None, description="调入货位")
    outposcode: Optional[str] = Field(None, description="调出货位")
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
    define34: Optional[float] = Field(None, description="表体自定义项34")
    define35: Optional[float] = Field(None, description="表体自定义项35")
    define36: Optional[str] = Field(None, description="表体自定义项36")
    define37: Optional[str] = Field(None, description="表体自定义项37")
    irowno: Optional[float] = Field(None, description="行号")
    inventoryname: Optional[str] = Field(None, description="存货名称")

# ===================== 获取调拨单列表 数据模型 =====================
class GetTransvouchListInput(BaseModel):
    """获取调拨单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    tvcode_begin: Optional[str] = Field(None, description="起始调拨单据号")
    tvcode_end: Optional[str] = Field(None, description="结束调拨单据号")
    idepcode: Optional[str] = Field(None, description="转入部门编码")
    idepname: Optional[str] = Field(None, description="转入部门名称关键字")
    odepcode: Optional[str] = Field(None, description="转出部门编码")
    odepname: Optional[str] = Field(None, description="转出部门名称关键字")
    irdcode: Optional[str] = Field(None, description="入库类别编码")
    irdname: Optional[str] = Field(None, description="入库类别名称关键字")
    ordcode: Optional[str] = Field(None, description="出库类别编码")
    ordname: Optional[str] = Field(None, description="出库类别名称关键字")
    iwhcode: Optional[str] = Field(None, description="转入仓库编码")
    iwhname: Optional[str] = Field(None, description="转入仓库名称关键字")
    owhcode: Optional[str] = Field(None, description="转出仓库编码")
    owhname: Optional[str] = Field(None, description="转出仓库名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    memory: Optional[str] = Field(None, description="备注关键字")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")


# ===================== 审核调拨单 数据模型 =====================
class VerifyTransvouchInput(BaseModel):
    """审核调拨单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code二选一传入")


# ===================== 弃审调拨单 数据模型 =====================
class UnverifyTransvouchInput(BaseModel):
    """弃审调拨单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个调拨单 数据模型 =====================
class GetTransvouchInput(BaseModel):
    """获取单个调拨单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="调拨单主表标识（必填）")


# ===================== 新增调拨单 数据模型 =====================
class AddTransvouchInput(BaseModel):
    """新增调拨单输入模型"""
    idepcode: Optional[str] = Field(None, description="转入部门编码")
    idepname: Optional[str] = Field(None, description="转入部门名称")
    odepcode: Optional[str] = Field(None, description="转出部门编码")
    odepname: Optional[str] = Field(None, description="转出部门名称")
    irdcode: Optional[str] = Field(None, description="入库类别编码")
    irdname: Optional[str] = Field(None, description="入库类别名称")
    ordcode: Optional[str] = Field(None, description="出库类别编码")
    ordname: Optional[str] = Field(None, description="出库类别名称")
    iwhcode: Optional[str] = Field(None, description="转入仓库编码")
    iwhname: Optional[str] = Field(None, description="转入仓库名称")
    owhcode: Optional[str] = Field(None, description="转出仓库编码")
    owhname: Optional[str] = Field(None, description="转出仓库名称")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称")
    tvcode: Optional[str] = Field(None, description="调拨单据号")
    date: Optional[str] = Field(None, description="单据日期")
    memory: Optional[str] = Field(None, description="备注")
    auditperson: Optional[str] = Field(None, description="审核人")
    auditdate: Optional[str] = Field(None, description="审核日期")
    maker: Optional[str] = Field(None, description="制单人")
    define1: Optional[str] = Field(None, description="自定义字段1")
    define2: Optional[str] = Field(None, description="自定义字段2")
    define3: Optional[str] = Field(None, description="自定义字段3")
    define4: Optional[str] = Field(None, description="自定义字段4")
    define5: Optional[float] = Field(None, description="自定义字段5")
    define6: Optional[str] = Field(None, description="自定义字段6")
    define7: Optional[float] = Field(None, description="自定义字段7")
    define8: Optional[str] = Field(None, description="自定义字段8")
    define9: Optional[str] = Field(None, description="自定义字段9")
    define10: Optional[str] = Field(None, description="自定义字段10")
    define11: Optional[str] = Field(None, description="自定义字段11")
    define12: Optional[str] = Field(None, description="自定义字段12")
    define13: Optional[str] = Field(None, description="自定义字段13")
    define14: Optional[str] = Field(None, description="自定义字段14")
    define15: Optional[float] = Field(None, description="自定义字段15")
    define16: Optional[float] = Field(None, description="自定义字段16")
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
    ordertype: Optional[str] = Field(None, description="订单类型")
    transappcode: Optional[str] = Field(None, description="调拨申请单号")
    csource: Optional[str] = Field(None, description="来源")
    entry: Optional[List[TransvouchEntry]] = Field(None, description="表体明细列表")


# ===================== 获取调拨申请单列表 数据模型 =====================
class GetTransvouchapplyListInput(BaseModel):
    """获取调拨申请单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始调拨申请单号")
    code_end: Optional[str] = Field(None, description="结束调拨申请单号")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    iwhcode: Optional[str] = Field(None, description="转入仓库编码")
    owhcode: Optional[str] = Field(None, description="转出仓库编码")
    personcode: Optional[str] = Field(None, description="业务员编码")
    memory: Optional[str] = Field(None, description="备注关键字")
    state: Optional[str] = Field(None, description="单据状态")


# ===================== 获取单个调拨申请单 数据模型 =====================
class GetTransvouchapplyInput(BaseModel):
    """获取单个调拨申请单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="调拨申请单主表标识（必填）")


# ===================== 获取采购入库单列表 数据模型 =====================
class GetPurchaseInListInput(BaseModel):
    """获取采购入库单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始单据号")
    code_end: Optional[str] = Field(None, description="结束单据号")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    auditdate_begin: Optional[str] = Field(None, description="起始审核日期")
    auditdate_end: Optional[str] = Field(None, description="结束审核日期")
    maker: Optional[str] = Field(None, description="制单人名称关键字")
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    memory: Optional[str] = Field(None, description="备注关键字")
    receivecode: Optional[str] = Field(None, description="收发类型编码")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    warehousename: Optional[str] = Field(None, description="仓库名称关键字")
    receivename: Optional[str] = Field(None, description="收发类型名称关键字")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    purchasetypename: Optional[str] = Field(None, description="采购类型名称关键字")
    bredvouch: Optional[str] = Field(None, description="红蓝标识（1为红字，0为蓝字）")


# ===================== 审核采购入库单 数据模型 =====================
class VerifyPurchaseInInput(BaseModel):
    """审核采购入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")


# ===================== 弃审采购入库单 数据模型 =====================
class UnverifyPurchaseInInput(BaseModel):
    """弃审采购入库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个采购入库单 数据模型 =====================
class GetPurchaseInInput(BaseModel):
    """获取单个采购入库单输入模型"""
    id: str = Field(..., description="单据号（必填）")


# ===================== 新增采购入库单 表体数据模型 =====================
class PurchaseInEntry(BaseModel):
    """采购入库单表体行"""
    inventorycode: str = Field(..., description="存货编码（必填）")
    inventoryname: Optional[str] = Field(None, description="存货")
    inventorystd: Optional[str] = Field(None, description="规格型号")
    quantity: float = Field(..., description="数量（必填）")
    price: Optional[float] = Field(None, description="本币单价")
    cost: Optional[float] = Field(None, description="本币金额")
    ioritaxprice: Optional[float] = Field(None, description="税额")
    iorisum: Optional[float] = Field(None, description="价税合计")
    taxprice: Optional[float] = Field(None, description="本币税额")
    isum: Optional[float] = Field(None, description="本币价税合计")
    ioritaxcost: Optional[float] = Field(None, description="含税单价，传入会自动重新计算相关价格及金额")
    ioricost: Optional[float] = Field(None, description="单价，传入会自动重新计算相关价格及金额。如果传入了含税单价，以含税单价为准自动计算")
    iorimoney: Optional[float] = Field(None, description="金额")
    taxrate: Optional[float] = Field(None, description="税率")
    cmassunitname: Optional[str] = Field(None, description="主计量单位名称")
    serial: Optional[str] = Field(None, description="批号")
    makedate: Optional[str] = Field(None, description="生产日期")
    validdate: Optional[str] = Field(None, description="失效日期")
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
    assitantunitname: Optional[str] = Field(None, description="辅计量单位名称")
    irate: Optional[float] = Field(None, description="换算率")
    number: Optional[float] = Field(None, description="件数")
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
    rowno: Optional[int] = Field(None, description="行号")


# ===================== 新增采购入库单 主数据模型 =====================
class AddPurchaseInInput(BaseModel):
    """新增采购入库单输入模型"""
    code: Optional[str] = Field(None, description="单据编号")
    date: Optional[str] = Field(None, description="制单日期")
    maker: Optional[str] = Field(None, description="制单人名称")
    warehousecode: str = Field(..., description="仓库编码（必填）")
    warehousename: Optional[str] = Field(None, description="仓库名称")
    vendorcode: str = Field(..., description="供货单位编码（必填）")
    vendorabbname: str = Field(..., description="供货单位简称（必填）")
    vendorname: Optional[str] = Field(None, description="供货单位")
    memory: Optional[str] = Field(None, description="备注")
    receivecode: str = Field(..., description="收发类型编码（必填）")
    receivename: Optional[str] = Field(None, description="收发类型")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
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
    taxrate: Optional[float] = Field(None, description="税率")
    entry: Optional[List[PurchaseInEntry]] = Field(None, description="表体明细列表")


# ===================== 获取销售出库单列表 数据模型 =====================

class GetSaleOutListInput(BaseModel):
    """获取销售出库单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    warehousename: Optional[str] = Field(None, description="仓库名称关键字")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    auditdate_begin: Optional[str] = Field(None, description="起始审核日期，格式:yyyy-MM-dd")
    auditdate_end: Optional[str] = Field(None, description="结束审核日期，格式:yyyy-MM-dd")
    code_begin: Optional[str] = Field(None, description="起始收发单据号")
    code_end: Optional[str] = Field(None, description="结束收发单据号")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    memory: Optional[str] = Field(None, description="备注关键字")
    maker: Optional[str] = Field(None, description="制单人关键字")
    bredvouch: Optional[str] = Field(None, description="红蓝标识")
    businesscode: Optional[str] = Field(None, description="对应业务单号")
    timestamp_begin: Optional[str] = Field(None, description="起始时间戳")
    timestamp_end: Optional[str] = Field(None, description="结束时间戳")
    define1: Optional[str] = Field(None, description="自定义项1")
    define2: Optional[str] = Field(None, description="自定义项2")
    define3: Optional[str] = Field(None, description="自定义项3")
    define4_begin: Optional[str] = Field(None, description="起始自定义项4")
    define4_end: Optional[str] = Field(None, description="结束自定义项4")
    define5: Optional[float] = Field(None, description="自定义项5")
    define6_begin: Optional[str] = Field(None, description="起始自定义项6")
    define6_end: Optional[str] = Field(None, description="结束自定义项6")
    define7: Optional[float] = Field(None, description="自定义项7")
    define8: Optional[str] = Field(None, description="自定义项8")
    define9: Optional[str] = Field(None, description="自定义项9")
    define10: Optional[str] = Field(None, description="自定义项10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[float] = Field(None, description="自定义项15")
    define16: Optional[float] = Field(None, description="自定义项16")


# ===================== 审核销售出库单 数据模型 =====================
class VerifySaleOutInput(BaseModel):
    """审核销售出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")


# ===================== 弃审销售出库单 数据模型 =====================
class UnverifySaleOutInput(BaseModel):
    """弃审销售出库单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个销售出库单 数据模型 =====================
class GetSaleOutInput(BaseModel):
    """获取单个销售出库单输入模型"""
    id: str = Field(..., description="收发记录主表标识（必填）")


# ===================== 新增销售出库单 表体数据模型 =====================
class SaleOutEntry(BaseModel):
    """销售出库单表体行"""
    barcode: Optional[str] = Field(None, description="条形码")
    inventorycode: str = Field(..., description="存货编码（必填）")
    free1: Optional[str] = Field(None, description="存货自由项1")
    free2: Optional[str] = Field(None, description="存货自由项2")
    free3: Optional[str] = Field(None, description="存货自由项3")
    free4: Optional[str] = Field(None, description="存货自由项4")
    free5: Optional[str] = Field(None, description="存货自由项5")
    free6: Optional[str] = Field(None, description="存货自由项6")
    free7: Optional[str] = Field(None, description="存货自由项7")
    free8: Optional[str] = Field(None, description="存货自由项8")
    free9: Optional[str] = Field(None, description="存货自由项9")
    free10: Optional[str] = Field(None, description="存货自由项10")
    shouldquantity: Optional[float] = Field(None, description="应发数量")
    shouldnumber: Optional[float] = Field(None, description="应发件数")
    quantity: float = Field(..., description="数量（必填）")
    cmassunitname: Optional[str] = Field(None, description="主计量单位")
    assitantunit: Optional[str] = Field(None, description="库存单位码")
    assitantunitname: Optional[str] = Field(None, description="库存单位")
    irate: Optional[float] = Field(None, description="换算率")
    number: Optional[float] = Field(None, description="件数")
    price: Optional[float] = Field(None, description="单价")
    cost: Optional[float] = Field(None, description="金额")
    serial: Optional[str] = Field(None, description="批号")
    makedate: Optional[str] = Field(None, description="生产日期")
    validdate: Optional[str] = Field(None, description="失效日期")
    define22: Optional[str] = Field(None, description="表体自定义项1")
    define23: Optional[str] = Field(None, description="表体自定义项2")
    define24: Optional[str] = Field(None, description="表体自定义项3")
    define25: Optional[str] = Field(None, description="表体自定义项4")
    define26: Optional[float] = Field(None, description="表体自定义项5")
    define27: Optional[float] = Field(None, description="表体自定义项6")
    define28: Optional[str] = Field(None, description="表体自定义项7")
    define29: Optional[str] = Field(None, description="表体自定义项8")
    define30: Optional[str] = Field(None, description="表体自定义项9")
    define31: Optional[str] = Field(None, description="表体自定义项10")
    define32: Optional[str] = Field(None, description="表体自定义项11")
    define33: Optional[str] = Field(None, description="表体自定义项12")
    define34: Optional[float] = Field(None, description="表体自定义项13")
    define35: Optional[float] = Field(None, description="表体自定义项14")
    define36: Optional[str] = Field(None, description="表体自定义项15")
    define37: Optional[str] = Field(None, description="表体自定义项16")
    rowno: Optional[int] = Field(None, description="行号")
    subconsignmentcode: Optional[str] = Field(None, description="发货单号")
    ordercode: Optional[str] = Field(None, description="订单号")


# ===================== 新增销售出库单 主数据模型 =====================
class AddSaleOutInput(BaseModel):
    """新增销售出库单输入模型"""
    businesstype: Optional[str] = Field(None, description="业务类型")
    source: Optional[str] = Field(None, description="单据来源")
    businesscode: Optional[str] = Field(None, description="对应业务单号")
    warehousecode: str = Field(..., description="仓库编码（必填）")
    warehousename: Optional[str] = Field(None, description="仓库名称")
    date: Optional[str] = Field(None, description="单据日期")
    code: Optional[str] = Field(None, description="单据号")
    receivecode: Optional[str] = Field(None, description="收发类别编码")
    receivename: Optional[str] = Field(None, description="收发类别名称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="业务员编码")
    saletypecode: Optional[str] = Field(None, description="销售类型编码")
    customercode: Optional[str] = Field(None, description="客户编码")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    arrivedate: Optional[str] = Field(None, description="到货日期")
    memory: Optional[str] = Field(None, description="备注")
    maker: Optional[str] = Field(None, description="制单人")
    define1: Optional[str] = Field(None, description="自定义项1")
    define2: Optional[str] = Field(None, description="自定义项2")
    define3: Optional[str] = Field(None, description="自定义项3")
    define4: Optional[str] = Field(None, description="自定义项4")
    define5: Optional[float] = Field(None, description="自定义项5")
    define6: Optional[str] = Field(None, description="自定义项6")
    define7: Optional[float] = Field(None, description="自定义项7")
    define8: Optional[str] = Field(None, description="自定义项8")
    define9: Optional[str] = Field(None, description="自定义项9")
    define10: Optional[str] = Field(None, description="自定义项10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[float] = Field(None, description="自定义项15")
    define16: Optional[float] = Field(None, description="自定义项16")
    entry: Optional[List[SaleOutEntry]] = Field(None, description="表体明细列表")




# ===================== 产成品入库单列表查询 Tool函数 =====================
def u8_productinlist_batch_get_tool(input_data: GetProductinListInput, client: U8OpenAPIClient) -> str:
    """
    获取产成品入库单列表，支持多条件筛选查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/productinlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取产成品入库单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取产成品入库单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "productinlist": result.get("productinlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核产成品入库单 Tool函数 =====================
def u8_productin_verify_tool(input_data: VerifyProductinInput, client: U8OpenAPIClient) -> str:
    """
    审核一张产成品入库单。
    """
    request_body: dict = {
        "productin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/productin/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "产成品入库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "产成品入库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审产成品入库单 Tool函数 =====================
def u8_productin_unverify_tool(input_data: UnverifyProductinInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张产成品入库单。
    """
    request_body: dict = {
        "productin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/productin/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "产成品入库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "产成品入库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个产成品入库单 Tool函数 =====================
def u8_productin_get_tool(input_data: GetProductinInput, client: U8OpenAPIClient) -> str:
    """
    获取单个产成品入库单详细信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/productin/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取产成品入库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取产成品入库单成功",
            "data": result.get("productin"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增产成品入库单 Tool函数 =====================
def u8_productin_add_tool(input_data: AddProductinInput, client: U8OpenAPIClient) -> str:
    """
    新增一张产成品入库单。
    """
    request_body: dict = {
        "productin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/productin/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "产成品入库单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "产成品入库单新增成功",
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


# ===================== 其他入库单列表查询 Tool函数 =====================
def u8_otherinlist_batch_get_tool(input_data: GetOtherinListInput, client: U8OpenAPIClient) -> str:
    """
    获取其它入库单列表，支持多条件筛选查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherinlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其它入库单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其它入库单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "otherinlist": result.get("otherinlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核其他入库单 Tool函数 =====================
def u8_otherin_verify_tool(input_data: VerifyOtherinInput, client: U8OpenAPIClient) -> str:
    """
    审核一张其他入库单。
    """
    request_body: dict = {
        "otherin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherin/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他入库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他入库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审其他入库单 Tool函数 =====================
def u8_otherin_unverify_tool(input_data: UnverifyOtherinInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张其他入库单。
    """
    request_body: dict = {
        "otherin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherin/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他入库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他入库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取其他入库单待办任务 Tool函数 =====================
def u8_otherin_tasks_tool(input_data: GetOtherinTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取其他入库单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherin/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他入库单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他入库单待办任务成功",
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


# ===================== 查看其他入库单审批进程 Tool函数 =====================
def u8_otherin_history_tool(input_data: GetOtherinHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看其他入库单审批进程，获取单据的审批历史记录。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherin/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他入库单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他入库单审批进程成功",
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


# ===================== 获取单个其它入库单 Tool函数 =====================
def u8_otherin_get_tool(input_data: GetOtherinInput, client: U8OpenAPIClient) -> str:
    """
    获取单张其它入库单详细信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherin/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其它入库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其它入库单成功",
            "data": result.get("otherin"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取其他入库单是否启用工作流 Tool函数 =====================
def u8_otherin_flowenabled_tool(input_data: GetOtherinFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取其他入库单是否启用工作流。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherin/flowenabled"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他入库单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他入库单工作流启用状态成功",
            "data": {
                "flowenabled": result.get("flowenabled")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取其他入库单工作流按钮是否可用状态 Tool函数 =====================
def u8_otherin_buttonstate_tool(input_data: GetOtherinButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取其他入库单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。
    只支持12.0版本，且需要打最新的WF补丁。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherin/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他入库单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他入库单工作流按钮状态成功",
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


# ===================== 审核其他入库单(工作流) Tool函数 =====================
def u8_otherin_audit_tool(input_data: AuditOtherinInput, client: U8OpenAPIClient) -> str:
    """
    审核其他入库单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "otherin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherin/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他入库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他入库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增其它入库单 Tool函数 =====================
def u8_otherin_add_tool(input_data: AddOtherinInput, client: U8OpenAPIClient) -> str:
    """
    新增一张其它入库单。
    """
    request_body: dict = {
        "otherin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherin/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其它入库单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其它入库单新增成功",
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


# ===================== 弃审其他入库单(工作流) Tool函数 =====================
def u8_otherin_abandon_tool(input_data: AbandonOtherinInput, client: U8OpenAPIClient) -> str:
    """
    弃审其他入库单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "otherin": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherin/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他入库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他入库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)



# ===================== 其他出库单列表查询 Tool函数 =====================
def u8_otheroutlist_batch_get_tool(input_data: GetOtheroutListInput, client: U8OpenAPIClient) -> str:
    """
    获取其它出库单列表信息，支持多条件筛选查询和分页。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otheroutlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其它出库单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其它出库单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "otheroutlist": result.get("otheroutlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核其他出库单 Tool函数 =====================
def u8_otherout_verify_tool(input_data: VerifyOtheroutInput, client: U8OpenAPIClient) -> str:
    """
    审核一张其他出库单。
    """
    request_body: dict = {
        "otherout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherout/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他出库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他出库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审其他出库单 Tool函数 =====================
def u8_otherout_unverify_tool(input_data: UnverifyOtheroutInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张其他出库单。
    """
    request_body: dict = {
        "otherout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherout/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他出库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他出库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取其他出库单待办任务 Tool函数 =====================
def u8_otherout_tasks_tool(input_data: GetOtheroutTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取其他出库单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherout/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他出库单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他出库单待办任务成功",
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


# ===================== 查看其他出库单审批进程 Tool函数 =====================
def u8_otherout_history_tool(input_data: GetOtheroutHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看其他出库单审批进程，获取单据的审批历史记录。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherout/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他出库单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他出库单审批进程成功",
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


# ===================== 获取单个其它出库单 Tool函数 =====================
def u8_otherout_get_tool(input_data: GetOtheroutInput, client: U8OpenAPIClient) -> str:
    """
    获取单张其它出库单详细信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherout/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其它出库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其它出库单成功",
            "data": result.get("otherout"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取其他出库单是否启用工作流 Tool函数 =====================
def u8_otherout_flowenabled_tool(input_data: GetOtheroutFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取其他出库单是否启用工作流。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherout/flowenabled"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他出库单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他出库单工作流启用状态成功",
            "data": {
                "flowenabled": result.get("flowenabled")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取其他出库单工作流按钮是否可用状态 Tool函数 =====================
def u8_otherout_buttonstate_tool(input_data: GetOtheroutButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取其他出库单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。
    只支持12.0版本，且需要打最新的WF补丁。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/otherout/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取其他出库单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取其他出库单工作流按钮状态成功",
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


# ===================== 审核其他出库单(工作流) Tool函数 =====================
def u8_otherout_audit_tool(input_data: AuditOtheroutInput, client: U8OpenAPIClient) -> str:
    """
    审核其他出库单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "otherout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherout/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他出库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他出库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增其它出库单 Tool函数 =====================
def u8_otherout_add_tool(input_data: AddOtheroutInput, client: U8OpenAPIClient) -> str:
    """
    新增一张其它出库单。
    """
    request_body: dict = {
        "otherout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherout/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其它出库单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其它出库单新增成功",
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


# ===================== 弃审其他出库单(工作流) Tool函数 =====================
def u8_otherout_abandon_tool(input_data: AbandonOtheroutInput, client: U8OpenAPIClient) -> str:
    """
    弃审其他出库单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "otherout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/otherout/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "其他出库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "其他出库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取库存结账状态 Tool函数 =====================
def u8_mendst_batch_get_tool(input_data: GetMendstListInput, client: U8OpenAPIClient) -> str:
    """
    批量获取库存结账状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/mendst/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取库存结账状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取库存结账状态成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "mendst": result.get("mendst", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取材料出库单列表 Tool函数 =====================
def u8_materialout_list_tool(input_data: GetMaterialoutListInput, client: U8OpenAPIClient) -> str:
    """
    获取材料出库单列表，支持多条件筛选查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/materialoutlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取材料出库单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取材料出库单列表成功",
            "data": {
                "materialoutlist": result.get("materialoutlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核材料出库单 Tool函数 =====================
def u8_materialout_verify_tool(input_data: VerifyMaterialoutInput, client: U8OpenAPIClient) -> str:
    """
    审核一张材料出库单。
    """
    request_body: dict = {
        "materialout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/materialout/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "审核材料出库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "审核材料出库单成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审材料出库单 Tool函数 =====================
def u8_materialout_unverify_tool(input_data: UnverifyMaterialoutInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张材料出库单。
    """
    request_body: dict = {
        "materialout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/materialout/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "弃审材料出库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "弃审材料出库单成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取材料出库单待办任务 Tool函数 =====================
def u8_materialout_tasks_tool(input_data: GetMaterialoutTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取材料出库单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/materialout/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取材料出库单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取材料出库单待办任务成功",
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


# ===================== 获取材料出库单审批进程 Tool函数 =====================
def u8_materialout_history_tool(input_data: GetMaterialoutHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看材料出库单审批进程，获取单据的审批历史记录。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/materialout/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取材料出库单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取材料出库单审批进程成功",
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


# ===================== 获取单个材料出库单 Tool函数 =====================
def u8_materialout_get_tool(input_data: GetMaterialoutInput, client: U8OpenAPIClient) -> str:
    """
    通过收发记录主表标识获取用友U8中的单个材料出库单详细信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/materialout/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取材料出库单详情失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取材料出库单详情成功",
            "data": {
                "materialout": result.get("materialout")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取材料出库单是否启用工作流 Tool函数 =====================
def u8_materialout_flowenabled_tool(input_data: GetMaterialoutFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取材料出库单是否启用工作流。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/materialout/flowenabled"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取材料出库单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取材料出库单工作流启用状态成功",
            "data": {
                "flowenabled": result.get("flowenabled")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取材料出库单工作流按钮是否可用状态 Tool函数 =====================
def u8_materialout_buttonstate_tool(input_data: GetMaterialoutButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取材料出库单工作流按钮是否可用状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/materialout/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取材料出库单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取材料出库单工作流按钮状态成功",
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


# ===================== 审核材料出库单（工作流） Tool函数 =====================
def u8_materialout_audit_tool(input_data: AuditMaterialoutInput, client: U8OpenAPIClient) -> str:
    """
    审核材料出库单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "materialout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/materialout/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "材料出库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "材料出库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增材料出库单 Tool函数 =====================
def u8_materialout_add_tool(input_data: AddMaterialoutInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增一张材料出库单。
    """
    request_body: dict = {
        "materialout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/materialout/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增材料出库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "材料出库单新增成功",
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


# ===================== 弃审材料出库单（工作流） Tool函数 =====================
def u8_materialout_abandon_tool(input_data: AbandonMaterialoutInput, client: U8OpenAPIClient) -> str:
    """
    弃审材料出库单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "materialout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/materialout/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "材料出库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "材料出库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 现存量查询 Tool函数 =====================
def u8_currentstock_batch_get_tool(input_data: GetCurrentstockInput, client: U8OpenAPIClient) -> str:
    """
    现存量查询，支持分页和多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/currentstock/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "现存量查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "现存量查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "currentstock": result.get("currentstock", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取调拨单列表 Tool函数 =====================
def u8_transvouch_list_tool(input_data: GetTransvouchListInput, client: U8OpenAPIClient) -> str:
    """
    获取调拨单列表，支持多条件筛选查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/transvouchlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调拨单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调拨单列表成功",
            "data": {
                "transvouchlist": result.get("transvouchlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核调拨单 Tool函数 =====================
def u8_transvouch_verify_tool(input_data: VerifyTransvouchInput, client: U8OpenAPIClient) -> str:
    """
    审核一张调拨单。
    """
    request_body: dict = {
        "transvouch": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/transvouch/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "审核调拨单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "审核调拨单成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审调拨单 Tool函数 =====================
def u8_transvouch_unverify_tool(input_data: UnverifyTransvouchInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张调拨单。
    """
    request_body: dict = {
        "transvouch": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/transvouch/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "弃审调拨单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "弃审调拨单成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个调拨单 Tool函数 =====================
def u8_transvouch_get_tool(input_data: GetTransvouchInput, client: U8OpenAPIClient) -> str:
    """
    通过调拨单主表标识获取用友U8中的单个调拨单详细信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/transvouch/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调拨单详情失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调拨单详情成功",
            "data": {
                "transvouch": result.get("transvouch")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增调拨单 Tool函数 =====================
def u8_transvouch_add_tool(input_data: AddTransvouchInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增一张调拨单。
    """
    request_body: dict = {
        "transvouch": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/transvouch/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增调拨单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "调拨单新增成功",
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


# ===================== 获取调拨申请单列表 Tool函数 =====================
def u8_transvouchapply_list_tool(input_data: GetTransvouchapplyListInput, client: U8OpenAPIClient) -> str:
    """
    获取调拨申请单列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/transvouchapplylist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调拨申请单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调拨申请单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "transvouchapplylist": result.get("transvouchapplylist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个调拨申请单 Tool函数 =====================
def u8_transvouchapply_get_tool(input_data: GetTransvouchapplyInput, client: U8OpenAPIClient) -> str:
    """
    通过调拨申请单主表标识获取用友U8中的单个调拨申请单详细信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/transvouchapply/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调拨申请单详情失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调拨申请单详情成功",
            "data": {
                "transvouchapply": result.get("transvouchapply")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取采购入库单列表 Tool函数 =====================
def u8_purchaseinlist_batch_get_tool(input_data: GetPurchaseInListInput, client: U8OpenAPIClient) -> str:
    """
    获取采购入库单列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseinlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购入库单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购入库单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purchaseinlist": result.get("purchaseinlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核采购入库单 Tool函数 =====================
def u8_purchasein_verify_tool(input_data: VerifyPurchaseInInput, client: U8OpenAPIClient) -> str:
    """
    审核一张采购入库单。
    执行审核动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "purchasein": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasein/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购入库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购入库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审采购入库单 Tool函数 =====================
def u8_purchasein_unverify_tool(input_data: UnverifyPurchaseInInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张采购入库单。
    """
    request_body: dict = {
        "purchasein": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasein/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购入库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购入库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个采购入库单 Tool函数 =====================
def u8_purchasein_get_tool(input_data: GetPurchaseInInput, client: U8OpenAPIClient) -> str:
    """
    获取单个采购入库单详细信息。
    """
    params = {
        "id": input_data.id
    }

    api_path = "/api/purchasein/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购入库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购入库单成功",
            "data": result.get("purchasein"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增采购入库单 Tool函数 =====================
def u8_purchasein_add_tool(input_data: AddPurchaseInInput, client: U8OpenAPIClient) -> str:
    """
    新增一张采购入库单。
    """
    request_body: dict = {
        "purchasein": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasein/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购入库单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购入库单新增成功",
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


# ===================== 获取销售出库单列表 Tool函数 =====================
def u8_saleoutlist_batch_get_tool(input_data: GetSaleOutListInput, client: U8OpenAPIClient) -> str:
    """
    获取销售出库单列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleoutlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取销售出库单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取销售出库单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "saleoutlist": result.get("saleoutlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核销售出库单 Tool函数 =====================
def u8_saleout_verify_tool(input_data: VerifySaleOutInput, client: U8OpenAPIClient) -> str:
    """
    审核一张销售出库单。
    执行审核动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "saleout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/saleout/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "销售出库单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "销售出库单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审销售出库单 Tool函数 =====================
def u8_saleout_unverify_tool(input_data: UnverifySaleOutInput, client: U8OpenAPIClient) -> str:
    """
    弃审一张销售出库单。
    """
    request_body: dict = {
        "saleout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/saleout/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "销售出库单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "销售出库单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个销售出库单 Tool函数 =====================
def u8_saleout_get_tool(input_data: GetSaleOutInput, client: U8OpenAPIClient) -> str:
    """
    获取单个销售出库单详细信息。
    """
    params = {
        "id": input_data.id
    }

    api_path = "/api/saleout/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取销售出库单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取销售出库单成功",
            "data": result.get("saleout"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增销售出库单 Tool函数 =====================
def u8_saleout_add_tool(input_data: AddSaleOutInput, client: U8OpenAPIClient) -> str:
    """
    新增一张销售出库单。
    """
    request_body: dict = {
        "saleout": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/saleout/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "销售出库单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "销售出库单新增成功",
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



# ===================== 产成品入库单列表查询 Schema定义 =====================
U8_PRODUCTINLIST_BATCH_GET_SCHEMA = {
    "name": "u8_productinlist_batch_get",
    "description": "获取产成品入库单列表，支持按单据号、日期、仓库、收发类别、部门等多条件筛选查询",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始单据号"},
            "code_end": {"type": "string", "description": "结束单据号"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "warehousecode": {"type": "string", "description": "仓库编码"},
            "warehousename": {"type": "string", "description": "仓库名称关键字"},
            "receivecode": {"type": "string", "description": "收发类别编码"},
            "receivename": {"type": "string", "description": "收发类别名称关键字"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "handler": {"type": "string", "description": "审核人关键字"},
            "memory": {"type": "string", "description": "备注关键字"},
            "maker": {"type": "string", "description": "制单人关键字"}
        },
        "required": []
    }
}

# ===================== 审核产成品入库单 Schema定义 =====================
U8_PRODUCTIN_VERIFY_SCHEMA = {
    "name": "u8_productin_verify",
    "description": "审核一张产成品入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审产成品入库单 Schema定义 =====================
U8_PRODUCTIN_UNVERIFY_SCHEMA = {
    "name": "u8_productin_unverify",
    "description": "弃审一张产成品入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个产成品入库单 Schema定义 =====================
U8_PRODUCTIN_GET_SCHEMA = {
    "name": "u8_productin_get",
    "description": "获取单个产成品入库单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "产成品入库单编码（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 新增产成品入库单 Schema定义 =====================
U8_PRODUCTIN_ADD_SCHEMA = {
    "name": "u8_productin_add",
    "description": "新增一张产成品入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据编号"},
            "date": {"type": "string", "description": "制单日期"},
            "maker": {"type": "string", "description": "制单人名称"},
            "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
            "memory": {"type": "string", "description": "备注"},
            "receivecode": {"type": "string", "description": "收发类型编码（必填）"},
            "departmentcode": {"type": "string", "description": "部门编码（必填）"},
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
            "entry": {
                "type": "array",
                "description": "表体信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "quantity": {"type": "number", "description": "数量"},
                        "assitantunit": {"type": "string", "description": "辅记量单位编码"},
                        "irate": {"type": "number", "description": "换算率"},
                        "number": {"type": "number", "description": "件数"},
                        "price": {"type": "number", "description": "单价"},
                        "cost": {"type": "number", "description": "金额"},
                        "serial": {"type": "string", "description": "批号"},
                        "makedate": {"type": "string", "description": "生产日期"},
                        "validdate": {"type": "string", "description": "失效日期"},
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
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        "rowno": {"type": "number", "description": "行号"}
                    }
                }
            }
        },
        "required": ["warehousecode", "receivecode", "departmentcode"]
    }
}



# ===================== 其他入库单列表查询 Schema定义 =====================
U8_OTHERINLIST_BATCH_GET_SCHEMA = {
    "name": "u8_otherinlist_batch_get",
    "description": "获取其它入库单列表，支持按单据号、日期、制单人、审核人、业务类型、仓库、部门等多条件筛选查询",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始制单日期"},
            "date_end": {"type": "string", "description": "结束制单日期"},
            "maker": {"type": "string", "description": "制单人名称关键字"},
            "handler": {"type": "string", "description": "审核人关键字"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "businesscode": {"type": "string", "description": "业务编码"},
            "warehousecode": {"type": "string", "description": "仓库编码"},
            "warehousename": {"type": "string", "description": "仓库名称关键字"},
            "memory": {"type": "string", "description": "备注关键字"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "auditdate_begin": {"type": "string", "description": "起始审核日期"},
            "auditdate_end": {"type": "string", "description": "结束审核日期"},
            "cvoucherstate": {"type": "string", "description": "状态"}
        },
        "required": []
    }
}

# ===================== 审核其他入库单 Schema定义 =====================
U8_OTHERIN_VERIFY_SCHEMA = {
    "name": "u8_otherin_verify",
    "description": "审核一张其他入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审其他入库单 Schema定义 =====================
U8_OTHERIN_UNVERIFY_SCHEMA = {
    "name": "u8_otherin_unverify",
    "description": "弃审一张其他入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取其他入库单待办任务 Schema定义 =====================
U8_OTHERIN_TASKS_SCHEMA = {
    "name": "u8_otherin_tasks",
    "description": "获取其他入库单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "string", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "integer", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "integer", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "integer", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "integer", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        },
        "required": []
    }
}

# ===================== 查看其他入库单审批进程 Schema定义 =====================
U8_OTHERIN_HISTORY_SCHEMA = {
    "name": "u8_otherin_history",
    "description": "查看其他入库单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "user_id": {"type": "string", "description": "审批人(用户编码)，user_id与person_code输入一个参数即可"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个其它入库单 Schema定义 =====================
U8_OTHERIN_GET_SCHEMA = {
    "name": "u8_otherin_get",
    "description": "获取单张其它入库单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "其他入库单编码"}
        },
        "required": []
    }
}

# ===================== 获取其他入库单是否启用工作流 Schema定义 =====================
U8_OTHERIN_FLOWENABLED_SCHEMA = {
    "name": "u8_otherin_flowenabled",
    "description": "获取其他入库单是否启用工作流",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"}
        },
        "required": []
    }
}

# ===================== 获取其他入库单工作流按钮是否可用状态 Schema定义 =====================
U8_OTHERIN_BUTTONSTATE_SCHEMA = {
    "name": "u8_otherin_buttonstate",
    "description": "获取其他入库单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 审核其他入库单(工作流) Schema定义 =====================
U8_OTHERIN_AUDIT_SCHEMA = {
    "name": "u8_otherin_audit",
    "description": "审核其他入库单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "opinion": {"type": "string", "description": "审批意见"},
            "agree": {"type": "integer", "description": "是否同意(1=同意;0=不同意)（必填）"}
        },
        "required": ["voucher_code", "agree"]
    }
}

# ===================== 新增其它入库单 Schema定义 =====================
U8_OTHERIN_ADD_SCHEMA = {
    "name": "u8_otherin_add",
    "description": "新增一张其它入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据编号"},
            "date": {"type": "string", "description": "制单日期"},
            "maker": {"type": "string", "description": "制单人名称"},
            "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
            "warehousename": {"type": "string", "description": "仓库名称"},
            "memory": {"type": "string", "description": "备注"},
            "receivecode": {"type": "string", "description": "收发类型编码（必填）"},
            "receivename": {"type": "string", "description": "收发类型"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
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
            "entry": {
                "type": "array",
                "description": "表体信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "inventoryname": {"type": "string", "description": "存货"},
                        "inventorystd": {"type": "string", "description": "规格型号"},
                        "quantity": {"type": "number", "description": "数量"},
                        "price": {"type": "number", "description": "单价"},
                        "cost": {"type": "number", "description": "金额（必填）"},
                        "cmassunitname": {"type": "string", "description": "主计量单位名称"},
                        "serial": {"type": "string", "description": "批号"},
                        "makedate": {"type": "string", "description": "生产日期"},
                        "validdate": {"type": "string", "description": "失效日期"},
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
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        "rowno": {"type": "number", "description": "行号"}
                    }
                }
            }
        },
        "required": ["warehousecode", "receivecode"]
    }
}

# ===================== 弃审其他入库单(工作流) Schema定义 =====================
U8_OTHERIN_ABANDON_SCHEMA = {
    "name": "u8_otherin_abandon",
    "description": "弃审其他入库单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "opinion": {"type": "string", "description": "审批意见"}
        },
        "required": ["voucher_code"]
    }
}



# ===================== 其他出库单列表查询 Schema定义 =====================
U8_OTHEROUTLIST_BATCH_GET_SCHEMA = {
    "name": "u8_otheroutlist_batch_get",
    "description": "获取其它出库单列表信息，支持按单据号、日期、状态、仓库、制单人、部门等多条件筛选查询和分页",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始制单日期，格式:yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束制单日期，格式:yyyy-MM-dd"},
            "auditdate_begin": {"type": "string", "description": "起始审核日期，格式:yyyy-MM-dd"},
            "auditdate_end": {"type": "string", "description": "结束审核日期，格式:yyyy-MM-dd"},
            "state": {"type": "string", "description": "单据状态"},
            "warehousecode": {"type": "string", "description": "仓库编码"},
            "warehousename": {"type": "string", "description": "仓库名称"},
            "maker": {"type": "string", "description": "制单人"},
            "departmentcode": {"type": "string", "description": "部门编码，可以通过api/department获取"},
            "departmentname": {"type": "string", "description": "部门名称关键字，可以通过api/department获取"},
            "memory": {"type": "string", "description": "备注关键字"}
        },
        "required": []
    }
}

# ===================== 审核其他出库单 Schema定义 =====================
U8_OTHEROUT_VERIFY_SCHEMA = {
    "name": "u8_otherout_verify",
    "description": "审核一张其他出库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审其他出库单 Schema定义 =====================
U8_OTHEROUT_UNVERIFY_SCHEMA = {
    "name": "u8_otherout_unverify",
    "description": "弃审一张其他出库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取其他出库单待办任务 Schema定义 =====================
U8_OTHEROUT_TASKS_SCHEMA = {
    "name": "u8_otherout_tasks",
    "description": "获取其他出库单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "string", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "integer", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "integer", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "integer", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "integer", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        },
        "required": []
    }
}

# ===================== 查看其他出库单审批进程 Schema定义 =====================
U8_OTHEROUT_HISTORY_SCHEMA = {
    "name": "u8_otherout_history",
    "description": "查看其他出库单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "user_id": {"type": "string", "description": "审批人(用户编码)，user_id与person_code输入一个参数即可"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个其它出库单 Schema定义 =====================
U8_OTHEROUT_GET_SCHEMA = {
    "name": "u8_otherout_get",
    "description": "获取单张其它出库单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 获取其他出库单是否启用工作流 Schema定义 =====================
U8_OTHEROUT_FLOWENABLED_SCHEMA = {
    "name": "u8_otherout_flowenabled",
    "description": "获取其他出库单是否启用工作流",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"}
        },
        "required": []
    }
}

# ===================== 获取其他出库单工作流按钮是否可用状态 Schema定义 =====================
U8_OTHEROUT_BUTTONSTATE_SCHEMA = {
    "name": "u8_otherout_buttonstate",
    "description": "获取其他出库单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 审核其他出库单(工作流) Schema定义 =====================
U8_OTHEROUT_AUDIT_SCHEMA = {
    "name": "u8_otherout_audit",
    "description": "审核其他出库单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "opinion": {"type": "string", "description": "审批意见"},
            "agree": {"type": "integer", "description": "是否同意(1=同意;0=不同意)（必填）"}
        },
        "required": ["voucher_code", "agree"]
    }
}

# ===================== 新增其它出库单 Schema定义 =====================
U8_OTHEROUT_ADD_SCHEMA = {
    "name": "u8_otherout_add",
    "description": "新增一张其它出库单",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据编号"},
            "date": {"type": "string", "description": "制单日期"},
            "maker": {"type": "string", "description": "制单人名称"},
            "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
            "warehousename": {"type": "string", "description": "仓库名称"},
            "memory": {"type": "string", "description": "备注"},
            "receivecode": {"type": "string", "description": "收发类型编码（必填）"},
            "receivename": {"type": "string", "description": "收发类型"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
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
            "entry": {
                "type": "array",
                "description": "表体信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "inventoryname": {"type": "string", "description": "存货"},
                        "inventorystd": {"type": "string", "description": "规格型号"},
                        "quantity": {"type": "number", "description": "数量"},
                        "price": {"type": "number", "description": "单价"},
                        "cost": {"type": "number", "description": "金额（必填）"},
                        "cmassunitname": {"type": "string", "description": "主计量单位名称"},
                        "serial": {"type": "string", "description": "批号"},
                        "makedate": {"type": "string", "description": "生产日期"},
                        "validdate": {"type": "string", "description": "失效日期"},
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
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        "rowno": {"type": "number", "description": "行号"}
                    }
                }
            }
        },
        "required": ["warehousecode", "receivecode"]
    }
}

# ===================== 弃审其他出库单(工作流) Schema定义 =====================
U8_OTHEROUT_ABANDON_SCHEMA = {
    "name": "u8_otherout_abandon",
    "description": "弃审其他出库单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "opinion": {"type": "string", "description": "审批意见"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 批量获取库存结账状态 Schema定义 =====================
U8_MENDST_BATCH_GET_SCHEMA = {
    "name": "u8_mendst_batch_get",
    "description": "批量获取库存结账状态",
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



# ===================== 获取材料出库单列表 Schema定义 =====================
U8_MATERIALOUT_LIST_SCHEMA = {
    "name": "u8_materialout_list",
    "description": "获取材料出库单列表，支持仓库、日期、部门、业务类型、供应商等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "warehousecode": {"type": "string", "description": "仓库编码"},
            "warehousename": {"type": "string", "description": "仓库名称关键字"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "auditdate_begin": {"type": "string", "description": "起始审核日期，格式:yyyy-MM-dd"},
            "auditdate_end": {"type": "string", "description": "结束审核日期，格式:yyyy-MM-dd"},
            "code_begin": {"type": "string", "description": "起始出库单号"},
            "code_end": {"type": "string", "description": "结束出库单号"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "memory": {"type": "string", "description": "备注关键字"},
            "maker": {"type": "string", "description": "制单人关键字"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "source": {"type": "string", "description": "单据来源"},
            "cmpocode": {"type": "string", "description": "生产订单号"},
            "serial": {"type": "string", "description": "生产批号"},
            "businesscode": {"type": "string", "description": "对应业务单号"},
            "receivecode": {"type": "string", "description": "收发类别编码"},
            "receivename": {"type": "string", "description": "收发类别名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "vendorabbname": {"type": "string", "description": "供应商简称关键字"},
            "handler": {"type": "string", "description": "审核人关键字"},
            "define1": {"type": "string", "description": "自定义项1"},
            "define2": {"type": "string", "description": "自定义项2"},
            "define3": {"type": "string", "description": "自定义项3"},
            "define4_begin": {"type": "string", "description": "起始自定义项4"},
            "define4_end": {"type": "string", "description": "结束自定义项4"},
            "define5": {"type": "number", "description": "自定义项5"},
            "define6_begin": {"type": "string", "description": "起始自定义项6"},
            "define6_end": {"type": "string", "description": "结束自定义项6"},
            "define7": {"type": "number", "description": "自定义项7"},
            "define8": {"type": "string", "description": "自定义项8"},
            "define9": {"type": "string", "description": "自定义项9"},
            "define10": {"type": "string", "description": "自定义项10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "number", "description": "自定义项15"},
            "define16": {"type": "number", "description": "自定义项16"}
        },
        "required": []
    }
}

# ===================== 审核材料出库单 Schema定义 =====================
U8_MATERIALOUT_VERIFY_SCHEMA = {
    "name": "u8_materialout_verify",
    "description": "审核一张材料出库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审材料出库单 Schema定义 =====================
U8_MATERIALOUT_UNVERIFY_SCHEMA = {
    "name": "u8_materialout_unverify",
    "description": "弃审一张材料出库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取材料出库单待办任务 Schema定义 =====================
U8_MATERIALOUT_TASKS_SCHEMA = {
    "name": "u8_materialout_tasks",
    "description": "获取材料出库单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "number", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "number", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "number", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "number", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "number", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        },
        "required": []
    }
}

# ===================== 获取材料出库单审批进程 Schema定义 =====================
U8_MATERIALOUT_HISTORY_SCHEMA = {
    "name": "u8_materialout_history",
    "description": "查看材料出库单审批进程，获取单据的审批历史记录",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "user_id": {"type": "string", "description": "审批人(用户编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个材料出库单 Schema定义 =====================
U8_MATERIALOUT_GET_SCHEMA = {
    "name": "u8_materialout_get",
    "description": "通过收发记录主表标识获取单个材料出库单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "收发记录主表标识（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 获取材料出库单是否启用工作流 Schema定义 =====================
U8_MATERIALOUT_FLOWENABLED_SCHEMA = {
    "name": "u8_materialout_flowenabled",
    "description": "获取材料出库单是否启用工作流",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"}
        },
        "required": []
    }
}

# ===================== 获取材料出库单工作流按钮是否可用状态 Schema定义 =====================
U8_MATERIALOUT_BUTTONSTATE_SCHEMA = {
    "name": "u8_materialout_buttonstate",
    "description": "获取材料出库单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 审核材料出库单（工作流） Schema定义 =====================
U8_MATERIALOUT_AUDIT_SCHEMA = {
    "name": "u8_materialout_audit",
    "description": "审核材料出库单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "opinion": {"type": "string", "description": "审批意见"},
            "agree": {"type": "number", "description": "是否同意(1=同意;0=不同意)（必填）"}
        },
        "required": ["voucher_code", "agree"]
    }
}

# ===================== 新增材料出库单 Schema定义 =====================
U8_MATERIALOUT_ADD_SCHEMA = {
    "name": "u8_materialout_add",
    "description": "新增一张材料出库单，支持单据头信息和表体明细录入",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据编号"},
            "date": {"type": "string", "description": "制单日期"},
            "maker": {"type": "string", "description": "制单人名称"},
            "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
            "memory": {"type": "string", "description": "备注"},
            "receivecode": {"type": "string", "description": "收发类型编码（必填）"},
            "departmentcode": {"type": "string", "description": "部门编码"},
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
            "entry": {
                "type": "array",
                "description": "表体明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "quantity": {"type": "number", "description": "数量"},
                        "assitantunit": {"type": "string", "description": "辅计量单位编码"},
                        "irate": {"type": "number", "description": "换算率"},
                        "number": {"type": "number", "description": "件数"},
                        "price": {"type": "number", "description": "单价"},
                        "cost": {"type": "number", "description": "金额"},
                        "serial": {"type": "string", "description": "批号"},
                        "makedate": {"type": "string", "description": "生产日期"},
                        "validdate": {"type": "string", "description": "失效日期"},
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
                        "memory": {"type": "string", "description": "表体备注"},
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
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        "rowno": {"type": "number", "description": "行号"}
                    }
                }
            }
        },
        "required": ["warehousecode", "receivecode"]
    }
}

# ===================== 弃审材料出库单（工作流） Schema定义 =====================
U8_MATERIALOUT_ABANDON_SCHEMA = {
    "name": "u8_materialout_abandon",
    "description": "弃审材料出库单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id"},
            "opinion": {"type": "string", "description": "审批意见"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 现存量查询 Schema定义 =====================
U8_CURRENTSTOCK_BATCH_GET_SCHEMA = {
    "name": "u8_currentstock_batch_get",
    "description": "现存量查询，支持按仓库、存货编码、存货名称、批号等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号(默认取应用的第一个数据源)"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "whcode_begin": {"type": "string", "description": "起始仓库编码"},
            "whcode_end": {"type": "string", "description": "结束仓库编码"},
            "whname": {"type": "string", "description": "仓库名称关键字"},
            "invcode_begin": {"type": "string", "description": "起始存货编码"},
            "invcode_end": {"type": "string", "description": "结束存货编码"},
            "invname": {"type": "string", "description": "存货名称关键字"},
            "batch": {"type": "string", "description": "批号"}
        },
        "required": []
    }
}

# ===================== 获取调拨单列表 Schema定义 =====================
U8_TRANSVOUCH_LIST_SCHEMA = {
    "name": "u8_transvouch_list",
    "description": "获取调拨单列表，支持调拨单号、部门、仓库、业务类型、日期等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "tvcode_begin": {"type": "string", "description": "起始调拨单据号"},
            "tvcode_end": {"type": "string", "description": "结束调拨单据号"},
            "idepcode": {"type": "string", "description": "转入部门编码"},
            "idepname": {"type": "string", "description": "转入部门名称关键字"},
            "odepcode": {"type": "string", "description": "转出部门编码"},
            "odepname": {"type": "string", "description": "转出部门名称关键字"},
            "irdcode": {"type": "string", "description": "入库类别编码"},
            "irdname": {"type": "string", "description": "入库类别名称关键字"},
            "ordcode": {"type": "string", "description": "出库类别编码"},
            "ordname": {"type": "string", "description": "出库类别名称关键字"},
            "iwhcode": {"type": "string", "description": "转入仓库编码"},
            "iwhname": {"type": "string", "description": "转入仓库名称关键字"},
            "owhcode": {"type": "string", "description": "转出仓库编码"},
            "owhname": {"type": "string", "description": "转出仓库名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "memory": {"type": "string", "description": "备注关键字"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"}
        },
        "required": []
    }
}

# ===================== 审核调拨单 Schema定义 =====================
U8_TRANSVOUCH_VERIFY_SCHEMA = {
    "name": "u8_transvouch_verify",
    "description": "审核一张调拨单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审调拨单 Schema定义 =====================
U8_TRANSVOUCH_UNVERIFY_SCHEMA = {
    "name": "u8_transvouch_unverify",
    "description": "弃审一张调拨单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个调拨单 Schema定义 =====================
U8_TRANSVOUCH_GET_SCHEMA = {
    "name": "u8_transvouch_get",
    "description": "通过调拨单主表标识获取单个调拨单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "调拨单主表标识（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 新增调拨单 Schema定义 =====================
U8_TRANSVOUCH_ADD_SCHEMA = {
    "name": "u8_transvouch_add",
    "description": "新增一张调拨单，支持单据头信息和表体明细录入",
    "parameters": {
        "type": "object",
        "properties": {
            "idepcode": {"type": "string", "description": "转入部门编码"},
            "idepname": {"type": "string", "description": "转入部门名称"},
            "odepcode": {"type": "string", "description": "转出部门编码"},
            "odepname": {"type": "string", "description": "转出部门名称"},
            "irdcode": {"type": "string", "description": "入库类别编码"},
            "irdname": {"type": "string", "description": "入库类别名称"},
            "ordcode": {"type": "string", "description": "出库类别编码"},
            "ordname": {"type": "string", "description": "出库类别名称"},
            "iwhcode": {"type": "string", "description": "转入仓库编码"},
            "iwhname": {"type": "string", "description": "转入仓库名称"},
            "owhcode": {"type": "string", "description": "转出仓库编码"},
            "owhname": {"type": "string", "description": "转出仓库名称"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称"},
            "tvcode": {"type": "string", "description": "调拨单据号"},
            "date": {"type": "string", "description": "单据日期"},
            "memory": {"type": "string", "description": "备注"},
            "auditperson": {"type": "string", "description": "审核人"},
            "auditdate": {"type": "string", "description": "审核日期"},
            "maker": {"type": "string", "description": "制单人"},
            "define1": {"type": "string", "description": "自定义字段1"},
            "define2": {"type": "string", "description": "自定义字段2"},
            "define3": {"type": "string", "description": "自定义字段3"},
            "define4": {"type": "string", "description": "自定义字段4"},
            "define5": {"type": "number", "description": "自定义字段5"},
            "define6": {"type": "string", "description": "自定义字段6"},
            "define7": {"type": "number", "description": "自定义字段7"},
            "define8": {"type": "string", "description": "自定义字段8"},
            "define9": {"type": "string", "description": "自定义字段9"},
            "define10": {"type": "string", "description": "自定义字段10"},
            "define11": {"type": "string", "description": "自定义字段11"},
            "define12": {"type": "string", "description": "自定义字段12"},
            "define13": {"type": "string", "description": "自定义字段13"},
            "define14": {"type": "string", "description": "自定义字段14"},
            "define15": {"type": "number", "description": "自定义字段15"},
            "define16": {"type": "number", "description": "自定义字段16"},
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
            "ordertype": {"type": "string", "description": "订单类型"},
            "transappcode": {"type": "string", "description": "调拨申请单号"},
            "csource": {"type": "string", "description": "来源"},
            "entry": {
                "type": "array",
                "description": "表体明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "barcode": {"type": "string", "description": "条形码"},
                        "inventorycode": {"type": "string", "description": "存货编码"},
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
                        "quantity": {"type": "number", "description": "数量（主记量数量）"},
                        "cmassunitname": {"type": "string", "description": "主计量单位名称"},
                        "assitantunit": {"type": "string", "description": "辅记量单位"},
                        "assitantunitname": {"type": "string", "description": "辅计量单位名称"},
                        "irate": {"type": "number", "description": "换算率"},
                        "number": {"type": "number", "description": "件数"},
                        "actualcost": {"type": "number", "description": "实际价格"},
                        "actualprice": {"type": "number", "description": "实际金额"},
                        "inposcode": {"type": "string", "description": "调入货位"},
                        "outposcode": {"type": "string", "description": "调出货位"},
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
                        "define34": {"type": "number", "description": "表体自定义项34"},
                        "define35": {"type": "number", "description": "表体自定义项35"},
                        "define36": {"type": "string", "description": "表体自定义项36"},
                        "define37": {"type": "string", "description": "表体自定义项37"},
                        "irowno": {"type": "number", "description": "行号"},
                        "inventoryname": {"type": "string", "description": "存货名称"}
                    }
                }
            }
        },
        "required": []
    }
}

# ===================== 获取调拨申请单列表 Schema定义 =====================
U8_TRANSVOUCHAPPLY_LIST_SCHEMA = {
    "name": "u8_transvouchapply_list",
    "description": "获取调拨申请单列表，支持分页和多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始调拨申请单号"},
            "code_end": {"type": "string", "description": "结束调拨申请单号"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "iwhcode": {"type": "string", "description": "转入仓库编码"},
            "owhcode": {"type": "string", "description": "转出仓库编码"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "memory": {"type": "string", "description": "备注关键字"},
            "state": {"type": "string", "description": "单据状态"}
        },
        "required": []
    }
}

# ===================== 获取单个调拨申请单 Schema定义 =====================
U8_TRANSVOUCHAPPLY_GET_SCHEMA = {
    "name": "u8_transvouchapply_get",
    "description": "通过调拨申请单主表标识获取单个调拨申请单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "调拨申请单主表标识（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 获取采购入库单列表 Schema定义 =====================
U8_PURCHASEINLIST_BATCH_GET_SCHEMA = {
    "name": "u8_purchaseinlist_batch_get",
    "description": "获取采购入库单列表，支持按单据号、日期、审核日期、制单人、仓库、收发类型、部门、采购类型、红蓝标识等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始单据号"},
            "code_end": {"type": "string", "description": "结束单据号"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "auditdate_begin": {"type": "string", "description": "起始审核日期"},
            "auditdate_end": {"type": "string", "description": "结束审核日期"},
            "maker": {"type": "string", "description": "制单人名称关键字"},
            "warehousecode": {"type": "string", "description": "仓库编码"},
            "memory": {"type": "string", "description": "备注关键字"},
            "receivecode": {"type": "string", "description": "收发类型编码"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "warehousename": {"type": "string", "description": "仓库名称关键字"},
            "receivename": {"type": "string", "description": "收发类型名称关键字"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "purchasetypename": {"type": "string", "description": "采购类型名称关键字"},
            "bredvouch": {"type": "string", "description": "红蓝标识（1为红字，0为蓝字）"}
        },
        "required": []
    }
}

# ===================== 审核采购入库单 Schema定义 =====================
U8_PURCHASEIN_VERIFY_SCHEMA = {
    "name": "u8_purchasein_verify",
    "description": "审核一张采购入库单。执行审核动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审采购入库单 Schema定义 =====================
U8_PURCHASEIN_UNVERIFY_SCHEMA = {
    "name": "u8_purchasein_unverify",
    "description": "弃审一张采购入库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个采购入库单 Schema定义 =====================
U8_PURCHASEIN_GET_SCHEMA = {
    "name": "u8_purchasein_get",
    "description": "获取单个采购入库单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "单据号（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 新增采购入库单 Schema定义 =====================
U8_PURCHASEIN_ADD_SCHEMA = {
    "name": "u8_purchasein_add",
    "description": "新增一张采购入库单，支持单据头信息和表体明细录入",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据编号"},
            "date": {"type": "string", "description": "制单日期"},
            "maker": {"type": "string", "description": "制单人名称"},
            "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
            "warehousename": {"type": "string", "description": "仓库名称"},
            "vendorcode": {"type": "string", "description": "供货单位编码（必填）"},
            "vendorabbname": {"type": "string", "description": "供货单位简称（必填）"},
            "vendorname": {"type": "string", "description": "供货单位"},
            "memory": {"type": "string", "description": "备注"},
            "receivecode": {"type": "string", "description": "收发类型编码（必填）"},
            "receivename": {"type": "string", "description": "收发类型"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
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
            "taxrate": {"type": "number", "description": "税率"},
            "entry": {
                "type": "array",
                "description": "表体明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "inventoryname": {"type": "string", "description": "存货"},
                        "inventorystd": {"type": "string", "description": "规格型号"},
                        "quantity": {"type": "number", "description": "数量（必填）"},
                        "price": {"type": "number", "description": "本币单价"},
                        "cost": {"type": "number", "description": "本币金额"},
                        "ioritaxprice": {"type": "number", "description": "税额"},
                        "iorisum": {"type": "number", "description": "价税合计"},
                        "taxprice": {"type": "number", "description": "本币税额"},
                        "isum": {"type": "number", "description": "本币价税合计"},
                        "ioritaxcost": {"type": "number", "description": "含税单价，传入会自动重新计算相关价格及金额"},
                        "ioricost": {"type": "number", "description": "单价，传入会自动重新计算相关价格及金额。如果传入了含税单价，以含税单价为准自动计算"},
                        "iorimoney": {"type": "number", "description": "金额"},
                        "taxrate": {"type": "number", "description": "税率"},
                        "cmassunitname": {"type": "string", "description": "主计量单位名称"},
                        "serial": {"type": "string", "description": "批号"},
                        "makedate": {"type": "string", "description": "生产日期"},
                        "validdate": {"type": "string", "description": "失效日期"},
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
                        "assitantunitname": {"type": "string", "description": "辅计量单位名称"},
                        "irate": {"type": "number", "description": "换算率"},
                        "number": {"type": "number", "description": "件数"},
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
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        "rowno": {"type": "integer", "description": "行号"}
                    }
                }
            }
        },
        "required": ["warehousecode", "vendorcode", "vendorabbname", "receivecode"]
    }
}


# ===================== 获取销售出库单列表 Schema定义 =====================
U8_SALEOUTLIST_BATCH_GET_SCHEMA = {
    "name": "u8_saleoutlist_batch_get",
    "description": "获取销售出库单列表，支持按仓库、单据日期、审核日期、单据号、部门、备注、制单人、红蓝标识、业务单号、时间戳及自定义项等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "warehousecode": {"type": "string", "description": "仓库编码"},
            "warehousename": {"type": "string", "description": "仓库名称关键字"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "auditdate_begin": {"type": "string", "description": "起始审核日期，格式:yyyy-MM-dd"},
            "auditdate_end": {"type": "string", "description": "结束审核日期，格式:yyyy-MM-dd"},
            "code_begin": {"type": "string", "description": "起始收发单据号"},
            "code_end": {"type": "string", "description": "结束收发单据号"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "memory": {"type": "string", "description": "备注关键字"},
            "maker": {"type": "string", "description": "制单人关键字"},
            "bredvouch": {"type": "string", "description": "红蓝标识"},
            "businesscode": {"type": "string", "description": "对应业务单号"},
            "timestamp_begin": {"type": "string", "description": "起始时间戳"},
            "timestamp_end": {"type": "string", "description": "结束时间戳"},
            "define1": {"type": "string", "description": "自定义项1"},
            "define2": {"type": "string", "description": "自定义项2"},
            "define3": {"type": "string", "description": "自定义项3"},
            "define4_begin": {"type": "string", "description": "起始自定义项4"},
            "define4_end": {"type": "string", "description": "结束自定义项4"},
            "define5": {"type": "number", "description": "自定义项5"},
            "define6_begin": {"type": "string", "description": "起始自定义项6"},
            "define6_end": {"type": "string", "description": "结束自定义项6"},
            "define7": {"type": "number", "description": "自定义项7"},
            "define8": {"type": "string", "description": "自定义项8"},
            "define9": {"type": "string", "description": "自定义项9"},
            "define10": {"type": "string", "description": "自定义项10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "number", "description": "自定义项15"},
            "define16": {"type": "number", "description": "自定义项16"}
        },
        "required": []
    }
}

# ===================== 审核销售出库单 Schema定义 =====================
U8_SALEOUT_VERIFY_SCHEMA = {
    "name": "u8_saleout_verify",
    "description": "审核一张销售出库单。执行审核动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审销售出库单 Schema定义 =====================
U8_SALEOUT_UNVERIFY_SCHEMA = {
    "name": "u8_saleout_unverify",
    "description": "弃审一张销售出库单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个销售出库单 Schema定义 =====================
U8_SALEOUT_GET_SCHEMA = {
    "name": "u8_saleout_get",
    "description": "获取单个销售出库单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "收发记录主表标识（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 新增销售出库单 Schema定义 =====================
U8_SALEOUT_ADD_SCHEMA = {
    "name": "u8_saleout_add",
    "description": "新增一张销售出库单，支持单据头信息和表体明细录入",
    "parameters": {
        "type": "object",
        "properties": {
            "businesstype": {"type": "string", "description": "业务类型"},
            "source": {"type": "string", "description": "单据来源"},
            "businesscode": {"type": "string", "description": "对应业务单号"},
            "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
            "warehousename": {"type": "string", "description": "仓库名称"},
            "date": {"type": "string", "description": "单据日期"},
            "code": {"type": "string", "description": "单据号"},
            "receivecode": {"type": "string", "description": "收发类别编码"},
            "receivename": {"type": "string", "description": "收发类别名称"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "saletypecode": {"type": "string", "description": "销售类型编码"},
            "customercode": {"type": "string", "description": "客户编码"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "arrivedate": {"type": "string", "description": "到货日期"},
            "memory": {"type": "string", "description": "备注"},
            "maker": {"type": "string", "description": "制单人"},
            "define1": {"type": "string", "description": "自定义项1"},
            "define2": {"type": "string", "description": "自定义项2"},
            "define3": {"type": "string", "description": "自定义项3"},
            "define4": {"type": "string", "description": "自定义项4"},
            "define5": {"type": "number", "description": "自定义项5"},
            "define6": {"type": "string", "description": "自定义项6"},
            "define7": {"type": "number", "description": "自定义项7"},
            "define8": {"type": "string", "description": "自定义项8"},
            "define9": {"type": "string", "description": "自定义项9"},
            "define10": {"type": "string", "description": "自定义项10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "number", "description": "自定义项15"},
            "define16": {"type": "number", "description": "自定义项16"},
            "entry": {
                "type": "array",
                "description": "表体明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "barcode": {"type": "string", "description": "条形码"},
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "free1": {"type": "string", "description": "存货自由项1"},
                        "free2": {"type": "string", "description": "存货自由项2"},
                        "free3": {"type": "string", "description": "存货自由项3"},
                        "free4": {"type": "string", "description": "存货自由项4"},
                        "free5": {"type": "string", "description": "存货自由项5"},
                        "free6": {"type": "string", "description": "存货自由项6"},
                        "free7": {"type": "string", "description": "存货自由项7"},
                        "free8": {"type": "string", "description": "存货自由项8"},
                        "free9": {"type": "string", "description": "存货自由项9"},
                        "free10": {"type": "string", "description": "存货自由项10"},
                        "shouldquantity": {"type": "number", "description": "应发数量"},
                        "shouldnumber": {"type": "number", "description": "应发件数"},
                        "quantity": {"type": "number", "description": "数量（必填）"},
                        "cmassunitname": {"type": "string", "description": "主计量单位"},
                        "assitantunit": {"type": "string", "description": "库存单位码"},
                        "assitantunitname": {"type": "string", "description": "库存单位"},
                        "irate": {"type": "number", "description": "换算率"},
                        "number": {"type": "number", "description": "件数"},
                        "price": {"type": "number", "description": "单价"},
                        "cost": {"type": "number", "description": "金额"},
                        "serial": {"type": "string", "description": "批号"},
                        "makedate": {"type": "string", "description": "生产日期"},
                        "validdate": {"type": "string", "description": "失效日期"},
                        "define22": {"type": "string", "description": "表体自定义项1"},
                        "define23": {"type": "string", "description": "表体自定义项2"},
                        "define24": {"type": "string", "description": "表体自定义项3"},
                        "define25": {"type": "string", "description": "表体自定义项4"},
                        "define26": {"type": "number", "description": "表体自定义项5"},
                        "define27": {"type": "number", "description": "表体自定义项6"},
                        "define28": {"type": "string", "description": "表体自定义项7"},
                        "define29": {"type": "string", "description": "表体自定义项8"},
                        "define30": {"type": "string", "description": "表体自定义项9"},
                        "define31": {"type": "string", "description": "表体自定义项10"},
                        "define32": {"type": "string", "description": "表体自定义项11"},
                        "define33": {"type": "string", "description": "表体自定义项12"},
                        "define34": {"type": "number", "description": "表体自定义项13"},
                        "define35": {"type": "number", "description": "表体自定义项14"},
                        "define36": {"type": "string", "description": "表体自定义项15"},
                        "define37": {"type": "string", "description": "表体自定义项16"},
                        "rowno": {"type": "integer", "description": "行号"},
                        "subconsignmentcode": {"type": "string", "description": "发货单号"},
                        "ordercode": {"type": "string", "description": "订单号"}
                    }
                }
            }
        },
        "required": ["warehousecode"]
    }
}
