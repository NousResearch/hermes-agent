import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)

# ============================================
# 数据模型 (Data Models)
# ============================================

# ===================== 发货单表体数据模型 =====================
class ConsignmentEntryInput(BaseModel):
    """发货单表体数据模型"""
    inventory_code: str = Field(..., description="存货编码（必填）")
    inventory_name: Optional[str] = Field(None, description="存货名称")
    warehouse_code: str = Field(..., description="仓库编码（必填）")
    warehouse_name: Optional[str] = Field(None, description="仓库名称")
    invstd: Optional[str] = Field(None, description="存货规格")
    ccomunitcode: Optional[str] = Field(None, description="主计量单位编码")
    cinvm_unit: Optional[str] = Field(None, description="主计量单位")
    quantity: float = Field(..., description="数量（必填）")
    price: Optional[float] = Field(None, description="单价")
    quotedprice: Optional[float] = Field(None, description="报价")
    taxprice: Optional[float] = Field(None, description="含税单价")
    money: Optional[float] = Field(None, description="无税金额")
    sum: Optional[float] = Field(None, description="价税合计")
    taxrate: Optional[float] = Field(None, description="税率")
    tax: Optional[float] = Field(None, description="税额")
    natprice: Optional[float] = Field(None, description="本币单价")
    natmoney: Optional[float] = Field(None, description="本币金额")
    nattax: Optional[float] = Field(None, description="本币税额")
    natsum: Optional[float] = Field(None, description="本币价税合计")
    discount: Optional[float] = Field(None, description="折扣额")
    natdiscount: Optional[float] = Field(None, description="本币折扣额")
    discount1: Optional[float] = Field(None, description="扣率(%)")
    discount2: Optional[float] = Field(None, description="扣率2(%)")
    socode: Optional[str] = Field(None, description="销售订单号")
    batch: Optional[str] = Field(None, description="批号")
    ExpirationDate: Optional[str] = Field(None, description="有效期至(yyyy-MM-dd)")
    cmassunit: Optional[str] = Field(None, description="保质期单位")
    ExpirationItem: Optional[str] = Field(None, description="有效期计算项(yyyy-MM-dd)")
    dmdate: Optional[str] = Field(None, description="生产日期(yyyy-MM-dd)")
    overdate: Optional[str] = Field(None, description="失效日期(yyyy-MM-dd)")
    ExpiratDateCalcu: Optional[float] = Field(None, description="有效期推算方式")
    imassdate: Optional[float] = Field(None, description="保质期")
    item_code: Optional[str] = Field(None, description="项目编码")
    unit_code: Optional[str] = Field(None, description="辅计量单位")
    num: Optional[float] = Field(None, description="件数")
    unitrate: Optional[float] = Field(None, description="换算率")
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
    define36: Optional[str] = Field(None, description="单据体自定义项15(yyyy-MM-dd)")
    define37: Optional[str] = Field(None, description="单据体自定义项16(yyyy-MM-dd)")
    rowno: float = Field(..., description="行号（必填）")


# ===================== 新增发货单 数据模型 =====================
class ConsignmentAddInput(BaseModel):
    """新增发货单输入模型"""
    code: Optional[str] = Field(None, description="单据号")
    date: Optional[str] = Field(None, description="单据日期(yyyy-MM-dd)")
    operation_type: str = Field(..., description="业务类型（必填）")
    saletype: str = Field(..., description="销售类型编码（必填）")
    saletypename: Optional[str] = Field(None, description="销售类型")
    state: Optional[str] = Field(None, description="订单状态")
    custcode: str = Field(..., description="客户编码（必填）")
    cusname: Optional[str] = Field(None, description="客户")
    cusabbname: Optional[str] = Field(None, description="客户简称")
    deptcode: str = Field(..., description="部门编码（必填）")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    cdeliverunit: Optional[str] = Field(None, description="收货单位")
    ccontactname: Optional[str] = Field(None, description="收货联系人")
    cofficephone: Optional[str] = Field(None, description="收货联系电话")
    cmobilephone: Optional[str] = Field(None, description="收货联系人手机")
    cdeliveradd: Optional[str] = Field(None, description="收货地址")
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4(yyyy-MM-dd)")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6(yyyy-MM-dd)")
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
    remark: Optional[str] = Field(None, description="备注")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")
    entry: List[ConsignmentEntryInput] = Field(..., description="表体数据（必填）")


# ===================== 获取单张发货单 数据模型 =====================
class ConsignmentGetInput(BaseModel):
    """获取单张发货单输入模型"""
    id: str = Field(..., description="发货单ID（必填）")


# ===================== 获取发货单列表 数据模型 =====================
class ConsignmentListInput(BaseModel):
    """获取发货单列表输入模型"""
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始制单日期(yyyy-MM-dd)")
    date_end: Optional[str] = Field(None, description="结束制单日期(yyyy-MM-dd)")
    state: Optional[str] = Field(None, description="订单状态")
    custcode: Optional[str] = Field(None, description="客户编码")
    cusname: Optional[str] = Field(None, description="客户名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    remark: Optional[str] = Field(None, description="备注关键字")
    socode: Optional[str] = Field(None, description="销售订单号")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")


# ===================== 审核发货单 数据模型 =====================
class ConsignmentVerifyInput(BaseModel):
    """审核发货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")


# ===================== 弃审发货单 数据模型 =====================
class ConsignmentUnverifyInput(BaseModel):
    """弃审发货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审批发货单(工作流) 数据模型 =====================
class ConsignmentAuditInput(BaseModel):
    """审批发货单(工作流)输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 弃审发货单(工作流) 数据模型 =====================
class ConsignmentAbandonInput(BaseModel):
    """弃审发货单(工作流)输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")


# ===================== 获取发货单工作流按钮状态 数据模型 =====================
class ConsignmentButtonStateInput(BaseModel):
    """获取发货单工作流按钮状态输入模型"""
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取发货单待办任务 数据模型 =====================
class ConsignmentTasksInput(BaseModel):
    """获取发货单待办任务输入模型"""
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[str] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[float] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[float] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[float] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[float] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 查看发货单审批进程 数据模型 =====================
class ConsignmentHistoryInput(BaseModel):
    """查看发货单审批进程输入模型"""
    user_id: Optional[str] = Field(None, description="审批人(用户编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 获取单个客户调价单 数据模型 =====================
class CuspricejustGetInput(BaseModel):
    """获取单个客户调价单输入模型"""
    id: str = Field(..., description="单据号（必填）")


# ===================== 批量获取客户调价单 数据模型 =====================
class CuspricejustListInput(BaseModel):
    """批量获取客户调价单输入模型"""
    ccode_begin: Optional[str] = Field(None, description="起始单据号")
    ccode_end: Optional[str] = Field(None, description="结束单据号")
    ddate_begin: Optional[str] = Field(None, description="起始单据日期(yyyy-MM-dd)")
    ddate_end: Optional[str] = Field(None, description="结束单据日期(yyyy-MM-dd)")
    cdepname: Optional[str] = Field(None, description="调价部门")
    cpersonname: Optional[str] = Field(None, description="调价业务员关键字")
    cmainmemo: Optional[str] = Field(None, description="表头备注")
    cmaker: Optional[str] = Field(None, description="制单人关键字")
    cverifier: Optional[str] = Field(None, description="审核人关键字")
    dverifydate: Optional[str] = Field(None, description="审核日期(yyyy-MM-dd)")
    drdate: Optional[str] = Field(None, description="系统日期(yyyy-MM-dd)")
    ccusabbname: Optional[str] = Field(None, description="客户简称关键字")
    cccname: Optional[str] = Field(None, description="客户大类")
    cinvcode: Optional[str] = Field(None, description="存货编码")
    cinvname: Optional[str] = Field(None, description="存货名称关键字")
    cinvstd: Optional[str] = Field(None, description="规格型号")
    ccomunitname: Optional[str] = Field(None, description="计量单位")
    fminquantity: Optional[float] = Field(None, description="数量下限")
    iinvscost: Optional[float] = Field(None, description="批发价")
    icusdisrate: Optional[float] = Field(None, description="客户扣率")
    iinvnowcost: Optional[float] = Field(None, description="成交价")
    iinvsalecost: Optional[float] = Field(None, description="零售单价")
    dstartdate: Optional[str] = Field(None, description="开始日期(yyyy-MM-dd)")
    denddate: Optional[str] = Field(None, description="结束日期(yyyy-MM-dd)")
    bsales: Optional[bool] = Field(None, description="是否促销价")
    fcusminprice: Optional[float] = Field(None, description="客户最低售价")


# ===================== 销售发票表体数据模型 =====================
class SaleinvoiceEntryInput(BaseModel):
    """销售发票表体数据模型"""
    warehousecode: str = Field(..., description="仓库编码（必填）")
    inventorycode: str = Field(..., description="存货编码（必填）")
    quantity: float = Field(..., description="数量（必填）")
    number: Optional[float] = Field(None, description="件数")
    quotedprice: Optional[float] = Field(None, description="报价")
    originalprice: float = Field(..., description="无税单价（必填）")
    originaltaxedprice: float = Field(..., description="含税单价（必填）")
    originalmoney: float = Field(..., description="无税金额（必填）")
    originaltax: float = Field(..., description="税额（必填）")
    originalsum: float = Field(..., description="价税合计（必填）")
    price: float = Field(..., description="本币单价（必填）")
    money: float = Field(..., description="本币金额（必填）")
    tax: float = Field(..., description="本币税额（必填）")
    sum: float = Field(..., description="本币价税合计（必填）")
    taxrate: float = Field(..., description="税率(%)（必填）")
    assistantunit: Optional[str] = Field(None, description="销售单位编码")
    originaldiscount: Optional[float] = Field(None, description="折扣额")
    discount: Optional[float] = Field(None, description="本币折扣额")
    memory: Optional[str] = Field(None, description="备注")
    serial: Optional[str] = Field(None, description="批号")
    accountrate1: Optional[float] = Field(None, description="扣率(%)")
    accountrate2: Optional[float] = Field(None, description="扣率2(%)")
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
    retailprice: Optional[float] = Field(None, description="零售单价")
    retailmoney: Optional[float] = Field(None, description="零售金额")
    itemclasscode: Optional[str] = Field(None, description="项目大类编码")
    itemcode: Optional[str] = Field(None, description="项目编码")
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
    batchproperty1: Optional[str] = Field(None, description="批次属性1")
    batchproperty2: Optional[str] = Field(None, description="批次属性2")
    batchproperty3: Optional[str] = Field(None, description="批次属性3")
    batchproperty4: Optional[str] = Field(None, description="批次属性4")
    batchproperty5: Optional[str] = Field(None, description="批次属性5")
    batchproperty6: Optional[str] = Field(None, description="批次属性6")
    batchproperty7: Optional[str] = Field(None, description="批次属性7")
    batchproperty8: Optional[str] = Field(None, description="批次属性8")
    batchproperty9: Optional[str] = Field(None, description="批次属性9")
    batchproperty10: Optional[str] = Field(None, description="批次属性10")
    exchangerate: Optional[float] = Field(None, description="换算率")
    unitid: Optional[str] = Field(None, description="销售单位编码")
    cmassunit: Optional[str] = Field(None, description="保质期单位")
    imassdate: Optional[float] = Field(None, description="保质期")
    dmdate: Optional[str] = Field(None, description="生产日期(yyyy-MM-dd)")
    invaliddate: Optional[str] = Field(None, description="失效日期(yyyy-MM-dd)")
    ExpirationDate: Optional[str] = Field(None, description="有效期至")
    ExpiratDateCalcu: Optional[float] = Field(None, description="有效期推算方式")
    ExpirationItem: Optional[str] = Field(None, description="有效期计算项(yyyy-MM-dd)")
    cvmivencode: Optional[str] = Field(None, description="供货商编码")
    irowno: Optional[float] = Field(None, description="行号")
    ReasonCode: Optional[str] = Field(None, description="退货原因编码")
    bsaleprice: Optional[bool] = Field(None, description="报价含税")
    bgift: Optional[bool] = Field(None, description="赠品")
    fcusminprice: Optional[float] = Field(None, description="最低售价")
    icalctype: Optional[float] = Field(None, description="发货模式")
    fchildqty: Optional[float] = Field(None, description="使用数量")
    fchildrate: Optional[float] = Field(None, description="权重比例")


# ===================== 新增销售发票 数据模型 =====================
class SaleinvoiceAddInput(BaseModel):
    """新增销售发票输入模型"""
    invoiceno: Optional[str] = Field(None, description="发票号")
    vouchertype: str = Field(..., description="单据类型（必填）")
    saletypecode: str = Field(..., description="销售类型编号（必填）")
    date: str = Field(..., description="日期(yyyy-MM-dd)（必填）")
    departmentcode: str = Field(..., description="部门编号（必填）")
    personcode: Optional[str] = Field(None, description="职员编号")
    customercode: str = Field(..., description="客商编号（必填）")
    payconditioncode: Optional[str] = Field(None, description="付款条件编码")
    foreigncurrency: Optional[str] = Field(None, description="外币名称")
    memory: Optional[str] = Field(None, description="备注")
    currencyrate: Optional[float] = Field(None, description="汇率")
    taxrate: Optional[float] = Field(None, description="税率")
    isnegative: Optional[bool] = Field(None, description="负发票-正发票")
    bankcode: Optional[str] = Field(None, description="本单位开户银行编号")
    invoiceversion: Optional[str] = Field(None, description="发票版别")
    maker: Optional[str] = Field(None, description="制单人")
    businesstype: Optional[str] = Field(None, description="业务类型")
    isfirst: Optional[bool] = Field(None, description="是否期初")
    itemclasscode: Optional[str] = Field(None, description="项目大类编号")
    itemcode: Optional[str] = Field(None, description="项目编码")
    define1: Optional[str] = Field(None, description="自定义字段1")
    define2: Optional[str] = Field(None, description="自定义字段2")
    define3: Optional[str] = Field(None, description="自定义字段3")
    define4: Optional[str] = Field(None, description="自定义字段4(yyyy-MM-dd)")
    define5: Optional[float] = Field(None, description="自定义字段5")
    define6: Optional[str] = Field(None, description="自定义字段6(yyyy-MM-dd)")
    define7: Optional[float] = Field(None, description="自定义字段7")
    define8: Optional[str] = Field(None, description="自定义字段8")
    define9: Optional[str] = Field(None, description="自定义字段9")
    define10: Optional[str] = Field(None, description="自定义字段10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[float] = Field(None, description="自定义项15")
    define16: Optional[float] = Field(None, description="自定义项16")
    ispayedfirst: Optional[float] = Field(None, description="1先发货;0先开票")
    customername: Optional[str] = Field(None, description="综合开票客户名称")
    ccusaccount: Optional[str] = Field(None, description="客户账号")
    cbaccount: Optional[str] = Field(None, description="本单位账号")
    cdeliverunit: Optional[str] = Field(None, description="收货单位名称")
    cdeliveradd: Optional[str] = Field(None, description="收货地址")
    ccontactname: Optional[str] = Field(None, description="收货联系人")
    cofficephone: Optional[str] = Field(None, description="收货联系电话")
    cmobilephone: Optional[str] = Field(None, description="收货联系手机")
    caddcode: Optional[str] = Field(None, description="收获地址编码")
    cgatheringplan: Optional[str] = Field(None, description="收付款协议编码")
    dcreditstart: Optional[str] = Field(None, description="立账日(yyyy-MM-dd)")
    icreditdays: Optional[float] = Field(None, description="账期")
    dgatheringdate: Optional[str] = Field(None, description="到期日(yyyy-MM-dd)")
    bcredit: Optional[bool] = Field(None, description="是否立账单据")
    csource: Optional[str] = Field(None, description="来源")
    ccusbank: Optional[str] = Field(None, description="客户开户银行")
    entry: List[SaleinvoiceEntryInput] = Field(..., description="表体数据（必填）")


# ===================== 获取单个销售发票 数据模型 =====================
class SaleinvoiceGetInput(BaseModel):
    """获取单个销售发票输入模型"""
    id: str = Field(..., description="销售发票ID（必填）")


# ===================== 获取销售发票列表 数据模型 =====================
class SaleinvoiceListInput(BaseModel):
    """获取销售发票列表输入模型"""
    invoiceno_begin: Optional[str] = Field(None, description="起始销售发票号")
    invoiceno_end: Optional[str] = Field(None, description="结束销售发票号")
    ccusname: Optional[str] = Field(None, description="客户名称关键字")
    cstname: Optional[str] = Field(None, description="销售类型")
    csocode: Optional[str] = Field(None, description="销售单号")
    cdlcode: Optional[str] = Field(None, description="发货单号")
    vouchertype: Optional[str] = Field(None, description="单据类型")
    saletypecode: Optional[str] = Field(None, description="销售类型编号")
    date_begin: Optional[str] = Field(None, description="起始日期(yyyy-MM-dd)")
    date_end: Optional[str] = Field(None, description="结束日期(yyyy-MM-dd)")
    departmentcode: Optional[str] = Field(None, description="部门编号")
    personcode: Optional[str] = Field(None, description="职员编号")
    customercode: Optional[str] = Field(None, description="客商编号")
    isnegative: Optional[float] = Field(None, description="负发票-正发票")
    maker: Optional[str] = Field(None, description="制单人关键字")
    businesstype: Optional[str] = Field(None, description="业务类型")
    isfirst: Optional[bool] = Field(None, description="是否期初")
    itemclasscode: Optional[str] = Field(None, description="项目大类编号")
    itemcode: Optional[str] = Field(None, description="项目编码")
    define1: Optional[str] = Field(None, description="自定义字段1")
    define2: Optional[str] = Field(None, description="自定义字段2")
    define3: Optional[str] = Field(None, description="自定义字段3")
    define4: Optional[str] = Field(None, description="自定义字段4(yyyy-MM-dd)")
    define5: Optional[float] = Field(None, description="自定义字段5")
    define6: Optional[str] = Field(None, description="自定义字段6(yyyy-MM-dd)")
    define7: Optional[float] = Field(None, description="自定义字段7")
    define8: Optional[str] = Field(None, description="自定义字段8")
    define9: Optional[str] = Field(None, description="自定义字段9")
    define10: Optional[str] = Field(None, description="自定义字段10")
    define11: Optional[str] = Field(None, description="自定义项11")
    define12: Optional[str] = Field(None, description="自定义项12")
    define13: Optional[str] = Field(None, description="自定义项13")
    define14: Optional[str] = Field(None, description="自定义项14")
    define15: Optional[float] = Field(None, description="自定义项15")
    define16: Optional[float] = Field(None, description="自定义项16")
    csource: Optional[str] = Field(None, description="来源")
    csaleout: Optional[str] = Field(None, description="出库单号")
    ccusabbname: Optional[str] = Field(None, description="客户简称关键字")
    cdepname: Optional[str] = Field(None, description="部门关键字")
    cpersonname: Optional[str] = Field(None, description="职员名称关键字")


# ===================== 批量获取销售结账状态 数据模型 =====================
class MendsaListInput(BaseModel):
    """批量获取销售结账状态输入模型"""
    iyear: Optional[float] = Field(None, description="会计年度")
    iperiod_begin: Optional[float] = Field(None, description="起始会计期间")
    iperiod_end: Optional[float] = Field(None, description="结束会计期间")

# ============================================
# 数据模型 (Data Models)
# ============================================

# ===================== 销售订单表体数据模型 =====================
class SaleorderEntryInput(BaseModel):
    """销售订单表体数据模型"""
    inventorycode: str = Field(..., description="存货编码（必填）")
    inventoryname: Optional[str] = Field(None, description="存货")
    invstd: Optional[str] = Field(None, description="规格型号")
    unitcode: Optional[str] = Field(None, description="销售单位编码")
    unitname: Optional[str] = Field(None, description="销售单位")
    unitrate: Optional[float] = Field(None, description="换算率")
    quantity: float = Field(..., description="数量（必填）")
    num: Optional[float] = Field(None, description="件数")
    unitprice: Optional[float] = Field(None, description="单价")
    quotedprice: Optional[float] = Field(None, description="报价")
    taxunitprice: Optional[float] = Field(None, description="含税单价")
    money: Optional[float] = Field(None, description="无税金额")
    taxrate: Optional[float] = Field(None, description="税率")
    sum: Optional[float] = Field(None, description="价税合计")
    discount: Optional[float] = Field(None, description="折扣额")
    natdiscount: Optional[float] = Field(None, description="本币折扣额")
    discountrate: Optional[float] = Field(None, description="扣率(%)")
    discountrate2: Optional[float] = Field(None, description="扣率2(%)")
    natmoney: Optional[float] = Field(None, description="本币金额")
    natunitprice: Optional[float] = Field(None, description="本币单价")
    tax: Optional[float] = Field(None, description="税额")
    nattax: Optional[float] = Field(None, description="本币税额")
    natsum: Optional[float] = Field(None, description="本币价税合计")
    ccontractid: Optional[str] = Field(None, description="合同编码")
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
    define36: Optional[str] = Field(None, description="单据体自定义项15(yyyy-MM-dd)")
    define37: Optional[str] = Field(None, description="单据体自定义项16(yyyy-MM-dd)")
    bgift: float = Field(..., description="是否赠品(0=非赠品;1=赠品)（必填）")
    rowno: float = Field(..., description="行号（必填）")


# ===================== 新增销售订单 数据模型 =====================
class SaleorderAddInput(BaseModel):
    """新增销售订单输入模型"""
    code: Optional[str] = Field(None, description="订单号")
    date: Optional[str] = Field(None, description="日期(yyyy-MM-dd)")
    businesstype: str = Field(..., description="业务类型（必填）")
    typecode: str = Field(..., description="销售类型编码（必填）")
    typename: Optional[str] = Field(None, description="销售类型")
    state: Optional[str] = Field(None, description="单据状态")
    custcode: str = Field(..., description="客户编码（必填）")
    cusname: Optional[str] = Field(None, description="客户名称")
    cusabbname: Optional[str] = Field(None, description="客户简称")
    deptcode: str = Field(..., description="部门编码（必填）")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    dpremodatebt: Optional[str] = Field(None, description="预完工日期(yyyy-MM-dd)")
    dpredatebt: Optional[str] = Field(None, description="预发货日期(yyyy-MM-dd)")
    sendaddress: Optional[str] = Field(None, description="发货地址")
    ccusperson: Optional[str] = Field(None, description="联系人")
    ccuspersoncode: Optional[str] = Field(None, description="联系人编码")
    caddcode: Optional[str] = Field(None, description="收货地址编码")
    taxrate: Optional[float] = Field(None, description="税率，默认16")
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4(yyyy-MM-dd)")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6(yyyy-MM-dd)")
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
    memo: Optional[str] = Field(None, description="备注")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")
    entry: List[SaleorderEntryInput] = Field(..., description="表体数据（必填）")


# ===================== 获取单个销售订单 数据模型 =====================
class SaleorderGetInput(BaseModel):
    """获取单个销售订单输入模型"""
    id: str = Field(..., description="订单编号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 查询销售订单列表 数据模型 =====================
class SaleorderListInput(BaseModel):
    """查询销售订单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始订单编号")
    code_end: Optional[str] = Field(None, description="结束订单编号")
    date_begin: Optional[str] = Field(None, description="起始订单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束订单日期，格式:yyyy-MM-dd")
    dpremodatebt_begin: Optional[str] = Field(None, description="起始预完工日期，格式:yyyy-MM-dd")
    dpremodatebt_end: Optional[str] = Field(None, description="结束预完工日期，格式:yyyy-MM-dd")
    dpredatebt_begin: Optional[str] = Field(None, description="起始预发货日期，格式:yyyy-MM-dd")
    dpredatebt_end: Optional[str] = Field(None, description="结束预发货日期，格式:yyyy-MM-dd")
    state: Optional[str] = Field(None, description="订单状态")
    custcode: Optional[str] = Field(None, description="客户编码")
    cusname: Optional[str] = Field(None, description="客户名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    memo: Optional[str] = Field(None, description="备注关键字")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")
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
    timestamp_begin: Optional[float] = Field(None, description="起始时间戳")
    timestamp_end: Optional[float] = Field(None, description="结束时间戳")
    fhstatus: Optional[float] = Field(None, description="发货状态(0=未发货;1=部分发货;2=全部发货)")


# ===================== 审核销售订单(verify) 数据模型 =====================
class SaleorderVerifyInput(BaseModel):
    """审核销售订单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 弃审销售订单(unverify) 数据模型 =====================
class SaleorderUnverifyInput(BaseModel):
    """弃审销售订单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 销售订单审批(audit) 数据模型 =====================
class SaleorderAuditInput(BaseModel):
    """销售订单审批输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: float = Field(..., description="是否同意(1=同意;0=不同意)（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 销售订单弃审(abandon) 数据模型 =====================
class SaleorderAbandonInput(BaseModel):
    """销售订单弃审输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 获取销售订单按钮状态 数据模型 =====================
class SaleorderButtonstateInput(BaseModel):
    """获取销售订单按钮状态输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 获取销售订单待办任务 数据模型 =====================
class SaleorderTasksInput(BaseModel):
    """获取销售订单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[float] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[float] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[float] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[float] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[float] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 获取销售订单审批历史 数据模型 =====================
class SaleorderHistoryInput(BaseModel):
    """获取销售订单审批历史输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 打开销售订单 数据模型 =====================
class SaleorderOpenInput(BaseModel):
    """打开销售订单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 关闭销售订单 数据模型 =====================
class SaleorderCloseInput(BaseModel):
    """关闭销售订单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 销售退货单表体数据模型 =====================
class ReturnorderEntryInput(BaseModel):
    """销售退货单表体数据模型"""
    inventory_code: str = Field(..., description="存货编码（必填）")
    inventory_name: Optional[str] = Field(None, description="存货名称")
    warehouse_code: str = Field(..., description="仓库编码（必填）")
    warehouse_name: Optional[str] = Field(None, description="仓库名称")
    invstd: Optional[str] = Field(None, description="存货规格")
    ccomunitcode: Optional[str] = Field(None, description="主计量单位编码")
    cinvm_unit: Optional[str] = Field(None, description="主计量单位")
    quantity: float = Field(..., description="数量（必填）")
    price: Optional[float] = Field(None, description="单价")
    quotedprice: Optional[float] = Field(None, description="报价")
    taxprice: Optional[float] = Field(None, description="含税单价")
    money: Optional[float] = Field(None, description="无税金额")
    sum: Optional[float] = Field(None, description="价税合计")
    taxrate: Optional[float] = Field(None, description="税率")
    tax: Optional[float] = Field(None, description="税额")
    natprice: Optional[float] = Field(None, description="本币单价")
    natmoney: Optional[float] = Field(None, description="本币金额")
    nattax: Optional[float] = Field(None, description="本币税额")
    natsum: Optional[float] = Field(None, description="本币价税合计")
    discount: Optional[float] = Field(None, description="折扣额")
    natdiscount: Optional[float] = Field(None, description="本币折扣额")
    discount1: Optional[float] = Field(None, description="扣率(%)")
    discount2: Optional[float] = Field(None, description="扣率2(%)")
    socode: Optional[str] = Field(None, description="销售订单号")
    ReasonCode: Optional[str] = Field(None, description="退货原因编码")
    ReasonName: Optional[str] = Field(None, description="退货原因")
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
    define36: Optional[str] = Field(None, description="单据体自定义项15(yyyy-MM-dd)")
    define37: Optional[str] = Field(None, description="单据体自定义项16(yyyy-MM-dd)")
    rowno: float = Field(..., description="行号（必填）")


# ===================== 新增销售退货单 数据模型 =====================
class ReturnorderAddInput(BaseModel):
    """新增销售退货单输入模型"""
    code: Optional[str] = Field(None, description="单据号")
    date: Optional[str] = Field(None, description="单据日期(yyyy-MM-dd)")
    operation_type: str = Field(..., description="业务类型（必填）")
    saletype: str = Field(..., description="销售类型编码（必填）")
    saletypename: Optional[str] = Field(None, description="销售类型")
    state: Optional[str] = Field(None, description="订单状态")
    custcode: str = Field(..., description="客户编码（必填）")
    cusname: Optional[str] = Field(None, description="客户")
    cusabbname: Optional[str] = Field(None, description="客户简称")
    deptcode: str = Field(..., description="部门编码（必填）")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4(yyyy-MM-dd)")
    define5: Optional[float] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6(yyyy-MM-dd)")
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
    remark: Optional[str] = Field(None, description="备注")
    entry: List[ReturnorderEntryInput] = Field(..., description="表体数据（必填）")


# ===================== 获取单个销售退货单 数据模型 =====================
class ReturnorderGetInput(BaseModel):
    """获取单个销售退货单输入模型"""
    id: str = Field(..., description="id（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 查询销售退货单列表 数据模型 =====================
class ReturnorderListInput(BaseModel):
    """查询销售退货单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始单据号")
    code_end: Optional[str] = Field(None, description="结束单据号")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    operation_type: Optional[str] = Field(None, description="业务类型关键字")
    saletype: Optional[str] = Field(None, description="销售类型编码")
    saletypename: Optional[str] = Field(None, description="销售类型关键字")
    state: Optional[str] = Field(None, description="订单状态")
    custcode: Optional[str] = Field(None, description="客户编码")
    cusname: Optional[str] = Field(None, description="客户名称关键字")
    cusabbname: Optional[str] = Field(None, description="客户简称关键字")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    remark: Optional[str] = Field(None, description="备注关键字")


# ===================== 审核销售退货单 数据模型 =====================
class ReturnorderVerifyInput(BaseModel):
    """审核销售退货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 弃审销售退货单 数据模型 =====================
class ReturnorderUnverifyInput(BaseModel):
    """弃审销售退货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ============================================
# Tool函数 (Tool Functions)
# ============================================

# ===================== 新增发货单 Tool函数 =====================
def u8_consignment_add_tool(input_data: ConsignmentAddInput, client: Any) -> str:
    """
    新增一张发货单。
    """
    request_body: Dict[str, Any] = {
        "consignment": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/consignment/add"
    
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "发货单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "发货单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单张发货单 Tool函数 =====================
def u8_consignment_get_tool(input_data: ConsignmentGetInput, client: Any) -> str:
    """
    获取单张发货单详情。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/consignment/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取发货单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取发货单成功",
            "data": result.get("consignment"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取发货单列表 Tool函数 =====================
def u8_consignment_list_tool(input_data: ConsignmentListInput, client: Any) -> str:
    """
    获取发货单列表，支持分页和条件查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/consignmentlist/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取发货单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取发货单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "consignmentlist": result.get("consignmentlist")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核发货单 Tool函数 =====================
def u8_consignment_verify_tool(input_data: ConsignmentVerifyInput, client: Any) -> str:
    """
    审核一张发货单。
    """
    request_body: Dict[str, Any] = {
        "consignment": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/consignment/verify"
    
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "发货单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "发货单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审发货单 Tool函数 =====================
def u8_consignment_unverify_tool(input_data: ConsignmentUnverifyInput, client: Any) -> str:
    """
    弃审一张发货单。
    """
    request_body: Dict[str, Any] = {
        "consignment": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/consignment/unverify"
    
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "发货单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "发货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审批发货单(工作流) Tool函数 =====================
def u8_consignment_audit_tool(input_data: ConsignmentAuditInput, client: Any) -> str:
    """
    审批发货单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: Dict[str, Any] = {
        "consignment": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/consignment/audit"
    
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "发货单审批失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "发货单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审发货单(工作流) Tool函数 =====================
def u8_consignment_abandon_tool(input_data: ConsignmentAbandonInput, client: Any) -> str:
    """
    弃审发货单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: Dict[str, Any] = {
        "consignment": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/consignment/abandon"
    
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "发货单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "发货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取发货单工作流按钮状态 Tool函数 =====================
def u8_consignment_buttonstate_tool(input_data: ConsignmentButtonStateInput, client: Any) -> str:
    """
    获取发货单工作流按钮是否可用状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/consignment/buttonstate"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取发货单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取发货单工作流按钮状态成功",
            "data": result.get("buttonstate"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取发货单待办任务 Tool函数 =====================
def u8_consignment_tasks_tool(input_data: ConsignmentTasksInput, client: Any) -> str:
    """
    获取发货单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/consignment/tasks"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取发货单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取发货单待办任务成功",
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


# ===================== 查看发货单审批进程 Tool函数 =====================
def u8_consignment_history_tool(input_data: ConsignmentHistoryInput, client: Any) -> str:
    """
    查看发货单审批进程，获取单据的审批历史记录。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/consignment/history"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取发货单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取发货单审批进程成功",
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


# ===================== 获取单个客户调价单 Tool函数 =====================
def u8_cuspricejust_get_tool(input_data: CuspricejustGetInput, client: Any) -> str:
    """
    获取单个客户调价单详情。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/cuspricejust/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取客户调价单失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取客户调价单成功",
            "data": result.get("cuspricejust"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取客户调价单 Tool函数 =====================
def u8_cuspricejust_list_tool(input_data: CuspricejustListInput, client: Any) -> str:
    """
    批量获取客户调价单列表，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/cuspricejust/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取客户调价单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取客户调价单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "cuspricejust": result.get("cuspricejust")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增销售发票 Tool函数 =====================
def u8_saleinvoice_add_tool(input_data: SaleinvoiceAddInput, client: Any) -> str:
    """
    新增一张销售发票。
    """
    request_body: Dict[str, Any] = {
        "saleinvoice": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleinvoice/add"
    
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "销售发票新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "销售发票新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个销售发票 Tool函数 =====================
def u8_saleinvoice_get_tool(input_data: SaleinvoiceGetInput, client: Any) -> str:
    """
    获取单个销售发票详情。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleinvoice/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取销售发票失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取销售发票成功",
            "data": result.get("saleinvoice"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取销售发票列表 Tool函数 =====================
def u8_saleinvoice_list_tool(input_data: SaleinvoiceListInput, client: Any) -> str:
    """
    获取销售发票列表，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleinvoicelist/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取销售发票列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取销售发票列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "saleinvoicelist": result.get("saleinvoicelist")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取销售结账状态 Tool函数 =====================
def u8_mendsa_list_tool(input_data: MendsaListInput, client: Any) -> str:
    """
    批量获取销售结账状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/mendsa/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取销售结账状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取销售结账状态成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "mendsa": result.get("mendsa")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)
    

# ============================================
# Tool函数 (Tool Functions)
# ============================================

# ===================== 新增销售订单 Tool函数 =====================
def u8_saleorder_add_tool(input_data: SaleorderAddInput, client: Any) -> str:
    """
    新增一张销售订单
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/add"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单新增成功",
            "data": {"id": result.get("id"), "tradeid": result.get("tradeid")},
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个销售订单 Tool函数 =====================
def u8_saleorder_get_tool(input_data: SaleorderGetInput, client: Any) -> str:
    """
    获取单张销售订单
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleorder/get"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售订单成功",
            "data": result.get("saleorder"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 查询销售订单列表 Tool函数 =====================
def u8_saleorder_list_tool(input_data: SaleorderListInput, client: Any) -> str:
    """
    获取销售订单列表信息
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleorderlist/batch_get"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售订单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "saleorderlist": result.get("saleorderlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核销售订单(verify) Tool函数 =====================
def u8_saleorder_verify_tool(input_data: SaleorderVerifyInput, client: Any) -> str:
    """
    审核一张销售订单
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/verify"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审销售订单(unverify) Tool函数 =====================
def u8_saleorder_unverify_tool(input_data: SaleorderUnverifyInput, client: Any) -> str:
    """
    弃审一张销售订单
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/unverify"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 销售订单审批(audit) Tool函数 =====================
def u8_saleorder_audit_tool(input_data: SaleorderAuditInput, client: Any) -> str:
    """
    审核销售订单（工作流审批）
    执行审批动作前，需要保证审批人已经进行ERP登录授权
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/audit"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 销售订单弃审(abandon) Tool函数 =====================
def u8_saleorder_abandon_tool(input_data: SaleorderAbandonInput, client: Any) -> str:
    """
    弃审销售订单（工作流弃审）
    执行弃审动作前，需要保证审批人已经进行ERP登录授权
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/abandon"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取销售订单按钮状态 Tool函数 =====================
def u8_saleorder_buttonstate_tool(input_data: SaleorderButtonstateInput, client: Any) -> str:
    """
    获取销售订单工作流按钮是否可用状态
    只支持12.0版本，且需要打最新的WF补丁
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleorder/buttonstate"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售订单按钮状态成功",
            "data": result.get("buttonstate"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取销售订单待办任务 Tool函数 =====================
def u8_saleorder_tasks_tool(input_data: SaleorderTasksInput, client: Any) -> str:
    """
    获取销售订单待办任务
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleorder/tasks"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售订单待办任务成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "tasks": result.get("tasks", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取销售订单审批历史 Tool函数 =====================
def u8_saleorder_history_tool(input_data: SaleorderHistoryInput, client: Any) -> str:
    """
    查看销售订单审批进程
    执行审批动作前，需要保证审批人已经进行ERP登录授权
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/saleorder/history"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售订单审批历史成功",
            "data": {"history": result.get("history", [])},
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 打开销售订单 Tool函数 =====================
def u8_saleorder_open_tool(input_data: SaleorderOpenInput, client: Any) -> str:
    """
    打开一张销售订单
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/open"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单打开成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 关闭销售订单 Tool函数 =====================
def u8_saleorder_close_tool(input_data: SaleorderCloseInput, client: Any) -> str:
    """
    关闭一张销售订单
    """
    request_body: Dict[str, Any] = {
        "saleorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/saleorder/close"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售订单关闭成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增销售退货单 Tool函数 =====================
def u8_returnorder_add_tool(input_data: ReturnorderAddInput, client: Any) -> str:
    """
    新增一张销售退货单
    """
    request_body: Dict[str, Any] = {
        "returnorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/returnorder/add"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售退货单新增成功",
            "data": {"id": result.get("id"), "tradeid": result.get("tradeid")},
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个销售退货单 Tool函数 =====================
def u8_returnorder_get_tool(input_data: ReturnorderGetInput, client: Any) -> str:
    """
    获取单个销售退货单
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/returnorder/get"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售退货单成功",
            "data": result.get("returnorder"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 查询销售退货单列表 Tool函数 =====================
def u8_returnorder_list_tool(input_data: ReturnorderListInput, client: Any) -> str:
    """
    获取销售退货单列表
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/returnorderlist/batch_get"
    try:
        result = client.request_api("GET", api_path, inparams=params)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "获取销售退货单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "returnorderlist": result.get("returnorderlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核销售退货单 Tool函数 =====================
def u8_returnorder_verify_tool(input_data: ReturnorderVerifyInput, client: Any) -> str:
    """
    审核一张销售退货单
    """
    request_body: Dict[str, Any] = {
        "returnorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/returnorder/verify"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售退货单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审销售退货单 Tool函数 =====================
def u8_returnorder_unverify_tool(input_data: ReturnorderUnverifyInput, client: Any) -> str:
    """
    弃审一张销售退货单
    """
    request_body: Dict[str, Any] = {
        "returnorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/returnorder/unverify"
    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "success": True,
            "message": "销售退货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ============================================
# Schema定义 (Schema Definitions)
# ============================================

# ===================== 新增发货单 Schema定义 =====================
U8_CONSIGNMENT_ADD_SCHEMA = {
    "name": "u8_consignment_add",
    "description": "新增一张发货单，用于创建销售发货记录",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据号"},
            "date": {"type": "string", "description": "单据日期(yyyy-MM-dd)"},
            "operation_type": {"type": "string", "description": "业务类型（必填）"},
            "saletype": {"type": "string", "description": "销售类型编码（必填）"},
            "saletypename": {"type": "string", "description": "销售类型"},
            "state": {"type": "string", "description": "订单状态"},
            "custcode": {"type": "string", "description": "客户编码（必填）"},
            "cusname": {"type": "string", "description": "客户"},
            "cusabbname": {"type": "string", "description": "客户简称"},
            "deptcode": {"type": "string", "description": "部门编码（必填）"},
            "deptname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "cdeliverunit": {"type": "string", "description": "收货单位"},
            "ccontactname": {"type": "string", "description": "收货联系人"},
            "cofficephone": {"type": "string", "description": "收货联系电话"},
            "cmobilephone": {"type": "string", "description": "收货联系人手机"},
            "cdeliveradd": {"type": "string", "description": "收货地址"},
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4(yyyy-MM-dd)"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6(yyyy-MM-dd)"},
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
            "remark": {"type": "string", "description": "备注"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"},
            "entry": {
                "type": "array",
                "description": "表体数据（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventory_code": {"type": "string", "description": "存货编码（必填）"},
                        "inventory_name": {"type": "string", "description": "存货名称"},
                        "warehouse_code": {"type": "string", "description": "仓库编码（必填）"},
                        "warehouse_name": {"type": "string", "description": "仓库名称"},
                        "invstd": {"type": "string", "description": "存货规格"},
                        "ccomunitcode": {"type": "string", "description": "主计量单位编码"},
                        "cinvm_unit": {"type": "string", "description": "主计量单位"},
                        "quantity": {"type": "number", "description": "数量（必填）"},
                        "price": {"type": "number", "description": "单价"},
                        "quotedprice": {"type": "number", "description": "报价"},
                        "taxprice": {"type": "number", "description": "含税单价"},
                        "money": {"type": "number", "description": "无税金额"},
                        "sum": {"type": "number", "description": "价税合计"},
                        "taxrate": {"type": "number", "description": "税率"},
                        "tax": {"type": "number", "description": "税额"},
                        "natprice": {"type": "number", "description": "本币单价"},
                        "natmoney": {"type": "number", "description": "本币金额"},
                        "nattax": {"type": "number", "description": "本币税额"},
                        "natsum": {"type": "number", "description": "本币价税合计"},
                        "discount": {"type": "number", "description": "折扣额"},
                        "natdiscount": {"type": "number", "description": "本币折扣额"},
                        "discount1": {"type": "number", "description": "扣率(%)"},
                        "discount2": {"type": "number", "description": "扣率2(%)"},
                        "socode": {"type": "string", "description": "销售订单号"},
                        "batch": {"type": "string", "description": "批号"},
                        "ExpirationDate": {"type": "string", "description": "有效期至(yyyy-MM-dd)"},
                        "cmassunit": {"type": "string", "description": "保质期单位"},
                        "ExpirationItem": {"type": "string", "description": "有效期计算项(yyyy-MM-dd)"},
                        "dmdate": {"type": "string", "description": "生产日期(yyyy-MM-dd)"},
                        "overdate": {"type": "string", "description": "失效日期(yyyy-MM-dd)"},
                        "ExpiratDateCalcu": {"type": "number", "description": "有效期推算方式"},
                        "imassdate": {"type": "number", "description": "保质期"},
                        "item_code": {"type": "string", "description": "项目编码"},
                        "unit_code": {"type": "string", "description": "辅计量单位"},
                        "num": {"type": "number", "description": "件数"},
                        "unitrate": {"type": "number", "description": "换算率"},
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
                        "define36": {"type": "string", "description": "单据体自定义项15(yyyy-MM-dd)"},
                        "define37": {"type": "string", "description": "单据体自定义项16(yyyy-MM-dd)"},
                        "rowno": {"type": "number", "description": "行号（必填）"}
                    },
                    "required": ["inventory_code", "warehouse_code", "quantity", "rowno"]
                }
            }
        },
        "required": ["operation_type", "saletype", "custcode", "deptcode", "entry"]
    }
}

# ===================== 获取单张发货单 Schema定义 =====================
U8_CONSIGNMENT_GET_SCHEMA = {
    "name": "u8_consignment_get",
    "description": "获取单张发货单详情",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "发货单ID（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 获取发货单列表 Schema定义 =====================
U8_CONSIGNMENT_LIST_SCHEMA = {
    "name": "u8_consignment_list",
    "description": "获取发货单列表信息，支持分页和条件查询",
    "parameters": {
        "type": "object",
        "properties": {
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始制单日期(yyyy-MM-dd)"},
            "date_end": {"type": "string", "description": "结束制单日期(yyyy-MM-dd)"},
            "state": {"type": "string", "description": "订单状态"},
            "custcode": {"type": "string", "description": "客户编码"},
            "cusname": {"type": "string", "description": "客户名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "remark": {"type": "string", "description": "备注关键字"},
            "socode": {"type": "string", "description": "销售订单号"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"}
        }
    }
}

# ===================== 审核发货单 Schema定义 =====================
U8_CONSIGNMENT_VERIFY_SCHEMA = {
    "name": "u8_consignment_verify",
    "description": "审核一张发货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审发货单 Schema定义 =====================
U8_CONSIGNMENT_UNVERIFY_SCHEMA = {
    "name": "u8_consignment_unverify",
    "description": "弃审一张发货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 审批发货单(工作流) Schema定义 =====================
U8_CONSIGNMENT_AUDIT_SCHEMA = {
    "name": "u8_consignment_audit",
    "description": "审批发货单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "opinion": {"type": "string", "description": "审批意见"},
            "agree": {"type": "number", "description": "是否同意(1=同意;0=不同意)（必填）"}
        },
        "required": ["voucher_code", "agree"]
    }
}

# ===================== 弃审发货单(工作流) Schema定义 =====================
U8_CONSIGNMENT_ABANDON_SCHEMA = {
    "name": "u8_consignment_abandon",
    "description": "弃审发货单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "opinion": {"type": "string", "description": "审批意见"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取发货单工作流按钮状态 Schema定义 =====================
U8_CONSIGNMENT_BUTTONSTATE_SCHEMA = {
    "name": "u8_consignment_buttonstate",
    "description": "获取发货单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
    "parameters": {
        "type": "object",
        "properties": {
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取发货单待办任务 Schema定义 =====================
U8_CONSIGNMENT_TASKS_SCHEMA = {
    "name": "u8_consignment_tasks",
    "description": "获取发货单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "string", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "number", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "number", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "number", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "number", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        }
    }
}

# ===================== 查看发货单审批进程 Schema定义 =====================
U8_CONSIGNMENT_HISTORY_SCHEMA = {
    "name": "u8_consignment_history",
    "description": "查看发货单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "审批人(用户编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "voucher_code": {"type": "string", "description": "单据编号（必填）"}
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取单个客户调价单 Schema定义 =====================
U8_CUSPRICEJUST_GET_SCHEMA = {
    "name": "u8_cuspricejust_get",
    "description": "获取单个客户调价单详情",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "单据号（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取客户调价单 Schema定义 =====================
U8_CUSPRICEJUST_LIST_SCHEMA = {
    "name": "u8_cuspricejust_list",
    "description": "批量获取客户调价单列表，支持多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ccode_begin": {"type": "string", "description": "起始单据号"},
            "ccode_end": {"type": "string", "description": "结束单据号"},
            "ddate_begin": {"type": "string", "description": "起始单据日期(yyyy-MM-dd)"},
            "ddate_end": {"type": "string", "description": "结束单据日期(yyyy-MM-dd)"},
            "cdepname": {"type": "string", "description": "调价部门"},
            "cpersonname": {"type": "string", "description": "调价业务员关键字"},
            "cmainmemo": {"type": "string", "description": "表头备注"},
            "cmaker": {"type": "string", "description": "制单人关键字"},
            "cverifier": {"type": "string", "description": "审核人关键字"},
            "dverifydate": {"type": "string", "description": "审核日期(yyyy-MM-dd)"},
            "drdate": {"type": "string", "description": "系统日期(yyyy-MM-dd)"},
            "ccusabbname": {"type": "string", "description": "客户简称关键字"},
            "cccname": {"type": "string", "description": "客户大类"},
            "cinvcode": {"type": "string", "description": "存货编码"},
            "cinvname": {"type": "string", "description": "存货名称关键字"},
            "cinvstd": {"type": "string", "description": "规格型号"},
            "ccomunitname": {"type": "string", "description": "计量单位"},
            "fminquantity": {"type": "number", "description": "数量下限"},
            "iinvscost": {"type": "number", "description": "批发价"},
            "icusdisrate": {"type": "number", "description": "客户扣率"},
            "iinvnowcost": {"type": "number", "description": "成交价"},
            "iinvsalecost": {"type": "number", "description": "零售单价"},
            "dstartdate": {"type": "string", "description": "开始日期(yyyy-MM-dd)"},
            "denddate": {"type": "string", "description": "结束日期(yyyy-MM-dd)"},
            "bsales": {"type": "boolean", "description": "是否促销价"},
            "fcusminprice": {"type": "number", "description": "客户最低售价"}
        }
    }
}

# ===================== 新增销售发票 Schema定义 =====================
U8_SALEINVOICE_ADD_SCHEMA = {
    "name": "u8_saleinvoice_add",
    "description": "新增一张销售发票",
    "parameters": {
        "type": "object",
        "properties": {
            "invoiceno": {"type": "string", "description": "发票号"},
            "vouchertype": {"type": "string", "description": "单据类型（必填）"},
            "saletypecode": {"type": "string", "description": "销售类型编号（必填）"},
            "date": {"type": "string", "description": "日期(yyyy-MM-dd)（必填）"},
            "departmentcode": {"type": "string", "description": "部门编号（必填）"},
            "personcode": {"type": "string", "description": "职员编号"},
            "customercode": {"type": "string", "description": "客商编号（必填）"},
            "payconditioncode": {"type": "string", "description": "付款条件编码"},
            "foreigncurrency": {"type": "string", "description": "外币名称"},
            "memory": {"type": "string", "description": "备注"},
            "currencyrate": {"type": "number", "description": "汇率"},
            "taxrate": {"type": "number", "description": "税率"},
            "isnegative": {"type": "boolean", "description": "负发票-正发票"},
            "bankcode": {"type": "string", "description": "本单位开户银行编号"},
            "invoiceversion": {"type": "string", "description": "发票版别"},
            "maker": {"type": "string", "description": "制单人"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "isfirst": {"type": "boolean", "description": "是否期初"},
            "itemclasscode": {"type": "string", "description": "项目大类编号"},
            "itemcode": {"type": "string", "description": "项目编码"},
            "define1": {"type": "string", "description": "自定义字段1"},
            "define2": {"type": "string", "description": "自定义字段2"},
            "define3": {"type": "string", "description": "自定义字段3"},
            "define4": {"type": "string", "description": "自定义字段4(yyyy-MM-dd)"},
            "define5": {"type": "number", "description": "自定义字段5"},
            "define6": {"type": "string", "description": "自定义字段6(yyyy-MM-dd)"},
            "define7": {"type": "number", "description": "自定义字段7"},
            "define8": {"type": "string", "description": "自定义字段8"},
            "define9": {"type": "string", "description": "自定义字段9"},
            "define10": {"type": "string", "description": "自定义字段10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "number", "description": "自定义项15"},
            "define16": {"type": "number", "description": "自定义项16"},
            "ispayedfirst": {"type": "number", "description": "1先发货;0先开票"},
            "customername": {"type": "string", "description": "综合开票客户名称"},
            "ccusaccount": {"type": "string", "description": "客户账号"},
            "cbaccount": {"type": "string", "description": "本单位账号"},
            "cdeliverunit": {"type": "string", "description": "收货单位名称"},
            "cdeliveradd": {"type": "string", "description": "收货地址"},
            "ccontactname": {"type": "string", "description": "收货联系人"},
            "cofficephone": {"type": "string", "description": "收货联系电话"},
            "cmobilephone": {"type": "string", "description": "收货联系手机"},
            "caddcode": {"type": "string", "description": "收获地址编码"},
            "cgatheringplan": {"type": "string", "description": "收付款协议编码"},
            "dcreditstart": {"type": "string", "description": "立账日(yyyy-MM-dd)"},
            "icreditdays": {"type": "number", "description": "账期"},
            "dgatheringdate": {"type": "string", "description": "到期日(yyyy-MM-dd)"},
            "bcredit": {"type": "boolean", "description": "是否立账单据"},
            "csource": {"type": "string", "description": "来源"},
            "ccusbank": {"type": "string", "description": "客户开户银行"},
            "entry": {
                "type": "array",
                "description": "表体数据（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "warehousecode": {"type": "string", "description": "仓库编码（必填）"},
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "quantity": {"type": "number", "description": "数量（必填）"},
                        "number": {"type": "number", "description": "件数"},
                        "quotedprice": {"type": "number", "description": "报价"},
                        "originalprice": {"type": "number", "description": "无税单价（必填）"},
                        "originaltaxedprice": {"type": "number", "description": "含税单价（必填）"},
                        "originalmoney": {"type": "number", "description": "无税金额（必填）"},
                        "originaltax": {"type": "number", "description": "税额（必填）"},
                        "originalsum": {"type": "number", "description": "价税合计（必填）"},
                        "price": {"type": "number", "description": "本币单价（必填）"},
                        "money": {"type": "number", "description": "本币金额（必填）"},
                        "tax": {"type": "number", "description": "本币税额（必填）"},
                        "sum": {"type": "number", "description": "本币价税合计（必填）"},
                        "taxrate": {"type": "number", "description": "税率(%)（必填）"},
                        "assistantunit": {"type": "string", "description": "销售单位编码"},
                        "originaldiscount": {"type": "number", "description": "折扣额"},
                        "discount": {"type": "number", "description": "本币折扣额"},
                        "memory": {"type": "string", "description": "备注"},
                        "serial": {"type": "string", "description": "批号"},
                        "accountrate1": {"type": "number", "description": "扣率(%)"},
                        "accountrate2": {"type": "number", "description": "扣率2(%)"},
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
                        "retailprice": {"type": "number", "description": "零售单价"},
                        "retailmoney": {"type": "number", "description": "零售金额"},
                        "itemclasscode": {"type": "string", "description": "项目大类编码"},
                        "itemcode": {"type": "string", "description": "项目编码"},
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
                        "batchproperty1": {"type": "string", "description": "批次属性1"},
                        "batchproperty2": {"type": "string", "description": "批次属性2"},
                        "batchproperty3": {"type": "string", "description": "批次属性3"},
                        "batchproperty4": {"type": "string", "description": "批次属性4"},
                        "batchproperty5": {"type": "string", "description": "批次属性5"},
                        "batchproperty6": {"type": "string", "description": "批次属性6"},
                        "batchproperty7": {"type": "string", "description": "批次属性7"},
                        "batchproperty8": {"type": "string", "description": "批次属性8"},
                        "batchproperty9": {"type": "string", "description": "批次属性9"},
                        "batchproperty10": {"type": "string", "description": "批次属性10"},
                        "exchangerate": {"type": "number", "description": "换算率"},
                        "unitid": {"type": "string", "description": "销售单位编码"},
                        "cmassunit": {"type": "string", "description": "保质期单位"},
                        "imassdate": {"type": "number", "description": "保质期"},
                        "dmdate": {"type": "string", "description": "生产日期(yyyy-MM-dd)"},
                        "invaliddate": {"type": "string", "description": "失效日期(yyyy-MM-dd)"},
                        "ExpirationDate": {"type": "string", "description": "有效期至"},
                        "ExpiratDateCalcu": {"type": "number", "description": "有效期推算方式"},
                        "ExpirationItem": {"type": "string", "description": "有效期计算项(yyyy-MM-dd)"},
                        "cvmivencode": {"type": "string", "description": "供货商编码"},
                        "irowno": {"type": "number", "description": "行号"},
                        "ReasonCode": {"type": "string", "description": "退货原因编码"},
                        "bsaleprice": {"type": "boolean", "description": "报价含税"},
                        "bgift": {"type": "boolean", "description": "赠品"},
                        "fcusminprice": {"type": "number", "description": "最低售价"},
                        "icalctype": {"type": "number", "description": "发货模式"},
                        "fchildqty": {"type": "number", "description": "使用数量"},
                        "fchildrate": {"type": "number", "description": "权重比例"}
                    },
                    "required": ["warehousecode", "inventorycode", "quantity", "originalprice", "originaltaxedprice", "originalmoney", "originaltax", "originalsum", "price", "money", "tax", "sum", "taxrate"]
                }
            }
        },
        "required": ["vouchertype", "saletypecode", "date", "departmentcode", "customercode", "entry"]
    }
}

# ===================== 获取单个销售发票 Schema定义 =====================
U8_SALEINVOICE_GET_SCHEMA = {
    "name": "u8_saleinvoice_get",
    "description": "获取单个销售发票详情",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "销售发票ID（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 获取销售发票列表 Schema定义 =====================
U8_SALEINVOICE_LIST_SCHEMA = {
    "name": "u8_saleinvoice_list",
    "description": "获取销售发票列表，支持多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "invoiceno_begin": {"type": "string", "description": "起始销售发票号"},
            "invoiceno_end": {"type": "string", "description": "结束销售发票号"},
            "ccusname": {"type": "string", "description": "客户名称关键字"},
            "cstname": {"type": "string", "description": "销售类型"},
            "csocode": {"type": "string", "description": "销售单号"},
            "cdlcode": {"type": "string", "description": "发货单号"},
            "vouchertype": {"type": "string", "description": "单据类型"},
            "saletypecode": {"type": "string", "description": "销售类型编号"},
            "date_begin": {"type": "string", "description": "起始日期(yyyy-MM-dd)"},
            "date_end": {"type": "string", "description": "结束日期(yyyy-MM-dd)"},
            "departmentcode": {"type": "string", "description": "部门编号"},
            "personcode": {"type": "string", "description": "职员编号"},
            "customercode": {"type": "string", "description": "客商编号"},
            "isnegative": {"type": "number", "description": "负发票-正发票"},
            "maker": {"type": "string", "description": "制单人关键字"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "isfirst": {"type": "boolean", "description": "是否期初"},
            "itemclasscode": {"type": "string", "description": "项目大类编号"},
            "itemcode": {"type": "string", "description": "项目编码"},
            "define1": {"type": "string", "description": "自定义字段1"},
            "define2": {"type": "string", "description": "自定义字段2"},
            "define3": {"type": "string", "description": "自定义字段3"},
            "define4": {"type": "string", "description": "自定义字段4(yyyy-MM-dd)"},
            "define5": {"type": "number", "description": "自定义字段5"},
            "define6": {"type": "string", "description": "自定义字段6(yyyy-MM-dd)"},
            "define7": {"type": "number", "description": "自定义字段7"},
            "define8": {"type": "string", "description": "自定义字段8"},
            "define9": {"type": "string", "description": "自定义字段9"},
            "define10": {"type": "string", "description": "自定义字段10"},
            "define11": {"type": "string", "description": "自定义项11"},
            "define12": {"type": "string", "description": "自定义项12"},
            "define13": {"type": "string", "description": "自定义项13"},
            "define14": {"type": "string", "description": "自定义项14"},
            "define15": {"type": "number", "description": "自定义项15"},
            "define16": {"type": "number", "description": "自定义项16"},
            "csource": {"type": "string", "description": "来源"},
            "csaleout": {"type": "string", "description": "出库单号"},
            "ccusabbname": {"type": "string", "description": "客户简称关键字"},
            "cdepname": {"type": "string", "description": "部门关键字"},
            "cpersonname": {"type": "string", "description": "职员名称关键字"}
        }
    }
}

# ===================== 批量获取销售结账状态 Schema定义 =====================
U8_MENDSA_LIST_SCHEMA = {
    "name": "u8_mendsa_list",
    "description": "批量获取销售结账状态",
    "parameters": {
        "type": "object",
        "properties": {
            "iyear": {"type": "number", "description": "会计年度"},
            "iperiod_begin": {"type": "number", "description": "起始会计期间"},
            "iperiod_end": {"type": "number", "description": "结束会计期间"}
        }
    }
}



# ===================== 销售订单表体 Schema =====================
U8_SALEORDER_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
        "inventoryname": {"type": "string", "description": "存货"},
        "invstd": {"type": "string", "description": "规格型号"},
        "unitcode": {"type": "string", "description": "销售单位编码"},
        "unitname": {"type": "string", "description": "销售单位"},
        "unitrate": {"type": "number", "description": "换算率"},
        "quantity": {"type": "number", "description": "数量（必填）"},
        "num": {"type": "number", "description": "件数"},
        "unitprice": {"type": "number", "description": "单价"},
        "quotedprice": {"type": "number", "description": "报价"},
        "taxunitprice": {"type": "number", "description": "含税单价"},
        "money": {"type": "number", "description": "无税金额"},
        "taxrate": {"type": "number", "description": "税率"},
        "sum": {"type": "number", "description": "价税合计"},
        "discount": {"type": "number", "description": "折扣额"},
        "natdiscount": {"type": "number", "description": "本币折扣额"},
        "discountrate": {"type": "number", "description": "扣率(%)"},
        "discountrate2": {"type": "number", "description": "扣率2(%)"},
        "natmoney": {"type": "number", "description": "本币金额"},
        "natunitprice": {"type": "number", "description": "本币单价"},
        "tax": {"type": "number", "description": "税额"},
        "nattax": {"type": "number", "description": "本币税额"},
        "natsum": {"type": "number", "description": "本币价税合计"},
        "ccontractid": {"type": "string", "description": "合同编码"},
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
        "define36": {"type": "string", "description": "单据体自定义项15(yyyy-MM-dd)"},
        "define37": {"type": "string", "description": "单据体自定义项16(yyyy-MM-dd)"},
        "bgift": {"type": "number", "description": "是否赠品(0=非赠品;1=赠品)（必填）"},
        "rowno": {"type": "number", "description": "行号（必填）"}
    },
    "required": ["inventorycode", "quantity", "bgift", "rowno"]
}


# ===================== 新增销售订单 Schema =====================
U8_SALEORDER_ADD_SCHEMA = {
    "name": "u8_saleorder_add",
    "description": "新增一张销售订单",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "订单号"},
            "date": {"type": "string", "description": "日期(yyyy-MM-dd)"},
            "businesstype": {"type": "string", "description": "业务类型（必填）"},
            "typecode": {"type": "string", "description": "销售类型编码（必填）"},
            "typename": {"type": "string", "description": "销售类型"},
            "state": {"type": "string", "description": "单据状态"},
            "custcode": {"type": "string", "description": "客户编码（必填）"},
            "cusname": {"type": "string", "description": "客户名称"},
            "cusabbname": {"type": "string", "description": "客户简称"},
            "deptcode": {"type": "string", "description": "部门编码（必填）"},
            "deptname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "dpremodatebt": {"type": "string", "description": "预完工日期(yyyy-MM-dd)"},
            "dpredatebt": {"type": "string", "description": "预发货日期(yyyy-MM-dd)"},
            "sendaddress": {"type": "string", "description": "发货地址"},
            "ccusperson": {"type": "string", "description": "联系人"},
            "ccuspersoncode": {"type": "string", "description": "联系人编码"},
            "caddcode": {"type": "string", "description": "收货地址编码"},
            "taxrate": {"type": "number", "description": "税率，默认16"},
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4(yyyy-MM-dd)"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6(yyyy-MM-dd)"},
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
            "memo": {"type": "string", "description": "备注"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"},
            "entry": {
                "type": "array",
                "description": "表体数据（必填）",
                "items": U8_SALEORDER_ENTRY_SCHEMA
            }
        },
        "required": ["businesstype", "typecode", "custcode", "deptcode", "entry"]
    }
}


# ===================== 获取单个销售订单 Schema =====================
U8_SALEORDER_GET_SCHEMA = {
    "name": "u8_saleorder_get",
    "description": "获取单张销售订单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "订单编号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}


# ===================== 查询销售订单列表 Schema =====================
U8_SALEORDER_LIST_SCHEMA = {
    "name": "u8_saleorder_list",
    "description": "获取销售订单列表信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始订单编号"},
            "code_end": {"type": "string", "description": "结束订单编号"},
            "date_begin": {"type": "string", "description": "起始订单日期，格式:yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束订单日期，格式:yyyy-MM-dd"},
            "dpremodatebt_begin": {"type": "string", "description": "起始预完工日期，格式:yyyy-MM-dd"},
            "dpremodatebt_end": {"type": "string", "description": "结束预完工日期，格式:yyyy-MM-dd"},
            "dpredatebt_begin": {"type": "string", "description": "起始预发货日期，格式:yyyy-MM-dd"},
            "dpredatebt_end": {"type": "string", "description": "结束预发货日期，格式:yyyy-MM-dd"},
            "state": {"type": "string", "description": "订单状态"},
            "custcode": {"type": "string", "description": "客户编码"},
            "cusname": {"type": "string", "description": "客户名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "memo": {"type": "string", "description": "备注关键字"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"},
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
            "define16": {"type": "number", "description": "自定义项16"},
            "timestamp_begin": {"type": "number", "description": "起始时间戳"},
            "timestamp_end": {"type": "number", "description": "结束时间戳"},
            "fhstatus": {"type": "number", "description": "发货状态(0=未发货;1=部分发货;2=全部发货)"}
        },
        "required": []
    }
}


# ===================== 审核销售订单(verify) Schema =====================
U8_SALEORDER_VERIFY_SCHEMA = {
    "name": "u8_saleorder_verify",
    "description": "审核一张销售订单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 弃审销售订单(unverify) Schema =====================
U8_SALEORDER_UNVERIFY_SCHEMA = {
    "name": "u8_saleorder_unverify",
    "description": "弃审一张销售订单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 销售订单审批(audit) Schema =====================
U8_SALEORDER_AUDIT_SCHEMA = {
    "name": "u8_saleorder_audit",
    "description": "审核销售订单（工作流审批）。执行审批动作前，需要保证审批人已经进行ERP登录授权",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "opinion": {"type": "string", "description": "审批意见"},
            "agree": {"type": "number", "description": "是否同意(1=同意;0=不同意)（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code", "agree"]
    }
}


# ===================== 销售订单弃审(abandon) Schema =====================
U8_SALEORDER_ABANDON_SCHEMA = {
    "name": "u8_saleorder_abandon",
    "description": "弃审销售订单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(操作员编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "opinion": {"type": "string", "description": "审批意见"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 获取销售订单按钮状态 Schema =====================
U8_SALEORDER_BUTTONSTATE_SCHEMA = {
    "name": "u8_saleorder_buttonstate",
    "description": "获取销售订单工作流按钮是否可用状态。只支持12.0版本，且需要打最新的WF补丁",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 获取销售订单待办任务 Schema =====================
U8_SALEORDER_TASKS_SCHEMA = {
    "name": "u8_saleorder_tasks",
    "description": "获取销售订单待办任务",
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


# ===================== 获取销售订单审批历史 Schema =====================
U8_SALEORDER_HISTORY_SCHEMA = {
    "name": "u8_saleorder_history",
    "description": "查看销售订单审批进程。执行审批动作前，需要保证审批人已经进行ERP登录授权",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "user_id": {"type": "string", "description": "审批人(用户编码)"},
            "person_code": {"type": "string", "description": "审批人(人员编码)"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 打开销售订单 Schema =====================
U8_SALEORDER_OPEN_SCHEMA = {
    "name": "u8_saleorder_open",
    "description": "打开一张销售订单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 关闭销售订单 Schema =====================
U8_SALEORDER_CLOSE_SCHEMA = {
    "name": "u8_saleorder_close",
    "description": "关闭一张销售订单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 销售退货单表体 Schema =====================
U8_RETURNORDER_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "inventory_code": {"type": "string", "description": "存货编码（必填）"},
        "inventory_name": {"type": "string", "description": "存货名称"},
        "warehouse_code": {"type": "string", "description": "仓库编码（必填）"},
        "warehouse_name": {"type": "string", "description": "仓库名称"},
        "invstd": {"type": "string", "description": "存货规格"},
        "ccomunitcode": {"type": "string", "description": "主计量单位编码"},
        "cinvm_unit": {"type": "string", "description": "主计量单位"},
        "quantity": {"type": "number", "description": "数量（必填）"},
        "price": {"type": "number", "description": "单价"},
        "quotedprice": {"type": "number", "description": "报价"},
        "taxprice": {"type": "number", "description": "含税单价"},
        "money": {"type": "number", "description": "无税金额"},
        "sum": {"type": "number", "description": "价税合计"},
        "taxrate": {"type": "number", "description": "税率"},
        "tax": {"type": "number", "description": "税额"},
        "natprice": {"type": "number", "description": "本币单价"},
        "natmoney": {"type": "number", "description": "本币金额"},
        "nattax": {"type": "number", "description": "本币税额"},
        "natsum": {"type": "number", "description": "本币价税合计"},
        "discount": {"type": "number", "description": "折扣额"},
        "natdiscount": {"type": "number", "description": "本币折扣额"},
        "discount1": {"type": "number", "description": "扣率(%)"},
        "discount2": {"type": "number", "description": "扣率2(%)"},
        "socode": {"type": "string", "description": "销售订单号"},
        "ReasonCode": {"type": "string", "description": "退货原因编码"},
        "ReasonName": {"type": "string", "description": "退货原因"},
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
        "define36": {"type": "string", "description": "单据体自定义项15(yyyy-MM-dd)"},
        "define37": {"type": "string", "description": "单据体自定义项16(yyyy-MM-dd)"},
        "rowno": {"type": "number", "description": "行号（必填）"}
    },
    "required": ["inventory_code", "warehouse_code", "quantity", "rowno"]
}


# ===================== 新增销售退货单 Schema =====================
U8_RETURNORDER_ADD_SCHEMA = {
    "name": "u8_returnorder_add",
    "description": "新增一张销售退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据号"},
            "date": {"type": "string", "description": "单据日期(yyyy-MM-dd)"},
            "operation_type": {"type": "string", "description": "业务类型（必填）"},
            "saletype": {"type": "string", "description": "销售类型编码（必填）"},
            "saletypename": {"type": "string", "description": "销售类型"},
            "state": {"type": "string", "description": "订单状态"},
            "custcode": {"type": "string", "description": "客户编码（必填）"},
            "cusname": {"type": "string", "description": "客户"},
            "cusabbname": {"type": "string", "description": "客户简称"},
            "deptcode": {"type": "string", "description": "部门编码（必填）"},
            "deptname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4(yyyy-MM-dd)"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6(yyyy-MM-dd)"},
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
            "remark": {"type": "string", "description": "备注"},
            "entry": {
                "type": "array",
                "description": "表体数据（必填）",
                "items": U8_RETURNORDER_ENTRY_SCHEMA
            }
        },
        "required": ["operation_type", "saletype", "custcode", "deptcode", "entry"]
    }
}


# ===================== 获取单个销售退货单 Schema =====================
U8_RETURNORDER_GET_SCHEMA = {
    "name": "u8_returnorder_get",
    "description": "获取单个销售退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "id（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}


# ===================== 查询销售退货单列表 Schema =====================
U8_RETURNORDER_LIST_SCHEMA = {
    "name": "u8_returnorder_list",
    "description": "获取销售退货单列表",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始单据号"},
            "code_end": {"type": "string", "description": "结束单据号"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "operation_type": {"type": "string", "description": "业务类型关键字"},
            "saletype": {"type": "string", "description": "销售类型编码"},
            "saletypename": {"type": "string", "description": "销售类型关键字"},
            "state": {"type": "string", "description": "订单状态"},
            "custcode": {"type": "string", "description": "客户编码"},
            "cusname": {"type": "string", "description": "客户名称关键字"},
            "cusabbname": {"type": "string", "description": "客户简称关键字"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "remark": {"type": "string", "description": "备注关键字"}
        },
        "required": []
    }
}


# ===================== 审核销售退货单 Schema =====================
U8_RETURNORDER_VERIFY_SCHEMA = {
    "name": "u8_returnorder_verify",
    "description": "审核一张销售退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "person_code": {"type": "string", "description": "审核人(人员编码)"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 弃审销售退货单 Schema =====================
U8_RETURNORDER_UNVERIFY_SCHEMA = {
    "name": "u8_returnorder_unverify",
    "description": "弃审一张销售退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}

