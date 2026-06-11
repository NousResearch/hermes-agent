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

# ===================== 供应商存货价格表 数据模型 =====================
class GetVenInvPriceInput(BaseModel):
    """批量获取供应商存货价格表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cvencode: Optional[str] = Field(None, description="供应商编码")
    cvenname: Optional[str] = Field(None, description="供应商名称")
    cvenabbname: Optional[str] = Field(None, description="供应商简称")
    cinvname: Optional[str] = Field(None, description="存货名称")
    cinvstd: Optional[str] = Field(None, description="存货规格")
    cinvccode: Optional[str] = Field(None, description="存货分类编码")
    cinvcname: Optional[str] = Field(None, description="存货分类名称")
    cinvcode_begin: Optional[str] = Field(None, description="起始存货编码")
    cinvcode_end: Optional[str] = Field(None, description="结束存货编码")
    denabledate_begin: Optional[str] = Field(None, description="起始生效日期，格式：yyyy-MM-dd")
    denabledate_end: Optional[str] = Field(None, description="结束生效日期，格式：yyyy-MM-dd")
    ddisabledate: Optional[str] = Field(None, description="失效日期")
    cexch_name: Optional[str] = Field(None, description="币种名称")
    bpromotion: Optional[str] = Field(None, description="促销")
    cmemo: Optional[str] = Field(None, description="备注")
    isupplytype: Optional[str] = Field(None, description="供货类型")
    btaxcost: Optional[str] = Field(None, description="含税")
    ctermcode: Optional[str] = Field(None, description="采购条件编码")
    ilowerlimit: Optional[float] = Field(None, description="价格下限")
    iupperlimit: Optional[float] = Field(None, description="价格上限")
    itaxrate: Optional[float] = Field(None, description="税率")


# ===================== 供应商存货调价单列表 数据模型 =====================
class GetVenPriceAdjustListInput(BaseModel):
    """获取调价单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始单据日期，格式：yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束单据日期，格式：yyyy-MM-dd")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称")
    depcode: Optional[str] = Field(None, description="部门编码")
    depname: Optional[str] = Field(None, description="部门名称")
    memo: Optional[str] = Field(None, description="备注")
    verifier: Optional[str] = Field(None, description="审核人")
    maker: Optional[str] = Field(None, description="制单人")


# ===================== 获取单张调价单 数据模型 =====================
class GetVenPriceAdjustInput(BaseModel):
    """获取单张调价单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="调价单编号（必填）")


# ===================== 调价单待办任务 数据模型 =====================
class GetVenPriceAdjustTasksInput(BaseModel):
    """获取调价单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[str] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[str] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[str] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[str] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[str] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 调价单审批进程 数据模型 =====================
class GetVenPriceAdjustHistoryInput(BaseModel):
    """查看调价单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 调价单工作流按钮状态 数据模型 =====================
class GetVenPriceAdjustButtonstateInput(BaseModel):
    """获取调价单工作流按钮状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审核调价单 数据模型 =====================
class AuditVenPriceAdjustInput(BaseModel):
    """审核调价单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: str = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 调价单体模型 =====================
class VenPriceAdjustEntry(BaseModel):
    """供应商存货调价单体模型"""
    cvencode: str = Field(..., description="供应商编码（必填）")
    cvenabbname: Optional[str] = Field(None, description="供应商简称")
    cvenname: Optional[str] = Field(None, description="供应商名称")
    cinvcode: str = Field(..., description="存货编码（必填）")
    cinvaddcode: Optional[str] = Field(None, description="存货代码")
    cinvname: Optional[str] = Field(None, description="存货名称")
    cinvstd: Optional[str] = Field(None, description="存货规格")
    dstartdate: Optional[str] = Field(None, description="生效日期，格式：yyyy-MM-dd")
    denddate: Optional[str] = Field(None, description="失效日期，格式：yyyy-MM-dd")
    cexch_name: Optional[str] = Field(None, description="币种名称")
    ctermcode: Optional[str] = Field(None, description="采购条件编码")
    ctermname: Optional[str] = Field(None, description="采购条件名称")
    bsales: Optional[str] = Field(None, description="销售")
    cbodymemo: Optional[str] = Field(None, description="备注")
    fminquantity: Optional[float] = Field(None, description="最小批量")
    iunitprice: str = Field(..., description="单价（必填）")
    itaxrate: Optional[float] = Field(None, description="税率")
    itaxunitprice: str = Field(..., description="含税单价（必填）")
    ivouchrowno: str = Field(..., description="行号（必填）")


# ===================== 新增调价单 数据模型 =====================
class AddVenPriceAdjustInput(BaseModel):
    """新增调价单输入模型"""
    ddate: Optional[str] = Field(None, description="单据日期，格式：yyyy-MM-dd")
    ccode: Optional[str] = Field(None, description="单据编号")
    cpersoncode: Optional[str] = Field(None, description="业务员编码")
    cpersonname: Optional[str] = Field(None, description="业务员名称")
    deptcode: Optional[str] = Field(None, description="部门编码")
    cdepname: Optional[str] = Field(None, description="部门名称")
    isupplytype: Optional[str] = Field(None, description="供货类型")
    memo: Optional[str] = Field(None, description="备注")
    btaxcost: Optional[str] = Field(None, description="含税")
    maker: Optional[str] = Field(None, description="制单人")
    entry: List[VenPriceAdjustEntry] = Field(..., description="调价单体列表（必填）")


# ===================== 弃审调价单 数据模型 =====================
class AbandonVenPriceAdjustInput(BaseModel):
    """弃审调价单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: str = Field(..., description="审批人(人员编码)（必填）")
    opinion: Optional[str] = Field(None, description="审批意见")


# ===================== 采购到货单列表 数据模型 =====================
class GetPurchaseReceiptListInput(BaseModel):
    """获取到货单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    date_begin: Optional[str] = Field(None, description="起始单据日期，格式：yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束单据日期，格式：yyyy-MM-dd")
    cauditdate_begin: Optional[str] = Field(None, description="起始审核日期，格式：yyyy-MM-dd")
    cauditdate_end: Optional[str] = Field(None, description="结束审核日期，格式：yyyy-MM-dd")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    purchasetypename: Optional[str] = Field(None, description="采购类型名称")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    vendorabbname: Optional[str] = Field(None, description="供应商简称")
    vendorname: Optional[str] = Field(None, description="供应商名称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称")
    payconditioncode: Optional[str] = Field(None, description="付款条件编码")
    payconditionname: Optional[str] = Field(None, description="付款条件名称")
    foreigncurrency: Optional[str] = Field(None, description="币种")
    memory: Optional[str] = Field(None, description="备注")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    ccloser: Optional[str] = Field(None, description="关闭人")
    shipcode: Optional[str] = Field(None, description="发货方式编码")
    shipname: Optional[str] = Field(None, description="发货方式名称")
    cauditdate: Optional[str] = Field(None, description="审核日期")
    cverifier: Optional[str] = Field(None, description="审核人")


# ===================== 获取单个到货单 数据模型 =====================
class GetPurchaseReceiptInput(BaseModel):
    """获取单个到货单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="到货单编号（必填）")


# ===================== 审核到货单 数据模型 =====================
class VerifyPurchaseReceiptInput(BaseModel):
    """审核到货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")


# ===================== 弃审到货单 数据模型 =====================
class UnVerifyPurchaseReceiptInput(BaseModel):
    """弃审到货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 到货单待办任务 数据模型 =====================
class GetPurchaseReceiptTasksInput(BaseModel):
    """获取到货单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    state: Optional[str] = Field(None, description="状态(0=待审;2=审批完成)")
    task_type_begin: Optional[str] = Field(None, description="起始类型值(1=正常;4=退回;5=退回到提交人)")
    task_type_end: Optional[str] = Field(None, description="结束类型值(1=正常;4=退回;5=退回到提交人)")
    task_desc: Optional[str] = Field(None, description="描述")
    submitter_code_begin: Optional[str] = Field(None, description="起始发起人编码")
    submitter_code_end: Optional[str] = Field(None, description="结束发起人编码")
    submitter_name: Optional[str] = Field(None, description="发起人名称关键字")


# ===================== 到货单审批进程 数据模型 =====================
class GetPurchaseReceiptHistoryInput(BaseModel):
    """查看到货单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 到货单是否启用工作流 数据模型 =====================
class GetPurchaseReceiptFlowenabledInput(BaseModel):
    """获取到货单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")


# ===================== 到货单工作流按钮状态 数据模型 =====================
class GetPurchaseReceiptButtonstateInput(BaseModel):
    """获取到货单工作流按钮状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")


# ===================== 审核到货单(工作流) 数据模型 =====================
class AuditPurchaseReceiptInput(BaseModel):
    """审核到货单(工作流)输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: str = Field(..., description="是否同意(1=同意;0=不同意)（必填）")


# ===================== 到货单体模型 =====================
class PurchaseReceiptEntry(BaseModel):
    """采购到货单体模型"""
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    inventorycode: str = Field(..., description="存货编码（必填）")
    cinva_unit: Optional[str] = Field(None, description="存货单位")
    iinvexchrate: Optional[float] = Field(None, description="换算率")
    serial: Optional[str] = Field(None, description="批号")
    originaltaxedprice: Optional[float] = Field(None, description="原币含税单价")
    quantity: str = Field(..., description="数量（必填）")
    number: Optional[str] = Field(None, description="件数")
    taxrate: Optional[float] = Field(None, description="税率")
    originalprice: Optional[float] = Field(None, description="原币无税单价")
    originalmoney: Optional[float] = Field(None, description="原币无税金额")
    originaltax: Optional[float] = Field(None, description="原币税额")
    originalsum: Optional[float] = Field(None, description="原币价税合计")
    price: Optional[float] = Field(None, description="本币无税单价")
    money: Optional[float] = Field(None, description="本币无税金额")
    tax: Optional[float] = Field(None, description="本币税额")
    sum: Optional[float] = Field(None, description="本币价税合计")
    ivouchrowno: Optional[str] = Field(None, description="行号")
    cbmemo: Optional[str] = Field(None, description="备注")


# ===================== 新增到货单 数据模型 =====================
class AddPurchaseReceiptInput(BaseModel):
    """新增到货单输入模型"""
    code: Optional[str] = Field(None, description="单据编号")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    vendorcode: str = Field(..., description="供应商编码（必填）")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    personcode: Optional[str] = Field(None, description="业务员编码")
    payconditioncode: Optional[str] = Field(None, description="付款条件编码")
    foreigncurrency: Optional[str] = Field(None, description="币种")
    foreigncurrencyrate: Optional[float] = Field(None, description="汇率")
    memory: Optional[str] = Field(None, description="备注")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    idiscounttaxtype: Optional[str] = Field(None, description="扣税类别")
    shipcode: Optional[str] = Field(None, description="发货方式编码")
    cvenpuomprotocol: Optional[str] = Field(None, description="供应商采购单位协议")
    entry: List[PurchaseReceiptEntry] = Field(..., description="到货单体列表（必填）")


# ===================== 弃审到货单(工作流) 数据模型 =====================
class AbandonPurchaseReceiptInput(BaseModel):
    """弃审到货单(工作流)输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: str = Field(..., description="审批人(操作员编码)（必填）")
    person_code: str = Field(..., description="审批人(人员编码)（必填）")
    opinion: Optional[str] = Field(None, description="审批意见")



# ===================== 采购发票表体数据模型 =====================
class PurchaseinvoiceEntryInput(BaseModel):
    """采购发票表体数据模型"""
    inventorycode: Optional[str] = Field(None, description="存货编码")
    quantity: Optional[float] = Field(None, description="数量")
    assistantunit: str = Field(..., description="辅计量单位（必填）")
    number: Optional[float] = Field(None, description="件数")
    originalprice: Optional[float] = Field(None, description="原币单价")
    oritaxcost: Optional[float] = Field(None, description="原币含税单价")
    originalmoney: Optional[float] = Field(None, description="原币金额")
    originaltax: Optional[float] = Field(None, description="原币税额")
    originalsum: Optional[float] = Field(None, description="原币价税合计")
    price: Optional[float] = Field(None, description="本币单价")
    money: float = Field(..., description="本币金额（必填）")
    tax: float = Field(..., description="本币税额（必填）")
    sum: Optional[float] = Field(None, description="本币价税合计")
    taxrate: float = Field(..., description="税率（必填）")
    define22: str = Field(..., description="表体自定义项1（必填）")
    define23: str = Field(..., description="表体自定义项2（必填）")
    define24: str = Field(..., description="表体自定义项3（必填）")
    define25: str = Field(..., description="表体自定义项4（必填）")
    define26: str = Field(..., description="表体自定义项5（必填）")
    define27: Optional[str] = Field(None, description="表体自定义项6")
    define28: Optional[str] = Field(None, description="表体自定义项7")
    define29: Optional[str] = Field(None, description="表体自定义项8")
    define30: Optional[str] = Field(None, description="表体自定义项9")
    define31: Optional[str] = Field(None, description="表体自定义项10")
    define32: Optional[str] = Field(None, description="表体自定义项11")
    define33: Optional[str] = Field(None, description="表体自定义项12")
    define34: Optional[str] = Field(None, description="表体自定义项13")
    define35: Optional[str] = Field(None, description="表体自定义项14")
    define36: Optional[str] = Field(None, description="表体自定义项15")
    define37: Optional[str] = Field(None, description="表体自定义项16")
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
    isfee: Optional[bool] = Field(None, description="是否为费用")
    ivouchrowno: Optional[float] = Field(None, description="行号")


# ===================== 新增采购发票 数据模型 =====================
class PurchaseinvoiceAddInput(BaseModel):
    """新增采购发票输入模型"""
    csource: Optional[str] = Field(None, description="单据来源")
    invoicetype: str = Field(..., description="发票类型（必填）")
    invoicecode: str = Field(..., description="发票号（必填）")
    purchasecode: str = Field(..., description="采购类型编号（必填）")
    date: str = Field(..., description="开票日期（必填）")
    vendorcode: Optional[str] = Field(None, description="供应商编号")
    delegatecode: str = Field(..., description="代垫单位编号（必填）")
    departmentcode: Optional[str] = Field(None, description="部门编号")
    personcode: Optional[str] = Field(None, description="职员编号")
    dsdate: Optional[str] = Field(None, description="结算日期")
    idiscounttaxtype: Optional[float] = Field(None, description="扣税类别")
    payconditioncode: Optional[str] = Field(None, description="付款条件编码")
    foreigncurrency: Optional[str] = Field(None, description="外币名称")
    foreigncurrencyrate: Optional[str] = Field(None, description="汇率")
    taxrate: Optional[float] = Field(None, description="税率")
    memory: Optional[str] = Field(None, description="备注")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    isnegative: Optional[bool] = Field(None, description="负发票标志")
    protocolcode: Optional[str] = Field(None, description="收付款协议编码")
    define1: Optional[str] = Field(None, description="自定义字段1")
    define2: Optional[str] = Field(None, description="自定义字段2")
    define3: Optional[str] = Field(None, description="自定义字段3")
    define4: Optional[str] = Field(None, description="自定义字段4")
    define5: Optional[str] = Field(None, description="自定义字段5")
    define6: Optional[str] = Field(None, description="自定义字段6")
    define7: Optional[str] = Field(None, description="自定义字段7")
    define8: Optional[str] = Field(None, description="自定义字段8")
    define9: Optional[str] = Field(None, description="自定义字段9")
    define10: Optional[str] = Field(None, description="自定义字段10")
    define11: Optional[str] = Field(None, description="自定义字段11")
    define12: Optional[str] = Field(None, description="自定义字段12")
    define13: Optional[str] = Field(None, description="自定义字段13")
    define14: Optional[str] = Field(None, description="自定义字段14")
    define15: Optional[str] = Field(None, description="自定义字段15")
    define16: Optional[str] = Field(None, description="自定义字段16")
    entry: Optional[List[PurchaseinvoiceEntryInput]] = Field(None, description="表体数据")


# ===================== 获取单个采购发票 数据模型 =====================
class PurinvoiceGetInput(BaseModel):
    """获取单个采购发票输入模型"""
    id: str = Field(..., description="采购发票号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 查询采购发票列表 数据模型 =====================
class PurinvoiceListInput(BaseModel):
    """查询采购发票列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    type: Optional[str] = Field(None, description="采购发票类型关键字")
    cpbvcode_begin: Optional[str] = Field(None, description="起始采购发票号")
    cpbvcode_end: Optional[str] = Field(None, description="结束采购发票号")
    dpbvdate_begin: Optional[str] = Field(None, description="起始开票日期")
    dpbvdate_end: Optional[str] = Field(None, description="结束开票日期")
    cexchrate: Optional[float] = Field(None, description="汇率")
    cexch_name: Optional[str] = Field(None, description="币种")
    cexch_code: Optional[str] = Field(None, description="币种编码")
    cdepname: Optional[str] = Field(None, description="部门名称关键字")
    cdepcode: Optional[str] = Field(None, description="部门名称编码")
    cptname: Optional[str] = Field(None, description="采购类型名称关键字")
    cptcode: Optional[str] = Field(None, description="采购类型编码")
    cpersonname: Optional[str] = Field(None, description="业务员名称")
    cpersoncode: Optional[str] = Field(None, description="业务员编码")
    cvenname: Optional[str] = Field(None, description="供应商名称关键字")
    cvencode: Optional[str] = Field(None, description="供应商编码")
    cvenabbname: Optional[str] = Field(None, description="供应商简称关键字")
    ipbvtaxrate: Optional[float] = Field(None, description="表头税率")
    cpbvbilltype: Optional[str] = Field(None, description="发票类型")
    cbustype: Optional[str] = Field(None, description="业务类型")


# ===================== 采购结账状态 数据模型 =====================
class MendpuBatchGetInput(BaseModel):
    """批量获取采购结账状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    iyear: Optional[float] = Field(None, description="会计年度")
    iperiod_begin: Optional[float] = Field(None, description="起始会计期间")
    iperiod_end: Optional[float] = Field(None, description="结束会计期间")


# ===================== 采购订单表体数据模型 =====================
class PurchaseorderEntryInput(BaseModel):
    """采购订单表体数据模型"""
    inventorycode: str = Field(..., description="存货编码（必填）")
    inventoryname: Optional[str] = Field(None, description="存货名称")
    inventorystd: Optional[str] = Field(None, description="规格型号")
    unitcode: Optional[str] = Field(None, description="采购单位编码")
    unitname: Optional[str] = Field(None, description="采购单位")
    quantity: float = Field(..., description="数量（必填）")
    arrivedate: Optional[str] = Field(None, description="计划到货日期")
    price: Optional[float] = Field(None, description="原币单价")
    quotedprice: Optional[float] = Field(None, description="报价")
    taxprice: Optional[float] = Field(None, description="含税单价")
    money: Optional[float] = Field(None, description="原币金额")
    tax: Optional[float] = Field(None, description="原币税额")
    sum: Optional[float] = Field(None, description="原币价税合计")
    discount: Optional[float] = Field(None, description="折扣额")
    natprice: Optional[float] = Field(None, description="本币单价")
    natmoney: Optional[float] = Field(None, description="本币金额")
    assistantunit: Optional[str] = Field(None, description="辅计量单位编码")
    nattax: Optional[float] = Field(None, description="本币税额")
    natsum: Optional[float] = Field(None, description="本币价税合计")
    natdiscount: Optional[float] = Field(None, description="本币折扣额")
    taxrate: Optional[float] = Field(None, description="税率")
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


# ===================== 新增采购订单 数据模型 =====================
class PurchaseorderAddInput(BaseModel):
    """新增采购订单输入模型"""
    code: Optional[str] = Field(None, description="订单编号")
    date: Optional[str] = Field(None, description="订单日期，默认取系统日期(yyyy-MM-dd)")
    operation_type_code: str = Field(..., description="采购业务类型，默认普通采购（必填）")
    state: Optional[str] = Field(None, description="订单状态")
    purchase_type_code: Optional[str] = Field(None, description="采购类型编码")
    purchase_type_name: Optional[str] = Field(None, description="采购类型")
    vendorcode: str = Field(..., description="供应商编码（必填）")
    vendorname: Optional[str] = Field(None, description="供应商名称")
    vendorabbname: Optional[str] = Field(None, description="供应商简称")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")
    currency_name: Optional[str] = Field(None, description="外币名称")
    currency_rate: Optional[float] = Field(None, description="汇率")
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
    entry: List[PurchaseorderEntryInput] = Field(..., description="表体数据（必填）")


# ===================== 获取单个采购订单 数据模型 =====================
class PurchaseorderGetInput(BaseModel):
    """获取单个采购订单输入模型"""
    id: str = Field(..., description="订单编号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 查询采购订单列表 数据模型 =====================
class PurchaseorderListInput(BaseModel):
    """查询采购订单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始订单编号")
    code_end: Optional[str] = Field(None, description="结束订单编号")
    date_begin: Optional[str] = Field(None, description="起始订单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束订单日期，格式:yyyy-MM-dd")
    state: Optional[str] = Field(None, description="订单状态")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    vendorname: Optional[str] = Field(None, description="供应商名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    remark: Optional[str] = Field(None, description="备注关键字")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")


# ===================== 查询采购订单列表2(以存货为单位) 数据模型 =====================
class PurchaseorderList2Input(BaseModel):
    """批量获取采购订单（以存货为单位）输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始订单编号")
    code_end: Optional[str] = Field(None, description="结束订单编号")
    date_begin: Optional[str] = Field(None, description="起始订单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束订单日期，格式:yyyy-MM-dd")
    state: Optional[str] = Field(None, description="订单状态")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    vendorname: Optional[str] = Field(None, description="供应商名称关键字")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员关键字")
    remark: Optional[str] = Field(None, description="备注关键字")
    maker: Optional[str] = Field(None, description="制单人")
    verifier: Optional[str] = Field(None, description="审核人")
    closer: Optional[str] = Field(None, description="关闭人")
    inventorycode: Optional[str] = Field(None, description="存货编码")
    inventoryname: Optional[str] = Field(None, description="存货名称关键字")
    arrivestate: Optional[str] = Field(None, description="到货状态")
    receivestate: Optional[str] = Field(None, description="入库状态")
    billstate: Optional[str] = Field(None, description="开票状态")
    paystate: Optional[str] = Field(None, description="付款状态")


# ===================== 审核采购订单(verify) 数据模型 =====================
class PurchaseorderVerifyInput(BaseModel):
    """审核采购订单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 弃审采购订单(unverify) 数据模型 =====================
class PurchaseorderUnverifyInput(BaseModel):
    """弃审采购订单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 采购订单审批(audit) 数据模型 =====================
class PurchaseorderAuditInput(BaseModel):
    """采购订单审批输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: float = Field(..., description="是否同意(1=同意;0=不同意)（必填）")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 采购订单弃审(abandon) 数据模型 =====================
class PurchaseorderAbandonInput(BaseModel):
    """采购订单弃审输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    opinion: Optional[str] = Field(None, description="审批意见")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 获取采购订单按钮状态 数据模型 =====================
class PurchaseorderButtonstateInput(BaseModel):
    """获取采购订单按钮状态输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 获取采购订单待办任务 数据模型 =====================
class PurchaseorderTasksInput(BaseModel):
    """获取采购订单待办任务输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
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


# ===================== 获取采购订单审批历史 数据模型 =====================
class PurchaseorderHistoryInput(BaseModel):
    """获取采购订单审批历史输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")

# ===================== 采购请购单通用数据模型 (复用于新增和查询) =====================

class PurchaseRequisitionEntry(BaseModel):
    """
    采购请购单体模型。
    注意：所有字段均为 Optional，以兼容查询返回（可能缺省）和新增传入（后端校验必填）。
    """
    inventorycode: Optional[str] = Field(None, description="存货编码")
    quantity: Optional[float] = Field(None, description="数量")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    price: Optional[float] = Field(None, description="单价")
    taxrate: Optional[float] = Field(None, description="税率")
    taxprice: Optional[float] = Field(None, description="含税单价")
    money: Optional[float] = Field(None, description="金额")
    tax: Optional[float] = Field(None, description="税额")
    sum: Optional[float] = Field(None, description="价税合计")
    requiredate: Optional[str] = Field(None, description="需求日期")
    arrivedate: Optional[str] = Field(None, description="到货日期")
    item_class: Optional[str] = Field(None, description="项目大类编码")
    item_code: Optional[str] = Field(None, description="项目编码")
    item_name: Optional[str] = Field(None, description="项目名称")
    btaxcost: Optional[float] = Field(None, description="本币金额")
    num: Optional[float] = Field(None, description="件数")
    unitid: Optional[str] = Field(None, description="单位编码")
    deptcodeexec: Optional[str] = Field(None, description="执行部门编码")
    personcodeexec: Optional[str] = Field(None, description="执行业务员编码")
    currency_name: Optional[str] = Field(None, description="币种名称")
    currency_rate: Optional[float] = Field(None, description="汇率")
    originalprice: Optional[float] = Field(None, description="原币单价")
    originaltaxedprice: Optional[float] = Field(None, description="原币含税单价")
    originalmoney: Optional[float] = Field(None, description="原币金额")
    originaltax: Optional[float] = Field(None, description="原币税额")
    originalsum: Optional[float] = Field(None, description="原币价税合计")
    ivouchrowno: Optional[int] = Field(None, description="行号")
    # 单据体自定义项
    define22: Optional[str] = Field(None, description="单据体自定义项1")
    define23: Optional[str] = Field(None, description="单据体自定义项2")
    define24: Optional[str] = Field(None, description="单据体自定义项3")
    define25: Optional[str] = Field(None, description="单据体自定义项4")
    define26: Optional[str] = Field(None, description="单据体自定义项5")
    define27: Optional[str] = Field(None, description="单据体自定义项6")
    define28: Optional[str] = Field(None, description="单据体自定义项7")
    define29: Optional[str] = Field(None, description="单据体自定义项8")
    define30: Optional[str] = Field(None, description="单据体自定义项9")
    define31: Optional[str] = Field(None, description="单据体自定义项10")
    define32: Optional[str] = Field(None, description="单据体自定义项11")
    define33: Optional[str] = Field(None, description="单据体自定义项12")
    define34: Optional[str] = Field(None, description="单据体自定义项13")
    define35: Optional[str] = Field(None, description="单据体自定义项14")
    define36: Optional[str] = Field(None, description="单据体自定义项15")
    define37: Optional[str] = Field(None, description="单据体自定义项16")
    # 自由项
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

class PurchaseRequisitionInfo(BaseModel):
    """
    采购请购单主表模型 (通用，用于新增输入和查询返回)
    """
    code: Optional[str] = Field(None, description="请购单号")
    date: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    personcode: Optional[str] = Field(None, description="业务员编码")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    memory: Optional[str] = Field(None, description="备注")
    # 单据头自定义项
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4")
    define5: Optional[str] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6")
    define7: Optional[str] = Field(None, description="单据头自定义项7")
    define8: Optional[str] = Field(None, description="单据头自定义项8")
    define9: Optional[str] = Field(None, description="单据头自定义项9")
    define10: Optional[str] = Field(None, description="单据头自定义项10")
    define11: Optional[str] = Field(None, description="单据头自定义项11")
    define12: Optional[str] = Field(None, description="单据头自定义项12")
    define13: Optional[str] = Field(None, description="单据头自定义项13")
    define14: Optional[str] = Field(None, description="单据头自定义项14")
    define15: Optional[str] = Field(None, description="单据头自定义项15")
    define16: Optional[str] = Field(None, description="单据头自定义项16")

    # 单据体列表
    entry: Optional[List[PurchaseRequisitionEntry]] = Field(None, description="采购请购单体列表")

# ===================== 新增一张采购请购单 数据模型 =====================
class AddPurchaseRequisitionInput(PurchaseRequisitionInfo):
    """新增采购请购单输入模型，继承自通用模型，可在此处强化必填项校验"""
    code: str = Field(..., description="请购单号（必填）")
    entry: List[PurchaseRequisitionEntry] = Field(..., description="采购请购单体列表（必填）")

# ===================== 获取单张采购请购单 数据模型 =====================
class GetPurchaseRequisitionInput(BaseModel):
    id: int = Field(..., description="主表id，用于查询采购请购单单据信息")

# ===================== 获取采购请购单列表信息 数据模型 =====================
class GetPurchaseRequisitionListInput(BaseModel):
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始制单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束制单日期，格式:yyyy-MM-dd")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    personcode: Optional[str] = Field(None, description="业务员编码")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    memory: Optional[str] = Field(None, description="备注")
    cvoucherstate: Optional[str] = Field(None, description="单据状态")

# ===================== 审核一张采购请购单 数据模型 =====================
class VerifyPurchaseRequisitionInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审一张采购请购单 数据模型 =====================
class UnVerifyPurchaseRequisitionInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 获取采购请购单待办任务 数据模型 =====================
class GetPurchaseRequisitionTasksInput(BaseModel):
    """获取采购请购单待办任务输入模型"""
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

# ===================== 获取采购请购单审批进程 数据模型 =====================
class GetPurchaseRequisitionHistoryInput(BaseModel):
    """获取采购请购单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 获取采购请购单是否启用工作流 数据模型 =====================
class GetPurchaseRequisitionFlowenabledInput(BaseModel):
    """获取采购请购单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")

# ===================== 获取采购请购单工作流按钮是否可用状态 数据模型 =====================
class GetPurchaseRequisitionButtonstateInput(BaseModel):
    """获取采购请购单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 审核采购请购单（工作流） 数据模型 =====================
class AuditPurchaseRequisitionInput(BaseModel):
    """审核采购请购单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")

# ===================== 弃审采购请购单 数据模型 =====================
class AbandonPurchaseRequisitionInput(BaseModel):
    """弃审采购请购单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")

# ===================== 采购退货单通用数据模型 (复用于新增和查询) =====================

class PurchaseReturnEntry(BaseModel):
    """
    采购退货单体模型。
    注意：所有字段均为 Optional，以兼容查询返回（可能缺省）和新增传入（后端校验必填）。
    """
    warehousecode: Optional[str] = Field(None, description="仓库编码")
    warehousename: Optional[str] = Field(None, description="仓库名称")
    inventorycode: Optional[str] = Field(None, description="存货编码")
    inventoryaddcode: Optional[str] = Field(None, description="存货代码")
    inventoryname: Optional[str] = Field(None, description="存货名称")
    inventorystd: Optional[str] = Field(None, description="存货规格")
    inventoryclasscode: Optional[str] = Field(None, description="存货大类编码")
    unitid: Optional[str] = Field(None, description="单位编码")
    ccomunitcode: Optional[str] = Field(None, description="主计量单位编码")
    cinvm_unit: Optional[str] = Field(None, description="辅计量单位编码")
    cinva_unit: Optional[str] = Field(None, description="采购单位编码")
    iinvexchrate: Optional[float] = Field(None, description="换算率")
    serial: Optional[str] = Field(None, description="批号")
    closer: Optional[str] = Field(None, description="关闭人")
    originaltaxedprice: Optional[float] = Field(None, description="原币含税单价")
    quantity: Optional[float] = Field(None, description="数量")
    number: Optional[float] = Field(None, description="件数")
    originalprice: Optional[float] = Field(None, description="原币单价")
    originalmoney: Optional[float] = Field(None, description="原币金额")
    originaltax: Optional[float] = Field(None, description="原币税额")
    originalsum: Optional[float] = Field(None, description="原币价税合计")
    price: Optional[float] = Field(None, description="本币单价")
    money: Optional[float] = Field(None, description="本币金额")
    tax: Optional[float] = Field(None, description="本币税额")
    sum: Optional[float] = Field(None, description="本币价税合计")
    cbcloser: Optional[str] = Field(None, description="行关闭人")
    free1: Optional[str] = Field(None, description="自由项1")
    free2: Optional[str] = Field(None, description="自由项2")
    define22: Optional[str] = Field(None, description="单据体自定义项1")
    define23: Optional[str] = Field(None, description="单据体自定义项2")
    define24: Optional[str] = Field(None, description="单据体自定义项3")
    define25: Optional[str] = Field(None, description="单据体自定义项4")
    define26: Optional[str] = Field(None, description="单据体自定义项5")
    define27: Optional[str] = Field(None, description="单据体自定义项6")
    define28: Optional[str] = Field(None, description="单据体自定义项7")
    define29: Optional[str] = Field(None, description="单据体自定义项8")
    define30: Optional[str] = Field(None, description="单据体自定义项9")
    define31: Optional[str] = Field(None, description="单据体自定义项10")
    define32: Optional[str] = Field(None, description="单据体自定义项11")
    define33: Optional[str] = Field(None, description="单据体自定义项12")
    define34: Optional[str] = Field(None, description="单据体自定义项13")
    define35: Optional[str] = Field(None, description="单据体自定义项14")
    define36: Optional[str] = Field(None, description="单据体自定义项15")
    define37: Optional[str] = Field(None, description="单据体自定义项16")
    taxrate: Optional[float] = Field(None, description="税率")
    iposid: Optional[int] = Field(None, description="POS机号")
    free3: Optional[str] = Field(None, description="自由项3")
    free4: Optional[str] = Field(None, description="自由项4")
    free5: Optional[str] = Field(None, description="自由项5")
    free6: Optional[str] = Field(None, description="自由项6")
    free7: Optional[str] = Field(None, description="自由项7")
    free8: Optional[str] = Field(None, description="自由项8")
    free9: Optional[str] = Field(None, description="自由项9")
    free10: Optional[str] = Field(None, description="自由项10")
    cordercode: Optional[str] = Field(None, description="订单号")
    vouchstate: Optional[str] = Field(None, description="单据状态")
    ivouchrowno: Optional[int] = Field(None, description="行号")
    cbmemo: Optional[str] = Field(None, description="备注")
    cbsysbarcode: Optional[str] = Field(None, description="条形码")

class PurchaseReturnInfo(BaseModel):
    """
    采购退货单主表模型 (通用，用于新增输入和查询返回)
    """
    code: Optional[str] = Field(None, description="采购退货单号")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    purchasetypename: Optional[str] = Field(None, description="采购类型名称")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    vendorabbname: Optional[str] = Field(None, description="供应商简称")
    vendorname: Optional[str] = Field(None, description="供应商名称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称")
    payconditioncode: Optional[str] = Field(None, description="付款条件编码")
    payconditionname: Optional[str] = Field(None, description="付款条件名称")
    foreigncurrency: Optional[str] = Field(None, description="外币币种")
    cexch_code: Optional[str] = Field(None, description="汇率编码")
    foreigncurrencyrate: Optional[float] = Field(None, description="汇率")
    memory: Optional[str] = Field(None, description="备注")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    ccloser: Optional[str] = Field(None, description="关闭人")
    idiscounttaxtype: Optional[str] = Field(None, description="扣税类别")
    define1: Optional[str] = Field(None, description="单据头自定义项1")
    define2: Optional[str] = Field(None, description="单据头自定义项2")
    define3: Optional[str] = Field(None, description="单据头自定义项3")
    define4: Optional[str] = Field(None, description="单据头自定义项4")
    define5: Optional[str] = Field(None, description="单据头自定义项5")
    define6: Optional[str] = Field(None, description="单据头自定义项6")
    define7: Optional[str] = Field(None, description="单据头自定义项7")
    define8: Optional[str] = Field(None, description="单据头自定义项8")
    define9: Optional[str] = Field(None, description="单据头自定义项9")
    define10: Optional[str] = Field(None, description="单据头自定义项10")
    define11: Optional[str] = Field(None, description="单据头自定义项11")
    define12: Optional[str] = Field(None, description="单据头自定义项12")
    define13: Optional[str] = Field(None, description="单据头自定义项13")
    define14: Optional[str] = Field(None, description="单据头自定义项14")
    define15: Optional[str] = Field(None, description="单据头自定义项15")
    define16: Optional[str] = Field(None, description="单据头自定义项16")
    shipcode: Optional[str] = Field(None, description="发货方式编码")
    shipname: Optional[str] = Field(None, description="发货方式名称")
    billtype: Optional[str] = Field(None, description="单据模板类型")
    cvouchtype: Optional[str] = Field(None, description="单据类型")
    cmodifydate: Optional[str] = Field(None, description="修改日期")
    creviser: Optional[str] = Field(None, description="修改人")
    cauditdate: Optional[str] = Field(None, description="审核日期")
    cverifier: Optional[str] = Field(None, description="审核人")
    cvenpuomprotocol: Optional[str] = Field(None, description="供应商 uom 协议编码")
    cvenpuomprotocolname: Optional[str] = Field(None, description="供应商 uom 协议名称")
    csysbarcode: Optional[str] = Field(None, description="条形码")

    # 单据体列表
    entry: Optional[List[PurchaseReturnEntry]] = Field(None, description="采购退货单体列表")

# ===================== 新增一张采购退货单 数据模型 =====================
class AddPurchaseReturnInput(PurchaseReturnInfo):
    """新增采购退货单输入模型，继承自通用模型，可在此处强化必填项校验"""
    purchasetypename: str = Field(..., description="采购类型名称（必填）")
    vendorcode: str = Field(..., description="供应商编码（必填）")
    departmentcode: str = Field(..., description="部门编码（必填）")
    personname: str = Field(..., description="业务员名称（必填）")
    cmodifydate: str = Field(..., description="修改日期（必填）")
    entry: List[PurchaseReturnEntry] = Field(..., description="采购退货单体列表（必填）")

# ===================== 获取单张采购退货单 数据模型 =====================
class GetPurchaseReturnInput(BaseModel):
    id: str = Field(..., description="采购退货单编号")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")

# ===================== 获取采购退货单列表信息 数据模型 =====================
class GetPurchaseReturnListInput(BaseModel):
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    date_begin: Optional[str] = Field(None, description="起始制单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束制单日期，格式:yyyy-MM-dd")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    purchasetypecode: Optional[str] = Field(None, description="采购类型编码")
    purchasetypename: Optional[str] = Field(None, description="采购类型名称")
    vendorcode: Optional[str] = Field(None, description="供应商编码")
    vendorabbname: Optional[str] = Field(None, description="供应商简称")
    vendorname: Optional[str] = Field(None, description="供应商名称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称")
    payconditioncode: Optional[str] = Field(None, description="付款条件编码")
    payconditionname: Optional[str] = Field(None, description="付款条件名称")
    foreigncurrency: Optional[str] = Field(None, description="外币币种")
    memory: Optional[str] = Field(None, description="备注")
    businesstype: Optional[str] = Field(None, description="业务类型")
    maker: Optional[str] = Field(None, description="制单人")
    ccloser: Optional[str] = Field(None, description="关闭人")
    shipcode: Optional[str] = Field(None, description="发货方式编码")
    shipname: Optional[str] = Field(None, description="发货方式名称")
    cauditdate: Optional[str] = Field(None, description="审核日期")
    cverifier: Optional[str] = Field(None, description="审核人")

# ===================== 审核一张采购退货单 数据模型 =====================
class VerifyPurchaseReturnInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审一张采购退货单 数据模型 =====================
class UnVerifyPurchaseReturnInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 获取采购退货单待办任务 数据模型 =====================
class GetPurchaseReturnTasksInput(BaseModel):
    """获取采购退货单待办任务输入模型"""
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

# ===================== 获取采购退货单审批进程 数据模型 =====================
class GetPurchaseReturnHistoryInput(BaseModel):
    """获取采购退货单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 获取采购退货单是否启用工作流 数据模型 =====================
class GetPurchaseReturnFlowenabledInput(BaseModel):
    """获取采购退货单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")

# ===================== 获取采购退货单工作流按钮是否可用状态 数据模型 =====================
class GetPurchaseReturnButtonstateInput(BaseModel):
    """获取采购退货单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 审核采购退货单（工作流） 数据模型 =====================
class AuditPurchaseReturnInput(BaseModel):
    """审核采购退货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: str = Field(..., description="审批人(操作员编码)（必填）")
    person_code: str = Field(..., description="审批人(人员编码)（必填）")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")

# ===================== 弃审采购退货单 数据模型 =====================
class AbandonPurchaseReturnInput(BaseModel):
    """弃审采购退货单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: str = Field(..., description="审批人(操作员编码)（必填）")
    person_code: str = Field(..., description="审批人(人员编码)（必填）")
    opinion: Optional[str] = Field(None, description="审批意见")

# ===================== 获取单个预算信息 数据模型 =====================
class GetBudgetInput(BaseModel):
    id: str = Field(..., description="预算项目编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")

# ===================== 批量获取预算信息 数据模型 =====================
class BatchGetBudgetInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cformcode: Optional[str] = Field(None, description="预算表编码")
    ccalibercode1: Optional[str] = Field(None, description="预算口径编码")
    cversioncode: Optional[str] = Field(None, description="版本编码")
    ctargetcode: Optional[str] = Field(None, description="预算目标编码")
    ctargetcode_ctl: Optional[str] = Field(None, description="预算目标控制编码")
    citemcode: Optional[str] = Field(None, description="预算项目编码")
    citemname: Optional[str] = Field(None, description="预算项目名称")
    fperiod13: Optional[float] = Field(None, description="13期预算金额")
    fperiod12: Optional[float] = Field(None, description="12期预算金额")
    freserve12: Optional[float] = Field(None, description="12期预留金额")
    pk: Optional[str] = Field(None, description="主键")


# =============================================================================
# Tool函数 - Tool Functions
# =============================================================================

# ===================== 批量获取供应商存货价格表 Tool函数 =====================
def u8_veninvprice_batch_get_tool(input_data: GetVenInvPriceInput, client: U8OpenAPIClient) -> str:
    """
    批量获取供应商存货价格表，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/veninvprice/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取供应商存货价格表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取供应商存货价格表成功",
            "data": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取调价单列表 Tool函数 =====================
def u8_venpriceadjustlist_batch_get_tool(input_data: GetVenPriceAdjustListInput, client: U8OpenAPIClient) -> str:
    """
    获取供应商存货调价单列表信息，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/venpriceadjustlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调价单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调价单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "venpriceadjustlist": result.get("venpriceadjustlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单张调价单 Tool函数 =====================
def u8_venpriceadjust_get_tool(input_data: GetVenPriceAdjustInput, client: U8OpenAPIClient) -> str:
    """
    通过调价单编号获取用友U8中的供应商存货调价单单据信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/venpriceadjust/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===================== 获取调价单待办任务 Tool函数 =====================
def u8_venpriceadjust_tasks_tool(input_data: GetVenPriceAdjustTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取供应商存货调价单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/venpriceadjust/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调价单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调价单待办任务成功",
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


# ===================== 查看调价单审批进程 Tool函数 =====================
def u8_venpriceadjust_history_tool(input_data: GetVenPriceAdjustHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看供应商存货调价单审批进程，获取单据的审批历史记录。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/venpriceadjust/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调价单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调价单审批进程成功",
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


# ===================== 获取调价单工作流按钮状态 Tool函数 =====================
def u8_venpriceadjust_buttonstate_tool(input_data: GetVenPriceAdjustButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取供应商存货调价单工作流按钮是否可用状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/venpriceadjust/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取调价单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取调价单工作流按钮状态成功",
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


# ===================== 审核调价单 Tool函数 =====================
def u8_venpriceadjust_audit_tool(input_data: AuditVenPriceAdjustInput, client: U8OpenAPIClient) -> str:
    """
    审核供应商存货调价单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "venpriceadjust": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/venpriceadjust/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "调价单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "调价单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增调价单 Tool函数 =====================
def u8_venpriceadjust_add_tool(input_data: AddVenPriceAdjustInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增供应商存货调价单，包含单据头和单据体（entry）完整信息。
    """
    request_body: dict = {
        "venpriceadjust": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/venpriceadjust/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "调价单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "调价单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审调价单 Tool函数 =====================
def u8_venpriceadjust_abandon_tool(input_data: AbandonVenPriceAdjustInput, client: U8OpenAPIClient) -> str:
    """
    弃审供应商存货调价单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "venpriceadjust": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/venpriceadjust/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "调价单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "调价单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取到货单列表 Tool函数 =====================
def u8_purchasereceiptlist_batch_get_tool(input_data: GetPurchaseReceiptListInput, client: U8OpenAPIClient) -> str:
    """
    获取采购到货单列表信息，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchasereceiptlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取到货单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取到货单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purchasereceiptlist": result.get("purchasereceiptlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个到货单 Tool函数 =====================
def u8_purchasereceipt_get_tool(input_data: GetPurchaseReceiptInput, client: U8OpenAPIClient) -> str:
    """
    通过到货单编号获取用友U8中的采购到货单单据信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchasereceipt/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===================== 审核到货单 Tool函数 =====================
def u8_purchasereceipt_verify_tool(input_data: VerifyPurchaseReceiptInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审核采购到货单。
    """
    request_body: dict = {
        "purchasereceipt": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasereceipt/verify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "到货单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "到货单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审到货单 Tool函数 =====================
def u8_purchasereceipt_unverify_tool(input_data: UnVerifyPurchaseReceiptInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审采购到货单。
    """
    request_body: dict = {
        "purchasereceipt": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasereceipt/unverify"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "到货单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "到货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取到货单待办任务 Tool函数 =====================
def u8_purchasereceipt_tasks_tool(input_data: GetPurchaseReceiptTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取采购到货单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchasereceipt/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取到货单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取到货单待办任务成功",
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


# ===================== 查看到货单审批进程 Tool函数 =====================
def u8_purchasereceipt_history_tool(input_data: GetPurchaseReceiptHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看采购到货单审批进程，获取单据的审批历史记录。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchasereceipt/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取到货单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取到货单审批进程成功",
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


# ===================== 获取到货单是否启用工作流 Tool函数 =====================
def u8_purchasereceipt_flowenabled_tool(input_data: GetPurchaseReceiptFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取采购到货单是否启用工作流。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchasereceipt/flowenabled"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取到货单是否启用工作流失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取到货单是否启用工作流成功",
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


# ===================== 获取到货单工作流按钮状态 Tool函数 =====================
def u8_purchasereceipt_buttonstate_tool(input_data: GetPurchaseReceiptButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取采购到货单工作流按钮是否可用状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchasereceipt/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取到货单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取到货单工作流按钮状态成功",
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


# ===================== 审核到货单(工作流) Tool函数 =====================
def u8_purchasereceipt_audit_tool(input_data: AuditPurchaseReceiptInput, client: U8OpenAPIClient) -> str:
    """
    审核采购到货单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "purchasereceipt": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasereceipt/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "到货单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "到货单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增到货单 Tool函数 =====================
def u8_purchasereceipt_add_tool(input_data: AddPurchaseReceiptInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增采购到货单，包含单据头和单据体（entry）完整信息。
    """
    request_body: dict = {
        "purchasereceipt": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasereceipt/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "到货单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "到货单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审到货单(工作流) Tool函数 =====================
def u8_purchasereceipt_abandon_tool(input_data: AbandonPurchaseReceiptInput, client: U8OpenAPIClient) -> str:
    """
    弃审采购到货单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "purchasereceipt": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/purchasereceipt/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "到货单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "到货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增采购发票 Tool函数 =====================
def u8_purchaseinvoice_add_tool(input_data: PurchaseinvoiceAddInput, client: Any) -> str:
    """
    新增一张采购发票
    """
    request_body: Dict[str, Any] = {
        "purchaseinvoice": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/purchaseinvoice/add"
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
            "message": "采购发票新增成功",
            "data": {"id": result.get("id"), "tradeid": result.get("tradeid")},
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个采购发票 Tool函数 =====================
def u8_purinvoice_get_tool(input_data: PurinvoiceGetInput, client: Any) -> str:
    """
    获取单个单张采购发票
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purinvoice/get"
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
            "message": "获取采购发票成功",
            "data": result.get("purinvoice"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 查询采购发票列表 Tool函数 =====================
def u8_purinvoice_list_tool(input_data: PurinvoiceListInput, client: Any) -> str:
    """
    获取采购发票列表信息
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purinvoicelist/batch_get"
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
            "message": "获取采购发票列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purinvoicelist": result.get("purinvoicelist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 采购结账状态 Tool函数 =====================
def u8_mendpu_batch_get_tool(input_data: MendpuBatchGetInput, client: Any) -> str:
    """
    批量获取采购结账状态
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/mendpu/batch_get"
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
            "message": "获取采购结账状态成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "mendpu": result.get("mendpu", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增采购订单 Tool函数 =====================
def u8_purchaseorder_add_tool(input_data: PurchaseorderAddInput, client: Any) -> str:
    """
    新增一张采购订单
    """
    request_body: Dict[str, Any] = {
        "purchaseorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/purchaseorder/add"
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
            "message": "采购订单新增成功",
            "data": {"id": result.get("id"), "tradeid": result.get("tradeid")},
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个采购订单 Tool函数 =====================
def u8_purchaseorder_get_tool(input_data: PurchaseorderGetInput, client: Any) -> str:
    """
    获取单张采购订单
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseorder/get"
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
            "message": "获取采购订单成功",
            "data": result.get("purchaseorder"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 查询采购订单列表 Tool函数 =====================
def u8_purchaseorder_list_tool(input_data: PurchaseorderListInput, client: Any) -> str:
    """
    获取采购订单列表信息
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseorderlist/batch_get"
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
            "message": "获取采购订单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purchaseorderlist": result.get("purchaseorderlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 查询采购订单列表2(以存货为单位) Tool函数 =====================
def u8_purchaseorder_list2_tool(input_data: PurchaseorderList2Input, client: Any) -> str:
    """
    批量获取采购订单（以存货为单位）
    获取采购订单列表，需要根据订单执行情况进行筛选
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseorderlist2/batch_get"
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
            "message": "获取采购订单列表(以存货为单位)成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purchaseorderlist2": result.get("purchaseorderlist2", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 审核采购订单(verify) Tool函数 =====================
def u8_purchaseorder_verify_tool(input_data: PurchaseorderVerifyInput, client: Any) -> str:
    """
    审核一张采购订单
    """
    request_body: Dict[str, Any] = {
        "purchaseorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/purchaseorder/verify"
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
            "message": "采购订单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 弃审采购订单(unverify) Tool函数 =====================
def u8_purchaseorder_unverify_tool(input_data: PurchaseorderUnverifyInput, client: Any) -> str:
    """
    弃审一张采购订单
    """
    request_body: Dict[str, Any] = {
        "purchaseorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/purchaseorder/unverify"
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
            "message": "采购订单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 采购订单审批(audit) Tool函数 =====================
def u8_purchaseorder_audit_tool(input_data: PurchaseorderAuditInput, client: Any) -> str:
    """
    审核采购订单（工作流审批）
    执行审批动作前，需要保证审批人已经进行ERP登录授权
    """
    request_body: Dict[str, Any] = {
        "purchaseorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/purchaseorder/audit"
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
            "message": "采购订单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 采购订单弃审(abandon) Tool函数 =====================
def u8_purchaseorder_abandon_tool(input_data: PurchaseorderAbandonInput, client: Any) -> str:
    """
    弃审采购订单（工作流弃审）
    执行弃审动作前，需要保证审批人已经进行ERP登录授权
    """
    request_body: Dict[str, Any] = {
        "purchaseorder": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/purchaseorder/abandon"
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
            "message": "采购订单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取采购订单按钮状态 Tool函数 =====================
def u8_purchaseorder_buttonstate_tool(input_data: PurchaseorderButtonstateInput, client: Any) -> str:
    """
    获取采购订单工作流按钮是否可用状态
    只支持12.0版本，且需要打最新的WF补丁
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseorder/buttonstate"
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
            "message": "获取采购订单按钮状态成功",
            "data": result.get("buttonstate"),
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取采购订单待办任务 Tool函数 =====================
def u8_purchaseorder_tasks_tool(input_data: PurchaseorderTasksInput, client: Any) -> str:
    """
    获取采购订单待办任务
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseorder/tasks"
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
            "message": "获取采购订单待办任务成功",
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


# ===================== 获取采购订单审批历史 Tool函数 =====================
def u8_purchaseorder_history_tool(input_data: PurchaseorderHistoryInput, client: Any) -> str:
    """
    查看采购订单审批进程
    执行审批动作前，需要保证审批人已经进行ERP登录授权
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/purchaseorder/history"
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
            "message": "获取采购订单审批历史成功",
            "data": {"history": result.get("history", [])},
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增一张采购请购单 Tool函数 =====================
def u8_purchaserequisition_add_tool(input_data: AddPurchaseRequisitionInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增采购请购单，包含单据头和单据体（entry）完整信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 purchaserequisition）
    request_body: dict = {
        "purchaserequisition": input_data.model_dump(exclude_none=True)
    }

    # 采购请购单添加接口路径
    api_path = "/api/purchaserequisition/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购请购单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购请购单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单张采购请购单 Tool函数 =====================
def u8_purchaserequisition_get_tool(input_data: GetPurchaseRequisitionInput, client: U8OpenAPIClient) -> str:
    """
    通过主表id获取用友U8中的采购请购单单据信息。
    """
    params = {
        "id": input_data.id
    }

    # 采购请购单查询接口路径
    api_path = "/api/purchaserequisition/get"

    try:
        # 使用 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)

        # 检查业务错误码
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取采购请购单列表信息 Tool函数 =====================
def u8_purchaserequisition_list_get_tool(input_data: GetPurchaseRequisitionListInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取采购请购单列表信息，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购请购单列表查询接口路径
    api_path = "/api/purchaserequisitionlist/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购请购单列表查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购请购单列表查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purchaserequisitionlist": result.get("purchaserequisitionlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 审核一张采购请购单 Tool函数 =====================
def u8_purchaserequisition_verify_tool(input_data: VerifyPurchaseRequisitionInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审核采购请购单。
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 purchaserequisition）
    request_body: dict = {
        "purchaserequisition": input_data.model_dump(exclude_none=True)
    }

    # 采购请购单审核接口路径
    api_path = "/api/purchaserequisition/verify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购请购单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购请购单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审一张采购请购单 Tool函数 =====================
def u8_purchaserequisition_unverify_tool(input_data: UnVerifyPurchaseRequisitionInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审采购请购单。
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 purchaserequisition）
    request_body: dict = {
        "purchaserequisition": input_data.model_dump(exclude_none=True)
    }

    # 采购请购单弃审接口路径
    api_path = "/api/purchaserequisition/unverify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购请购单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购请购单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取采购请购单待办任务 Tool函数 =====================
def u8_purchaserequisition_tasks_tool(input_data: GetPurchaseRequisitionTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取采购请购单待办任务列表，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购请购单待办任务接口路径
    api_path = "/api/purchaserequisition/tasks"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购请购单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购请购单待办任务成功",
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

# ===================== 获取采购请购单审批进程 Tool函数 =====================
def u8_purchaserequisition_history_tool(input_data: GetPurchaseRequisitionHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看采购请购单审批进程，获取单据的审批历史记录。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购请购单审批进程接口路径
    api_path = "/api/purchaserequisition/history"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购请购单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购请购单审批进程成功",
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

# ===================== 获取采购请购单是否启用工作流 Tool函数 =====================
def u8_purchaserequisition_flowenabled_tool(input_data: GetPurchaseRequisitionFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取采购请购单是否启用工作流。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购请购单是否启用工作流接口路径
    api_path = "/api/purchaserequisition/flowenabled"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购请购单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购请购单工作流启用状态成功",
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

# ===================== 获取采购请购单工作流按钮是否可用状态 Tool函数 =====================
def u8_purchaserequisition_buttonstate_tool(input_data: GetPurchaseRequisitionButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取采购请购单工作流按钮是否可用状态。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购请购单工作流按钮状态接口路径
    api_path = "/api/purchaserequisition/buttonstate"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购请购单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购请购单工作流按钮状态成功",
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

# ===================== 审核采购请购单（工作流） Tool函数 =====================
def u8_purchaserequisition_audit_tool(input_data: AuditPurchaseRequisitionInput, client: U8OpenAPIClient) -> str:
    """
    审核采购请购单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 purchaserequisition）
    request_body: dict = {
        "purchaserequisition": input_data.model_dump(exclude_none=True)
    }

    # 采购请购单审核接口路径
    api_path = "/api/purchaserequisition/audit"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购请购单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购请购单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审采购请购单 Tool函数 =====================
def u8_purchaserequisition_abandon_tool(input_data: AbandonPurchaseRequisitionInput, client: U8OpenAPIClient) -> str:
    """
    弃审采购请购单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 purchaserequisition）
    request_body: dict = {
        "purchaserequisition": input_data.model_dump(exclude_none=True)
    }

    # 采购请购单弃审接口路径
    api_path = "/api/purchaserequisition/abandon"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购请购单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购请购单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增一张采购退货单 Tool函数 =====================
def u8_purchasereturn_add_tool(input_data: AddPurchaseReturnInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增采购退货单，包含单据头和单据体（entry）完整信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 purchasereturn）
    request_body: dict = {
        "purchasereturn": input_data.model_dump(exclude_none=True)
    }

    # 采购退货单添加接口路径
    api_path = "/api/purchasereturn/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购退货单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购退货单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单张采购退货单 Tool函数 =====================
def u8_purchasereturn_get_tool(input_data: GetPurchaseReturnInput, client: U8OpenAPIClient) -> str:
    """
    通过采购退货单编号获取用友U8中的采购退货单单据信息。
    """
    params = {
        "id": input_data.id
    }

    # 采购退货单查询接口路径
    api_path = "/api/purchasereturn/get"

    try:
        # 使用 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)

        # 检查业务错误码
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取采购退货单列表信息 Tool函数 =====================
def u8_purchasereturn_list_get_tool(input_data: GetPurchaseReturnListInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取采购退货单列表信息，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购退货单列表查询接口路径
    api_path = "/api/purchasereturnlist/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购退货单列表查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购退货单列表查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "purchasereturnlist": result.get("purchasereturnlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 审核一张采购退货单 Tool函数 =====================
def u8_purchasereturn_verify_tool(input_data: VerifyPurchaseReturnInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审核采购退货单。
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 purchasereturn）
    request_body: dict = {
        "purchasereturn": input_data.model_dump(exclude_none=True)
    }

    # 采购退货单审核接口路径
    api_path = "/api/purchasereturn/verify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购退货单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购退货单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审一张采购退货单 Tool函数 =====================
def u8_purchasereturn_unverify_tool(input_data: UnVerifyPurchaseReturnInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审采购退货单。
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 purchasereturn）
    request_body: dict = {
        "purchasereturn": input_data.model_dump(exclude_none=True)
    }

    # 采购退货单弃审接口路径
    api_path = "/api/purchasereturn/unverify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购退货单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购退货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取采购退货单待办任务 Tool函数 =====================
def u8_purchasereturn_tasks_tool(input_data: GetPurchaseReturnTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取采购退货单待办任务列表，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购退货单待办任务接口路径
    api_path = "/api/purchasereturn/tasks"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购退货单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购退货单待办任务成功",
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

# ===================== 获取采购退货单审批进程 Tool函数 =====================
def u8_purchasereturn_history_tool(input_data: GetPurchaseReturnHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看采购退货单审批进程，获取单据的审批历史记录。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购退货单审批进程接口路径
    api_path = "/api/purchasereturn/history"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购退货单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购退货单审批进程成功",
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

# ===================== 获取采购退货单是否启用工作流 Tool函数 =====================
def u8_purchasereturn_flowenabled_tool(input_data: GetPurchaseReturnFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取采购退货单是否启用工作流。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购退货单是否启用工作流接口路径
    api_path = "/api/purchasereturn/flowenabled"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购退货单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购退货单工作流启用状态成功",
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

# ===================== 获取采购退货单工作流按钮是否可用状态 Tool函数 =====================
def u8_purchasereturn_buttonstate_tool(input_data: GetPurchaseReturnButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取采购退货单工作流按钮是否可用状态。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 采购退货单工作流按钮状态接口路径
    api_path = "/api/purchasereturn/buttonstate"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取采购退货单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取采购退货单工作流按钮状态成功",
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

# ===================== 审核采购退货单（工作流） Tool函数 =====================
def u8_purchasereturn_audit_tool(input_data: AuditPurchaseReturnInput, client: U8OpenAPIClient) -> str:
    """
    审核采购退货单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 purchasereturn）
    request_body: dict = {
        "purchasereturn": input_data.model_dump(exclude_none=True)
    }

    # 采购退货单审核接口路径
    api_path = "/api/purchasereturn/audit"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购退货单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购退货单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审采购退货单 Tool函数 =====================
def u8_purchasereturn_abandon_tool(input_data: AbandonPurchaseReturnInput, client: U8OpenAPIClient) -> str:
    """
    弃审采购退货单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 purchasereturn）
    request_body: dict = {
        "purchasereturn": input_data.model_dump(exclude_none=True)
    }

    # 采购退货单弃审接口路径
    api_path = "/api/purchasereturn/abandon"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "采购退货单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "采购退货单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个预算信息 Tool函数 =====================
def u8_budget_get_tool(input_data: GetBudgetInput, client: U8OpenAPIClient) -> str:
    """
    通过预算项目编码获取用友U8中的单个预算信息。
    """
    params = {
        "id": input_data.id
    }

    # 预算查询接口路径
    api_path = "/api/budget/get"

    try:
        # 使用 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)

        # 检查业务错误码
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取预算信息 Tool函数 =====================
def u8_budget_batch_get_tool(input_data: BatchGetBudgetInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取预算信息，支持多条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 预算批量查询接口路径
    api_path = "/api/budget/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "预算批量查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "预算批量查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "budget": result.get("budget", [])
            },
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

# ===================== 批量获取供应商存货价格表 Schema定义 =====================
U8_VENINVPRICE_BATCH_GET_SCHEMA = {
    "name": "u8_veninvprice_batch_get",
    "description": "在用友U8 OpenAPI中批量获取供应商存货价格表，支持供应商、存货、日期范围、价格区间等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cvencode": {"type": "string", "description": "供应商编码"},
            "cvenname": {"type": "string", "description": "供应商名称"},
            "cvenabbname": {"type": "string", "description": "供应商简称"},
            "cinvname": {"type": "string", "description": "存货名称"},
            "cinvstd": {"type": "string", "description": "存货规格"},
            "cinvccode": {"type": "string", "description": "存货分类编码"},
            "cinvcname": {"type": "string", "description": "存货分类名称"},
            "cinvcode_begin": {"type": "string", "description": "起始存货编码"},
            "cinvcode_end": {"type": "string", "description": "结束存货编码"},
            "denabledate_begin": {"type": "string", "description": "起始生效日期，格式：yyyy-MM-dd"},
            "denabledate_end": {"type": "string", "description": "结束生效日期，格式：yyyy-MM-dd"},
            "ddisabledate": {"type": "string", "description": "失效日期"},
            "cexch_name": {"type": "string", "description": "币种名称"},
            "bpromotion": {"type": "string", "description": "促销"},
            "cmemo": {"type": "string", "description": "备注"},
            "isupplytype": {"type": "string", "description": "供货类型"},
            "btaxcost": {"type": "string", "description": "含税"},
            "ctermcode": {"type": "string", "description": "采购条件编码"},
            "ilowerlimit": {"type": "number", "description": "价格下限"},
            "iupperlimit": {"type": "number", "description": "价格上限"},
            "itaxrate": {"type": "number", "description": "税率"}
        },
        "required": []
    }
}

# ===================== 获取调价单列表 Schema定义 =====================
U8_VENPRICEADJUSTLIST_BATCH_GET_SCHEMA = {
    "name": "u8_venpriceadjustlist_batch_get",
    "description": "在用友U8 OpenAPI中查询供应商存货调价单列表信息，支持分页、单据号/日期范围、部门/业务员等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始单据日期，格式：yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束单据日期，格式：yyyy-MM-dd"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称"},
            "depcode": {"type": "string", "description": "部门编码"},
            "depname": {"type": "string", "description": "部门名称"},
            "memo": {"type": "string", "description": "备注"},
            "verifier": {"type": "string", "description": "审核人"},
            "maker": {"type": "string", "description": "制单人"}
        },
        "required": []
    }
}

# ===================== 获取单张调价单 Schema定义 =====================
U8_VENPRICEADJUST_GET_SCHEMA = {
    "name": "u8_venpriceadjust_get",
    "description": "通过调价单编号获取单张供应商存货调价单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {
                "type": "string",
                "description": "调价单编号（必填）"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取调价单待办任务 Schema定义 =====================
U8_VENPRICEADJUST_TASKS_SCHEMA = {
    "name": "u8_venpriceadjust_tasks",
    "description": "获取供应商存货调价单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "string", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "string", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "string", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "string", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "string", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        },
        "required": []
    }
}

# ===================== 查看调价单审批进程 Schema定义 =====================
U8_VENPRICEADJUST_HISTORY_SCHEMA = {
    "name": "u8_venpriceadjust_history",
    "description": "查看供应商存货调价单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取调价单工作流按钮状态 Schema定义 =====================
U8_VENPRICEADJUST_BUTTONSTATE_SCHEMA = {
    "name": "u8_venpriceadjust_buttonstate",
    "description": "获取供应商存货调价单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
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

# ===================== 审核调价单 Schema定义 =====================
U8_VENPRICEADJUST_AUDIT_SCHEMA = {
    "name": "u8_venpriceadjust_audit",
    "description": "审核供应商存货调价单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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
                "type": "string",
                "description": "是否同意(1=同意;0=不同意)（必填）"
            }
        },
        "required": [
            "voucher_code",
            "agree"
        ]
    }
}

# ===================== 新增调价单 Schema定义 =====================
U8_VENPRICEADJUST_ADD_SCHEMA = {
    "name": "u8_venpriceadjust_add",
    "description": "在用友U8 OpenAPI中新增供应商存货调价单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            "ddate": {"type": "string", "description": "单据日期，格式：yyyy-MM-dd"},
            "ccode": {"type": "string", "description": "单据编号"},
            "cpersoncode": {"type": "string", "description": "业务员编码"},
            "cpersonname": {"type": "string", "description": "业务员名称"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "cdepname": {"type": "string", "description": "部门名称"},
            "isupplytype": {"type": "string", "description": "供货类型"},
            "memo": {"type": "string", "description": "备注"},
            "btaxcost": {"type": "string", "description": "含税"},
            "maker": {"type": "string", "description": "制单人"},
            "entry": {
                "type": "array",
                "description": "调价单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "cvencode": {"type": "string", "description": "供应商编码（必填）"},
                        "cvenabbname": {"type": "string", "description": "供应商简称"},
                        "cvenname": {"type": "string", "description": "供应商名称"},
                        "cinvcode": {"type": "string", "description": "存货编码（必填）"},
                        "cinvaddcode": {"type": "string", "description": "存货代码"},
                        "cinvname": {"type": "string", "description": "存货名称"},
                        "cinvstd": {"type": "string", "description": "存货规格"},
                        "dstartdate": {"type": "string", "description": "生效日期，格式：yyyy-MM-dd"},
                        "denddate": {"type": "string", "description": "失效日期，格式：yyyy-MM-dd"},
                        "cexch_name": {"type": "string", "description": "币种名称"},
                        "ctermcode": {"type": "string", "description": "采购条件编码"},
                        "ctermname": {"type": "string", "description": "采购条件名称"},
                        "bsales": {"type": "string", "description": "销售"},
                        "cbodymemo": {"type": "string", "description": "备注"},
                        "fminquantity": {"type": "number", "description": "最小批量"},
                        "iunitprice": {"type": "string", "description": "单价（必填）"},
                        "itaxrate": {"type": "number", "description": "税率"},
                        "itaxunitprice": {"type": "string", "description": "含税单价（必填）"},
                        "ivouchrowno": {"type": "string", "description": "行号（必填）"}
                    },
                    "required": ["cvencode", "cinvcode", "iunitprice", "itaxunitprice", "ivouchrowno"]
                }
            }
        },
        "required": ["entry"]
    }
}

# ===================== 弃审调价单 Schema定义 =====================
U8_VENPRICEADJUST_ABANDON_SCHEMA = {
    "name": "u8_venpriceadjust_abandon",
    "description": "弃审供应商存货调价单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审批人(人员编码)（必填）"
            },
            "opinion": {
                "type": "string",
                "description": "审批意见"
            }
        },
        "required": [
            "voucher_code",
            "person_code"
        ]
    }
}

# ===================== 获取到货单列表 Schema定义 =====================
U8_PURCHASERECEIPTLIST_BATCH_GET_SCHEMA = {
    "name": "u8_purchasereceiptlist_batch_get",
    "description": "在用友U8 OpenAPI中查询采购到货单列表信息，支持分页、单据号/日期范围、供应商/部门/业务员等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "date_begin": {"type": "string", "description": "起始单据日期，格式：yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束单据日期，格式：yyyy-MM-dd"},
            "cauditdate_begin": {"type": "string", "description": "起始审核日期，格式：yyyy-MM-dd"},
            "cauditdate_end": {"type": "string", "description": "结束审核日期，格式：yyyy-MM-dd"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "purchasetypename": {"type": "string", "description": "采购类型名称"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "vendorabbname": {"type": "string", "description": "供应商简称"},
            "vendorname": {"type": "string", "description": "供应商名称"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称"},
            "payconditioncode": {"type": "string", "description": "付款条件编码"},
            "payconditionname": {"type": "string", "description": "付款条件名称"},
            "foreigncurrency": {"type": "string", "description": "币种"},
            "memory": {"type": "string", "description": "备注"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "ccloser": {"type": "string", "description": "关闭人"},
            "shipcode": {"type": "string", "description": "发货方式编码"},
            "shipname": {"type": "string", "description": "发货方式名称"},
            "cauditdate": {"type": "string", "description": "审核日期"},
            "cverifier": {"type": "string", "description": "审核人"}
        },
        "required": []
    }
}

# ===================== 获取单个到货单 Schema定义 =====================
U8_PURCHASERECEIPT_GET_SCHEMA = {
    "name": "u8_purchasereceipt_get",
    "description": "通过到货单编号获取单张采购到货单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {
                "type": "string",
                "description": "到货单编号（必填）"
            }
        },
        "required": ["id"]
    }
}

# ===================== 审核到货单 Schema定义 =====================
U8_PURCHASERECEIPT_VERIFY_SCHEMA = {
    "name": "u8_purchasereceipt_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审核采购到货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数"
            },
            "user_id": {
                "type": "string",
                "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"
            }
        },
        "required": ["voucher_code"]
    }
}

# ===================== 弃审到货单 Schema定义 =====================
U8_PURCHASERECEIPT_UNVERIFY_SCHEMA = {
    "name": "u8_purchasereceipt_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审采购到货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            }
        },
        "required": ["voucher_code"]
    }
}

# ===================== 获取到货单待办任务 Schema定义 =====================
U8_PURCHASERECEIPT_TASKS_SCHEMA = {
    "name": "u8_purchasereceipt_tasks",
    "description": "获取采购到货单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"},
            "state": {"type": "string", "description": "状态(0=待审;2=审批完成)"},
            "task_type_begin": {"type": "string", "description": "起始类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_type_end": {"type": "string", "description": "结束类型值(1=正常;4=退回;5=退回到提交人)"},
            "task_desc": {"type": "string", "description": "描述"},
            "submitter_code_begin": {"type": "string", "description": "起始发起人编码"},
            "submitter_code_end": {"type": "string", "description": "结束发起人编码"},
            "submitter_name": {"type": "string", "description": "发起人名称关键字"}
        },
        "required": []
    }
}

# ===================== 查看到货单审批进程 Schema定义 =====================
U8_PURCHASERECEIPT_HISTORY_SCHEMA = {
    "name": "u8_purchasereceipt_history",
    "description": "查看采购到货单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取到货单是否启用工作流 Schema定义 =====================
U8_PURCHASERECEIPT_FLOWENABLED_SCHEMA = {
    "name": "u8_purchasereceipt_flowenabled",
    "description": "获取采购到货单是否启用工作流",
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

# ===================== 获取到货单工作流按钮状态 Schema定义 =====================
U8_PURCHASERECEIPT_BUTTONSTATE_SCHEMA = {
    "name": "u8_purchasereceipt_buttonstate",
    "description": "获取采购到货单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
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

# ===================== 审核到货单(工作流) Schema定义 =====================
U8_PURCHASERECEIPT_AUDIT_SCHEMA = {
    "name": "u8_purchasereceipt_audit",
    "description": "审核采购到货单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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
                "type": "string",
                "description": "是否同意(1=同意;0=不同意)（必填）"
            }
        },
        "required": [
            "voucher_code",
            "agree"
        ]
    }
}

# ===================== 新增到货单 Schema定义 =====================
U8_PURCHASERECEIPT_ADD_SCHEMA = {
    "name": "u8_purchasereceipt_add",
    "description": "在用友U8 OpenAPI中新增采购到货单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "单据编号"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "vendorcode": {"type": "string", "description": "供应商编码（必填）"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "payconditioncode": {"type": "string", "description": "付款条件编码"},
            "foreigncurrency": {"type": "string", "description": "币种"},
            "foreigncurrencyrate": {"type": "number", "description": "汇率"},
            "memory": {"type": "string", "description": "备注"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "idiscounttaxtype": {"type": "string", "description": "扣税类别"},
            "shipcode": {"type": "string", "description": "发货方式编码"},
            "cvenpuomprotocol": {"type": "string", "description": "供应商采购单位协议"},
            "entry": {
                "type": "array",
                "description": "到货单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "warehousecode": {"type": "string", "description": "仓库编码"},
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "cinva_unit": {"type": "string", "description": "存货单位"},
                        "iinvexchrate": {"type": "number", "description": "换算率"},
                        "serial": {"type": "string", "description": "批号"},
                        "originaltaxedprice": {"type": "number", "description": "原币含税单价"},
                        "quantity": {"type": "string", "description": "数量（必填）"},
                        "number": {"type": "string", "description": "件数"},
                        "taxrate": {"type": "number", "description": "税率"},
                        "originalprice": {"type": "number", "description": "原币无税单价"},
                        "originalmoney": {"type": "number", "description": "原币无税金额"},
                        "originaltax": {"type": "number", "description": "原币税额"},
                        "originalsum": {"type": "number", "description": "原币价税合计"},
                        "price": {"type": "number", "description": "本币无税单价"},
                        "money": {"type": "number", "description": "本币无税金额"},
                        "tax": {"type": "number", "description": "本币税额"},
                        "sum": {"type": "number", "description": "本币价税合计"},
                        "ivouchrowno": {"type": "string", "description": "行号"},
                        "cbmemo": {"type": "string", "description": "备注"}
                    },
                    "required": ["inventorycode", "quantity"]
                }
            }
        },
        "required": ["vendorcode", "entry"]
    }
}

# ===================== 弃审到货单(工作流) Schema定义 =====================
U8_PURCHASERECEIPT_ABANDON_SCHEMA = {
    "name": "u8_purchasereceipt_abandon",
    "description": "弃审采购到货单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "user_id": {
                "type": "string",
                "description": "审批人(操作员编码)（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审批人(人员编码)（必填）"
            },
            "opinion": {
                "type": "string",
                "description": "审批意见"
            }
        },
        "required": [
            "voucher_code",
            "user_id",
            "person_code"
        ]
    }
}


# ===================== 采购发票表体 Schema =====================
U8_PURCHASEINVOICE_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "inventorycode": {"type": "string", "description": "存货编码"},
        "quantity": {"type": "number", "description": "数量"},
        "assistantunit": {"type": "string", "description": "辅计量单位（必填）"},
        "number": {"type": "number", "description": "件数"},
        "originalprice": {"type": "number", "description": "原币单价"},
        "oritaxcost": {"type": "number", "description": "原币含税单价"},
        "originalmoney": {"type": "number", "description": "原币金额"},
        "originaltax": {"type": "number", "description": "原币税额"},
        "originalsum": {"type": "number", "description": "原币价税合计"},
        "price": {"type": "number", "description": "本币单价"},
        "money": {"type": "number", "description": "本币金额（必填）"},
        "tax": {"type": "number", "description": "本币税额（必填）"},
        "sum": {"type": "number", "description": "本币价税合计"},
        "taxrate": {"type": "number", "description": "税率（必填）"},
        "define22": {"type": "string", "description": "表体自定义项1（必填）"},
        "define23": {"type": "string", "description": "表体自定义项2（必填）"},
        "define24": {"type": "string", "description": "表体自定义项3（必填）"},
        "define25": {"type": "string", "description": "表体自定义项4（必填）"},
        "define26": {"type": "string", "description": "表体自定义项5（必填）"},
        "define27": {"type": "string", "description": "表体自定义项6"},
        "define28": {"type": "string", "description": "表体自定义项7"},
        "define29": {"type": "string", "description": "表体自定义项8"},
        "define30": {"type": "string", "description": "表体自定义项9"},
        "define31": {"type": "string", "description": "表体自定义项10"},
        "define32": {"type": "string", "description": "表体自定义项11"},
        "define33": {"type": "string", "description": "表体自定义项12"},
        "define34": {"type": "string", "description": "表体自定义项13"},
        "define35": {"type": "string", "description": "表体自定义项14"},
        "define36": {"type": "string", "description": "表体自定义项15"},
        "define37": {"type": "string", "description": "表体自定义项16"},
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
        "isfee": {"type": "boolean", "description": "是否为费用"},
        "ivouchrowno": {"type": "number", "description": "行号"}
    },
    "required": ["assistantunit", "money", "tax", "taxrate", "define22", "define23", "define24", "define25", "define26"]
}


# ===================== 新增采购发票 Schema =====================
U8_PURCHASEINVOICE_ADD_SCHEMA = {
    "name": "u8_purchaseinvoice_add",
    "description": "新增一张采购发票",
    "parameters": {
        "type": "object",
        "properties": {
            "csource": {"type": "string", "description": "单据来源"},
            "invoicetype": {"type": "string", "description": "发票类型（必填）"},
            "invoicecode": {"type": "string", "description": "发票号（必填）"},
            "purchasecode": {"type": "string", "description": "采购类型编号（必填）"},
            "date": {"type": "string", "description": "开票日期（必填）"},
            "vendorcode": {"type": "string", "description": "供应商编号"},
            "delegatecode": {"type": "string", "description": "代垫单位编号（必填）"},
            "departmentcode": {"type": "string", "description": "部门编号"},
            "personcode": {"type": "string", "description": "职员编号"},
            "dsdate": {"type": "string", "description": "结算日期"},
            "idiscounttaxtype": {"type": "number", "description": "扣税类别"},
            "payconditioncode": {"type": "string", "description": "付款条件编码"},
            "foreigncurrency": {"type": "string", "description": "外币名称"},
            "foreigncurrencyrate": {"type": "string", "description": "汇率"},
            "taxrate": {"type": "number", "description": "税率"},
            "memory": {"type": "string", "description": "备注"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "isnegative": {"type": "boolean", "description": "负发票标志"},
            "protocolcode": {"type": "string", "description": "收付款协议编码"},
            "define1": {"type": "string", "description": "自定义字段1"},
            "define2": {"type": "string", "description": "自定义字段2"},
            "define3": {"type": "string", "description": "自定义字段3"},
            "define4": {"type": "string", "description": "自定义字段4"},
            "define5": {"type": "string", "description": "自定义字段5"},
            "define6": {"type": "string", "description": "自定义字段6"},
            "define7": {"type": "string", "description": "自定义字段7"},
            "define8": {"type": "string", "description": "自定义字段8"},
            "define9": {"type": "string", "description": "自定义字段9"},
            "define10": {"type": "string", "description": "自定义字段10"},
            "define11": {"type": "string", "description": "自定义字段11"},
            "define12": {"type": "string", "description": "自定义字段12"},
            "define13": {"type": "string", "description": "自定义字段13"},
            "define14": {"type": "string", "description": "自定义字段14"},
            "define15": {"type": "string", "description": "自定义字段15"},
            "define16": {"type": "string", "description": "自定义字段16"},
            "entry": {
                "type": "array",
                "description": "表体数据",
                "items": U8_PURCHASEINVOICE_ENTRY_SCHEMA
            }
        },
        "required": ["invoicetype", "invoicecode", "purchasecode", "date", "delegatecode"]
    }
}


# ===================== 获取单个采购发票 Schema =====================
U8_PURINVOICE_GET_SCHEMA = {
    "name": "u8_purinvoice_get",
    "description": "获取单个单张采购发票",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "采购发票号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}


# ===================== 查询采购发票列表 Schema =====================
U8_PURINVOICE_LIST_SCHEMA = {
    "name": "u8_purinvoice_list",
    "description": "获取采购发票列表信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "type": {"type": "string", "description": "采购发票类型关键字"},
            "cpbvcode_begin": {"type": "string", "description": "起始采购发票号"},
            "cpbvcode_end": {"type": "string", "description": "结束采购发票号"},
            "dpbvdate_begin": {"type": "string", "description": "起始开票日期"},
            "dpbvdate_end": {"type": "string", "description": "结束开票日期"},
            "cexchrate": {"type": "number", "description": "汇率"},
            "cexch_name": {"type": "string", "description": "币种"},
            "cexch_code": {"type": "string", "description": "币种编码"},
            "cdepname": {"type": "string", "description": "部门名称关键字"},
            "cdepcode": {"type": "string", "description": "部门名称编码"},
            "cptname": {"type": "string", "description": "采购类型名称关键字"},
            "cptcode": {"type": "string", "description": "采购类型编码"},
            "cpersonname": {"type": "string", "description": "业务员名称"},
            "cpersoncode": {"type": "string", "description": "业务员编码"},
            "cvenname": {"type": "string", "description": "供应商名称关键字"},
            "cvencode": {"type": "string", "description": "供应商编码"},
            "cvenabbname": {"type": "string", "description": "供应商简称关键字"},
            "ipbvtaxrate": {"type": "number", "description": "表头税率"},
            "cpbvbilltype": {"type": "string", "description": "发票类型"},
            "cbustype": {"type": "string", "description": "业务类型"}
        },
        "required": []
    }
}


# ===================== 采购结账状态 Schema =====================
U8_MENDPU_BATCH_GET_SCHEMA = {
    "name": "u8_mendpu_batch_get",
    "description": "批量获取采购结账状态",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "iyear": {"type": "number", "description": "会计年度"},
            "iperiod_begin": {"type": "number", "description": "起始会计期间"},
            "iperiod_end": {"type": "number", "description": "结束会计期间"}
        },
        "required": []
    }
}


# ===================== 采购订单表体 Schema =====================
U8_PURCHASEORDER_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
        "inventoryname": {"type": "string", "description": "存货名称"},
        "inventorystd": {"type": "string", "description": "规格型号"},
        "unitcode": {"type": "string", "description": "采购单位编码"},
        "unitname": {"type": "string", "description": "采购单位"},
        "quantity": {"type": "number", "description": "数量（必填）"},
        "arrivedate": {"type": "string", "description": "计划到货日期"},
        "price": {"type": "number", "description": "原币单价"},
        "quotedprice": {"type": "number", "description": "报价"},
        "taxprice": {"type": "number", "description": "含税单价"},
        "money": {"type": "number", "description": "原币金额"},
        "tax": {"type": "number", "description": "原币税额"},
        "sum": {"type": "number", "description": "原币价税合计"},
        "discount": {"type": "number", "description": "折扣额"},
        "natprice": {"type": "number", "description": "本币单价"},
        "natmoney": {"type": "number", "description": "本币金额"},
        "assistantunit": {"type": "string", "description": "辅计量单位编码"},
        "nattax": {"type": "number", "description": "本币税额"},
        "natsum": {"type": "number", "description": "本币价税合计"},
        "natdiscount": {"type": "number", "description": "本币折扣额"},
        "taxrate": {"type": "number", "description": "税率"},
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
    "required": ["inventorycode", "quantity", "rowno"]
}


# ===================== 新增采购订单 Schema =====================
U8_PURCHASEORDER_ADD_SCHEMA = {
    "name": "u8_purchaseorder_add",
    "description": "新增一张采购订单",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "订单编号"},
            "date": {"type": "string", "description": "订单日期，默认取系统日期(yyyy-MM-dd)"},
            "operation_type_code": {"type": "string", "description": "采购业务类型，默认普通采购（必填）"},
            "state": {"type": "string", "description": "订单状态"},
            "purchase_type_code": {"type": "string", "description": "采购类型编码"},
            "purchase_type_name": {"type": "string", "description": "采购类型"},
            "vendorcode": {"type": "string", "description": "供应商编码（必填）"},
            "vendorname": {"type": "string", "description": "供应商名称"},
            "vendorabbname": {"type": "string", "description": "供应商简称"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"},
            "currency_name": {"type": "string", "description": "外币名称"},
            "currency_rate": {"type": "number", "description": "汇率"},
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
                "items": U8_PURCHASEORDER_ENTRY_SCHEMA
            }
        },
        "required": ["operation_type_code", "vendorcode", "entry"]
    }
}


# ===================== 获取单个采购订单 Schema =====================
U8_PURCHASEORDER_GET_SCHEMA = {
    "name": "u8_purchaseorder_get",
    "description": "获取单张采购订单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "订单编号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}


# ===================== 查询采购订单列表 Schema =====================
U8_PURCHASEORDER_LIST_SCHEMA = {
    "name": "u8_purchaseorder_list",
    "description": "获取采购订单列表信息",
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
            "state": {"type": "string", "description": "订单状态"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "vendorname": {"type": "string", "description": "供应商名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "remark": {"type": "string", "description": "备注关键字"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"}
        },
        "required": []
    }
}


# ===================== 查询采购订单列表2 Schema =====================
U8_PURCHASEORDER_LIST2_SCHEMA = {
    "name": "u8_purchaseorder_list2",
    "description": "批量获取采购订单（以存货为单位）。获取采购订单列表，需要根据订单执行情况进行筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始订单编号"},
            "code_end": {"type": "string", "description": "结束订单编号"},
            "date_begin": {"type": "string", "description": "起始订单日期，格式:yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束订单日期，格式:yyyy-MM-dd"},
            "state": {"type": "string", "description": "订单状态"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "vendorname": {"type": "string", "description": "供应商名称关键字"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员关键字"},
            "remark": {"type": "string", "description": "备注关键字"},
            "maker": {"type": "string", "description": "制单人"},
            "verifier": {"type": "string", "description": "审核人"},
            "closer": {"type": "string", "description": "关闭人"},
            "inventorycode": {"type": "string", "description": "存货编码"},
            "inventoryname": {"type": "string", "description": "存货名称关键字"},
            "arrivestate": {"type": "string", "description": "到货状态"},
            "receivestate": {"type": "string", "description": "入库状态"},
            "billstate": {"type": "string", "description": "开票状态"},
            "paystate": {"type": "string", "description": "付款状态"}
        },
        "required": []
    }
}


# ===================== 审核采购订单(verify) Schema =====================
U8_PURCHASEORDER_VERIFY_SCHEMA = {
    "name": "u8_purchaseorder_verify",
    "description": "审核一张采购订单",
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


# ===================== 弃审采购订单(unverify) Schema =====================
U8_PURCHASEORDER_UNVERIFY_SCHEMA = {
    "name": "u8_purchaseorder_unverify",
    "description": "弃审一张采购订单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {"type": "string", "description": "单据编号（必填）"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["voucher_code"]
    }
}


# ===================== 采购订单审批(audit) Schema =====================
U8_PURCHASEORDER_AUDIT_SCHEMA = {
    "name": "u8_purchaseorder_audit",
    "description": "审核采购订单（工作流审批）。执行审批动作前，需要保证审批人已经进行ERP登录授权",
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


# ===================== 采购订单弃审(abandon) Schema =====================
U8_PURCHASEORDER_ABANDON_SCHEMA = {
    "name": "u8_purchaseorder_abandon",
    "description": "弃审采购订单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权",
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


# ===================== 获取采购订单按钮状态 Schema =====================
U8_PURCHASEORDER_BUTTONSTATE_SCHEMA = {
    "name": "u8_purchaseorder_buttonstate",
    "description": "获取采购订单工作流按钮是否可用状态。只支持12.0版本，且需要打最新的WF补丁",
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


# ===================== 获取采购订单待办任务 Schema =====================
U8_PURCHASEORDER_TASKS_SCHEMA = {
    "name": "u8_purchaseorder_tasks",
    "description": "获取采购订单待办任务",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
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
        },
        "required": []
    }
}


# ===================== 获取采购订单审批历史 Schema =====================
U8_PURCHASEORDER_HISTORY_SCHEMA = {
    "name": "u8_purchaseorder_history",
    "description": "查看采购订单审批进程。执行审批动作前，需要保证审批人已经进行ERP登录授权",
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



# ===================== 新增一张采购请购单 Schema定义 =====================
U8_PURCHASEREQUISITION_ADD_SCHEMA = {
    "name": "u8_purchaserequisition_add",
    "description": "在用友U8 OpenAPI中新增采购请购单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 单据头参数
            "code": {"type": "string", "description": "请购单号（必填）"},
            "date": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "memory": {"type": "string", "description": "备注"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4"},
            "define5": {"type": "string", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6"},
            "define7": {"type": "string", "description": "单据头自定义项7"},
            "define8": {"type": "string", "description": "单据头自定义项8"},
            "define9": {"type": "string", "description": "单据头自定义项9"},
            "define10": {"type": "string", "description": "单据头自定义项10"},
            "define11": {"type": "string", "description": "单据头自定义项11"},
            "define12": {"type": "string", "description": "单据头自定义项12"},
            "define13": {"type": "string", "description": "单据头自定义项13"},
            "define14": {"type": "string", "description": "单据头自定义项14"},
            "define15": {"type": "string", "description": "单据头自定义项15"},
            "define16": {"type": "string", "description": "单据头自定义项16"},

            # 单据体（entry）列表
            "entry": {
                "type": "array",
                "description": "采购请购单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "inventorycode": {"type": "string", "description": "存货编码（必填）"},
                        "quantity": {"type": "number", "description": "数量（必填）"},
                        "vendorcode": {"type": "string", "description": "供应商编码"},
                        "price": {"type": "number", "description": "单价"},
                        "taxrate": {"type": "number", "description": "税率"},
                        "taxprice": {"type": "number", "description": "含税单价"},
                        "money": {"type": "number", "description": "金额"},
                        "tax": {"type": "number", "description": "税额"},
                        "sum": {"type": "number", "description": "价税合计"},
                        "requiredate": {"type": "string", "description": "需求日期"},
                        "arrivedate": {"type": "string", "description": "到货日期"},
                        "item_class": {"type": "string", "description": "项目大类编码"},
                        "item_code": {"type": "string", "description": "项目编码"},
                        "item_name": {"type": "string", "description": "项目名称"},
                        "btaxcost": {"type": "number", "description": "本币金额"},
                        "num": {"type": "number", "description": "件数"},
                        "unitid": {"type": "string", "description": "单位编码"},
                        "deptcodeexec": {"type": "string", "description": "执行部门编码"},
                        "personcodeexec": {"type": "string", "description": "执行业务员编码"},
                        "currency_name": {"type": "string", "description": "币种名称"},
                        "currency_rate": {"type": "number", "description": "汇率"},
                        "originalprice": {"type": "number", "description": "原币单价"},
                        "originaltaxedprice": {"type": "number", "description": "原币含税单价"},
                        "originalmoney": {"type": "number", "description": "原币金额"},
                        "originaltax": {"type": "number", "description": "原币税额"},
                        "originalsum": {"type": "number", "description": "原币价税合计"},
                        "ivouchrowno": {"type": "integer", "description": "行号"},
                        # 单据体自定义项 1-16
                        "define22": {"type": "string", "description": "单据体自定义项1"},
                        "define23": {"type": "string", "description": "单据体自定义项2"},
                        "define24": {"type": "string", "description": "单据体自定义项3"},
                        "define25": {"type": "string", "description": "单据体自定义项4"},
                        "define26": {"type": "string", "description": "单据体自定义项5"},
                        "define27": {"type": "string", "description": "单据体自定义项6"},
                        "define28": {"type": "string", "description": "单据体自定义项7"},
                        "define29": {"type": "string", "description": "单据体自定义项8"},
                        "define30": {"type": "string", "description": "单据体自定义项9"},
                        "define31": {"type": "string", "description": "单据体自定义项10"},
                        "define32": {"type": "string", "description": "单据体自定义项11"},
                        "define33": {"type": "string", "description": "单据体自定义项12"},
                        "define34": {"type": "string", "description": "单据体自定义项13"},
                        "define35": {"type": "string", "description": "单据体自定义项14"},
                        "define36": {"type": "string", "description": "单据体自定义项15"},
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        # 自由项
                        "free1": {"type": "string", "description": "自由项1"},
                        "free2": {"type": "string", "description": "自由项2"},
                        "free3": {"type": "string", "description": "自由项3"},
                        "free4": {"type": "string", "description": "自由项4"},
                        "free5": {"type": "string", "description": "自由项5"},
                        "free6": {"type": "string", "description": "自由项6"},
                        "free7": {"type": "string", "description": "自由项7"},
                        "free8": {"type": "string", "description": "自由项8"},
                        "free9": {"type": "string", "description": "自由项9"},
                        "free10": {"type": "string", "description": "自由项10"}
                    },
                    "required": ["inventorycode", "quantity"]
                }
            }
        },
        "required": [
            "code",
            "entry"
        ]
    }
}

# ===================== 获取单张采购请购单 Schema定义 =====================
U8_PURCHASEREQUISITION_GET_SCHEMA = {
    "name": "u8_purchaserequisition_get",
    "description": "通过主表id获得采购请购单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "number",
                "description": "主表id"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取采购请购单列表信息 Schema定义 =====================
U8_PURCHASEREQUISITION_LIST_GET_SCHEMA = {
    "name": "u8_purchaserequisition_list_get",
    "description": "在用友U8 OpenAPI中查询采购请购单列表信息，支持分页、单据号/日期范围、部门/业务员/采购类型等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始制单日期，格式:yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束制单日期，格式:yyyy-MM-dd"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "memory": {"type": "string", "description": "备注"},
            "cvoucherstate": {"type": "string", "description": "单据状态"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 审核一张采购请购单 Schema定义 =====================
U8_PURCHASEREQUISITION_VERIFY_SCHEMA = {
    "name": "u8_purchaserequisition_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审核采购请购单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审核人员编码，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"
            },
            "user_id": {
                "type": "string",
                "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"
            }
        },
        "required": [
            "voucher_code"
        ]
    }
}

# ===================== 弃审一张采购请购单 Schema定义 =====================
U8_PURCHASEREQUISITION_UNVERIFY_SCHEMA = {
    "name": "u8_purchaserequisition_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审采购请购单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审核人员编码，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"
            },
            "user_id": {
                "type": "string",
                "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"
            }
        },
        "required": [
            "voucher_code"
        ]
    }
}

# ===================== 获取采购请购单待办任务 Schema定义 =====================
U8_PURCHASEREQUISITION_TASKS_SCHEMA = {
    "name": "u8_purchaserequisition_tasks",
    "description": "获取采购请购单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
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

# ===================== 获取采购请购单审批进程 Schema定义 =====================
U8_PURCHASEREQUISITION_HISTORY_SCHEMA = {
    "name": "u8_purchaserequisition_history",
    "description": "查看采购请购单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取采购请购单是否启用工作流 Schema定义 =====================
U8_PURCHASEREQUISITION_FLOWENABLED_SCHEMA = {
    "name": "u8_purchaserequisition_flowenabled",
    "description": "获取采购请购单是否启用工作流",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 获取采购请购单工作流按钮是否可用状态 Schema定义 =====================
U8_PURCHASEREQUISITION_BUTTONSTATE_SCHEMA = {
    "name": "u8_purchaserequisition_buttonstate",
    "description": "获取采购请购单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
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

# ===================== 审核采购请购单（工作流） Schema定义 =====================
U8_PURCHASEREQUISITION_AUDIT_SCHEMA = {
    "name": "u8_purchaserequisition_audit",
    "description": "审核采购请购单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 弃审采购请购单 Schema定义 =====================
U8_PURCHASEREQUISITION_ABANDON_SCHEMA = {
    "name": "u8_purchaserequisition_abandon",
    "description": "弃审采购请购单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
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



# ===================== 新增一张采购退货单 Schema定义 =====================
U8_PURCHASERETURN_ADD_SCHEMA = {
    "name": "u8_purchasereturn_add",
    "description": "在用友U8 OpenAPI中新增采购退货单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 单据头参数
            "code": {"type": "string", "description": "采购退货单号"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "purchasetypename": {"type": "string", "description": "采购类型名称（必填）"},
            "vendorcode": {"type": "string", "description": "供应商编码（必填）"},
            "vendorabbname": {"type": "string", "description": "供应商简称"},
            "vendorname": {"type": "string", "description": "供应商名称"},
            "departmentcode": {"type": "string", "description": "部门编码（必填）"},
            "departmentname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称（必填）"},
            "payconditioncode": {"type": "string", "description": "付款条件编码"},
            "payconditionname": {"type": "string", "description": "付款条件名称"},
            "foreigncurrency": {"type": "string", "description": "外币币种"},
            "cexch_code": {"type": "string", "description": "汇率编码"},
            "foreigncurrencyrate": {"type": "number", "description": "汇率"},
            "memory": {"type": "string", "description": "备注"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "ccloser": {"type": "string", "description": "关闭人"},
            "idiscounttaxtype": {"type": "string", "description": "扣税类别"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4"},
            "define5": {"type": "string", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6"},
            "define7": {"type": "string", "description": "单据头自定义项7"},
            "define8": {"type": "string", "description": "单据头自定义项8"},
            "define9": {"type": "string", "description": "单据头自定义项9"},
            "define10": {"type": "string", "description": "单据头自定义项10"},
            "define11": {"type": "string", "description": "单据头自定义项11"},
            "define12": {"type": "string", "description": "单据头自定义项12"},
            "define13": {"type": "string", "description": "单据头自定义项13"},
            "define14": {"type": "string", "description": "单据头自定义项14"},
            "define15": {"type": "string", "description": "单据头自定义项15"},
            "define16": {"type": "string", "description": "单据头自定义项16"},
            "shipcode": {"type": "string", "description": "发货方式编码"},
            "shipname": {"type": "string", "description": "发货方式名称"},
            "billtype": {"type": "string", "description": "单据模板类型"},
            "cvouchtype": {"type": "string", "description": "单据类型"},
            "cmodifydate": {"type": "string", "description": "修改日期（必填）"},
            "creviser": {"type": "string", "description": "修改人"},
            "cauditdate": {"type": "string", "description": "审核日期"},
            "cverifier": {"type": "string", "description": "审核人"},
            "cvenpuomprotocol": {"type": "string", "description": "供应商 uom 协议编码"},
            "cvenpuomprotocolname": {"type": "string", "description": "供应商 uom 协议名称"},
            "csysbarcode": {"type": "string", "description": "条形码"},

            # 单据体（entry）列表
            "entry": {
                "type": "array",
                "description": "采购退货单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "warehousecode": {"type": "string", "description": "仓库编码"},
                        "warehousename": {"type": "string", "description": "仓库名称"},
                        "inventorycode": {"type": "string", "description": "存货编码"},
                        "inventoryaddcode": {"type": "string", "description": "存货代码"},
                        "inventoryname": {"type": "string", "description": "存货名称"},
                        "inventorystd": {"type": "string", "description": "存货规格"},
                        "inventoryclasscode": {"type": "string", "description": "存货大类编码"},
                        "unitid": {"type": "string", "description": "单位编码"},
                        "ccomunitcode": {"type": "string", "description": "主计量单位编码"},
                        "cinvm_unit": {"type": "string", "description": "辅计量单位编码"},
                        "cinva_unit": {"type": "string", "description": "采购单位编码"},
                        "iinvexchrate": {"type": "number", "description": "换算率"},
                        "serial": {"type": "string", "description": "批号"},
                        "closer": {"type": "string", "description": "关闭人"},
                        "originaltaxedprice": {"type": "number", "description": "原币含税单价"},
                        "quantity": {"type": "number", "description": "数量"},
                        "number": {"type": "number", "description": "件数"},
                        "originalprice": {"type": "number", "description": "原币单价"},
                        "originalmoney": {"type": "number", "description": "原币金额"},
                        "originaltax": {"type": "number", "description": "原币税额"},
                        "originalsum": {"type": "number", "description": "原币价税合计"},
                        "price": {"type": "number", "description": "本币单价"},
                        "money": {"type": "number", "description": "本币金额"},
                        "tax": {"type": "number", "description": "本币税额"},
                        "sum": {"type": "number", "description": "本币价税合计"},
                        "cbcloser": {"type": "string", "description": "行关闭人"},
                        "free1": {"type": "string", "description": "自由项1（必填）"},
                        "free2": {"type": "string", "description": "自由项2"},
                        # 单据体自定义项 1-16
                        "define22": {"type": "string", "description": "单据体自定义项1"},
                        "define23": {"type": "string", "description": "单据体自定义项2"},
                        "define24": {"type": "string", "description": "单据体自定义项3"},
                        "define25": {"type": "string", "description": "单据体自定义项4"},
                        "define26": {"type": "string", "description": "单据体自定义项5"},
                        "define27": {"type": "string", "description": "单据体自定义项6"},
                        "define28": {"type": "string", "description": "单据体自定义项7"},
                        "define29": {"type": "string", "description": "单据体自定义项8"},
                        "define30": {"type": "string", "description": "单据体自定义项9"},
                        "define31": {"type": "string", "description": "单据体自定义项10"},
                        "define32": {"type": "string", "description": "单据体自定义项11"},
                        "define33": {"type": "string", "description": "单据体自定义项12"},
                        "define34": {"type": "string", "description": "单据体自定义项13"},
                        "define35": {"type": "string", "description": "单据体自定义项14"},
                        "define36": {"type": "string", "description": "单据体自定义项15"},
                        "define37": {"type": "string", "description": "单据体自定义项16"},
                        "taxrate": {"type": "number", "description": "税率"},
                        "iposid": {"type": "integer", "description": "POS机号"},
                        "free3": {"type": "string", "description": "自由项3"},
                        "free4": {"type": "string", "description": "自由项4"},
                        "free5": {"type": "string", "description": "自由项5"},
                        "free6": {"type": "string", "description": "自由项6"},
                        "free7": {"type": "string", "description": "自由项7"},
                        "free8": {"type": "string", "description": "自由项8"},
                        "free9": {"type": "string", "description": "自由项9"},
                        "free10": {"type": "string", "description": "自由项10"},
                        "cordercode": {"type": "string", "description": "订单号"},
                        "vouchstate": {"type": "string", "description": "单据状态"},
                        "ivouchrowno": {"type": "integer", "description": "行号"},
                        "cbmemo": {"type": "string", "description": "备注（必填）"},
                        "cbsysbarcode": {"type": "string", "description": "条形码"}
                    },
                    "required": ["free1", "cbmemo"]
                }
            }
        },
        "required": [
            "purchasetypename",
            "vendorcode",
            "departmentcode",
            "personname",
            "cmodifydate",
            "entry"
        ]
    }
}

# ===================== 获取单张采购退货单 Schema定义 =====================
U8_PURCHASERETURN_GET_SCHEMA = {
    "name": "u8_purchasereturn_get",
    "description": "通过采购退货单编号获得采购退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "采购退货单编号"
            },
            "ds_sequence": {
                "type": "integer",
                "description": "数据源序号（默认取应用的第一个数据源）"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取采购退货单列表信息 Schema定义 =====================
U8_PURCHASERETURN_LIST_GET_SCHEMA = {
    "name": "u8_purchasereturn_list_get",
    "description": "在用友U8 OpenAPI中查询采购退货单列表信息，支持分页、单据号/日期范围、供应商/部门/业务员等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "date_begin": {"type": "string", "description": "起始制单日期，格式:yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束制单日期，格式:yyyy-MM-dd"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "purchasetypecode": {"type": "string", "description": "采购类型编码"},
            "purchasetypename": {"type": "string", "description": "采购类型名称"},
            "vendorcode": {"type": "string", "description": "供应商编码"},
            "vendorabbname": {"type": "string", "description": "供应商简称"},
            "vendorname": {"type": "string", "description": "供应商名称"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称"},
            "payconditioncode": {"type": "string", "description": "付款条件编码"},
            "payconditionname": {"type": "string", "description": "付款条件名称"},
            "foreigncurrency": {"type": "string", "description": "外币币种"},
            "memory": {"type": "string", "description": "备注"},
            "businesstype": {"type": "string", "description": "业务类型"},
            "maker": {"type": "string", "description": "制单人"},
            "ccloser": {"type": "string", "description": "关闭人"},
            "shipcode": {"type": "string", "description": "发货方式编码"},
            "shipname": {"type": "string", "description": "发货方式名称"},
            "cauditdate": {"type": "string", "description": "审核日期"},
            "cverifier": {"type": "string", "description": "审核人"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 审核一张采购退货单 Schema定义 =====================
U8_PURCHASERETURN_VERIFY_SCHEMA = {
    "name": "u8_purchasereturn_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审核采购退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审核人员编码，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取"
            },
            "user_id": {
                "type": "string",
                "description": "审批人用户编码，同person_code参数，且与person_code二选一传入"
            }
        },
        "required": [
            "voucher_code"
        ]
    }
}

# ===================== 弃审一张采购退货单 Schema定义 =====================
U8_PURCHASERETURN_UNVERIFY_SCHEMA = {
    "name": "u8_purchasereturn_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审采购退货单",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            }
        },
        "required": [
            "voucher_code"
        ]
    }
}

# ===================== 获取采购退货单待办任务 Schema定义 =====================
U8_PURCHASERETURN_TASKS_SCHEMA = {
    "name": "u8_purchasereturn_tasks",
    "description": "获取采购退货单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
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

# ===================== 获取采购退货单审批进程 Schema定义 =====================
U8_PURCHASERETURN_HISTORY_SCHEMA = {
    "name": "u8_purchasereturn_history",
    "description": "查看采购退货单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取采购退货单是否启用工作流 Schema定义 =====================
U8_PURCHASERETURN_FLOWENABLED_SCHEMA = {
    "name": "u8_purchasereturn_flowenabled",
    "description": "获取采购退货单是否启用工作流",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "person_code": {"type": "string", "description": "审批人(人员编码)，可以通过api/person获取"},
            "user_id": {"type": "string", "description": "审批人用户编码"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 获取采购退货单工作流按钮是否可用状态 Schema定义 =====================
U8_PURCHASERETURN_BUTTONSTATE_SCHEMA = {
    "name": "u8_purchasereturn_buttonstate",
    "description": "获取采购退货单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
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

# ===================== 审核采购退货单（工作流） Schema定义 =====================
U8_PURCHASERETURN_AUDIT_SCHEMA = {
    "name": "u8_purchasereturn_audit",
    "description": "审核采购退货单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "user_id": {
                "type": "string",
                "description": "审批人(操作员编码)（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审批人(人员编码)（必填）"
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
            "user_id",
            "person_code",
            "agree"
        ]
    }
}

# ===================== 弃审采购退货单 Schema定义 =====================
U8_PURCHASERETURN_ABANDON_SCHEMA = {
    "name": "u8_purchasereturn_abandon",
    "description": "弃审采购退货单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "user_id": {
                "type": "string",
                "description": "审批人(操作员编码)（必填）"
            },
            "person_code": {
                "type": "string",
                "description": "审批人(人员编码)（必填）"
            },
            "opinion": {
                "type": "string",
                "description": "审批意见"
            }
        },
        "required": [
            "voucher_code",
            "user_id",
            "person_code"
        ]
    }
}

# ===================== 获取单个预算信息 Schema定义 =====================
U8_BUDGET_GET_SCHEMA = {
    "name": "u8_budget_get",
    "description": "通过预算项目编码获得单个预算信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "预算项目编码"
            },
            "ds_sequence": {
                "type": "integer",
                "description": "数据源序号（默认取应用的第一个数据源）"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取预算信息 Schema定义 =====================
U8_BUDGET_BATCH_GET_SCHEMA = {
    "name": "u8_budget_batch_get",
    "description": "在用友U8 OpenAPI中批量获取预算信息，支持预算表、口径、版本、目标、项目等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cformcode": {"type": "string", "description": "预算表编码"},
            "ccalibercode1": {"type": "string", "description": "预算口径编码"},
            "cversioncode": {"type": "string", "description": "版本编码"},
            "ctargetcode": {"type": "string", "description": "预算目标编码"},
            "ctargetcode_ctl": {"type": "string", "description": "预算目标控制编码"},
            "citemcode": {"type": "string", "description": "预算项目编码"},
            "citemname": {"type": "string", "description": "预算项目名称"},
            "fperiod13": {"type": "number", "description": "13期预算金额"},
            "fperiod12": {"type": "number", "description": "12期预算金额"},
            "freserve12": {"type": "number", "description": "12期预留金额"},
            "pk": {"type": "string", "description": "主键"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}
