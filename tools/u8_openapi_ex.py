import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 出口订单 entry 子数据模型 =====================
class ExportorderEntry(BaseModel):
    """出口订单 entry 子表数据模型"""
    pricecontailtax: Optional[bool] = Field(None, description="报价是否含税，默认值1")
    comcriterion: Optional[str] = Field(None, description="佣金计提标准")
    comrule: Optional[str] = Field(None, description="佣金规则")
    comunitcode: Optional[str] = Field(None, description="计量单位编码")
    configstatus: Optional[str] = Field(None, description="选配状态")
    cusinvcode: Optional[str] = Field(None, description="客户存货编码")
    cusinvname: Optional[str] = Field(None, description="客户存货名称")
    invcode: Optional[str] = Field(None, description="存货编码")
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
    materialsize: Optional[str] = Field(None, description="材积")
    memo: Optional[str] = Field(None, description="备注")
    packinfo: Optional[str] = Field(None, description="包装要求")
    packsize: Optional[str] = Field(None, description="尺寸")
    sacomunitcode: Optional[str] = Field(None, description="销售计量单位编码")
    vcomunitcode: Optional[str] = Field(None, description="体积单位编码")
    volumes: Optional[str] = Field(None, description="体积")
    wcomunitcode: Optional[str] = Field(None, description="重量单位编码")
    completedate: Optional[str] = Field(None, description="预完工日期")
    shippingdate: Optional[str] = Field(None, description="预发货日期")
    changrate: Optional[float] = Field(None, description="换算率")
    comcoefficient: Optional[float] = Field(None, description="佣金系数")
    commoney: Optional[float] = Field(None, description="佣金")
    comrate: Optional[float] = Field(None, description="佣金率（％）")
    discount: Optional[float] = Field(None, description="原币折扣额")
    fobmoney: Optional[float] = Field(None, description="FOB金额")
    freight: Optional[float] = Field(None, description="运费")
    gweight: Optional[float] = Field(None, description="单位毛重")
    gweights: Optional[float] = Field(None, description="毛重")
    insurance: Optional[float] = Field(None, description="保险费")
    money: Optional[float] = Field(None, description="原币无税金额")
    natcommoney: Optional[float] = Field(None, description="本币佣金")
    natdiscount: Optional[float] = Field(None, description="本币折扣额")
    natfobmoney: Optional[float] = Field(None, description="FOB本币金额")
    natfreight: Optional[float] = Field(None, description="本币运费")
    natinsurance: Optional[float] = Field(None, description="本币保险费")
    natmoney: Optional[float] = Field(None, description="本币无税金额")
    natprice: Optional[float] = Field(None, description="本币无税单价")
    nattax: Optional[float] = Field(None, description="本币税额")
    nattaxmoney: Optional[float] = Field(None, description="本币金额")
    nattaxprice: Optional[float] = Field(None, description="本币单价")
    num: Optional[float] = Field(None, description="件数")
    nweight: Optional[float] = Field(None, description="单位净重")
    nweights: Optional[float] = Field(None, description="净重")
    overflowrange: Optional[float] = Field(None, description="溢短装")
    packqty: Optional[float] = Field(None, description="包装数量")
    ppartqty: Optional[float] = Field(None, description="母件数量")
    price: Optional[float] = Field(None, description="原币无税单价")
    quantity: Optional[float] = Field(None, description="数量")
    quotedprice: Optional[float] = Field(None, description="报价")
    tax: Optional[float] = Field(None, description="原币税额")
    taxmoney: Optional[float] = Field(None, description="原币金额")
    taxprice: Optional[float] = Field(None, description="原币单价")
    taxrate: Optional[float] = Field(None, description="税率（％）")
    volume: Optional[float] = Field(None, description="单位体积")
    ppartseqid: Optional[int] = Field(None, description="选配序号")
    rowno: Optional[int] = Field(None, description="订单行号")
    fcost: Optional[float] = Field(None, description="成本")
    fgrossrate: Optional[float] = Field(None, description="毛利")
    cvencode: Optional[str] = Field(None, description="供应商编码")
    centerprise: Optional[str] = Field(None, description="生产厂家")
    fimqty: Optional[float] = Field(None, description="累计下达进口数量")
    fcompensatemoney: Optional[float] = Field(None, description="补差发票原币金额")
    fnatcompensatemoney: Optional[float] = Field(None, description="补差发票本币金额")
    flength: Optional[float] = Field(None, description="长")
    fwidth: Optional[float] = Field(None, description="宽")
    fheight: Optional[float] = Field(None, description="高")
    citem_class: Optional[str] = Field(None, description="项目大类编码")
    citem_cname: Optional[str] = Field(None, description="项目大类名称")
    citemcode: Optional[str] = Field(None, description="项目编码")
    citemname: Optional[str] = Field(None, description="项目名称")
    iciqbookid: Optional[int] = Field(None, description="海关手册ID，默认值-1")
    cciqbookcode: Optional[str] = Field(None, description="海关手册编码")
    cciqcode: Optional[str] = Field(None, description="海关编码")
    fciqchangrate: Optional[float] = Field(None, description="海关换算率，默认值1")
    fbacktaxrate: Optional[float] = Field(None, description="退税率")
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


# ===================== 批量获取出口订单列表 数据模型 =====================
class GetExportorderlistBatchInput(BaseModel):
    """批量获取出口订单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始出口订单号")
    code_end: Optional[str] = Field(None, description="结束出口订单号")
    aportcode: Optional[str] = Field(None, description="目的港编码")
    cuscode: Optional[str] = Field(None, description="客户编号")
    cusordercode: Optional[str] = Field(None, description="客户订单号")
    memos: Optional[str] = Field(None, description="备注关键字")
    personcode: Optional[str] = Field(None, description="业务员编号")
    receiveperson: Optional[str] = Field(None, description="收货人关键字")
    sccode: Optional[str] = Field(None, description="发运方式编码")
    shipvencode: Optional[str] = Field(None, description="船公司编码")
    sportcode: Optional[str] = Field(None, description="装运港编码")
    sscode: Optional[str] = Field(None, description="结算方式编码")
    stcode: Optional[str] = Field(None, description="销售类型编号")
    svencode: Optional[str] = Field(None, description="承运商编码")
    tmcode: Optional[str] = Field(None, description="贸易方式代码")
    tportcode: Optional[str] = Field(None, description="转运港编码")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    cverifier: Optional[str] = Field(None, description="审批人关键字")
    depname: Optional[str] = Field(None, description="部门名称关键字")
    personname: Optional[str] = Field(None, description="业务员关键字")
    scname: Optional[str] = Field(None, description="发运方式关键字")
    shipvenname: Optional[str] = Field(None, description="船公司关键字")
    sportname: Optional[str] = Field(None, description="装运港关键字")
    cssname: Optional[str] = Field(None, description="结算方式关键字")
    stname: Optional[str] = Field(None, description="销售类型关键字")
    svenname: Optional[str] = Field(None, description="承运商关键字")
    tmname: Optional[str] = Field(None, description="贸易方式关键字")
    tportname: Optional[str] = Field(None, description="转运港关键字")


# ===================== 获取出口订单 数据模型 =====================
class GetExportorderInput(BaseModel):
    """获取出口订单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="出口订单号（必填）")


# ===================== 新增出口订单 数据模型 =====================
class AddExportorderInput(BaseModel):
    """新增出口订单输入模型"""
    cal: Optional[bool] = Field(None, description="原币折算方式")
    canchangeprice: Optional[bool] = Field(None, description="后续单据是否可修改单价")
    farecontrol: Optional[bool] = Field(None, description="控制费用")
    aportcode: Optional[str] = Field(None, description="目的港编码")
    batchshipping: Optional[str] = Field(None, description="分批装运")
    bustype: Optional[str] = Field(None, description="业务类型")
    chinesesummoney: Optional[str] = Field(None, description="原币总额中文大写")
    code: Optional[str] = Field(None, description="出口订单号")
    comvencode: Optional[str] = Field(None, description="佣金支付对象编码")
    containersize: Optional[str] = Field(None, description="货柜尺寸")
    cusaddress: Optional[str] = Field(None, description="客户地址")
    cuscode: Optional[str] = Field(None, description="客户编号")
    cusenaddr1: Optional[str] = Field(None, description="英文地址1")
    cusenaddr2: Optional[str] = Field(None, description="英文地址2")
    cusenaddr3: Optional[str] = Field(None, description="英文地址3")
    cusenaddr4: Optional[str] = Field(None, description="英文地址4")
    cusoaddress: Optional[str] = Field(None, description="发货地址")
    cusordercode: Optional[str] = Field(None, description="客户订单号")
    cusperson: Optional[str] = Field(None, description="联系人")
    define1: Optional[str] = Field(None, description="表头自定义项1")
    define2: Optional[str] = Field(None, description="表头自定义项2")
    define3: Optional[str] = Field(None, description="表头自定义项3")
    define4: Optional[str] = Field(None, description="表头自定义项4")
    define5: Optional[str] = Field(None, description="表头自定义项5")
    define6: Optional[str] = Field(None, description="表头自定义项6")
    define7: Optional[str] = Field(None, description="表头自定义项7")
    define8: Optional[str] = Field(None, description="表头自定义项8")
    define9: Optional[str] = Field(None, description="表头自定义项9")
    define10: Optional[str] = Field(None, description="表头自定义项10")
    define11: Optional[str] = Field(None, description="表头自定义项11")
    define12: Optional[str] = Field(None, description="表头自定义项12")
    define13: Optional[str] = Field(None, description="表头自定义项13")
    define14: Optional[str] = Field(None, description="表头自定义项14")
    define15: Optional[str] = Field(None, description="表头自定义项15")
    define16: Optional[str] = Field(None, description="表头自定义项16")
    depcode: Optional[str] = Field(None, description="部门编码")
    engaddress1: Optional[str] = Field(None, description="单位英文地址1")
    engaddress2: Optional[str] = Field(None, description="单位英文地址2")
    engaddress3: Optional[str] = Field(None, description="单位英文地址3")
    engaddress4: Optional[str] = Field(None, description="单位英文地址4")
    englishsummoney: Optional[str] = Field(None, description="原币总额英文大写")
    exch_name: Optional[str] = Field(None, description="币种")
    farecontroltype: Optional[str] = Field(None, description="费用控制方式")
    incotermcode: Optional[str] = Field(None, description="贸易术语，默认值FOB")
    incotermremark: Optional[str] = Field(None, description="说明项，默认值FOB")
    invgeneraldesc: Optional[str] = Field(None, description="总品名描述")
    memos: Optional[str] = Field(None, description="备注")
    outaftercredit: Optional[str] = Field(None, description="收到信用证后才能发货，默认值'否'")
    paycode: Optional[str] = Field(None, description="付款条件编码")
    personcode: Optional[str] = Field(None, description="业务员编号")
    receiveaddress: Optional[str] = Field(None, description="收货地址1")
    receiveaddress2: Optional[str] = Field(None, description="收货地址2")
    receiveaddress3: Optional[str] = Field(None, description="收货地址3")
    receiveaddress4: Optional[str] = Field(None, description="收货地址4")
    receivecompany: Optional[str] = Field(None, description="收货单位")
    receiveperson: Optional[str] = Field(None, description="收货人")
    sccode: Optional[str] = Field(None, description="发运方式编码")
    shipvencode: Optional[str] = Field(None, description="船公司编码")
    sportcode: Optional[str] = Field(None, description="装运港编码")
    sscode: Optional[str] = Field(None, description="结算方式编码")
    stcode: Optional[str] = Field(None, description="销售类型编号")
    svencode: Optional[str] = Field(None, description="承运商编码")
    tmcode: Optional[str] = Field(None, description="贸易方式代码")
    tportcode: Optional[str] = Field(None, description="转运港编码")
    transport: Optional[str] = Field(None, description="转运，默认值'不允许'")
    unitenglish: Optional[str] = Field(None, description="单位英文名称")
    unitname: Optional[str] = Field(None, description="本单位名称")
    unittel: Optional[str] = Field(None, description="联系电话")
    date: Optional[str] = Field(None, description="单据日期")
    lastedshippingdate: Optional[str] = Field(None, description="最迟装船期")
    containerqty: Optional[float] = Field(None, description="货柜数量")
    exchrate: Optional[float] = Field(None, description="汇率，默认值1")
    fobsummoney: Optional[float] = Field(None, description="FOB原币金额")
    generalcommoney: Optional[float] = Field(None, description="佣金")
    generalcomrate: Optional[float] = Field(None, description="佣金率")
    generalcomremark: Optional[float] = Field(None, description="佣金描述")
    generaloverflowitem: Optional[float] = Field(None, description="溢短装条款")
    generaloverflowrange: Optional[float] = Field(None, description="溢短装%")
    generaltaxrate: Optional[float] = Field(None, description="税率（％）")
    natfobsummoney: Optional[float] = Field(None, description="FOB本币金额")
    natgeneralcommoney: Optional[float] = Field(None, description="本币佣金")
    natsummoney: Optional[float] = Field(None, description="本币总额")
    summoney: Optional[float] = Field(None, description="原币总额")
    dec: Optional[float] = Field(None, description="汇率小数位数，默认值5")
    prearriveday: Optional[float] = Field(None, description="信用证必须在装船前几天内收到")
    ffreightsum: Optional[float] = Field(None, description="运费")
    fnatfreightsum: Optional[float] = Field(None, description="本币运费")
    bfreighttofare: Optional[bool] = Field(None, description="运费转费用单，默认值0")
    finsurancesum: Optional[float] = Field(None, description="保险费")
    fnatinsurancesum: Optional[float] = Field(None, description="本币保险费")
    binsurancetofare: Optional[bool] = Field(None, description="保险费转费用单，默认值0")
    cinsurancevencode: Optional[str] = Field(None, description="保险公司编码")
    bcommisiontype: Optional[bool] = Field(None, description="佣金类型")
    bcommisiontofare: Optional[bool] = Field(None, description="佣金转费用单")
    cgatheringplan: Optional[str] = Field(None, description="收付款协议编码")
    iverifystate: Optional[int] = Field(None, description="审批状态")
    cverifier: Optional[str] = Field(None, description="审批人")
    dverifierdate: Optional[str] = Field(None, description="审批时间")
    ivtid: Optional[int] = Field(None, description="单据模版号")
    cvouchtype: Optional[str] = Field(None, description="单据类型")
    idrawbackstyle: Optional[int] = Field(None, description="退税方式，默认值1")
    cmaker: Optional[str] = Field(None, description="制单人，默认值demo")
    entry: Optional[List[ExportorderEntry]] = Field(None, description="出口订单明细列表")


# ===================== 批量获取出口订单列表 Tool函数 =====================
def u8_exportorderlist_batch_get_tool(input_data: GetExportorderlistBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取出口订单列表信息，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/exportorderlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "出口订单列表获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "出口订单列表获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "exportorderlist": result.get("exportorderlist", {})
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


# ===================== 获取出口订单 Tool函数 =====================
def u8_exportorder_get_tool(input_data: GetExportorderInput, client: U8OpenAPIClient) -> str:
    """
    通过出口订单号获取用友U8中的出口订单信息，包含表头及表体明细。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/exportorder/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "出口订单信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "出口订单信息获取成功",
            "data": {
                "exportorder": result.get("exportorder", {}),
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


# ===================== 新增出口订单 Tool函数 =====================
def u8_exportorder_add_tool(input_data: AddExportorderInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增出口订单信息。
    """
    request_body: Dict[str, Any] = {
        "exportorder": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/exportorder/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "出口订单新增失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "出口订单新增成功",
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



# ===================== 批量获取出口订单列表 Schema定义 =====================
U8_EXPORTORDERLIST_BATCH_GET_SCHEMA = {
    "name": "u8_exportorderlist_batch_get",
    "description": "在用友U8 OpenAPI中批量获取出口订单列表信息，支持按订单号、目的港、客户、业务员、单据日期等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始出口订单号"},
            "code_end": {"type": "string", "description": "结束出口订单号"},
            "aportcode": {"type": "string", "description": "目的港编码"},
            "cuscode": {"type": "string", "description": "客户编号"},
            "cusordercode": {"type": "string", "description": "客户订单号"},
            "memos": {"type": "string", "description": "备注关键字"},
            "personcode": {"type": "string", "description": "业务员编号"},
            "receiveperson": {"type": "string", "description": "收货人关键字"},
            "sccode": {"type": "string", "description": "发运方式编码"},
            "shipvencode": {"type": "string", "description": "船公司编码"},
            "sportcode": {"type": "string", "description": "装运港编码"},
            "sscode": {"type": "string", "description": "结算方式编码"},
            "stcode": {"type": "string", "description": "销售类型编号"},
            "svencode": {"type": "string", "description": "承运商编码"},
            "tmcode": {"type": "string", "description": "贸易方式代码"},
            "tportcode": {"type": "string", "description": "转运港编码"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "cverifier": {"type": "string", "description": "审批人关键字"},
            "depname": {"type": "string", "description": "部门名称关键字"},
            "personname": {"type": "string", "description": "业务员关键字"},
            "scname": {"type": "string", "description": "发运方式关键字"},
            "shipvenname": {"type": "string", "description": "船公司关键字"},
            "sportname": {"type": "string", "description": "装运港关键字"},
            "cssname": {"type": "string", "description": "结算方式关键字"},
            "stname": {"type": "string", "description": "销售类型关键字"},
            "svenname": {"type": "string", "description": "承运商关键字"},
            "tmname": {"type": "string", "description": "贸易方式关键字"},
            "tportname": {"type": "string", "description": "转运港关键字"}
        },
        "required": []
    }
}


# ===================== 获取出口订单 Schema定义 =====================
U8_EXPORTORDER_GET_SCHEMA = {
    "name": "u8_exportorder_get",
    "description": "在用友U8 OpenAPI中通过出口订单号获取出口订单信息，包含表头及表体明细(entry)",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "出口订单号（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 新增出口订单 Schema定义 =====================
U8_EXPORTORDER_ADD_SCHEMA = {
    "name": "u8_exportorder_add",
    "description": "在用友U8 OpenAPI中新增出口订单，支持录入订单表头信息（客户、币种、贸易术语、装运港、目的港等）及表体明细（存货编码、数量、单价、FOB金额等）",
    "parameters": {
        "type": "object",
        "properties": {
            "cal": {"type": "boolean", "description": "原币折算方式"},
            "canchangeprice": {"type": "boolean", "description": "后续单据是否可修改单价"},
            "farecontrol": {"type": "boolean", "description": "控制费用"},
            "aportcode": {"type": "string", "description": "目的港编码"},
            "batchshipping": {"type": "string", "description": "分批装运"},
            "bustype": {"type": "string", "description": "业务类型"},
            "chinesesummoney": {"type": "string", "description": "原币总额中文大写"},
            "code": {"type": "string", "description": "出口订单号"},
            "comvencode": {"type": "string", "description": "佣金支付对象编码"},
            "containersize": {"type": "string", "description": "货柜尺寸"},
            "cusaddress": {"type": "string", "description": "客户地址"},
            "cuscode": {"type": "string", "description": "客户编号"},
            "cusenaddr1": {"type": "string", "description": "英文地址1"},
            "cusenaddr2": {"type": "string", "description": "英文地址2"},
            "cusenaddr3": {"type": "string", "description": "英文地址3"},
            "cusenaddr4": {"type": "string", "description": "英文地址4"},
            "cusoaddress": {"type": "string", "description": "发货地址"},
            "cusordercode": {"type": "string", "description": "客户订单号"},
            "cusperson": {"type": "string", "description": "联系人"},
            "define1": {"type": "string", "description": "表头自定义项1"},
            "define2": {"type": "string", "description": "表头自定义项2"},
            "define3": {"type": "string", "description": "表头自定义项3"},
            "define4": {"type": "string", "description": "表头自定义项4"},
            "define5": {"type": "string", "description": "表头自定义项5"},
            "define6": {"type": "string", "description": "表头自定义项6"},
            "define7": {"type": "string", "description": "表头自定义项7"},
            "define8": {"type": "string", "description": "表头自定义项8"},
            "define9": {"type": "string", "description": "表头自定义项9"},
            "define10": {"type": "string", "description": "表头自定义项10"},
            "define11": {"type": "string", "description": "表头自定义项11"},
            "define12": {"type": "string", "description": "表头自定义项12"},
            "define13": {"type": "string", "description": "表头自定义项13"},
            "define14": {"type": "string", "description": "表头自定义项14"},
            "define15": {"type": "string", "description": "表头自定义项15"},
            "define16": {"type": "string", "description": "表头自定义项16"},
            "depcode": {"type": "string", "description": "部门编码"},
            "engaddress1": {"type": "string", "description": "单位英文地址1"},
            "engaddress2": {"type": "string", "description": "单位英文地址2"},
            "engaddress3": {"type": "string", "description": "单位英文地址3"},
            "engaddress4": {"type": "string", "description": "单位英文地址4"},
            "englishsummoney": {"type": "string", "description": "原币总额英文大写"},
            "exch_name": {"type": "string", "description": "币种"},
            "farecontroltype": {"type": "string", "description": "费用控制方式"},
            "incotermcode": {"type": "string", "description": "贸易术语，默认值FOB"},
            "incotermremark": {"type": "string", "description": "说明项，默认值FOB"},
            "invgeneraldesc": {"type": "string", "description": "总品名描述"},
            "memos": {"type": "string", "description": "备注"},
            "outaftercredit": {"type": "string", "description": "收到信用证后才能发货，默认值'否'"},
            "paycode": {"type": "string", "description": "付款条件编码"},
            "personcode": {"type": "string", "description": "业务员编号"},
            "receiveaddress": {"type": "string", "description": "收货地址1"},
            "receiveaddress2": {"type": "string", "description": "收货地址2"},
            "receiveaddress3": {"type": "string", "description": "收货地址3"},
            "receiveaddress4": {"type": "string", "description": "收货地址4"},
            "receivecompany": {"type": "string", "description": "收货单位"},
            "receiveperson": {"type": "string", "description": "收货人"},
            "sccode": {"type": "string", "description": "发运方式编码"},
            "shipvencode": {"type": "string", "description": "船公司编码"},
            "sportcode": {"type": "string", "description": "装运港编码"},
            "sscode": {"type": "string", "description": "结算方式编码"},
            "stcode": {"type": "string", "description": "销售类型编号"},
            "svencode": {"type": "string", "description": "承运商编码"},
            "tmcode": {"type": "string", "description": "贸易方式代码"},
            "tportcode": {"type": "string", "description": "转运港编码"},
            "transport": {"type": "string", "description": "转运，默认值'不允许'"},
            "unitenglish": {"type": "string", "description": "单位英文名称"},
            "unitname": {"type": "string", "description": "本单位名称"},
            "unittel": {"type": "string", "description": "联系电话"},
            "date": {"type": "string", "description": "单据日期"},
            "lastedshippingdate": {"type": "string", "description": "最迟装船期"},
            "containerqty": {"type": "number", "description": "货柜数量"},
            "exchrate": {"type": "number", "description": "汇率，默认值1"},
            "fobsummoney": {"type": "number", "description": "FOB原币金额"},
            "generalcommoney": {"type": "number", "description": "佣金"},
            "generalcomrate": {"type": "number", "description": "佣金率"},
            "generalcomremark": {"type": "number", "description": "佣金描述"},
            "generaloverflowitem": {"type": "number", "description": "溢短装条款"},
            "generaloverflowrange": {"type": "number", "description": "溢短装%"},
            "generaltaxrate": {"type": "number", "description": "税率（％）"},
            "natfobsummoney": {"type": "number", "description": "FOB本币金额"},
            "natgeneralcommoney": {"type": "number", "description": "本币佣金"},
            "natsummoney": {"type": "number", "description": "本币总额"},
            "summoney": {"type": "number", "description": "原币总额"},
            "dec": {"type": "number", "description": "汇率小数位数，默认值5"},
            "prearriveday": {"type": "number", "description": "信用证必须在装船前几天内收到"},
            "ffreightsum": {"type": "number", "description": "运费"},
            "fnatfreightsum": {"type": "number", "description": "本币运费"},
            "bfreighttofare": {"type": "boolean", "description": "运费转费用单，默认值0"},
            "finsurancesum": {"type": "number", "description": "保险费"},
            "fnatinsurancesum": {"type": "number", "description": "本币保险费"},
            "binsurancetofare": {"type": "boolean", "description": "保险费转费用单，默认值0"},
            "cinsurancevencode": {"type": "string", "description": "保险公司编码"},
            "bcommisiontype": {"type": "boolean", "description": "佣金类型"},
            "bcommisiontofare": {"type": "boolean", "description": "佣金转费用单"},
            "cgatheringplan": {"type": "string", "description": "收付款协议编码"},
            "iverifystate": {"type": "integer", "description": "审批状态"},
            "cverifier": {"type": "string", "description": "审批人"},
            "dverifierdate": {"type": "string", "description": "审批时间"},
            "ivtid": {"type": "integer", "description": "单据模版号"},
            "cvouchtype": {"type": "string", "description": "单据类型"},
            "idrawbackstyle": {"type": "integer", "description": "退税方式，默认值1"},
            "cmaker": {"type": "string", "description": "制单人，默认值demo"},
            "entry": {
                "type": "array",
                "description": "出口订单明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "pricecontailtax": {"type": "boolean", "description": "报价是否含税，默认值1"},
                        "comcriterion": {"type": "string", "description": "佣金计提标准"},
                        "comrule": {"type": "string", "description": "佣金规则"},
                        "comunitcode": {"type": "string", "description": "计量单位编码"},
                        "configstatus": {"type": "string", "description": "选配状态"},
                        "cusinvcode": {"type": "string", "description": "客户存货编码"},
                        "cusinvname": {"type": "string", "description": "客户存货名称"},
                        "invcode": {"type": "string", "description": "存货编码"},
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
                        "materialsize": {"type": "string", "description": "材积"},
                        "memo": {"type": "string", "description": "备注"},
                        "packinfo": {"type": "string", "description": "包装要求"},
                        "packsize": {"type": "string", "description": "尺寸"},
                        "sacomunitcode": {"type": "string", "description": "销售计量单位编码"},
                        "vcomunitcode": {"type": "string", "description": "体积单位编码"},
                        "volumes": {"type": "string", "description": "体积"},
                        "wcomunitcode": {"type": "string", "description": "重量单位编码"},
                        "completedate": {"type": "string", "description": "预完工日期"},
                        "shippingdate": {"type": "string", "description": "预发货日期"},
                        "changrate": {"type": "number", "description": "换算率"},
                        "comcoefficient": {"type": "number", "description": "佣金系数"},
                        "commoney": {"type": "number", "description": "佣金"},
                        "comrate": {"type": "number", "description": "佣金率（％）"},
                        "discount": {"type": "number", "description": "原币折扣额"},
                        "fobmoney": {"type": "number", "description": "FOB金额"},
                        "freight": {"type": "number", "description": "运费"},
                        "gweight": {"type": "number", "description": "单位毛重"},
                        "gweights": {"type": "number", "description": "毛重"},
                        "insurance": {"type": "number", "description": "保险费"},
                        "money": {"type": "number", "description": "原币无税金额"},
                        "natcommoney": {"type": "number", "description": "本币佣金"},
                        "natdiscount": {"type": "number", "description": "本币折扣额"},
                        "natfobmoney": {"type": "number", "description": "FOB本币金额"},
                        "natfreight": {"type": "number", "description": "本币运费"},
                        "natinsurance": {"type": "number", "description": "本币保险费"},
                        "natmoney": {"type": "number", "description": "本币无税金额"},
                        "natprice": {"type": "number", "description": "本币无税单价"},
                        "nattax": {"type": "number", "description": "本币税额"},
                        "nattaxmoney": {"type": "number", "description": "本币金额"},
                        "nattaxprice": {"type": "number", "description": "本币单价"},
                        "num": {"type": "number", "description": "件数"},
                        "nweight": {"type": "number", "description": "单位净重"},
                        "nweights": {"type": "number", "description": "净重"},
                        "overflowrange": {"type": "number", "description": "溢短装"},
                        "packqty": {"type": "number", "description": "包装数量"},
                        "ppartqty": {"type": "number", "description": "母件数量"},
                        "price": {"type": "number", "description": "原币无税单价"},
                        "quantity": {"type": "number", "description": "数量"},
                        "quotedprice": {"type": "number", "description": "报价"},
                        "tax": {"type": "number", "description": "原币税额"},
                        "taxmoney": {"type": "number", "description": "原币金额"},
                        "taxprice": {"type": "number", "description": "原币单价"},
                        "taxrate": {"type": "number", "description": "税率（％）"},
                        "volume": {"type": "number", "description": "单位体积"},
                        "ppartseqid": {"type": "integer", "description": "选配序号"},
                        "rowno": {"type": "integer", "description": "订单行号"},
                        "fcost": {"type": "number", "description": "成本"},
                        "fgrossrate": {"type": "number", "description": "毛利"},
                        "cvencode": {"type": "string", "description": "供应商编码"},
                        "centerprise": {"type": "string", "description": "生产厂家"},
                        "fimqty": {"type": "number", "description": "累计下达进口数量"},
                        "fcompensatemoney": {"type": "number", "description": "补差发票原币金额"},
                        "fnatcompensatemoney": {"type": "number", "description": "补差发票本币金额"},
                        "flength": {"type": "number", "description": "长"},
                        "fwidth": {"type": "number", "description": "宽"},
                        "fheight": {"type": "number", "description": "高"},
                        "citem_class": {"type": "string", "description": "项目大类编码"},
                        "citem_cname": {"type": "string", "description": "项目大类名称"},
                        "citemcode": {"type": "string", "description": "项目编码"},
                        "citemname": {"type": "string", "description": "项目名称"},
                        "iciqbookid": {"type": "integer", "description": "海关手册ID，默认值-1"},
                        "cciqbookcode": {"type": "string", "description": "海关手册编码"},
                        "cciqcode": {"type": "string", "description": "海关编码"},
                        "fciqchangrate": {"type": "number", "description": "海关换算率，默认值1"},
                        "fbacktaxrate": {"type": "number", "description": "退税率"},
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


