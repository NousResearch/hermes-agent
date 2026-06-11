import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 店铺商品档案 entry 子数据模型 =====================
class EbIteminvcontraposeEntry(BaseModel):
    """店铺商品档案 entry 子表数据模型"""
    cShopCode: Optional[str] = Field(None, description="店铺编码")
    cShopName: Optional[str] = Field(None, description="店铺名称")
    cEBItemCode: Optional[str] = Field(None, description="店铺商品编码")
    cEBItemName: Optional[str] = Field(None, description="店铺商品名称")
    cEBSKUID: Optional[str] = Field(None, description="店铺商品SKU")
    cEBItemMemo: Optional[str] = Field(None, description="店铺商品描述")
    cOutIID: Optional[str] = Field(None, description="商家编码")
    cOutSKUID: Optional[str] = Field(None, description="商家SKU")


# ===================== 批量获取店铺商品档案 数据模型 =====================
class GetEbIteminvcontraposeBatchInput(BaseModel):
    """批量获取店铺商品档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cShopCode: Optional[str] = Field(None, description="店铺编码")
    cShopName: Optional[str] = Field(None, description="店铺名称关键字")
    cEBItemCode_begin: Optional[str] = Field(None, description="起始店铺商品编码")
    cEBItemCode_end: Optional[str] = Field(None, description="结束店铺商品编码")
    cEBItemName: Optional[str] = Field(None, description="店铺商品名称关键字")
    cEBSKUID_begin: Optional[str] = Field(None, description="起始店铺商品SKU")
    cEBSKUID_end: Optional[str] = Field(None, description="结束店铺商品SKU")
    cEBItemMemo: Optional[str] = Field(None, description="店铺商品描述")
    cOutIID: Optional[str] = Field(None, description="商家编码")
    cOutSKUID: Optional[str] = Field(None, description="商家SKU")


# ===================== 新增店铺商品档案 数据模型 =====================
class AddEbIteminvcontraposeInput(BaseModel):
    """新增店铺商品档案输入模型"""
    cShopCode: str = Field(..., description="店铺编码（必填）")
    cShopName: Optional[str] = Field(None, description="店铺名称")
    cEBItemCode: str = Field(..., description="店铺商品编码（必填）")
    cEBItemName: Optional[str] = Field(None, description="店铺商品名称")
    cEBSKUID: Optional[str] = Field(None, description="店铺商品SKU")
    cEBItemMemo: Optional[str] = Field(None, description="店铺商品描述")
    cOutIID: Optional[str] = Field(None, description="商家编码")
    cOutSKUID: Optional[str] = Field(None, description="商家SKU")


# ===================== 电商订单 entry 子数据模型 =====================
class EbTradeEntry(BaseModel):
    """电商订单 entry 子表数据模型"""
    title: Optional[str] = Field(None, description="商品标题")
    num_iid: Optional[str] = Field(None, description="商品数字ID")
    outer_iid: Optional[str] = Field(None, description="商家外部编码")
    sku_id: Optional[str] = Field(None, description="商品Sku的id")
    outer_sku_id: Optional[str] = Field(None, description="外部网店自己定义的Sku编号")
    sku_properties_name: Optional[str] = Field(None, description="SKU值")
    cItemCode: Optional[str] = Field(None, description="商品编码")
    cItemName: Optional[str] = Field(None, description="商品名称")
    price: Optional[float] = Field(None, description="商品价格")
    num: Optional[float] = Field(None, description="购买数量")
    discount_fee: Optional[float] = Field(None, description="订单优惠金额")
    adjust_fee: Optional[float] = Field(None, description="手工调整金额")
    post_fee: Optional[float] = Field(None, description="运费")
    cWhCode: Optional[str] = Field(None, description="发货仓库")
    isDiscount: Optional[int] = Field(None, description="是否折扣(0=非折扣;1=折扣)")
    isPostFee: Optional[int] = Field(None, description="是否运费(0=非运费;1=运费)")
    isGift: Optional[int] = Field(None, description="是否赠品(0=非赠品;1=赠品)")
    cDefine22: Optional[str] = Field(None, description="表体自定义项22")
    cDefine23: Optional[str] = Field(None, description="表体自定义项23")
    cDefine24: Optional[str] = Field(None, description="表体自定义项24")
    cDefine25: Optional[str] = Field(None, description="表体自定义项25")
    cDefine26: Optional[float] = Field(None, description="表体自定义项26")
    cDefine27: Optional[float] = Field(None, description="表体自定义项27")
    cDefine28: Optional[str] = Field(None, description="表体自定义项28")
    cDefine29: Optional[str] = Field(None, description="表体自定义项29")
    cDefine30: Optional[str] = Field(None, description="表体自定义项30")
    cDefine31: Optional[str] = Field(None, description="表体自定义项31")
    cDefine32: Optional[str] = Field(None, description="表体自定义项32")
    cDefine33: Optional[str] = Field(None, description="表体自定义项33")
    cDefine34: Optional[int] = Field(None, description="表体自定义项34")
    cDefine35: Optional[int] = Field(None, description="表体自定义项35")
    cDefine36: Optional[str] = Field(None, description="表体自定义项36")
    cDefine37: Optional[str] = Field(None, description="表体自定义项37")


# ===================== 批量获取电商订单(v2) 数据模型 =====================
class GetEbTradelistV2BatchInput(BaseModel):
    """批量获取电商订单(v2)输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cshopcode: Optional[str] = Field(None, description="店铺编码")
    tid: Optional[str] = Field(None, description="交易编号")
    title: Optional[str] = Field(None, description="交易标题关键字")
    buyer_nick: Optional[str] = Field(None, description="买家会员号")
    receiver_name: Optional[str] = Field(None, description="收货人的姓名关键字")
    trade_memo: Optional[str] = Field(None, description="交易备注关键字")
    cexpresscode: Optional[str] = Field(None, description="快递单号")
    cshipcode: Optional[str] = Field(None, description="发货单编号")


# ===================== 获取电商订单(v2) 数据模型 =====================
class GetEbTradeV2Input(BaseModel):
    """获取电商订单(v2)输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="店铺编码_商品编码（必填）")


# ===================== 批量获取电商订单(v2) 数据模型 =====================
class GetEbTradeV2BatchInput(BaseModel):
    """批量获取电商订单(v2)输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cshopcode: Optional[str] = Field(None, description="店铺编码")
    tid: Optional[str] = Field(None, description="交易编号")
    title: Optional[str] = Field(None, description="交易标题关键字")
    buyer_nick: Optional[str] = Field(None, description="买家会员号")
    receiver_name: Optional[str] = Field(None, description="收货人的姓名关键字")
    trade_memo: Optional[str] = Field(None, description="交易备注关键字")
    cexpresscode: Optional[str] = Field(None, description="快递单号")
    cshipcode: Optional[str] = Field(None, description="发货单编号")


# ===================== 批量获取电商订单 数据模型 =====================
class GetEbTradelistBatchInput(BaseModel):
    """批量获取电商订单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cshopcode: Optional[str] = Field(None, description="店铺编码")
    tid_begin: Optional[str] = Field(None, description="起始交易编号")
    tid_end: Optional[str] = Field(None, description="结束交易编号")
    title: Optional[str] = Field(None, description="交易标题")
    buyer_nick: Optional[str] = Field(None, description="买家会员号")
    receiver_name: Optional[str] = Field(None, description="收货人的姓名关键字")
    receiver_state: Optional[str] = Field(None, description="收货人的所在省份")
    receiver_city: Optional[str] = Field(None, description="收货人的所在城市")
    receiver_district: Optional[str] = Field(None, description="收货人的所在地区")
    receiver_address: Optional[str] = Field(None, description="收货人的详细地址")
    receiver_mobile: Optional[str] = Field(None, description="收货人的手机号码")
    isinvoice: Optional[bool] = Field(None, description="是否开票")
    receiver_phone: Optional[str] = Field(None, description="收货人的电话号码")
    buyer_email: Optional[str] = Field(None, description="买家邮件地址")
    seller_memo: Optional[str] = Field(None, description="卖家备注")
    created_begin: Optional[str] = Field(None, description="起始交易时间")
    created_end: Optional[str] = Field(None, description="结束交易时间")
    pay_time_begin: Optional[str] = Field(None, description="起始付款时间")
    pay_time_end: Optional[str] = Field(None, description="结束付款时间")
    buyer_message: Optional[str] = Field(None, description="买家留言")
    trade_memo: Optional[str] = Field(None, description="交易备注")
    cexpresscode: Optional[str] = Field(None, description="快递单号")
    cdepcode: Optional[str] = Field(None, description="部门编码")
    cdepname: Optional[str] = Field(None, description="部门")
    cpersonname: Optional[str] = Field(None, description="业务员关键字")
    ispickself: Optional[int] = Field(None, description="是否自提")
    sysstatus: Optional[str] = Field(None, description="单据状态")
    ishold: Optional[bool] = Field(None, description="是否挂起")
    isclosed: Optional[bool] = Field(None, description="是否关闭")


# ===================== 获取电商订单 数据模型 =====================
class GetEbTradeInput(BaseModel):
    """获取电商订单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="店铺编码_交易号（必填）")


# ===================== 新增电商订单 数据模型 =====================
class AddEbTradeInput(BaseModel):
    """新增电商订单输入模型"""
    cShopCode: str = Field(..., description="店铺编码（必填）")
    tid: str = Field(..., description="交易编号（必填）")
    title: str = Field(..., description="交易标题（必填）")
    buyer_nick: str = Field(..., description="买家会员号（必填）")
    receiver_name: Optional[str] = Field(None, description="收货人的姓名")
    receiver_state: Optional[str] = Field(None, description="收货人的所在省份")
    receiver_city: Optional[str] = Field(None, description="收货人的所在城市")
    receiver_district: Optional[str] = Field(None, description="收货人的所在地区")
    receiver_address: Optional[str] = Field(None, description="收货人的详细地址")
    receiver_zip: Optional[str] = Field(None, description="收货的人邮编")
    receiver_mobile: Optional[str] = Field(None, description="收货人的手机号码")
    isInvoice: Optional[int] = Field(None, description="是否开票(0=不开票;1=开票)")
    receiver_phone: Optional[str] = Field(None, description="收货人的电话号码")
    buyer_email: Optional[str] = Field(None, description="买家邮件地址")
    seller_memo: Optional[str] = Field(None, description="卖家备注")
    created: str = Field(..., description="交易时间（必填）")
    pay_time: Optional[str] = Field(None, description="付款时间")
    promotion: Optional[str] = Field(None, description="交易促销信息")
    buyer_message: Optional[str] = Field(None, description="买家留言")
    trade_memo: Optional[str] = Field(None, description="交易备注")
    invoice_name: Optional[str] = Field(None, description="发票抬头")
    cInvoiceCode: Optional[str] = Field(None, description="发票号")
    cExpressCoCode: Optional[str] = Field(None, description="快递公司编码")
    cExpressCoName: Optional[str] = Field(None, description="快递公司名称")
    cExpressCode: Optional[str] = Field(None, description="快递单号")
    cDepCode: Optional[str] = Field(None, description="部门编码")
    cDepName: Optional[str] = Field(None, description="部门")
    cPersonName: Optional[str] = Field(None, description="业务员")
    cSTName: Optional[str] = Field(None, description="销售类型")
    cSSName: Optional[str] = Field(None, description="结算方式")
    cShipMode: Optional[str] = Field(None, description="发货模式")
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
    entry: Optional[List[EbTradeEntry]] = Field(None, description="订单明细列表")


# ===================== 批量获取店铺商品档案 Tool函数 =====================
def u8_eb_iteminvcontrapose_batch_get_tool(input_data: GetEbIteminvcontraposeBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取店铺商品档案信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eb_iteminvcontrapose/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "店铺商品档案获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "店铺商品档案获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "eb_iteminvcontrapose": result.get("eb_iteminvcontrapose", {})
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


# ===================== 新增店铺商品档案 Tool函数 =====================
def u8_eb_iteminvcontrapose_add_tool(input_data: AddEbIteminvcontraposeInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增店铺商品档案信息。
    """
    request_body: Dict[str, Any] = {
        "eb_iteminvcontrapose": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/eb_iteminvcontrapose/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "店铺商品档案新增失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "店铺商品档案新增成功",
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


# ===================== 批量获取电商订单(v2) Tool函数 =====================
def u8_eb_tradelist_v2_batch_get_tool(input_data: GetEbTradelistV2BatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取电商订单(v2)列表信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eb_tradelist_v2/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "电商订单(v2)列表获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "电商订单(v2)列表获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "eb_tradelist_v2": result.get("eb_tradelist_v2", {})
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


# ===================== 获取电商订单(v2) Tool函数 =====================
def u8_eb_trade_v2_get_tool(input_data: GetEbTradeV2Input, client: U8OpenAPIClient) -> str:
    """
    通过店铺编码_商品编码获取用友U8中的电商订单(v2)信息，包含表头及表体明细。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eb_trade_v2/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "电商订单(v2)信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "电商订单(v2)信息获取成功",
            "data": {
                "eb_trade_v2": result.get("eb_trade_v2", {}),
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


# ===================== 批量获取电商订单(v2) Tool函数 =====================
def u8_eb_trade_v2_batch_get_tool(input_data: GetEbTradeV2BatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取电商订单(v2)信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eb_trade_v2/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "电商订单(v2)批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "电商订单(v2)批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "eb_trade_v2": result.get("eb_trade_v2", {})
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


# ===================== 批量获取电商订单 Tool函数 =====================
def u8_eb_tradelist_batch_get_tool(input_data: GetEbTradelistBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取电商订单信息（旧版），支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eb_tradelist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "电商订单列表获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "电商订单列表获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "eb_tradelist": result.get("eb_tradelist", {})
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


# ===================== 获取电商订单 Tool函数 =====================
def u8_eb_trade_get_tool(input_data: GetEbTradeInput, client: U8OpenAPIClient) -> str:
    """
    通过店铺编码_交易号获取用友U8中的电商订单信息（旧版），包含表头及表体明细。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/eb_trade/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "电商订单信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "电商订单信息获取成功",
            "data": {
                "eb_trade": result.get("eb_trade", {}),
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


# ===================== 新增电商订单 Tool函数 =====================
def u8_eb_trade_add_tool(input_data: AddEbTradeInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增电商订单信息。
    """
    request_body: Dict[str, Any] = {
        "eb_trade": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/eb_trade/add"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "电商订单新增失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "电商订单新增成功",
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


# ===================== 批量获取店铺商品档案 Schema定义 =====================
U8_EB_ITEMINVCONTRAPOSE_BATCH_GET_SCHEMA = {
    "name": "u8_eb_iteminvcontrapose_batch_get",
    "description": "在用友U8 OpenAPI中批量获取店铺商品档案信息，返回店铺编码、店铺名称、店铺商品编码、名称、SKU、商家编码等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cShopCode": {"type": "string", "description": "店铺编码"},
            "cShopName": {"type": "string", "description": "店铺名称关键字"},
            "cEBItemCode_begin": {"type": "string", "description": "起始店铺商品编码"},
            "cEBItemCode_end": {"type": "string", "description": "结束店铺商品编码"},
            "cEBItemName": {"type": "string", "description": "店铺商品名称关键字"},
            "cEBSKUID_begin": {"type": "string", "description": "起始店铺商品SKU"},
            "cEBSKUID_end": {"type": "string", "description": "结束店铺商品SKU"},
            "cEBItemMemo": {"type": "string", "description": "店铺商品描述"},
            "cOutIID": {"type": "string", "description": "商家编码"},
            "cOutSKUID": {"type": "string", "description": "商家SKU"}
        },
        "required": []
    }
}


# ===================== 新增店铺商品档案 Schema定义 =====================
U8_EB_ITEMINVCONTRAPOSE_ADD_SCHEMA = {
    "name": "u8_eb_iteminvcontrapose_add",
    "description": "在用友U8 OpenAPI中新增店铺商品档案，支持录入店铺编码、店铺商品编码、名称、SKU、商家编码等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "cShopCode": {"type": "string", "description": "店铺编码（必填）"},
            "cShopName": {"type": "string", "description": "店铺名称"},
            "cEBItemCode": {"type": "string", "description": "店铺商品编码（必填）"},
            "cEBItemName": {"type": "string", "description": "店铺商品名称"},
            "cEBSKUID": {"type": "string", "description": "店铺商品SKU"},
            "cEBItemMemo": {"type": "string", "description": "店铺商品描述"},
            "cOutIID": {"type": "string", "description": "商家编码"},
            "cOutSKUID": {"type": "string", "description": "商家SKU"}
        },
        "required": ["cShopCode", "cEBItemCode"]
    }
}


# ===================== 批量获取电商订单(v2) Schema定义 =====================
U8_EB_TRADELIST_V2_BATCH_GET_SCHEMA = {
    "name": "u8_eb_tradelist_v2_batch_get",
    "description": "在用友U8 OpenAPI中批量获取电商订单(v2)列表信息，支持按店铺编码、交易编号、买家会员号、收货人、快递单号等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cshopcode": {"type": "string", "description": "店铺编码"},
            "tid": {"type": "string", "description": "交易编号"},
            "title": {"type": "string", "description": "交易标题关键字"},
            "buyer_nick": {"type": "string", "description": "买家会员号"},
            "receiver_name": {"type": "string", "description": "收货人的姓名关键字"},
            "trade_memo": {"type": "string", "description": "交易备注关键字"},
            "cexpresscode": {"type": "string", "description": "快递单号"},
            "cshipcode": {"type": "string", "description": "发货单编号"}
        },
        "required": []
    }
}


# ===================== 获取电商订单(v2) Schema定义 =====================
U8_EB_TRADE_V2_GET_SCHEMA = {
    "name": "u8_eb_trade_v2_get",
    "description": "在用友U8 OpenAPI中通过店铺编码_商品编码获取电商订单(v2)信息，包含表头及表体明细(entry)",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "店铺编码_商品编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取电商订单(v2) Schema定义 =====================
U8_EB_TRADE_V2_BATCH_GET_SCHEMA = {
    "name": "u8_eb_trade_v2_batch_get",
    "description": "在用友U8 OpenAPI中批量获取电商订单(v2)信息，支持按店铺编码、交易编号、买家会员号、收货人、快递单号等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cshopcode": {"type": "string", "description": "店铺编码"},
            "tid": {"type": "string", "description": "交易编号"},
            "title": {"type": "string", "description": "交易标题关键字"},
            "buyer_nick": {"type": "string", "description": "买家会员号"},
            "receiver_name": {"type": "string", "description": "收货人的姓名关键字"},
            "trade_memo": {"type": "string", "description": "交易备注关键字"},
            "cexpresscode": {"type": "string", "description": "快递单号"},
            "cshipcode": {"type": "string", "description": "发货单编号"}
        },
        "required": []
    }
}


# ===================== 批量获取电商订单 Schema定义 =====================
U8_EB_TRADELIST_BATCH_GET_SCHEMA = {
    "name": "u8_eb_tradelist_batch_get",
    "description": "在用友U8 OpenAPI中批量获取电商订单信息（旧版），支持按店铺、交易编号、收货人、交易时间、付款时间等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cshopcode": {"type": "string", "description": "店铺编码"},
            "tid_begin": {"type": "string", "description": "起始交易编号"},
            "tid_end": {"type": "string", "description": "结束交易编号"},
            "title": {"type": "string", "description": "交易标题"},
            "buyer_nick": {"type": "string", "description": "买家会员号"},
            "receiver_name": {"type": "string", "description": "收货人的姓名关键字"},
            "receiver_state": {"type": "string", "description": "收货人的所在省份"},
            "receiver_city": {"type": "string", "description": "收货人的所在城市"},
            "receiver_district": {"type": "string", "description": "收货人的所在地区"},
            "receiver_address": {"type": "string", "description": "收货人的详细地址"},
            "receiver_mobile": {"type": "string", "description": "收货人的手机号码"},
            "isinvoice": {"type": "boolean", "description": "是否开票"},
            "receiver_phone": {"type": "string", "description": "收货人的电话号码"},
            "buyer_email": {"type": "string", "description": "买家邮件地址"},
            "seller_memo": {"type": "string", "description": "卖家备注"},
            "created_begin": {"type": "string", "description": "起始交易时间"},
            "created_end": {"type": "string", "description": "结束交易时间"},
            "pay_time_begin": {"type": "string", "description": "起始付款时间"},
            "pay_time_end": {"type": "string", "description": "结束付款时间"},
            "buyer_message": {"type": "string", "description": "买家留言"},
            "trade_memo": {"type": "string", "description": "交易备注"},
            "cexpresscode": {"type": "string", "description": "快递单号"},
            "cdepcode": {"type": "string", "description": "部门编码"},
            "cdepname": {"type": "string", "description": "部门"},
            "cpersonname": {"type": "string", "description": "业务员关键字"},
            "ispickself": {"type": "integer", "description": "是否自提"},
            "sysstatus": {"type": "string", "description": "单据状态"},
            "ishold": {"type": "boolean", "description": "是否挂起"},
            "isclosed": {"type": "boolean", "description": "是否关闭"}
        },
        "required": []
    }
}


# ===================== 获取电商订单 Schema定义 =====================
U8_EB_TRADE_GET_SCHEMA = {
    "name": "u8_eb_trade_get",
    "description": "在用友U8 OpenAPI中通过店铺编码_交易号获取电商订单信息（旧版），包含表头及表体明细(entry)",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "店铺编码_交易号（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 新增电商订单 Schema定义 =====================
U8_EB_TRADE_ADD_SCHEMA = {
    "name": "u8_eb_trade_add",
    "description": "在用友U8 OpenAPI中新增电商订单，支持录入订单表头信息（店铺、交易编号、收货人、快递公司等）及表体明细（商品编码、价格、数量等）",
    "parameters": {
        "type": "object",
        "properties": {
            "cShopCode": {"type": "string", "description": "店铺编码（必填）"},
            "tid": {"type": "string", "description": "交易编号（必填）"},
            "title": {"type": "string", "description": "交易标题（必填）"},
            "buyer_nick": {"type": "string", "description": "买家会员号（必填）"},
            "receiver_name": {"type": "string", "description": "收货人的姓名"},
            "receiver_state": {"type": "string", "description": "收货人的所在省份"},
            "receiver_city": {"type": "string", "description": "收货人的所在城市"},
            "receiver_district": {"type": "string", "description": "收货人的所在地区"},
            "receiver_address": {"type": "string", "description": "收货人的详细地址"},
            "receiver_zip": {"type": "string", "description": "收货的人邮编"},
            "receiver_mobile": {"type": "string", "description": "收货人的手机号码"},
            "isInvoice": {"type": "integer", "description": "是否开票(0=不开票;1=开票)"},
            "receiver_phone": {"type": "string", "description": "收货人的电话号码"},
            "buyer_email": {"type": "string", "description": "买家邮件地址"},
            "seller_memo": {"type": "string", "description": "卖家备注"},
            "created": {"type": "string", "description": "交易时间（必填）"},
            "pay_time": {"type": "string", "description": "付款时间"},
            "promotion": {"type": "string", "description": "交易促销信息"},
            "buyer_message": {"type": "string", "description": "买家留言"},
            "trade_memo": {"type": "string", "description": "交易备注"},
            "invoice_name": {"type": "string", "description": "发票抬头"},
            "cInvoiceCode": {"type": "string", "description": "发票号"},
            "cExpressCoCode": {"type": "string", "description": "快递公司编码"},
            "cExpressCoName": {"type": "string", "description": "快递公司名称"},
            "cExpressCode": {"type": "string", "description": "快递单号"},
            "cDepCode": {"type": "string", "description": "部门编码"},
            "cDepName": {"type": "string", "description": "部门"},
            "cPersonName": {"type": "string", "description": "业务员"},
            "cSTName": {"type": "string", "description": "销售类型"},
            "cSSName": {"type": "string", "description": "结算方式"},
            "cShipMode": {"type": "string", "description": "发货模式"},
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
            "entry": {
                "type": "array",
                "description": "订单明细列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "商品标题"},
                        "num_iid": {"type": "string", "description": "商品数字ID"},
                        "outer_iid": {"type": "string", "description": "商家外部编码"},
                        "sku_id": {"type": "string", "description": "商品Sku的id"},
                        "outer_sku_id": {"type": "string", "description": "外部网店自己定义的Sku编号"},
                        "sku_properties_name": {"type": "string", "description": "SKU值"},
                        "cItemCode": {"type": "string", "description": "商品编码"},
                        "cItemName": {"type": "string", "description": "商品名称"},
                        "price": {"type": "number", "description": "商品价格"},
                        "num": {"type": "number", "description": "购买数量"},
                        "discount_fee": {"type": "number", "description": "订单优惠金额"},
                        "adjust_fee": {"type": "number", "description": "手工调整金额"},
                        "post_fee": {"type": "number", "description": "运费"},
                        "cWhCode": {"type": "string", "description": "发货仓库"},
                        "isDiscount": {"type": "integer", "description": "是否折扣(0=非折扣;1=折扣)"},
                        "isPostFee": {"type": "integer", "description": "是否运费(0=非运费;1=运费)"},
                        "isGift": {"type": "integer", "description": "是否赠品(0=非赠品;1=赠品)"},
                        "cDefine22": {"type": "string", "description": "表体自定义项22"},
                        "cDefine23": {"type": "string", "description": "表体自定义项23"},
                        "cDefine24": {"type": "string", "description": "表体自定义项24"},
                        "cDefine25": {"type": "string", "description": "表体自定义项25"},
                        "cDefine26": {"type": "number", "description": "表体自定义项26"},
                        "cDefine27": {"type": "number", "description": "表体自定义项27"},
                        "cDefine28": {"type": "string", "description": "表体自定义项28"},
                        "cDefine29": {"type": "string", "description": "表体自定义项29"},
                        "cDefine30": {"type": "string", "description": "表体自定义项30"},
                        "cDefine31": {"type": "string", "description": "表体自定义项31"},
                        "cDefine32": {"type": "string", "description": "表体自定义项32"},
                        "cDefine33": {"type": "string", "description": "表体自定义项33"},
                        "cDefine34": {"type": "integer", "description": "表体自定义项34"},
                        "cDefine35": {"type": "integer", "description": "表体自定义项35"},
                        "cDefine36": {"type": "string", "description": "表体自定义项36"},
                        "cDefine37": {"type": "string", "description": "表体自定义项37"}
                    }
                }
            }
        },
        "required": ["cShopCode", "tid", "title", "buyer_nick", "created"]
    }
}
