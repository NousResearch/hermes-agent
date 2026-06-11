import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== U易联积分增加 数据模型 =====================
class PointsChangeInput(BaseModel):
    """U易联积分增加输入模型"""
    mid: str = Field(..., description="会员统一id（必填）")
    action_type: str = Field(..., description="积分计算业务类型（5：商城送积分 51：商城抵现 16：退货扣积分 17：退货返还抵扣积分 200：积分结转）（必填）")
    ori_money: str = Field(..., description="业务发生金额（必填）")
    calc_money: str = Field(..., description="积分计算金额、退货积分返回金额（必填）")
    ori_points: str = Field(..., description="抵扣积分（正数）、退货扣除积分（正数）、结转积分（必填）")
    cur_points: Optional[str] = Field(None, description="会员当前积分")
    multiple: Optional[str] = Field(None, description="积分核算倍数")
    extra_points: Optional[str] = Field(None, description="额外赠送积分")
    extra_memo: Optional[str] = Field(None, description="额外赠送积分备注")
    source_code: str = Field(..., description="来源依据（订单编号）（必填）")
    source1: Optional[str] = Field(None, description="来源依据1、退货时原订单编号")
    source2: Optional[str] = Field(None, description="来源依据2")


# ===================== U易联积分明细查询 conditions 子数据模型 =====================
class PointsQueryCondition(BaseModel):
    """U易联积分明细查询条件子模型"""
    name: Optional[str] = Field(None, description="条件属性")
    value1: Optional[str] = Field(None, description="条件属性值")
    type: Optional[str] = Field(None, description="条件属性类型(string/number/date/datetime/datetime2/refer/tree)")
    op: Optional[str] = Field(None, description="条件匹配方式(eq/neq/gt/egt/lt/elt/between/in/not in/like)")


# ===================== U易联积分明细查询 pager 子数据模型 =====================
class PointsQueryPager(BaseModel):
    """U易联积分明细查询分页子模型"""
    pageIndex: Optional[int] = Field(None, description="当前页")
    pageSize: Optional[int] = Field(None, description="单页显示个数")


# ===================== U易联积分明细查询 order 子数据模型 =====================
class PointsQueryOrder(BaseModel):
    """U易联积分明细查询排序子模型"""
    name: Optional[str] = Field(None, description="需要排序字段")
    entity: Optional[str] = Field(None, description="对应平台字段")
    order: Optional[str] = Field(None, description="排序方式(desc/asc)")


# ===================== U易联积分明细查询 field 子数据模型 =====================
class PointsQueryField(BaseModel):
    """U易联积分明细查询返回字段子模型"""
    name: Optional[str] = Field(None, description="返回显示字段")
    entity: Optional[str] = Field(None, description="对应平台字段")
    format: Optional[str] = Field(None, description="返回显示格式(json/xml)")


# ===================== U易联积分明细查询 数据模型 =====================
class PointsQueryInput(BaseModel):
    """U易联积分明细查询输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    conditions: Optional[List[PointsQueryCondition]] = Field(None, description="查询条件列表")
    pager: Optional[PointsQueryPager] = Field(None, description="分页信息")
    orders: Optional[List[PointsQueryOrder]] = Field(None, description="排序信息列表")
    fields: Optional[List[PointsQueryField]] = Field(None, description="返回字段列表")


# ===================== U易联订单交易记录查询 数据模型 =====================
class OrdersQueryInput(BaseModel):
    """U易联订单交易记录查询输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    conditions: Optional[List[PointsQueryCondition]] = Field(None, description="查询条件列表")
    pager: Optional[PointsQueryPager] = Field(None, description="分页信息")
    orders: Optional[List[PointsQueryOrder]] = Field(None, description="排序信息列表")
    fields: Optional[List[PointsQueryField]] = Field(None, description="返回字段列表")


# ===================== U商城商品列表 where 子数据模型 =====================
class ProductlistWhere(BaseModel):
    """U商城商品列表查询条件子模型"""
    cgoodsname: Optional[str] = Field(None, description="商品名称条件")
    valuefrom: Optional[str] = Field(None, description="条件属性值")


# ===================== U商城商品列表 pager 子数据模型 =====================
class ProductlistPager(BaseModel):
    """U商城商品列表分页子模型"""
    pageIndex: Optional[int] = Field(None, description="当前页")
    pageSize: Optional[int] = Field(None, description="单页显示个数")
    totalPage: Optional[int] = Field(None, description="总共页数")
    totalCount: Optional[int] = Field(None, description="总共数据数")


# ===================== U商城商品列表 数据模型 =====================
class ProductlistQueryInput(BaseModel):
    """U商城商品列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    where: Optional[ProductlistWhere] = Field(None, description="查询条件")
    pager: Optional[ProductlistPager] = Field(None, description="分页信息")


# ===================== U订货批量获取订单信息 数据模型 =====================
class UdhOrderlistBatchGetInput(BaseModel):
    """U订货批量获取订单信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    ordernos: Optional[str] = Field(None, description="U易联订单编号")


# ===================== U易联积分增加 Tool函数 =====================
def u8_points_change_tool(input_data: PointsChangeInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8 U易联系统中增加会员积分。
    """
    request_body: Dict[str, Any] = {
        "points": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/points/change"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "积分增加失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "积分增加成功",
            "data": {
                "id": result.get("id"),
                "mid": result.get("mid"),
                "action_type": result.get("action_type"),
                "before_points": result.get("before_points"),
                "calc_points": result.get("calc_points"),
                "after_points": result.get("after_points")
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


# ===================== U易联积分明细查询 Tool函数 =====================
def u8_points_query_tool(input_data: PointsQueryInput, client: U8OpenAPIClient) -> str:
    """
    查询用友U8 U易联系统中的会员积分明细记录。
    """
    request_body: Dict[str, Any] = input_data.model_dump(exclude_none=True)

    api_path = "/api/points/query"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "积分明细查询失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "积分明细查询成功",
            "data": {
                "data": result.get("data", []),
                "pager": result.get("pager", {})
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


# ===================== U易联订单交易记录查询 Tool函数 =====================
def u8_orders_query_tool(input_data: OrdersQueryInput, client: U8OpenAPIClient) -> str:
    """
    查询用友U8 U易联系统中的订单交易记录。
    """
    request_body: Dict[str, Any] = input_data.model_dump(exclude_none=True)

    api_path = "/api/orders/query"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "订单交易记录查询失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "订单交易记录查询成功",
            "data": {
                "data": result.get("data", []),
                "pager": result.get("pager", {})
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


# ===================== U商城商品列表 Tool函数 =====================
def u8_productlist_query_tool(input_data: ProductlistQueryInput, client: U8OpenAPIClient) -> str:
    """
    查询用友U8 U商城系统中的商品列表信息。
    """
    request_body: Dict[str, Any] = input_data.model_dump(exclude_none=True)

    api_path = "/umall/productlist/query"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "商品列表查询失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "商品列表查询成功",
            "data": {
                "data": result.get("data", []),
                "currentPage": result.get("currentPage"),
                "pageNum": result.get("pageNum"),
                "totalPage": result.get("totalPage"),
                "totalCount": result.get("totalCount")
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


# ===================== U订货批量获取订单信息 Tool函数 =====================
def u8_udh_orderlist_batch_get_tool(input_data: UdhOrderlistBatchGetInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8 U订货系统中批量获取订单信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/udh/orderlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "U订货订单批量获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "U订货订单批量获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "orderlist": result.get("orderlist", {})
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


# ===================== U易联积分增加 Schema定义 =====================
U8_POINTS_CHANGE_SCHEMA = {
    "name": "u8_points_change",
    "description": "在用友U8 U易联系统中增加会员积分，支持商城送积分、商城抵现、退货扣积分、退货返还抵扣积分、积分结转等业务类型",
    "parameters": {
        "type": "object",
        "properties": {
            "mid": {"type": "string", "description": "会员统一id（必填）"},
            "action_type": {"type": "string", "description": "积分计算业务类型（5：商城送积分 51：商城抵现 16：退货扣积分 17：退货返还抵扣积分 200：积分结转）（必填）"},
            "ori_money": {"type": "string", "description": "业务发生金额（必填）"},
            "calc_money": {"type": "string", "description": "积分计算金额、退货积分返回金额（必填）"},
            "ori_points": {"type": "string", "description": "抵扣积分（正数）、退货扣除积分（正数）、结转积分（必填）"},
            "cur_points": {"type": "string", "description": "会员当前积分"},
            "multiple": {"type": "string", "description": "积分核算倍数"},
            "extra_points": {"type": "string", "description": "额外赠送积分"},
            "extra_memo": {"type": "string", "description": "额外赠送积分备注"},
            "source_code": {"type": "string", "description": "来源依据（订单编号）（必填）"},
            "source1": {"type": "string", "description": "来源依据1、退货时原订单编号"},
            "source2": {"type": "string", "description": "来源依据2"}
        },
        "required": ["mid", "action_type", "ori_money", "calc_money", "ori_points", "source_code"]
    }
}


# ===================== U易联积分明细查询 Schema定义 =====================
U8_POINTS_QUERY_SCHEMA = {
    "name": "u8_points_query",
    "description": "在用友U8 U易联系统中查询会员积分明细记录，支持条件筛选、分页查询和排序",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "conditions": {
                "type": "array",
                "description": "查询条件列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "条件属性"},
                        "value1": {"type": "string", "description": "条件属性值"},
                        "type": {"type": "string", "description": "条件属性类型(string/number/date/datetime/datetime2/refer/tree)"},
                        "op": {"type": "string", "description": "条件匹配方式(eq/neq/gt/egt/lt/elt/between/in/not in/like)"}
                    }
                }
            },
            "pager": {
                "type": "object",
                "description": "分页信息",
                "properties": {
                    "pageIndex": {"type": "integer", "description": "当前页"},
                    "pageSize": {"type": "integer", "description": "单页显示个数"}
                }
            },
            "orders": {
                "type": "array",
                "description": "排序信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "需要排序字段"},
                        "entity": {"type": "string", "description": "对应平台字段"},
                        "order": {"type": "string", "description": "排序方式(desc/asc)"}
                    }
                }
            },
            "fields": {
                "type": "array",
                "description": "返回字段列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "返回显示字段"},
                        "entity": {"type": "string", "description": "对应平台字段"},
                        "format": {"type": "string", "description": "返回显示格式(json/xml)"}
                    }
                }
            }
        },
        "required": []
    }
}


# ===================== U易联订单交易记录查询 Schema定义 =====================
U8_ORDERS_QUERY_SCHEMA = {
    "name": "u8_orders_query",
    "description": "在用友U8 U易联系统中查询订单交易记录，支持条件筛选、分页查询和排序",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "conditions": {
                "type": "array",
                "description": "查询条件列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "条件属性"},
                        "value1": {"type": "string", "description": "条件属性值"},
                        "type": {"type": "string", "description": "条件属性类型"},
                        "op": {"type": "string", "description": "条件匹配方式"}
                    }
                }
            },
            "pager": {
                "type": "object",
                "description": "分页信息",
                "properties": {
                    "pageIndex": {"type": "integer", "description": "当前页"},
                    "pageSize": {"type": "integer", "description": "单页显示个数"}
                }
            },
            "orders": {
                "type": "array",
                "description": "排序信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "需要排序字段"},
                        "entity": {"type": "string", "description": "对应平台字段"},
                        "order": {"type": "string", "description": "排序方式(desc/asc)"}
                    }
                }
            },
            "fields": {
                "type": "array",
                "description": "返回字段列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "返回显示字段"},
                        "entity": {"type": "string", "description": "对应平台字段"}
                    }
                }
            }
        },
        "required": []
    }
}


# ===================== U商城商品列表 Schema定义 =====================
U8_PRODUCTLIST_QUERY_SCHEMA = {
    "name": "u8_productlist_query",
    "description": "在用友U8 U商城系统中查询商品列表信息，支持按商品名称条件筛选和分页查询",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "where": {
                "type": "object",
                "description": "查询条件",
                "properties": {
                    "cgoodsname": {"type": "string", "description": "商品名称条件"},
                    "valuefrom": {"type": "string", "description": "条件属性值"}
                }
            },
            "pager": {
                "type": "object",
                "description": "分页信息",
                "properties": {
                    "pageIndex": {"type": "integer", "description": "当前页"},
                    "pageSize": {"type": "integer", "description": "单页显示个数"},
                    "totalPage": {"type": "integer", "description": "总共页数"},
                    "totalCount": {"type": "integer", "description": "总共数据数"}
                }
            }
        },
        "required": []
    }
}


# ===================== U订货批量获取订单信息 Schema定义 =====================
U8_UDH_ORDERLIST_BATCH_GET_SCHEMA = {
    "name": "u8_udh_orderlist_batch_get",
    "description": "在用友U8 U订货系统中批量获取订单信息，支持按订单编号筛选和分页查询",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "ordernos": {"type": "string", "description": "U易联订单编号"}
        },
        "required": []
    }
}
