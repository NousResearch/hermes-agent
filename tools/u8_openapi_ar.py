import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 收款单通用数据模型 (复用于新增和查询) =====================

class AcceptEntry(BaseModel):
    """
    收款单体模型。
    注意：所有字段均为 Optional，以兼容查询返回（可能缺省）和新增传入（后端校验必填）。
    """
    type: Optional[int] = Field(None, description="款项类型(0-应收款;1-预收款;2-其他费用)")
    customercode: Optional[str] = Field(None, description="客商编码")
    customerabbname: Optional[str] = Field(None, description="客商简称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    digest: Optional[str] = Field(None, description="摘要")
    project: Optional[str] = Field(None, description="项目编号")
    projectclass: Optional[str] = Field(None, description="项目大类编号")
    itemcode: Optional[str] = Field(None, description="科目")
    itemname: Optional[str] = Field(None, description="项目名称")
    amount: Optional[float] = Field(None, description="本币金额")
    originalamount: Optional[float] = Field(None, description="原币金额")
    iamt_s: Optional[float] = Field(None, description="数量")
    iramt_s: Optional[float] = Field(None, description="剩余数量")
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

class AcceptInfo(BaseModel):
    """
    收款单主表模型 (通用，用于新增输入和查询返回)
    """
    vouchcode: Optional[str] = Field(None, description="应收单号")
    vouchdate: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    vouchtype: Optional[str] = Field(None, description="单据类型(48=收款单;49=付款单)")
    customercode: Optional[str] = Field(None, description="客商编码")
    customername: Optional[str] = Field(None, description="客商名称")
    customerabbname: Optional[str] = Field(None, description="客商简称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    amount: Optional[float] = Field(None, description="本币金额")
    originalamount: Optional[float] = Field(None, description="原币金额")
    digest: Optional[str] = Field(None, description="摘要")
    balancecode: Optional[str] = Field(None, description="结算方式编码")
    balancename: Optional[str] = Field(None, description="结算方式")
    balanceitemcode: Optional[str] = Field(None, description="结算科目编码")
    itemclasscode: Optional[str] = Field(None, description="项目大类编号")
    itemcode: Optional[str] = Field(None, description="项目编码")
    itemname: Optional[str] = Field(None, description="项目名称")
    oppositebankname: Optional[str] = Field(None, description="对方单位银行名称")
    oppositebankcode: Optional[str] = Field(None, description="对方单位银行帐号")
    bankname: Optional[str] = Field(None, description="本单位银行名称")
    bankaccount: Optional[str] = Field(None, description="本单位银行帐号")
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
    entry: Optional[List[AcceptEntry]] = Field(None, description="收款单体列表")

# ===================== 新增一张收款单 数据模型 =====================
class AddAcceptInfoInput(AcceptInfo):
    """新增收款单输入模型，继承自通用模型，可在此处强化必填项校验"""
    vouchtype: str = Field(..., description="单据类型(48=收款单;49=付款单)（必填）")
    customercode: str = Field(..., description="客商编码（必填）")
    amount: float = Field(..., description="本币金额（必填）")
    digest: str = Field(..., description="摘要（必填）")
    entry: List[AcceptEntry] = Field(..., description="收款单体列表（必填）")

# ===================== 获取单张收款单 数据模型 =====================
class GetAcceptInfoInput(BaseModel):
    id: str = Field(..., description="单据编码，用于查询收款单单据信息")

# ===================== 获取收款单列表信息 数据模型 =====================
class GetAcceptListInfoInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    vouchcode_begin: Optional[str] = Field(None, description="起始单据编号")
    vouchcode_end: Optional[str] = Field(None, description="结束单据编号")
    vouchdate_begin: Optional[str] = Field(None, description="起始制单日期，格式:yyyy-MM-dd")
    vouchdate_end: Optional[str] = Field(None, description="结束制单日期，格式:yyyy-MM-dd")
    vouchtype: Optional[str] = Field(None, description="单据类型(48=收款单;49=付款单)")
    customercode: Optional[str] = Field(None, description="客户或供应商编码")
    customername: Optional[str] = Field(None, description="客户或供应商名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    digest: Optional[str] = Field(None, description="摘要关键字")

# ===================== 审批一张收款单 数据模型 =====================
class VerifyAcceptInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    voucher_type: str = Field(..., description="单据类型（48=收款；49=付款）（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审一张收款单 数据模型 =====================
class UnVerifyAcceptInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    voucher_type: str = Field(..., description="单据类型（48=收款；49=付款）（必填）")







# ===================== 新增一张收款单 Tool函数 =====================
def u8_accept_add_tool(input_data: AddAcceptInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增收款单，包含单据头和单据体（entry）完整信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 accept）
    request_body: dict = {
        "accept": input_data.model_dump(exclude_none=True)
    }

    # 收款单添加接口路径
    api_path = "/api/accept/add" 
    
    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body,is_tradeid=True)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "收款单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "收款单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单张收款单 Tool函数 =====================
def u8_accept_get_tool(input_data: GetAcceptInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编码获取用友U8中的收款单单据信息。
    """
    params = {
        "id": input_data.id 
    }
    
    # 修改点3: 更新API路径
    api_path = "/api/accept/get" 
    
    try:
        # 使用 GET 或 POST 取决于具体接口要求，这里假设是 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)
        
        # 检查业务错误码 (可选，根据实际返回结构调整)
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取收款单列表信息 Tool函数 =====================
def u8_accept_list_get_tool(input_data: GetAcceptListInfoInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取收款单列表信息，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)
    
    # 收款单列表查询接口路径
    api_path = "/api/acceptlist/batch_get" 
    
    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "收款单列表查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "收款单列表查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "acceptlist": result.get("acceptlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 审批一张收款单 Tool函数 =====================
def u8_accept_verify_tool(input_data: VerifyAcceptInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审批收款单（支持收款单/付款单审核）
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 accept）
    request_body: dict = {
        "accept": input_data.model_dump(exclude_none=True)
    }

    # 收款单审批接口路径
    api_path = "/api/accept/verify" 
    
    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body,is_tradeid=True,is_user_login_v2=True)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "收款单审批失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "收款单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审一张收款单 Tool函数 =====================
def u8_accept_unverify_tool(input_data: VerifyAcceptInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审收款单（支持收款单/付款单弃审）
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 accept）
    request_body: dict = {
        "accept": input_data.model_dump(exclude_none=True)
    }

    # 收款单审批接口路径
    api_path = "/api/accept/unverify" 
    
    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body,is_tradeid=True,is_user_login_v2=True)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "收款单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "收款单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)














# ===================== 新增一张收款单 Schema定义 =====================
U8_ACCEPT_ADD_SCHEMA = {
    "name": "u8_accept_add",
    "description": "在用友U8 OpenAPI中新增收款单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 单据头参数
            "vouchcode": {"type": "string", "description": "应收单号"},
            "vouchdate": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）"},
            "vouchtype": {
                "type": "string",
                "description": "单据类型(48=收款单;49=付款单)（必填）"
            },
            "customercode": {"type": "string", "description": "客商编码（必填）"},
            "customername": {"type": "string", "description": "客商名称"},
            "customerabbname": {"type": "string", "description": "客商简称"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "amount": {"type": "number", "description": "本币金额（必填）"},
            "originalamount": {"type": "number", "description": "原币金额"},
            "digest": {"type": "string", "description": "摘要（必填）"},
            "balancecode": {"type": "string", "description": "结算方式编码"},
            "balancename": {"type": "string", "description": "结算方式"},
            "balanceitemcode": {"type": "string", "description": "结算科目编码"},
            "itemclasscode": {"type": "string", "description": "项目大类编号"},
            "itemcode": {"type": "string", "description": "项目编码"},
            "itemname": {"type": "string", "description": "项目名称"},
            "oppositebankname": {"type": "string", "description": "对方单位银行名称"},
            "oppositebankcode": {"type": "string", "description": "对方单位银行帐号"},
            "bankname": {"type": "string", "description": "本单位银行名称"},
            "bankaccount": {"type": "string", "description": "本单位银行帐号"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4（日期格式：yyyy-MM-dd）"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6（日期格式：yyyy-MM-dd）"},
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
                "description": "收款单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "integer", "description": "款项类型(0-应收款;1-预收款;2-其他费用)，默认0"},
                        "customercode": {"type": "string", "description": "客商编码（必填）"},
                        "customerabbname": {"type": "string", "description": "客商简称"},
                        "departmentcode": {"type": "string", "description": "部门编码"},
                        "departmentname": {"type": "string", "description": "部门名称"},
                        "personcode": {"type": "string", "description": "人员编码"},
                        "personname": {"type": "string", "description": "人员"},
                        "digest": {"type": "string", "description": "摘要"},
                        "project": {"type": "string", "description": "项目编号"},
                        "projectclass": {"type": "string", "description": "项目大类编号"},
                        "itemcode": {"type": "string", "description": "科目"},
                        "itemname": {"type": "string", "description": "项目名称"},
                        "amount": {"type": "number", "description": "本币金额（必填）"},
                        "originalamount": {"type": "number", "description": "原币金额"},
                        "iamt_s": {"type": "number", "description": "数量（必填）"},
                        "iramt_s": {"type": "number", "description": "剩余数量"},
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
                        "define36": {"type": "string", "description": "单据体自定义项15（日期格式：yyyy-MM-dd）"},
                        "define37": {"type": "string", "description": "单据体自定义项16（日期格式：yyyy-MM-dd）"}
                    },
                    "required": ["customercode", "amount", "iamt_s"]
                }
            }
        },
        "required": [
            "vouchtype", 
            "customercode", 
            "amount", 
            "digest", 
            "entry"
        ]
    }
}

# ===================== 获取单张收款单 Schema定义 =====================
U8_ACCEPT_GET_SCHEMA = {
    "name": "u8_accept_get",
    "description": "通过单据编码获得收款单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "单据编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取收款单列表信息 Schema定义 =====================
U8_ACCEPT_LIST_GET_SCHEMA = {
    "name": "u8_accept_list_get",
    "description": "在用友U8 OpenAPI中查询收款单列表信息，支持分页、单据号/日期范围、客商/部门/业务员等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "vouchcode_begin": {"type": "string", "description": "起始单据编号"},
            "vouchcode_end": {"type": "string", "description": "结束单据编号"},
            "vouchdate_begin": {"type": "string", "description": "起始制单日期，格式:yyyy-MM-dd"},
            "vouchdate_end": {"type": "string", "description": "结束制单日期，格式:yyyy-MM-dd"},
            "vouchtype": {"type": "string", "description": "单据类型(48=收款单;49=付款单)"},
            "customercode": {"type": "string", "description": "客户或供应商编码"},
            "customername": {"type": "string", "description": "客户或供应商名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "digest": {"type": "string", "description": "摘要关键字"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 审批一张收款单 Schema定义 =====================
U8_ACCEPT_VERIFY_SCHEMA = {
    "name": "u8_accept_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审批收款单（支持收款单/付款单审核）",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "voucher_type": {
                "type": "string",
                "description": "单据类型（48=收款单；49=付款单）（必填）"
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
            "voucher_code",
            "voucher_type"
        ]
    }
}

# ===================== 弃审一张收款单 Schema定义 =====================
U8_ACCEPT_UNVERIFY_SCHEMA = {
    "name": "u8_accept_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审收款单（支持收款单/付款单弃审）",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "voucher_type": {
                "type": "string",
                "description": "单据类型（48=收款单；49=付款单）（必填）"
            }
        },
        "required": [
            "voucher_code",
            "voucher_type"
        ]
    }
}















# ===================== 付款单通用数据模型 (复用于新增和查询) =====================

class PayEntry(BaseModel):
    """
    付款单体模型。
    注意：所有字段均为 Optional，以兼容查询返回（可能缺省）和新增传入（后端校验必填）。
    """
    customercode: Optional[str] = Field(None, description="客商编码")
    customerabbname: Optional[str] = Field(None, description="客商简称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    digest: Optional[str] = Field(None, description="摘要")
    project: Optional[str] = Field(None, description="项目编号")
    projectclass: Optional[str] = Field(None, description="项目大类编号")
    itemcode: Optional[str] = Field(None, description="科目")
    itemname: Optional[str] = Field(None, description="项目名称")
    amount: Optional[float] = Field(None, description="本币金额")
    originalamount: Optional[float] = Field(None, description="原币金额")
    iamt_s: Optional[float] = Field(None, description="数量")
    iramt_s: Optional[float] = Field(None, description="剩余数量")
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

class PayInfo(BaseModel):
    """
    付款单主表模型 (通用，用于新增输入和查询返回)
    """
    vouchcode: Optional[str] = Field(None, description="付款单号")
    vouchdate: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    vouchtype: Optional[str] = Field(None, description="单据类型(48=收款单;49=付款单)")
    customercode: Optional[str] = Field(None, description="客商编码")
    customername: Optional[str] = Field(None, description="客商名称")
    customerabbname: Optional[str] = Field(None, description="客商简称")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    amount: Optional[float] = Field(None, description="本币金额")
    originalamount: Optional[float] = Field(None, description="原币金额")
    digest: Optional[str] = Field(None, description="摘要")
    balancecode: Optional[str] = Field(None, description="结算方式编码")
    balancename: Optional[str] = Field(None, description="结算方式")
    balanceitemcode: Optional[str] = Field(None, description="结算科目编码")
    itemclasscode: Optional[str] = Field(None, description="项目大类编号")
    itemcode: Optional[str] = Field(None, description="项目编码")
    checkman: Optional[str] = Field(None, description="审核人")
    operator: Optional[str] = Field(None, description="录入人")
    itemname: Optional[str] = Field(None, description="项目名称")
    oppositebankname: Optional[str] = Field(None, description="对方单位银行名称")
    oppositebankcode: Optional[str] = Field(None, description="对方单位银行帐号")
    bankname: Optional[str] = Field(None, description="本单位银行名称")
    bankaccount: Optional[str] = Field(None, description="本单位银行帐号")
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
    entry: Optional[List[PayEntry]] = Field(None, description="付款单体列表")

# ===================== 新增一张付款单 数据模型 =====================
class AddPayInfoInput(PayInfo):
    """新增付款单输入模型，继承自通用模型，可在此处强化必填项校验"""
    vouchtype: str = Field(..., description="单据类型(48=收款单;49=付款单)（必填）")
    customercode: str = Field(..., description="客商编码（必填）")
    amount: float = Field(..., description="本币金额（必填）")
    digest: str = Field(..., description="摘要（必填）")
    entry: List[PayEntry] = Field(..., description="付款单体列表（必填）")

# ===================== 获取单张付款单 数据模型 =====================
class GetPayInfoInput(BaseModel):
    id: str = Field(..., description="单据编码，用于查询付款单单据信息")

# ===================== 获取付款单列表信息 数据模型 =====================
class GetPayListInfoInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    vouchcode_begin: Optional[str] = Field(None, description="起始单据编号")
    vouchcode_end: Optional[str] = Field(None, description="结束单据编号")
    vouchdate_begin: Optional[str] = Field(None, description="起始制单日期，格式:yyyy-MM-dd")
    vouchdate_end: Optional[str] = Field(None, description="结束制单日期，格式:yyyy-MM-dd")
    vouchtype: Optional[str] = Field(None, description="单据类型(48=收款单;49=付款单)")
    vendorcode: Optional[str] = Field(None, description="客户或供应商编码")
    vendorname: Optional[str] = Field(None, description="客户或供应商名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    departmentcode: Optional[str] = Field(None, description="部门编码")
    departmentname: Optional[str] = Field(None, description="部门名称关键字")
    digest: Optional[str] = Field(None, description="摘要关键字")
    checkman: Optional[str] = Field(None, description="审核人")
    operator: Optional[str] = Field(None, description="录入人")

# ===================== 审批一张付款单 数据模型 =====================
class VerifyPayInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    voucher_type: str = Field(..., description="单据类型（48=收款；49=付款）（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审一张付款单 数据模型 =====================
class UnVerifyPayInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    voucher_type: str = Field(..., description="单据类型（48=收款；49=付款）（必填）")




# ===================== 获取付款单待办任务 数据模型 =====================
class GetPayTasksInput(BaseModel):
    """获取付款单待办任务输入模型"""
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

# ===================== 获取付款单审批进程 数据模型 =====================
class GetPayHistoryInput(BaseModel):
    """获取付款单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 获取付款单是否启用工作流 数据模型 =====================
class GetPayFlowenabledInput(BaseModel):
    """获取付款单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")

# ===================== 获取付款单工作流按钮是否可用状态 数据模型 =====================
class GetPayButtonstateInput(BaseModel):
    """获取付款单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 审核付款单 数据模型 =====================
class AuditPayInfoInput(BaseModel):
    """审核付款单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")

# ===================== 弃审付款单 数据模型 =====================
class AbandonPayInfoInput(BaseModel):
    """弃审付款单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")











# ===================== 新增一张付款单 Tool函数 =====================
def u8_pay_add_tool(input_data: AddPayInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增付款单，包含单据头和单据体（entry）完整信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 pay）
    request_body: dict = {
        "pay": input_data.model_dump(exclude_none=True)
    }

    # 付款单添加接口路径
    api_path = "/api/pay/add" 
    
    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单张付款单 Tool函数 =====================
def u8_pay_get_tool(input_data: GetPayInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编码获取用友U8中的付款单单据信息。
    """
    params = {
        "id": input_data.id 
    }
    
    # 付款单查询接口路径
    api_path = "/api/pay/get" 
    
    try:
        # 使用 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)
        
        # 检查业务错误码
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取付款单列表信息 Tool函数 =====================
def u8_pay_list_get_tool(input_data: GetPayListInfoInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取付款单列表信息，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)
    
    # 付款单列表查询接口路径
    api_path = "/api/paylist/batch_get" 
    
    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款单列表查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款单列表查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "paylist": result.get("paylist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 审批一张付款单 Tool函数 =====================
def u8_pay_verify_tool(input_data: VerifyPayInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审批付款单（支持收款单/付款单审核）
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 pay）
    request_body: dict = {
        "pay": input_data.model_dump(exclude_none=True)
    }

    # 付款单审批接口路径
    api_path = "/api/pay/verify" 
    
    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款单审批失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审一张付款单 Tool函数 =====================
def u8_pay_unverify_tool(input_data: UnVerifyPayInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审付款单（支持收款单/付款单弃审）
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 pay）
    request_body: dict = {
        "pay": input_data.model_dump(exclude_none=True)
    }

    # 付款单弃审接口路径
    api_path = "/api/pay/unverify" 
    
    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)
        
        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)





# ===================== 获取付款单待办任务 Tool函数 =====================
def u8_pay_tasks_tool(input_data: GetPayTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取付款单待办任务列表，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 付款单待办任务接口路径
    api_path = "/api/pay/tasks"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款单待办任务成功",
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

# ===================== 获取付款单审批进程 Tool函数 =====================
def u8_pay_history_tool(input_data: GetPayHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看付款单审批进程，获取单据的审批历史记录。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 付款单审批进程接口路径
    api_path = "/api/pay/history"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款单审批进程成功",
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

# ===================== 获取付款单是否启用工作流 Tool函数 =====================
def u8_pay_flowenabled_tool(input_data: GetPayFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取付款单是否启用工作流。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 付款单是否启用工作流接口路径
    api_path = "/api/pay/flowenabled"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款单工作流启用状态成功",
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

# ===================== 获取付款单工作流按钮是否可用状态 Tool函数 =====================
def u8_pay_buttonstate_tool(input_data: GetPayButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取付款单工作流按钮是否可用状态。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 付款单工作流按钮状态接口路径
    api_path = "/api/pay/buttonstate"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款单工作流按钮状态成功",
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

# ===================== 审核付款单 Tool函数 =====================
def u8_pay_audit_tool(input_data: AuditPayInfoInput, client: U8OpenAPIClient) -> str:
    """
    审核付款单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 pay）
    request_body: dict = {
        "pay": input_data.model_dump(exclude_none=True)
    }

    # 付款单审核接口路径
    api_path = "/api/pay/audit"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审付款单 Tool函数 =====================
def u8_pay_abandon_tool(input_data: AbandonPayInfoInput, client: U8OpenAPIClient) -> str:
    """
    弃审付款单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 pay）
    request_body: dict = {
        "pay": input_data.model_dump(exclude_none=True)
    }

    # 付款单弃审接口路径
    api_path = "/api/pay/abandon"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)







# ===================== 新增一张付款单 Schema定义 =====================
U8_PAY_ADD_SCHEMA = {
    "name": "u8_pay_add",
    "description": "在用友U8 OpenAPI中新增付款单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 单据头参数
            "vouchcode": {"type": "string", "description": "付款单号"},
            "vouchdate": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）"},
            "vouchtype": {
                "type": "string",
                "description": "单据类型(48=收款单;49=付款单)（必填）"
            },
            "customercode": {"type": "string", "description": "客商编码（必填）"},
            "customername": {"type": "string", "description": "客商名称"},
            "customerabbname": {"type": "string", "description": "客商简称"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "amount": {"type": "number", "description": "本币金额（必填）"},
            "originalamount": {"type": "number", "description": "原币金额"},
            "digest": {"type": "string", "description": "摘要（必填）"},
            "balancecode": {"type": "string", "description": "结算方式编码"},
            "balancename": {"type": "string", "description": "结算方式"},
            "balanceitemcode": {"type": "string", "description": "结算科目编码"},
            "itemclasscode": {"type": "string", "description": "项目大类编号"},
            "itemcode": {"type": "string", "description": "项目编码"},
            "checkman": {"type": "string", "description": "审核人"},
            "operator": {"type": "string", "description": "录入人"},
            "itemname": {"type": "string", "description": "项目名称"},
            "oppositebankname": {"type": "string", "description": "对方单位银行名称"},
            "oppositebankcode": {"type": "string", "description": "对方单位银行帐号"},
            "bankname": {"type": "string", "description": "本单位银行名称"},
            "bankaccount": {"type": "string", "description": "本单位银行帐号"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4（日期格式：yyyy-MM-dd）"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6（日期格式：yyyy-MM-dd）"},
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
                "description": "付款单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "customercode": {"type": "string", "description": "客商编码（必填）"},
                        "customerabbname": {"type": "string", "description": "客商简称"},
                        "departmentcode": {"type": "string", "description": "部门编码"},
                        "departmentname": {"type": "string", "description": "部门名称"},
                        "personcode": {"type": "string", "description": "人员编码"},
                        "personname": {"type": "string", "description": "人员"},
                        "digest": {"type": "string", "description": "摘要"},
                        "project": {"type": "string", "description": "项目编号"},
                        "projectclass": {"type": "string", "description": "项目大类编号"},
                        "itemcode": {"type": "string", "description": "科目"},
                        "itemname": {"type": "string", "description": "项目名称"},
                        "amount": {"type": "number", "description": "本币金额（必填）"},
                        "originalamount": {"type": "number", "description": "原币金额"},
                        "iamt_s": {"type": "number", "description": "数量（必填）"},
                        "iramt_s": {"type": "number", "description": "剩余数量"},
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
                        "define36": {"type": "string", "description": "单据体自定义项15（日期格式：yyyy-MM-dd）"},
                        "define37": {"type": "string", "description": "单据体自定义项16（日期格式：yyyy-MM-dd）"}
                    },
                    "required": ["customercode", "amount", "iamt_s"]
                }
            }
        },
        "required": [
            "vouchtype", 
            "customercode", 
            "amount", 
            "digest", 
            "entry"
        ]
    }
}

# ===================== 获取单张付款单 Schema定义 =====================
U8_PAY_GET_SCHEMA = {
    "name": "u8_pay_get",
    "description": "通过单据编码获得付款单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "单据编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取付款单列表信息 Schema定义 =====================
U8_PAY_LIST_GET_SCHEMA = {
    "name": "u8_pay_list_get",
    "description": "在用友U8 OpenAPI中查询付款单列表信息，支持分页、单据号/日期范围、客商/部门/业务员等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "vouchcode_begin": {"type": "string", "description": "起始单据编号"},
            "vouchcode_end": {"type": "string", "description": "结束单据编号"},
            "vouchdate_begin": {"type": "string", "description": "起始制单日期，格式:yyyy-MM-dd"},
            "vouchdate_end": {"type": "string", "description": "结束制单日期，格式:yyyy-MM-dd"},
            "vouchtype": {"type": "string", "description": "单据类型(48=收款单;49=付款单)"},
            "vendorcode": {"type": "string", "description": "客户或供应商编码"},
            "vendorname": {"type": "string", "description": "客户或供应商名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "departmentcode": {"type": "string", "description": "部门编码"},
            "departmentname": {"type": "string", "description": "部门名称关键字"},
            "digest": {"type": "string", "description": "摘要关键字"},
            "checkman": {"type": "string", "description": "审核人"},
            "operator": {"type": "string", "description": "录入人"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 审批一张付款单 Schema定义 =====================
U8_PAY_VERIFY_SCHEMA = {
    "name": "u8_pay_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审批付款单（支持收款单/付款单审核）",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "voucher_type": {
                "type": "string",
                "description": "单据类型（48=收款单；49=付款单）（必填）"
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
            "voucher_code",
            "voucher_type"
        ]
    }
}

# ===================== 弃审一张付款单 Schema定义 =====================
U8_PAY_UNVERIFY_SCHEMA = {
    "name": "u8_pay_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审付款单（支持收款单/付款单弃审）",
    "parameters": {
        "type": "object",
        "properties": {
            "voucher_code": {
                "type": "string",
                "description": "单据编号（必填）"
            },
            "voucher_type": {
                "type": "string",
                "description": "单据类型（48=收款单；49=付款单）（必填）"
            }
        },
        "required": [
            "voucher_code",
            "voucher_type"
        ]
    }
}







# ===================== 获取付款单待办任务 Schema定义 =====================
U8_PAY_TASKS_SCHEMA = {
    "name": "u8_pay_tasks",
    "description": "获取付款单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
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

# ===================== 获取付款单审批进程 Schema定义 =====================
U8_PAY_HISTORY_SCHEMA = {
    "name": "u8_pay_history",
    "description": "查看付款单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取付款单是否启用工作流 Schema定义 =====================
U8_PAY_FLOWENABLED_SCHEMA = {
    "name": "u8_pay_flowenabled",
    "description": "获取付款单是否启用工作流",
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

# ===================== 获取付款单工作流按钮是否可用状态 Schema定义 =====================
U8_PAY_BUTTONSTATE_SCHEMA = {
    "name": "u8_pay_buttonstate",
    "description": "获取付款单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
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

# ===================== 审核付款单 Schema定义 =====================
U8_PAY_AUDIT_SCHEMA = {
    "name": "u8_pay_audit",
    "description": "审核付款单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 弃审付款单 Schema定义 =====================
U8_PAY_ABANDON_SCHEMA = {
    "name": "u8_pay_abandon",
    "description": "弃审付款单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
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







# ===================== 获取付款申请单列表 数据模型 =====================
class GetPayrequestListInfoInput(BaseModel):
    """获取付款申请单列表输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    cvouchtype: Optional[str] = Field(None, description="单据类型")
    cdwcode: Optional[str] = Field(None, description="供应商编码")
    cvenabbname: Optional[str] = Field(None, description="供应商")
    cdepcode: Optional[str] = Field(None, description="部门编码")
    cdepname: Optional[str] = Field(None, description="部门")
    csscode: Optional[str] = Field(None, description="结算方式编号")
    cssname: Optional[str] = Field(None, description="结算方式")
    ccode: Optional[str] = Field(None, description="结算科目")
    citemcode: Optional[str] = Field(None, description="项目编码")
    citemname: Optional[str] = Field(None, description="项目名称关键字")
    citem_class: Optional[str] = Field(None, description="项目大类编号")
    citem_name: Optional[str] = Field(None, description="项目大类")
    cdigest: Optional[str] = Field(None, description="备注关键字")
    istate: Optional[int] = Field(None, description="单据状态")
    iverifystate: Optional[int] = Field(None, description="审批状态")
    ibustype: Optional[int] = Field(None, description="业务类型编号")
    cbustype: Optional[str] = Field(None, description="业务类型")
    ipaytype: Optional[int] = Field(None, description="付款类型编号")
    cpaytype: Optional[str] = Field(None, description="付款类型")
    cagentcuscode: Optional[str] = Field(None, description="代理商编码")

# ===================== 获取付款申请单待办任务 数据模型 =====================
class GetPayrequestTasksInput(BaseModel):
    """获取付款申请单待办任务输入模型"""
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

# ===================== 撤销付款申请单 数据模型 =====================
class ReturnPayrequestInfoInput(BaseModel):
    """撤销付款申请单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")

# ===================== 获取付款申请单审批进程 数据模型 =====================
class GetPayrequestHistoryInput(BaseModel):
    """获取付款申请单审批进程输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 获取单个付款申请单 数据模型 =====================
class GetPayrequestInfoInput(BaseModel):
    """获取单个付款申请单输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: Optional[str] = Field(None, description="付款申请单编号")

# ===================== 获取付款申请单是否启用工作流 数据模型 =====================
class GetPayrequestFlowenabledInput(BaseModel):
    """获取付款申请单是否启用工作流输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")

# ===================== 获取付款申请单工作流按钮是否可用状态 数据模型 =====================
class GetPayrequestButtonstateInput(BaseModel):
    """获取付款申请单工作流按钮是否可用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码")
    voucher_code: str = Field(..., description="单据编号（必填）")

# ===================== 审核付款申请单（工作流） 数据模型 =====================
class AuditPayrequestInfoInput(BaseModel):
    """审核付款申请单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")
    agree: int = Field(..., description="是否同意(1=同意;0=不同意)（必填）")

# ===================== 弃审付款申请单 数据模型 =====================
class AbandonPayrequestInfoInput(BaseModel):
    """弃审付款申请单输入模型"""
    voucher_code: str = Field(..., description="单据编号（必填）")
    user_id: Optional[str] = Field(None, description="审批人(操作员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    person_code: Optional[str] = Field(None, description="审批人(人员编码)，user_id与person_code输入一个参数即可，同时传入取user_id")
    opinion: Optional[str] = Field(None, description="审批意见")







# ===================== 获取付款申请单列表 Tool函数 =====================
def u8_payrequest_list_get_tool(input_data: GetPayrequestListInfoInput, client: U8OpenAPIClient) -> str:
    """
    获取付款申请单列表信息，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payrequestlist/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款申请单列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款申请单列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "payrequestlist": result.get("payrequestlist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取付款申请单待办任务 Tool函数 =====================
def u8_payrequest_tasks_tool(input_data: GetPayrequestTasksInput, client: U8OpenAPIClient) -> str:
    """
    获取付款申请单待办任务列表，支持多条件筛选和分页查询。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payrequest/tasks"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款申请单待办任务失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款申请单待办任务成功",
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

# ===================== 撤销付款申请单 Tool函数 =====================
def u8_payrequest_return_tool(input_data: ReturnPayrequestInfoInput, client: U8OpenAPIClient) -> str:
    """
    撤销付款申请单（工作流撤销）。
    执行撤销动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "payrequest": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/payrequest/return"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款申请单撤销失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款申请单撤销成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取付款申请单审批进程 Tool函数 =====================
def u8_payrequest_history_tool(input_data: GetPayrequestHistoryInput, client: U8OpenAPIClient) -> str:
    """
    查看付款申请单审批进程，获取单据的审批历史记录。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payrequest/history"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款申请单审批进程失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款申请单审批进程成功",
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

# ===================== 获取单个付款申请单 Tool函数 =====================
def u8_payrequest_get_tool(input_data: GetPayrequestInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过付款申请单编号获取用友U8中的付款申请单单据信息。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payrequest/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取付款申请单是否启用工作流 Tool函数 =====================
def u8_payrequest_flowenabled_tool(input_data: GetPayrequestFlowenabledInput, client: U8OpenAPIClient) -> str:
    """
    获取付款申请单是否启用工作流。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payrequest/flowenabled"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款申请单工作流启用状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款申请单工作流启用状态成功",
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

# ===================== 获取付款申请单工作流按钮是否可用状态 Tool函数 =====================
def u8_payrequest_buttonstate_tool(input_data: GetPayrequestButtonstateInput, client: U8OpenAPIClient) -> str:
    """
    获取付款申请单工作流按钮是否可用状态。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payrequest/buttonstate"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取付款申请单工作流按钮状态失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取付款申请单工作流按钮状态成功",
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

# ===================== 审核付款申请单（工作流） Tool函数 =====================
def u8_payrequest_audit_tool(input_data: AuditPayrequestInfoInput, client: U8OpenAPIClient) -> str:
    """
    审核付款申请单（工作流审批，支持同意或不同意）。
    执行审批动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "payrequest": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/payrequest/audit"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款申请单审核失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款申请单审核成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审付款申请单 Tool函数 =====================
def u8_payrequest_abandon_tool(input_data: AbandonPayrequestInfoInput, client: U8OpenAPIClient) -> str:
    """
    弃审付款申请单（工作流弃审）。
    执行弃审动作前，需要保证审批人已经进行ERP登录授权。
    """
    request_body: dict = {
        "payrequest": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/payrequest/abandon"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "付款申请单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "付款申请单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取付款申请单列表 Schema定义 =====================
U8_PAYREQUEST_LIST_GET_SCHEMA = {
    "name": "u8_payrequest_list_get",
    "description": "获取付款申请单列表信息，支持单据号/日期范围、供应商/部门/项目/业务类型/付款类型等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始单据编号"},
            "code_end": {"type": "string", "description": "结束单据编号"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "cvouchtype": {"type": "string", "description": "单据类型"},
            "cdwcode": {"type": "string", "description": "供应商编码"},
            "cvenabbname": {"type": "string", "description": "供应商"},
            "cdepcode": {"type": "string", "description": "部门编码"},
            "cdepname": {"type": "string", "description": "部门"},
            "csscode": {"type": "string", "description": "结算方式编号"},
            "cssname": {"type": "string", "description": "结算方式"},
            "ccode": {"type": "string", "description": "结算科目"},
            "citemcode": {"type": "string", "description": "项目编码"},
            "citemname": {"type": "string", "description": "项目名称关键字"},
            "citem_class": {"type": "string", "description": "项目大类编号"},
            "citem_name": {"type": "string", "description": "项目大类"},
            "cdigest": {"type": "string", "description": "备注关键字"},
            "istate": {"type": "number", "description": "单据状态"},
            "iverifystate": {"type": "number", "description": "审批状态"},
            "ibustype": {"type": "number", "description": "业务类型编号"},
            "cbustype": {"type": "string", "description": "业务类型"},
            "ipaytype": {"type": "number", "description": "付款类型编号"},
            "cpaytype": {"type": "string", "description": "付款类型"},
            "cagentcuscode": {"type": "string", "description": "代理商编码"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 获取付款申请单待办任务 Schema定义 =====================
U8_PAYREQUEST_TASKS_SCHEMA = {
    "name": "u8_payrequest_tasks",
    "description": "获取付款申请单待办任务列表，支持分页、状态、类型、发起人等多条件筛选",
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

# ===================== 撤销付款申请单 Schema定义 =====================
U8_PAYREQUEST_RETURN_SCHEMA = {
    "name": "u8_payrequest_return",
    "description": "撤销付款申请单（工作流撤销）。执行撤销动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取付款申请单审批进程 Schema定义 =====================
U8_PAYREQUEST_HISTORY_SCHEMA = {
    "name": "u8_payrequest_history",
    "description": "查看付款申请单审批进程，获取单据的审批历史记录。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 获取单个付款申请单 Schema定义 =====================
U8_PAYREQUEST_GET_SCHEMA = {
    "name": "u8_payrequest_get",
    "description": "通过付款申请单编号获取单个付款申请单详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {
                "type": "string",
                "description": "付款申请单编号"
            }
        },
        "required": []  # 所有参数均为可选
    }
}

# ===================== 获取付款申请单是否启用工作流 Schema定义 =====================
U8_PAYREQUEST_FLOWENABLED_SCHEMA = {
    "name": "u8_payrequest_flowenabled",
    "description": "获取付款申请单是否启用工作流",
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

# ===================== 获取付款申请单工作流按钮是否可用状态 Schema定义 =====================
U8_PAYREQUEST_BUTTONSTATE_SCHEMA = {
    "name": "u8_payrequest_buttonstate",
    "description": "获取付款申请单工作流按钮是否可用状态（同意、不同意、弃审、转签、重新提交）。只支持12.0版本，且需要打最新的WF补丁。",
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

# ===================== 审核付款申请单（工作流） Schema定义 =====================
U8_PAYREQUEST_AUDIT_SCHEMA = {
    "name": "u8_payrequest_audit",
    "description": "审核付款申请单（工作流审批，支持同意或不同意）。执行审批动作前，需要保证审批人已经进行ERP登录授权。",
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

# ===================== 弃审付款申请单 Schema定义 =====================
U8_PAYREQUEST_ABANDON_SCHEMA = {
    "name": "u8_payrequest_abandon",
    "description": "弃审付款申请单（工作流弃审）。执行弃审动作前，需要保证审批人已经进行ERP登录授权。",
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















# ===================== 应付单通用数据模型 (复用于新增和查询) =====================

class OughtpayEntry(BaseModel):
    """
    应付单体模型。
    注意：所有字段均为 Optional，以兼容查询返回（可能缺省）和新增传入（后端校验必填）。
    """
    cust_vendor_code: Optional[str] = Field(None, description="客商编码")
    cusabbname: Optional[str] = Field(None, description="客商简称")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    digest: Optional[str] = Field(None, description="摘要")
    subjectcode: Optional[str] = Field(None, description="科目编码")
    currency_name: Optional[str] = Field(None, description="币种")
    currency_rate: Optional[float] = Field(None, description="汇率")
    natamount: Optional[float] = Field(None, description="本币金额")
    amount: Optional[float] = Field(None, description="原币金额")
    bdebitcredit: Optional[bool] = Field(None, description="借贷方向（0贷1借，默认为1）")
    item_classcode: Optional[str] = Field(None, description="项目大类编码")
    item_code: Optional[str] = Field(None, description="项目编码")
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

class OughtpayInfo(BaseModel):
    """
    应付单主表模型 (通用，用于新增输入和查询返回)
    """
    code: Optional[str] = Field(None, description="应付单号")
    date: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    state: Optional[str] = Field(None, description="单据状态")
    bdebitcredit: Optional[bool] = Field(None, description="单据类型（0蓝单1红单，默认为0）")
    cust_vendor_code: Optional[str] = Field(None, description="客商编码")
    cusname: Optional[str] = Field(None, description="客商名称")
    cusabbname: Optional[str] = Field(None, description="客商简称")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称")
    subjectcode: Optional[str] = Field(None, description="科目编码")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    amount: Optional[float] = Field(None, description="原币金额")
    natamount: Optional[float] = Field(None, description="本币金额")
    digest: Optional[str] = Field(None, description="摘要")
    item_classcode: Optional[str] = Field(None, description="项目大类编码")
    item_code: Optional[str] = Field(None, description="项目编码")
    currency_name: Optional[str] = Field(None, description="币种")
    currency_rate: Optional[float] = Field(None, description="汇率")
    paycondition_code: Optional[str] = Field(None, description="付款条件")
    operator: Optional[str] = Field(None, description="操作员姓名")
    flag: Optional[str] = Field(None, description="标志")
    quantity: Optional[float] = Field(None, description="数量")
    startflag: Optional[bool] = Field(None, description="期初标志")
    relatevouchercode: Optional[str] = Field(None, description="对应单据号")
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
    entry: Optional[List[OughtpayEntry]] = Field(None, description="应付单体列表")

# ===================== 新增一张应付单 数据模型 =====================
class AddOughtpayInfoInput(OughtpayInfo):
    """新增应付单输入模型，继承自通用模型，可在此处强化必填项校验"""
    cust_vendor_code: str = Field(..., description="客商编码（必填）")
    amount: float = Field(..., description="原币金额（必填）")
    entry: List[OughtpayEntry] = Field(..., description="应付单体列表（必填）")

# ===================== 获取单张应付单 数据模型 =====================
class GetOughtpayInfoInput(BaseModel):
    id: str = Field(..., description="单据编码，用于查询应付单单据信息")

# ===================== 获取应付单列表信息 数据模型 =====================
class GetOughtpayListInfoInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始应收付单据号")
    code_end: Optional[str] = Field(None, description="结束应收付单据号")
    state: Optional[str] = Field(None, description="状态")
    date_begin: Optional[str] = Field(None, description="起始单据日期")
    date_end: Optional[str] = Field(None, description="结束单据日期")
    cust_vendor_code: Optional[str] = Field(None, description="客商编号")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    personcode: Optional[str] = Field(None, description="职员编码")
    personname: Optional[str] = Field(None, description="职员名称关键字")
    item_code: Optional[str] = Field(None, description="项目编码")
    digest: Optional[str] = Field(None, description="摘要")
    operator: Optional[str] = Field(None, description="操作员姓名关键字")
    define1: Optional[str] = Field(None, description="表头自定义项1")
    define2: Optional[str] = Field(None, description="表头自定义项2")
    define3: Optional[str] = Field(None, description="表头自定义项3")
    define4: Optional[str] = Field(None, description="表头自定义项4")
    define5: Optional[float] = Field(None, description="表头自定义项5")
    define6: Optional[str] = Field(None, description="表头自定义项6")
    define7: Optional[float] = Field(None, description="表头自定义项7")
    define8: Optional[str] = Field(None, description="表头自定义项8")
    define9: Optional[str] = Field(None, description="表头自定义项9")
    define10: Optional[str] = Field(None, description="表头自定义项10")
    define11: Optional[str] = Field(None, description="表头自定义项11")
    define12: Optional[str] = Field(None, description="表头自定义项12")
    define13: Optional[str] = Field(None, description="表头自定义项13")
    define14: Optional[str] = Field(None, description="表头自定义项14")
    define15: Optional[float] = Field(None, description="表头自定义项15")
    define16: Optional[float] = Field(None, description="表头自定义项16")
    ufts_begin: Optional[str] = Field(None, description="起始时间戳")
    ufts_end: Optional[str] = Field(None, description="结束时间戳")

# ===================== 审批一张应付单 数据模型 =====================
class VerifyOughtpayInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审一张应付单 数据模型 =====================
class UnVerifyOughtpayInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")




# ===================== 新增一张应付单 Tool函数 =====================
def u8_oughtpay_add_tool(input_data: AddOughtpayInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增应付单，包含单据头和单据体（entry）完整信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 oughtpay）
    request_body: dict = {
        "oughtpay": input_data.model_dump(exclude_none=True)
    }

    # 应付单添加接口路径
    api_path = "/api/oughtpay/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应付单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应付单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单张应付单 Tool函数 =====================
def u8_oughtpay_get_tool(input_data: GetOughtpayInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编码获取用友U8中的应付单单据信息。
    """
    params = {
        "id": input_data.id
    }

    # 应付单查询接口路径
    api_path = "/api/oughtpay/get"

    try:
        # 使用 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)

        # 检查业务错误码
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取应付单列表信息 Tool函数 =====================
def u8_oughtpay_list_get_tool(input_data: GetOughtpayListInfoInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取应付单列表信息，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 应付单列表查询接口路径
    api_path = "/api/oughtpaylist/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应付单列表查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应付单列表查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "oughtpaylist": result.get("oughtpaylist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 审批一张应付单 Tool函数 =====================
def u8_oughtpay_verify_tool(input_data: VerifyOughtpayInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审批应付单
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 oughtpay）
    request_body: dict = {
        "oughtpay": input_data.model_dump(exclude_none=True)
    }

    # 应付单审批接口路径
    api_path = "/api/oughtpay/verify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应付单审批失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应付单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审一张应付单 Tool函数 =====================
def u8_oughtpay_unverify_tool(input_data: UnVerifyOughtpayInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审应付单
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 oughtpay）
    request_body: dict = {
        "oughtpay": input_data.model_dump(exclude_none=True)
    }

    # 应付单弃审接口路径
    api_path = "/api/oughtpay/unverify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应付单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应付单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)









# ===================== 新增一张应付单 Schema定义 =====================
U8_OUGHTPAY_ADD_SCHEMA = {
    "name": "u8_oughtpay_add",
    "description": "在用友U8 OpenAPI中新增应付单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 单据头参数
            "code": {"type": "string", "description": "应付单号"},
            "date": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）"},
            "state": {"type": "string", "description": "单据状态"},
            "bdebitcredit": {"type": "boolean", "description": "单据类型（0蓝单1红单，默认为0）"},
            "cust_vendor_code": {"type": "string", "description": "客商编码（必填）"},
            "cusname": {"type": "string", "description": "客商名称"},
            "cusabbname": {"type": "string", "description": "客商简称"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称"},
            "subjectcode": {"type": "string", "description": "科目编码"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "amount": {"type": "number", "description": "原币金额（必填）"},
            "natamount": {"type": "number", "description": "本币金额"},
            "digest": {"type": "string", "description": "摘要"},
            "item_classcode": {"type": "string", "description": "项目大类编码"},
            "item_code": {"type": "string", "description": "项目编码"},
            "currency_name": {"type": "string", "description": "币种"},
            "currency_rate": {"type": "number", "description": "汇率"},
            "paycondition_code": {"type": "string", "description": "付款条件"},
            "operator": {"type": "string", "description": "操作员姓名"},
            "flag": {"type": "string", "description": "标志"},
            "quantity": {"type": "number", "description": "数量"},
            "startflag": {"type": "boolean", "description": "期初标志"},
            "relatevouchercode": {"type": "string", "description": "对应单据号"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4（日期格式：yyyy-MM-dd）"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6（日期格式：yyyy-MM-dd）"},
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
                "description": "应付单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "cust_vendor_code": {"type": "string", "description": "客商编码（必填）"},
                        "cusabbname": {"type": "string", "description": "客商简称"},
                        "deptcode": {"type": "string", "description": "部门编码"},
                        "deptname": {"type": "string", "description": "部门名称"},
                        "personcode": {"type": "string", "description": "人员编码"},
                        "personname": {"type": "string", "description": "人员"},
                        "digest": {"type": "string", "description": "摘要"},
                        "subjectcode": {"type": "string", "description": "科目编码"},
                        "currency_name": {"type": "string", "description": "币种"},
                        "currency_rate": {"type": "number", "description": "汇率"},
                        "natamount": {"type": "number", "description": "本币金额（必填）"},
                        "amount": {"type": "number", "description": "原币金额"},
                        "bdebitcredit": {"type": "boolean", "description": "借贷方向（0贷1借，默认为1）"},
                        "item_classcode": {"type": "string", "description": "项目大类编码"},
                        "item_code": {"type": "string", "description": "项目编码"},
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
                        "define36": {"type": "string", "description": "单据体自定义项15（日期格式：yyyy-MM-dd）"},
                        "define37": {"type": "string", "description": "单据体自定义项16（日期格式：yyyy-MM-dd）"}
                    },
                    "required": ["cust_vendor_code", "natamount"]
                }
            }
        },
        "required": [
            "cust_vendor_code",
            "amount",
            "entry"
        ]
    }
}

# ===================== 获取单张应付单 Schema定义 =====================
U8_OUGHTPAY_GET_SCHEMA = {
    "name": "u8_oughtpay_get",
    "description": "通过单据编码获得应付单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "单据编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取应付单列表信息 Schema定义 =====================
U8_OUGHTPAY_LIST_GET_SCHEMA = {
    "name": "u8_oughtpay_list_get",
    "description": "在用友U8 OpenAPI中查询应付单列表信息，支持分页、单据号/日期范围、客商/部门/业务员等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始应收付单据号"},
            "code_end": {"type": "string", "description": "结束应收付单据号"},
            "state": {"type": "string", "description": "状态"},
            "date_begin": {"type": "string", "description": "起始单据日期"},
            "date_end": {"type": "string", "description": "结束单据日期"},
            "cust_vendor_code": {"type": "string", "description": "客商编号"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "personcode": {"type": "string", "description": "职员编码"},
            "personname": {"type": "string", "description": "职员名称关键字"},
            "item_code": {"type": "string", "description": "项目编码"},
            "digest": {"type": "string", "description": "摘要"},
            "operator": {"type": "string", "description": "操作员姓名关键字"},
            "define1": {"type": "string", "description": "表头自定义项1"},
            "define2": {"type": "string", "description": "表头自定义项2"},
            "define3": {"type": "string", "description": "表头自定义项3"},
            "define4": {"type": "string", "description": "表头自定义项4"},
            "define5": {"type": "number", "description": "表头自定义项5"},
            "define6": {"type": "string", "description": "表头自定义项6"},
            "define7": {"type": "number", "description": "表头自定义项7"},
            "define8": {"type": "string", "description": "表头自定义项8"},
            "define9": {"type": "string", "description": "表头自定义项9"},
            "define10": {"type": "string", "description": "表头自定义项10"},
            "define11": {"type": "string", "description": "表头自定义项11"},
            "define12": {"type": "string", "description": "表头自定义项12"},
            "define13": {"type": "string", "description": "表头自定义项13"},
            "define14": {"type": "string", "description": "表头自定义项14"},
            "define15": {"type": "number", "description": "表头自定义项15"},
            "define16": {"type": "number", "description": "表头自定义项16"},
            "ufts_begin": {"type": "string", "description": "起始时间戳"},
            "ufts_end": {"type": "string", "description": "结束时间戳"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 审批一张应付单 Schema定义 =====================
U8_OUGHTPAY_VERIFY_SCHEMA = {
    "name": "u8_oughtpay_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审批应付单",
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

# ===================== 弃审一张应付单 Schema定义 =====================
U8_OUGHTPAY_UNVERIFY_SCHEMA = {
    "name": "u8_oughtpay_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审应付单",
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











# ===================== 应收单通用数据模型 (复用于新增和查询) =====================

class OughtreceiveEntry(BaseModel):
    """
    应收单体模型。
    注意：所有字段均为 Optional，以兼容查询返回（可能缺省）和新增传入（后端校验必填）。
    """
    cust_vendor_code: Optional[str] = Field(None, description="客商编码")
    cusabbname: Optional[str] = Field(None, description="客商简称")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    digest: Optional[str] = Field(None, description="摘要")
    subjectcode: Optional[str] = Field(None, description="科目编码")
    currency_name: Optional[str] = Field(None, description="币种")
    currency_rate: Optional[float] = Field(None, description="汇率")
    natamount: Optional[float] = Field(None, description="本币金额")
    amount: Optional[float] = Field(None, description="原币金额")
    bdebitcredit: Optional[bool] = Field(None, description="借贷方向（0贷1借，默认为0）")
    item_classcode: Optional[str] = Field(None, description="项目大类编码")
    item_code: Optional[str] = Field(None, description="项目编码")
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

class OughtreceiveInfo(BaseModel):
    """
    应收单主表模型 (通用，用于新增输入和查询返回)
    """
    code: Optional[str] = Field(None, description="应收单号")
    date: Optional[str] = Field(None, description="单据日期（格式：yyyy-MM-dd）")
    state: Optional[str] = Field(None, description="单据状态")
    bdebitcredit: Optional[bool] = Field(None, description="单据类型（0红单1蓝单，默认为1）")
    cust_vendor_code: Optional[str] = Field(None, description="客商编码")
    cusname: Optional[str] = Field(None, description="客商名称")
    cusabbname: Optional[str] = Field(None, description="客商简称")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称")
    personcode: Optional[str] = Field(None, description="人员编码")
    personname: Optional[str] = Field(None, description="人员")
    amount: Optional[float] = Field(None, description="原币金额")
    natamount: Optional[float] = Field(None, description="本币金额")
    digest: Optional[str] = Field(None, description="摘要")
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
    entry: Optional[List[OughtreceiveEntry]] = Field(None, description="应收单体列表")

# ===================== 新增一张应收单 数据模型 =====================
class AddOughtreceiveInfoInput(OughtreceiveInfo):
    """新增应收单输入模型，继承自通用模型，可在此处强化必填项校验"""
    cust_vendor_code: str = Field(..., description="客商编码（必填）")
    amount: float = Field(..., description="原币金额（必填）")
    entry: List[OughtreceiveEntry] = Field(..., description="应收单体列表（必填）")

# ===================== 获取单张应收单 数据模型 =====================
class GetOughtreceiveInfoInput(BaseModel):
    id: str = Field(..., description="单据编号，用于查询应收单单据信息")

# ===================== 获取应收单列表信息 数据模型 =====================
class GetOughtreceiveListInfoInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始单据编号")
    code_end: Optional[str] = Field(None, description="结束单据编号")
    date_begin: Optional[str] = Field(None, description="起始制单日期，格式:yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束制单日期，格式:yyyy-MM-dd")
    state: Optional[str] = Field(None, description="订单状态")
    cust_vendor_code: Optional[str] = Field(None, description="客户或供应商编码")
    cusname: Optional[str] = Field(None, description="客户或供应商名称关键字")
    personcode: Optional[str] = Field(None, description="业务员编码")
    personname: Optional[str] = Field(None, description="业务员名称关键字")
    deptcode: Optional[str] = Field(None, description="部门编码")
    deptname: Optional[str] = Field(None, description="部门名称关键字")
    digest: Optional[str] = Field(None, description="摘要关键字")

# ===================== 审批一张应收单 数据模型 =====================
class VerifyOughtreceiveInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审一张应收单 数据模型 =====================
class UnVerifyOughtreceiveInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")




# ===================== 新增一张应收单 Tool函数 =====================
def u8_oughtreceive_add_tool(input_data: AddOughtreceiveInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中新增应收单，包含单据头和单据体（entry）完整信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 oughtreceive）
    request_body: dict = {
        "oughtreceive": input_data.model_dump(exclude_none=True)
    }

    # 应收单添加接口路径
    api_path = "/api/oughtreceive/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应收单新增失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应收单新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单张应收单 Tool函数 =====================
def u8_oughtreceive_get_tool(input_data: GetOughtreceiveInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过单据编号获取用友U8中的应收单单据信息。
    """
    params = {
        "id": input_data.id
    }

    # 应收单查询接口路径
    api_path = "/api/oughtreceive/get"

    try:
        # 使用 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)

        # 检查业务错误码
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 获取应收单列表信息 Tool函数 =====================
def u8_oughtreceive_list_get_tool(input_data: GetOughtreceiveListInfoInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取应收单列表信息，支持多条件筛选和分页查询。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 应收单列表查询接口路径
    api_path = "/api/oughtreceivelist/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应收单列表查询失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应收单列表查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "oughtreceivelist": result.get("oughtreceivelist", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 审批一张应收单 Tool函数 =====================
def u8_oughtreceive_verify_tool(input_data: VerifyOughtreceiveInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中审批应收单
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 oughtreceive）
    request_body: dict = {
        "oughtreceive": input_data.model_dump(exclude_none=True)
    }

    # 应收单审批接口路径
    api_path = "/api/oughtreceive/verify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应收单审批失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应收单审批成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 弃审一张应收单 Tool函数 =====================
def u8_oughtreceive_unverify_tool(input_data: UnVerifyOughtreceiveInfoInput, client: U8OpenAPIClient) -> str:
    """
    在用友U8系统中弃审应收单
    """

    # 构造接口要求的标准 JSON 结构（外层包一层 oughtreceive）
    request_body: dict = {
        "oughtreceive": input_data.model_dump(exclude_none=True)
    }

    # 应收单弃审接口路径
    api_path = "/api/oughtreceive/unverify"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True, is_user_login_v2=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "应收单弃审失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "应收单弃审成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)










# ===================== 新增一张应收单 Schema定义 =====================
U8_OUGHTRECEIVE_ADD_SCHEMA = {
    "name": "u8_oughtreceive_add",
    "description": "在用友U8 OpenAPI中新增应收单，支持单据头、单据体（entry）完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 单据头参数
            "code": {"type": "string", "description": "应收单号"},
            "date": {"type": "string", "description": "单据日期（格式：yyyy-MM-dd）"},
            "state": {"type": "string", "description": "单据状态"},
            "bdebitcredit": {"type": "boolean", "description": "单据类型（0红单1蓝单，默认为1）"},
            "cust_vendor_code": {"type": "string", "description": "客商编码（必填）"},
            "cusname": {"type": "string", "description": "客商名称"},
            "cusabbname": {"type": "string", "description": "客商简称"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称"},
            "personcode": {"type": "string", "description": "人员编码"},
            "personname": {"type": "string", "description": "人员"},
            "amount": {"type": "number", "description": "原币金额（必填）"},
            "natamount": {"type": "number", "description": "本币金额"},
            "digest": {"type": "string", "description": "摘要"},
            # 单据头自定义项 1-16
            "define1": {"type": "string", "description": "单据头自定义项1"},
            "define2": {"type": "string", "description": "单据头自定义项2"},
            "define3": {"type": "string", "description": "单据头自定义项3"},
            "define4": {"type": "string", "description": "单据头自定义项4（日期格式：yyyy-MM-dd）"},
            "define5": {"type": "number", "description": "单据头自定义项5"},
            "define6": {"type": "string", "description": "单据头自定义项6（日期格式：yyyy-MM-dd）"},
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
                "description": "应收单体列表（必填）",
                "items": {
                    "type": "object",
                    "properties": {
                        "cust_vendor_code": {"type": "string", "description": "客商编码（必填）"},
                        "cusabbname": {"type": "string", "description": "客商简称"},
                        "deptcode": {"type": "string", "description": "部门编码"},
                        "deptname": {"type": "string", "description": "部门名称"},
                        "personcode": {"type": "string", "description": "人员编码"},
                        "personname": {"type": "string", "description": "人员"},
                        "digest": {"type": "string", "description": "摘要"},
                        "subjectcode": {"type": "string", "description": "科目编码"},
                        "currency_name": {"type": "string", "description": "币种"},
                        "currency_rate": {"type": "number", "description": "汇率"},
                        "natamount": {"type": "number", "description": "本币金额（必填）"},
                        "amount": {"type": "number", "description": "原币金额"},
                        "bdebitcredit": {"type": "boolean", "description": "借贷方向（0贷1借，默认为0）"},
                        "item_classcode": {"type": "string", "description": "项目大类编码"},
                        "item_code": {"type": "string", "description": "项目编码"},
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
                        "define36": {"type": "string", "description": "单据体自定义项15（日期格式：yyyy-MM-dd）"},
                        "define37": {"type": "string", "description": "单据体自定义项16（日期格式：yyyy-MM-dd）"}
                    },
                    "required": ["cust_vendor_code", "natamount"]
                }
            }
        },
        "required": [
            "cust_vendor_code",
            "amount",
            "entry"
        ]
    }
}

# ===================== 获取单张应收单 Schema定义 =====================
U8_OUGHTRECEIVE_GET_SCHEMA = {
    "name": "u8_oughtreceive_get",
    "description": "通过单据编号获得应收单",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "单据编号"
            }
        },
        "required": ["id"]
    }
}

# ===================== 获取应收单列表信息 Schema定义 =====================
U8_OUGHTRECEIVE_LIST_GET_SCHEMA = {
    "name": "u8_oughtreceive_list_get",
    "description": "在用友U8 OpenAPI中查询应收单列表信息，支持分页、单据号/日期范围、客商/部门/业务员等多条件筛选",
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
            "state": {"type": "string", "description": "订单状态"},
            "cust_vendor_code": {"type": "string", "description": "客户或供应商编码"},
            "cusname": {"type": "string", "description": "客户或供应商名称关键字"},
            "personcode": {"type": "string", "description": "业务员编码"},
            "personname": {"type": "string", "description": "业务员名称关键字"},
            "deptcode": {"type": "string", "description": "部门编码"},
            "deptname": {"type": "string", "description": "部门名称关键字"},
            "digest": {"type": "string", "description": "摘要关键字"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# ===================== 审批一张应收单 Schema定义 =====================
U8_OUGHTRECEIVE_VERIFY_SCHEMA = {
    "name": "u8_oughtreceive_verify",
    "description": "在用友U8 OpenAPI中通过单据编号审批应收单",
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

# ===================== 弃审一张应收单 Schema定义 =====================
U8_OUGHTRECEIVE_UNVERIFY_SCHEMA = {
    "name": "u8_oughtreceive_unverify",
    "description": "在用友U8 OpenAPI中通过单据编号弃审应收单",
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








