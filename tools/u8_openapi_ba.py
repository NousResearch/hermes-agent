import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# region U8帐套

# ===================== 获取单个U8帐套信息 数据模型 =====================
class GetAccountInput(BaseModel):
    """获取单个U8帐套信息输入模型"""
    id: str = Field(..., description="帐套号")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取U8帐套信息 数据模型 =====================
class GetAccountBatchInput(BaseModel):
    """批量获取U8帐套信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始帐套号")
    code_end: Optional[str] = Field(None, description="结束帐套号")
    name: Optional[str] = Field(None, description="帐套名关键字")


# ===================== 获取单个U8帐套信息 Tool函数 =====================
def u8_account_get_tool(input_data: GetAccountInput, client: U8OpenAPIClient) -> str:
    """
    获取单个U8帐套信息
    适用场景: 多组织应用下，获取所有组织数据源
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/account/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取U8帐套信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取U8帐套信息成功",
            "data": {
                "account": result.get("account")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取U8帐套信息 Tool函数 =====================
def u8_account_batch_get_tool(input_data: GetAccountBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取U8帐套信息
    适用场景: 多组织应用下，获取所有组织数据源
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/account/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取U8帐套信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取U8帐套信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "accounts": result.get("account")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)



# ===================== 获取单个U8帐套信息 Schema定义 =====================
U8_ACCOUNT_GET_SCHEMA = {
    "name": "u8_account_get",
    "description": "获取单个U8帐套信息。适用场景: 多组织应用下，获取所有组织数据源。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "帐套号"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取U8帐套信息 Schema定义 =====================
U8_ACCOUNT_BATCH_GET_SCHEMA = {
    "name": "u8_account_batch_get",
    "description": "批量获取U8帐套信息。适用场景: 多组织应用下，获取所有组织数据源。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始帐套号"},
            "code_end": {"type": "string", "description": "结束帐套号"},
            "name": {"type": "string", "description": "帐套名关键字"}
        },
        "required": []
    }
}

# endregion

# region 交易分类

# ===================== 获取单个交易分类信息 数据模型 =====================
class GetPayunitclassInput(BaseModel):
    """获取单个交易分类信息输入模型"""
    id: str = Field(..., description="交易方分类编号")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取交易分类信息 数据模型 =====================
class GetPayunitclassBatchInput(BaseModel):
    """批量获取交易分类信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")

# ===================== 获取单个交易分类信息 Tool函数 =====================
def u8_payunitclass_get_tool(input_data: GetPayunitclassInput, client: U8OpenAPIClient) -> str:
    """
    获取单个交易分类信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payunitclass/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取交易分类信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取交易分类信息成功",
            "data": {
                "payunitclass": result.get("payunitclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取交易分类信息 Tool函数 =====================
def u8_payunitclass_batch_get_tool(input_data: GetPayunitclassBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取交易分类信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payunitclass/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取交易分类信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取交易分类信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "payunitclasss": result.get("payunitclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个交易分类信息 Schema定义 =====================
U8_PAYUNITCLASS_GET_SCHEMA = {
    "name": "u8_payunitclass_get",
    "description": "获取单个交易分类信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "交易方分类编号"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取交易分类信息 Schema定义 =====================
U8_PAYUNITCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_payunitclass_batch_get",
    "description": "批量获取交易分类信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"}
        },
        "required": []
    }
}



# endregion

# region 交易单位

# ===================== 获取单个交易单位信息 数据模型 =====================
class GetPayunitInput(BaseModel):
    """获取单个交易单位信息输入模型"""
    id: str = Field(..., description="交易方编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取交易单位信息 数据模型 =====================
class GetPayunitBatchInput(BaseModel):
    """批量获取交易单位信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")


# ===================== 获取单个交易单位信息 Tool函数 =====================
def u8_payunit_get_tool(input_data: GetPayunitInput, client: U8OpenAPIClient) -> str:
    """
    获取单个交易单位信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payunit/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取交易单位信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取交易单位信息成功",
            "data": {
                "payunit": result.get("payunit")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取交易单位信息 Tool函数 =====================
def u8_payunit_batch_get_tool(input_data: GetPayunitBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取交易单位信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/payunit/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取交易单位信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取交易单位信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "payunits": result.get("payunit")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)



# ===================== 获取单个交易单位信息 Schema定义 =====================
U8_PAYUNIT_GET_SCHEMA = {
    "name": "u8_payunit_get",
    "description": "获取单个交易单位信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "交易方编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取交易单位信息 Schema定义 =====================
U8_PAYUNIT_BATCH_GET_SCHEMA = {
    "name": "u8_payunit_batch_get",
    "description": "批量获取交易单位信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"}
        },
        "required": []
    }
}


# endregion

# region 人员

# =============================================================================
# 数据模型 - Data Models
# =============================================================================

# ===================== 获取单个人员信息 数据模型 =====================
class GetPersonInput(BaseModel):
    """获取单个人员信息输入模型"""
    id: str = Field(..., description="人员编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取人员信息 数据模型 =====================
class GetPersonBatchInput(BaseModel):
    """批量获取人员信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")
    user_id: Optional[str] = Field(None, description="操作员编码")
    property: Optional[str] = Field(None, description="属性")
    timestamp_begin: Optional[str] = Field(None, description="起始时间戳")
    timestamp_end: Optional[str] = Field(None, description="结束时间戳")


# ===================== 修改人员 数据模型 =====================
class EditPersonInput(BaseModel):
    """修改人员输入模型"""
    code: str = Field(..., description="人员编码")
    name: str = Field(..., description="人员名称")
    cdept_num: str = Field(..., description="部门编码")
    cdept_name: str = Field(..., description="部门名称")
    rsex: str = Field(..., description="人员性别（1:男 2:女）")
    rpersontype: int = Field(..., description="人员类型")
    rIDType: int = Field(..., description="证件类型（0：身份证 1：护照 2：军人证 3：港澳身份证 4：台胞证 9：其他）")
    rEmployState: int = Field(..., description="雇佣状态（10：在职 20：离退 30：离职）")
    bpsnperson: str = Field(..., description="0=非业务员；1=业务员")
    cpsnproperty: Optional[str] = Field(None, description="人员属性")
    cpsnmobilephone: Optional[str] = Field(None, description="人员手机号")
    cpsnemail: Optional[str] = Field(None, description="人员邮箱")
    cjobcode: Optional[str] = Field(None, description="职位编码")
    vjobname: Optional[str] = Field(None, description="职位名称")
    cpsnpostaddr: Optional[str] = Field(None, description="通讯地址")
    rpersontypename: Optional[str] = Field(None, description="人员类型名称")
    cpsnqqcode: Optional[int] = Field(None, description="QQ号")
    vIDNo: Optional[str] = Field(None, description="证件号码")


# ===================== 新增人员 数据模型 =====================
class AddPersonInput(BaseModel):
    """新增人员输入模型"""
    code: str = Field(..., description="人员编码")
    name: str = Field(..., description="人员名称")
    cdept_num: str = Field(..., description="部门编码")
    cdept_name: str = Field(..., description="部门名称")
    rsex: str = Field(..., description="人员性别(1:男 2:女)")
    rIDType: int = Field(..., description="证件类型(0:身份证 1:护照 2:军人证 3:港澳身份证 4:台胞证 9:其他)")
    rEmployState: int = Field(..., description="雇佣状态(10:在职 20:离退 30:离职)")
    cpsnproperty: Optional[str] = Field(None, description="人员属性")
    cpsnmobilephone: Optional[str] = Field(None, description="人员手机号")
    cpsnemail: Optional[str] = Field(None, description="人员邮箱")
    cjobcode: Optional[str] = Field(None, description="职位编码")
    vjobname: Optional[str] = Field(None, description="职位名称")
    cpsnpostaddr: Optional[str] = Field(None, description="通讯地址")
    rpersontype: Optional[int] = Field(None, description="人员类型")
    rpersontypename: Optional[str] = Field(None, description="人员类型名称")
    cpsnqqcode: Optional[int] = Field(None, description="QQ号")
    vIDNo: Optional[str] = Field(None, description="证件号码")
    bpsnperson: Optional[str] = Field(None, description="0=非业务员;1=业务员")
    cdepcode: Optional[str] = Field(None, description="业务或费用部门")


# =============================================================================
# Tool函数 - Tool Functions
# =============================================================================

# ===================== 获取单个人员信息 Tool函数 =====================
def u8_person_get_tool(input_data: GetPersonInput, client: U8OpenAPIClient) -> str:
    """
    获取单个人员信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/person/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取人员信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取人员信息成功",
            "data": {
                "person": result.get("person")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取人员信息 Tool函数 =====================
def u8_person_batch_get_tool(input_data: GetPersonBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取人员信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/person/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取人员信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取人员信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "persons": result.get("person")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 修改人员 Tool函数 =====================
def u8_person_edit_tool(input_data: EditPersonInput, client: U8OpenAPIClient) -> str:
    """
    修改人员
    适用场景: 基础数据同步或前端展示
    """
    params = {"person": input_data.model_dump(exclude_none=True)}
    api_path = "/api/person/edit"
    
    try:
        result = client.request_api("POST", api_path, body=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "修改人员失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "修改人员成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增人员 Tool函数 =====================
def u8_person_add_tool(input_data: AddPersonInput, client: U8OpenAPIClient) -> str:
    """
    新增人员
    适用场景: 基础数据同步或前端展示
    """
    params = {"person": input_data.model_dump(exclude_none=True)}
    api_path = "/api/person/add"
    
    try:
        result = client.request_api("POST", api_path, body=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增人员失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "新增人员成功",
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


# =============================================================================
# Schema定义 - Schema Definitions
# =============================================================================

# ===================== 获取单个人员信息 Schema定义 =====================
U8_PERSON_GET_SCHEMA = {
    "name": "u8_person_get",
    "description": "获取单个人员信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "人员编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取人员信息 Schema定义 =====================
U8_PERSON_BATCH_GET_SCHEMA = {
    "name": "u8_person_batch_get",
    "description": "批量获取人员信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"},
            "user_id": {"type": "string", "description": "操作员编码"},
            "property": {"type": "string", "description": "属性"},
            "timestamp_begin": {"type": "string", "description": "起始时间戳"},
            "timestamp_end": {"type": "string", "description": "结束时间戳"}
        },
        "required": []
    }
}

# ===================== 修改人员 Schema定义 =====================
U8_PERSON_EDIT_SCHEMA = {
    "name": "u8_person_edit",
    "description": "修改人员。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "人员编码"},
            "name": {"type": "string", "description": "人员名称"},
            "cdept_num": {"type": "string", "description": "部门编码"},
            "cdept_name": {"type": "string", "description": "部门名称"},
            "rsex": {"type": "string", "description": "人员性别（1:男 2:女）"},
            "rpersontype": {"type": "integer", "description": "人员类型"},
            "rIDType": {"type": "integer", "description": "证件类型（0：身份证 1：护照 2：军人证 3：港澳身份证 4：台胞证 9：其他）"},
            "rEmployState": {"type": "integer", "description": "雇佣状态（10：在职 20：离退 30：离职）"},
            "bpsnperson": {"type": "string", "description": "0=非业务员；1=业务员"},
            "cpsnproperty": {"type": "string", "description": "人员属性"},
            "cpsnmobilephone": {"type": "string", "description": "人员手机号"},
            "cpsnemail": {"type": "string", "description": "人员邮箱"},
            "cjobcode": {"type": "string", "description": "职位编码"},
            "vjobname": {"type": "string", "description": "职位名称"},
            "cpsnpostaddr": {"type": "string", "description": "通讯地址"},
            "rpersontypename": {"type": "string", "description": "人员类型名称"},
            "cpsnqqcode": {"type": "integer", "description": "QQ号"},
            "vIDNo": {"type": "string", "description": "证件号码"}
        },
        "required": ["code", "name", "cdept_num", "cdept_name", "rsex", "rpersontype", "rIDType", "rEmployState", "bpsnperson"]
    }
}

# ===================== 新增人员 Schema定义 =====================
U8_PERSON_ADD_SCHEMA = {
    "name": "u8_person_add",
    "description": "新增人员。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "人员编码"},
            "name": {"type": "string", "description": "人员名称"},
            "cdept_num": {"type": "string", "description": "部门编码"},
            "cdept_name": {"type": "string", "description": "部门名称"},
            "rsex": {"type": "string", "description": "人员性别(1:男 2:女)"},
            "rIDType": {"type": "integer", "description": "证件类型(0:身份证 1:护照 2:军人证 3:港澳身份证 4:台胞证 9:其他)"},
            "rEmployState": {"type": "integer", "description": "雇佣状态(10:在职 20:离退 30:离职)"},
            "cpsnproperty": {"type": "string", "description": "人员属性"},
            "cpsnmobilephone": {"type": "string", "description": "人员手机号"},
            "cpsnemail": {"type": "string", "description": "人员邮箱"},
            "cjobcode": {"type": "string", "description": "职位编码"},
            "vjobname": {"type": "string", "description": "职位名称"},
            "cpsnpostaddr": {"type": "string", "description": "通讯地址"},
            "rpersontype": {"type": "integer", "description": "人员类型"},
            "rpersontypename": {"type": "string", "description": "人员类型名称"},
            "cpsnqqcode": {"type": "integer", "description": "QQ号"},
            "vIDNo": {"type": "string", "description": "证件号码"},
            "bpsnperson": {"type": "string", "description": "0=非业务员;1=业务员"},
            "cdepcode": {"type": "string", "description": "业务或费用部门"}
        },
        "required": ["code", "name", "cdept_num", "cdept_name", "rsex", "rIDType", "rEmployState"]
    }
}


# endregion

# region 人员类别


# =============================================================================
# 数据模型 - Data Models
# =============================================================================

# ===================== 获取单个人员类别 数据模型 =====================
class GetPersontypeInput(BaseModel):
    """获取单个人员类别输入模型"""
    id: str = Field(..., description="人员类别编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取人员类别 数据模型 =====================
class GetPersontypeBatchInput(BaseModel):
    """批量获取人员类别输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始人员类别编码")
    code_end: Optional[str] = Field(None, description="结束人员类别编码")
    name: Optional[str] = Field(None, description="人员类别名称")
    levels: Optional[int] = Field(None, description="级别")
    pcodeid: Optional[str] = Field(None, description="上级编码")
    childflag: Optional[bool] = Field(None, description="是否有下级")
    memo: Optional[str] = Field(None, description="备注关键字")


# =============================================================================
# Tool函数 - Tool Functions
# =============================================================================

# ===================== 获取单个人员类别 Tool函数 =====================
def u8_persontype_get_tool(input_data: GetPersontypeInput, client: U8OpenAPIClient) -> str:
    """
    获取单个人员类别
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/persontype/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取人员类别失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取人员类别成功",
            "data": {
                "persontype": result.get("persontype")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取人员类别 Tool函数 =====================
def u8_persontype_batch_get_tool(input_data: GetPersontypeBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取人员类别
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/persontype/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取人员类别失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取人员类别成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "persontypes": result.get("persontype")
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

# ===================== 获取单个人员类别 Schema定义 =====================
U8_PERSONTYPE_GET_SCHEMA = {
    "name": "u8_persontype_get",
    "description": "获取单个人员类别。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "人员类别编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取人员类别 Schema定义 =====================
U8_PERSONTYPE_BATCH_GET_SCHEMA = {
    "name": "u8_persontype_batch_get",
    "description": "批量获取人员类别。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始人员类别编码"},
            "code_end": {"type": "string", "description": "结束人员类别编码"},
            "name": {"type": "string", "description": "人员类别名称"},
            "levels": {"type": "integer", "description": "级别"},
            "pcodeid": {"type": "string", "description": "上级编码"},
            "childflag": {"type": "boolean", "description": "是否有下级"},
            "memo": {"type": "string", "description": "备注关键字"}
        },
        "required": []
    }
}



# endregion

# region 仓库
# =============================================================================
# 数据模型 - Data Models
# =============================================================================

# ===================== 获取单个仓库信息 数据模型 =====================
class GetWarehouseInput(BaseModel):
    """获取单个仓库信息输入模型"""
    id: str = Field(..., description="仓库编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取仓库信息 数据模型 =====================
class GetWarehouseBatchInput(BaseModel):
    """批量获取仓库信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cwhcode_begin: Optional[str] = Field(None, description="起始仓库编码")
    cwhcode_end: Optional[str] = Field(None, description="结束仓库编码")
    cwhname: Optional[str] = Field(None, description="仓库名称关键字")
    cwhcode: Optional[str] = Field(None, description="仓库编码")
    code_begin: Optional[str] = Field(None, description="起始code")
    code_end: Optional[str] = Field(None, description="结束code")
    name: Optional[str] = Field(None, description="name")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    page_index: Optional[str] = Field(None, description="第几页")


# ===================== 新增仓库 数据模型 =====================
class AddWarehouseInput(BaseModel):
    """新增仓库输入模型"""
    code: str = Field(..., description="仓库编码")
    name: str = Field(..., description="仓库名称")


# =============================================================================
# Tool函数 - Tool Functions
# =============================================================================

# ===================== 获取单个仓库信息 Tool函数 =====================
def u8_warehouse_get_tool(input_data: GetWarehouseInput, client: U8OpenAPIClient) -> str:
    """
    获取单个仓库信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/warehouse/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取仓库信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取仓库信息成功",
            "data": {
                "warehouse": result.get("warehouse")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取仓库信息 Tool函数 =====================
def u8_warehouse_batch_get_tool(input_data: GetWarehouseBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取仓库信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/warehouse/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取仓库信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取仓库信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "warehouses": result.get("warehouse")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增仓库 Tool函数 =====================
def u8_warehouse_add_tool(input_data: AddWarehouseInput, client: U8OpenAPIClient) -> str:
    """
    新增仓库
    适用场景: 基础数据同步或前端展示
    """
    params = {"warehouse": input_data.model_dump(exclude_none=True)}
    api_path = "/api/warehouse/add"
    
    try:
        result = client.request_api("POST", api_path, body=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增仓库失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "新增仓库成功",
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


# =============================================================================
# Schema定义 - Schema Definitions
# =============================================================================

# ===================== 获取单个仓库信息 Schema定义 =====================
U8_WAREHOUSE_GET_SCHEMA = {
    "name": "u8_warehouse_get",
    "description": "获取单个仓库信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "仓库编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取仓库信息 Schema定义 =====================
U8_WAREHOUSE_BATCH_GET_SCHEMA = {
    "name": "u8_warehouse_batch_get",
    "description": "批量获取仓库信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cwhcode_begin": {"type": "string", "description": "起始仓库编码"},
            "cwhcode_end": {"type": "string", "description": "结束仓库编码"},
            "cwhname": {"type": "string", "description": "仓库名称关键字"},
            "cwhcode": {"type": "string", "description": "仓库编码"},
            "code_begin": {"type": "string", "description": "起始code"},
            "code_end": {"type": "string", "description": "结束code"},
            "name": {"type": "string", "description": "name"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "page_index": {"type": "string", "description": "第几页"}
        },
        "required": []
    }
}

# ===================== 新增仓库 Schema定义 =====================
U8_WAREHOUSE_ADD_SCHEMA = {
    "name": "u8_warehouse_add",
    "description": "新增仓库。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "仓库编码"},
            "name": {"type": "string", "description": "仓库名称"}
        },
        "required": ["code", "name"]
    }
}

# endregion

# region 会计期间

# ===================== 批量获取会计期间 数据模型 =====================
class GetPeriodBatchInput(BaseModel):
    """批量获取会计期间输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cacc_id: Optional[str] = Field(None, description="账套号")
    iyear: Optional[int] = Field(None, description="账簿年度")
    iid: Optional[int] = Field(None, description="会计期间")
    dbegin: Optional[str] = Field(None, description="开始日期")
    dend: Optional[str] = Field(None, description="结束日期")




# ===================== 批量获取会计期间 Tool函数 =====================
def u8_period_batch_get_tool(input_data: GetPeriodBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取会计期间，支持按账套号、年度、期间等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 会计期间接口路径
    api_path = "/api/period/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取会计期间失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取会计期间成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "period": result.get("period")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 批量获取会计期间 Schema定义 =====================
U8_PERIOD_BATCH_GET_SCHEMA = {
    "name": "u8_period_batch_get",
    "description": "批量获取会计期间，支持按账套号、年度、会计期间、日期范围等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cacc_id": {"type": "string", "description": "账套号"},
            "iyear": {"type": "number", "description": "账簿年度"},
            "iid": {"type": "number", "description": "会计期间"},
            "dbegin": {"type": "string", "description": "开始日期"},
            "dend": {"type": "string", "description": "结束日期"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}

# endregion

# region 供应商
# =============================================================================
# 数据模型 - Data Models
# =============================================================================

# ===================== 获取单个供应商信息 数据模型 =====================
class GetVendorInput(BaseModel):
    """获取单个供应商信息输入模型"""
    id: str = Field(..., description="供应商编码")
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")


# ===================== 批量获取供应商信息 数据模型 =====================
class GetVendorBatchInput(BaseModel):
    """批量获取供应商信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")
    timestamp_begin: Optional[str] = Field(None, description="起始时间戳")
    timestamp_end: Optional[str] = Field(None, description="结束时间戳")


# ===================== 修改供应商 数据模型 =====================
class EditVendorInput(BaseModel):
    """修改供应商输入模型"""
    code: str = Field(..., description="供应商编码")
    name: str = Field(..., description="供应商名称")
    abbrname: Optional[str] = Field(None, description="供应商简称，省略取供应商名称")
    sort_code: Optional[str] = Field(None, description="所属分类码")
    industry: Optional[str] = Field(None, description="所属行业")
    address: Optional[str] = Field(None, description="地址")
    bank_open: Optional[str] = Field(None, description="开户银行")
    bank_acc_number: Optional[str] = Field(None, description="银行帐号")
    phone: Optional[str] = Field(None, description="电话")
    fax: Optional[str] = Field(None, description="传真")
    email: Optional[str] = Field(None, description="Email地址")
    contact: Optional[str] = Field(None, description="联系人")
    mobile: Optional[str] = Field(None, description="手机")
    receive_site: Optional[str] = Field(None, description="到货地址")
    end_date: Optional[str] = Field(None, description="停用日期")
    bvencargo: Optional[int] = Field(None, description="是否采购(0=否;1=是)")
    bproxyforeign: Optional[int] = Field(None, description="是否委外(0=否;1=是)")
    bvenservice: Optional[int] = Field(None, description="是否服务(0=否;1=是)")
    memo: Optional[str] = Field(None, description="备注")


# ===================== 新增供应商 数据模型 =====================
class AddVendorInput(BaseModel):
    """新增供应商输入模型"""
    code: str = Field(..., description="供应商编码")
    name: str = Field(..., description="供应商名称")
    sort_code: str = Field(..., description="所属分类码")
    abbrname: Optional[str] = Field(None, description="供应商简称，省略取供应商名称")
    industry: Optional[str] = Field(None, description="所属行业")
    address: Optional[str] = Field(None, description="地址")
    bank_open: Optional[str] = Field(None, description="开户银行")
    bank_acc_number: Optional[str] = Field(None, description="银行帐号")
    phone: Optional[str] = Field(None, description="电话")
    fax: Optional[str] = Field(None, description="传真")
    email: Optional[str] = Field(None, description="Email地址")
    contact: Optional[str] = Field(None, description="联系人")
    mobile: Optional[str] = Field(None, description="手机")
    receive_site: Optional[str] = Field(None, description="到货地址")
    end_date: Optional[str] = Field(None, description="停用日期")
    tax_in_price_flag: Optional[int] = Field(None, description="单价是否含税(0=不含税;1=含税)")
    bvencargo: Optional[int] = Field(None, description="是否采购(0=否;1=是)")
    bproxyforeign: Optional[int] = Field(None, description="是否委外(0=否;1=是)")
    bvenservice: Optional[int] = Field(None, description="是否服务(0=否;1=是)")
    iventaxrate: Optional[float] = Field(None, description="税率")
    memo: Optional[str] = Field(None, description="备注")


# =============================================================================
# Tool函数 - Tool Functions
# =============================================================================

# ===================== 获取单个供应商信息 Tool函数 =====================
def u8_vendor_get_tool(input_data: GetVendorInput, client: U8OpenAPIClient) -> str:
    """
    获取单个供应商信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/vendor/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取供应商信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "获取供应商信息成功",
            "data": {
                "vendor": result.get("vendor")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取供应商信息 Tool函数 =====================
def u8_vendor_batch_get_tool(input_data: GetVendorBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取供应商信息
    适用场景: 基础数据同步或前端展示
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/vendor/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取供应商信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "批量获取供应商信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "vendors": result.get("vendor")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 修改供应商 Tool函数 =====================
def u8_vendor_edit_tool(input_data: EditVendorInput, client: U8OpenAPIClient) -> str:
    """
    修改供应商
    适用场景: 基础数据同步或前端展示
    """
    params = {"vendor": input_data.model_dump(exclude_none=True)}
    api_path = "/api/vendor/edit"
    
    try:
        result = client.request_api("POST", api_path, body=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "修改供应商失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "修改供应商成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增供应商 Tool函数 =====================
def u8_vendor_add_tool(input_data: AddVendorInput, client: U8OpenAPIClient) -> str:
    """
    新增供应商
    适用场景: 基础数据同步或前端展示
    """
    params = {"vendor": input_data.model_dump(exclude_none=True)}
    api_path = "/api/vendor/add"
    
    try:
        result = client.request_api("POST", api_path, body=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增供应商失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": True,
            "message": "新增供应商成功",
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


# =============================================================================
# Schema定义 - Schema Definitions
# =============================================================================

# ===================== 获取单个供应商信息 Schema定义 =====================
U8_VENDOR_GET_SCHEMA = {
    "name": "u8_vendor_get",
    "description": "获取单个供应商信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "供应商编码"},
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取供应商信息 Schema定义 =====================
U8_VENDOR_BATCH_GET_SCHEMA = {
    "name": "u8_vendor_batch_get",
    "description": "批量获取供应商信息。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"},
            "timestamp_begin": {"type": "string", "description": "起始时间戳"},
            "timestamp_end": {"type": "string", "description": "结束时间戳"}
        },
        "required": []
    }
}

# ===================== 修改供应商 Schema定义 =====================
U8_VENDOR_EDIT_SCHEMA = {
    "name": "u8_vendor_edit",
    "description": "修改供应商。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "供应商编码"},
            "name": {"type": "string", "description": "供应商名称"},
            "abbrname": {"type": "string", "description": "供应商简称，省略取供应商名称"},
            "sort_code": {"type": "string", "description": "所属分类码"},
            "industry": {"type": "string", "description": "所属行业"},
            "address": {"type": "string", "description": "地址"},
            "bank_open": {"type": "string", "description": "开户银行"},
            "bank_acc_number": {"type": "string", "description": "银行帐号"},
            "phone": {"type": "string", "description": "电话"},
            "fax": {"type": "string", "description": "传真"},
            "email": {"type": "string", "description": "Email地址"},
            "contact": {"type": "string", "description": "联系人"},
            "mobile": {"type": "string", "description": "手机"},
            "receive_site": {"type": "string", "description": "到货地址"},
            "end_date": {"type": "string", "description": "停用日期"},
            "bvencargo": {"type": "integer", "description": "是否采购(0=否;1=是)"},
            "bproxyforeign": {"type": "integer", "description": "是否委外(0=否;1=是)"},
            "bvenservice": {"type": "integer", "description": "是否服务(0=否;1=是)"},
            "memo": {"type": "string", "description": "备注"}
        },
        "required": ["code", "name"]
    }
}

# ===================== 新增供应商 Schema定义 =====================
U8_VENDOR_ADD_SCHEMA = {
    "name": "u8_vendor_add",
    "description": "新增供应商。适用场景: 基础数据同步或前端展示。",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "供应商编码"},
            "name": {"type": "string", "description": "供应商名称"},
            "sort_code": {"type": "string", "description": "所属分类码"},
            "abbrname": {"type": "string", "description": "供应商简称，省略取供应商名称"},
            "industry": {"type": "string", "description": "所属行业"},
            "address": {"type": "string", "description": "地址"},
            "bank_open": {"type": "string", "description": "开户银行"},
            "bank_acc_number": {"type": "string", "description": "银行帐号"},
            "phone": {"type": "string", "description": "电话"},
            "fax": {"type": "string", "description": "传真"},
            "email": {"type": "string", "description": "Email地址"},
            "contact": {"type": "string", "description": "联系人"},
            "mobile": {"type": "string", "description": "手机"},
            "receive_site": {"type": "string", "description": "到货地址"},
            "end_date": {"type": "string", "description": "停用日期"},
            "tax_in_price_flag": {"type": "integer", "description": "单价是否含税(0=不含税;1=含税)"},
            "bvencargo": {"type": "integer", "description": "是否采购(0=否;1=是)"},
            "bproxyforeign": {"type": "integer", "description": "是否委外(0=否;1=是)"},
            "bvenservice": {"type": "integer", "description": "是否服务(0=否;1=是)"},
            "iventaxrate": {"type": "number", "description": "税率"},
            "memo": {"type": "string", "description": "备注"}
        },
        "required": ["code", "name", "sort_code"]
    }
}

# endregion

# region 供应商分类

# ===================== 获取单个供应商分类 数据模型 =====================
class GetVendorclassInput(BaseModel):
    """获取单个供应商分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="供应商分类编码（必填）")


# ===================== 批量获取供应商分类 数据模型 =====================
class BatchGetVendorclassInput(BaseModel):
    """批量获取供应商分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")
    timestamp_begin: Optional[int] = Field(None, description="起始时间戳")
    timestamp_end: Optional[int] = Field(None, description="结束时间戳")


# ===================== 添加供应商分类 数据模型 =====================
class AddVendorclassInput(BaseModel):
    """添加供应商分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code: str = Field(..., description="供应商分类编码（必填）")
    name: str = Field(..., description="供应商分类名称（必填）")
    rank: Optional[int] = Field(None, description="供应商分类编码级次")
    end_rank_flag: Optional[bool] = Field(None, description="末级标志")
    biz_id: str = Field(..., description="上游id，需要保证biz_id与ERP主键唯一对应关系（必填）")
    sync: Optional[int] = Field(None, description="0=异步新增（默认）;1=同步新增")




# ===================== 获取单个供应商分类 Tool函数 =====================
def u8_vendorclass_get_tool(input_data: GetVendorclassInput, client: U8OpenAPIClient) -> str:
    """
    获取单个供应商分类信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 供应商分类接口路径
    api_path = "/api/vendorclass/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取供应商分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取供应商分类成功",
            "data": {
                "vendorclass": result.get("vendorclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取供应商分类 Tool函数 =====================
def u8_vendorclass_batch_get_tool(input_data: BatchGetVendorclassInput, client: U8OpenAPIClient) -> str:
    """
    批量获取供应商分类列表，支持分页、编码范围、名称等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 供应商分类接口路径
    api_path = "/api/vendorclass/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取供应商分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取供应商分类成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "vendorclass": result.get("vendorclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 添加供应商分类 Tool函数 =====================
def u8_vendorclass_add_tool(input_data: AddVendorclassInput, client: U8OpenAPIClient) -> str:
    """
    添加一个新供应商分类。
    """
    # 提取biz_id和sync参数
    params_data = input_data.model_dump(exclude_none=True)
    biz_id = params_data.pop("biz_id", None)
    sync = params_data.pop("sync", None)
    ds_sequence = params_data.pop("ds_sequence", None)

    # 构造接口要求的标准 JSON 结构（外层包一层 vendorclass）
    request_body: dict = {
        "vendorclass": params_data
    }

    # 构造URL参数
    inparams = {"biz_id": biz_id}
    if sync is not None:
        inparams["sync"] = sync
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    # 供应商分类接口路径
    api_path = "/api/vendorclass/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "添加供应商分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "添加供应商分类成功",
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




# ===================== 获取单个供应商分类 Schema定义 =====================
U8_VENDORCLASS_GET_SCHEMA = {
    "name": "u8_vendorclass_get",
    "description": "获取单个供应商分类信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "供应商分类编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取供应商分类 Schema定义 =====================
U8_VENDORCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_vendorclass_batch_get",
    "description": "批量获取供应商分类列表，支持分页、编码范围、名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"},
            "timestamp_begin": {"type": "number", "description": "起始时间戳"},
            "timestamp_end": {"type": "number", "description": "结束时间戳"}
        },
        "required": []
    }
}


# ===================== 添加供应商分类 Schema定义 =====================
U8_VENDORCLASS_ADD_SCHEMA = {
    "name": "u8_vendorclass_add",
    "description": "添加一个新供应商分类",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code": {"type": "string", "description": "供应商分类编码（必填）"},
            "name": {"type": "string", "description": "供应商分类名称（必填）"},
            "rank": {"type": "number", "description": "供应商分类编码级次"},
            "end_rank_flag": {"type": "boolean", "description": "末级标志"},
            "biz_id": {"type": "string", "description": "上游id，需要保证biz_id与ERP主键唯一对应关系（必填）"},
            "sync": {"type": "number", "description": "0=异步新增（默认）;1=同步新增"}
        },
        "required": ["code", "name", "biz_id"]
    }
}


# endregion

# region 供应商银行
# ===================== 获取单个供应商银行 数据模型 =====================
class GetVendorBankInput(BaseModel):
    """获取单个供应商银行输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="供应商编码（必填）")


# ===================== 批量获取供应商银行 数据模型 =====================
class BatchGetVendorBankInput(BaseModel):
    """批量获取供应商银行输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始供应商编码")
    code_end: Optional[str] = Field(None, description="结束供应商编码")
    cvenname: Optional[str] = Field(None, description="供应商名称关键字")
    cbank: Optional[str] = Field(None, description="所属银行")
    cbranch: Optional[str] = Field(None, description="开户银行")




# ===================== 获取单个供应商银行 Tool函数 =====================
def u8_vendor_bank_get_tool(input_data: GetVendorBankInput, client: U8OpenAPIClient) -> str:
    """
    获取单个供应商银行信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 供应商银行接口路径
    api_path = "/api/vendor_bank/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取供应商银行失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取供应商银行成功",
            "data": {
                "vendor_bank": result.get("vendor_bank")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取供应商银行 Tool函数 =====================
def u8_vendor_bank_batch_get_tool(input_data: BatchGetVendorBankInput, client: U8OpenAPIClient) -> str:
    """
    批量获取供应商银行列表，支持按供应商编码范围、名称、银行等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 供应商银行接口路径
    api_path = "/api/vendor_bank/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取供应商银行失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取供应商银行成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "vendor_bank": result.get("vendor_bank")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取单个供应商银行 Schema定义 =====================
U8_VENDOR_BANK_GET_SCHEMA = {
    "name": "u8_vendor_bank_get",
    "description": "获取单个供应商银行信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "供应商编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取供应商银行 Schema定义 =====================
U8_VENDOR_BANK_BATCH_GET_SCHEMA = {
    "name": "u8_vendor_bank_batch_get",
    "description": "批量获取供应商银行列表，支持按供应商编码范围、名称、银行等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始供应商编码"},
            "code_end": {"type": "string", "description": "结束供应商编码"},
            "cvenname": {"type": "string", "description": "供应商名称关键字"},
            "cbank": {"type": "string", "description": "所属银行"},
            "cbranch": {"type": "string", "description": "开户银行"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}


# endregion

# region 凭证类别

# ===================== 获取单个凭证类别 数据模型 =====================
class GetDsignInput(BaseModel):
    """获取单个凭证类别输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="凭证类别标识（必填）")


# ===================== 批量获取凭证类别 数据模型 =====================
class BatchGetDsignInput(BaseModel):
    """批量获取凭证类别输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    id_begin: Optional[str] = Field(None, description="起始凭证类别标识")
    id_end: Optional[str] = Field(None, description="结束凭证类别标识")
    type: Optional[str] = Field(None, description="凭证类别字关键字")
    type_name: Optional[str] = Field(None, description="凭证类别名称关键字")




# ===================== 获取单个凭证类别 Tool函数 =====================
def u8_dsign_get_tool(input_data: GetDsignInput, client: U8OpenAPIClient) -> str:
    """
    获取单个凭证类别信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 凭证类别接口路径
    api_path = "/api/dsign/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取凭证类别失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取凭证类别成功",
            "data": {
                "dsign": result.get("dsign")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取凭证类别 Tool函数 =====================
def u8_dsign_batch_get_tool(input_data: BatchGetDsignInput, client: U8OpenAPIClient) -> str:
    """
    批量获取凭证类别列表，支持分页、凭证类别标识范围、类别字、名称等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 凭证类别接口路径
    api_path = "/api/dsign/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取凭证类别失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取凭证类别成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "dsign": result.get("dsign")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取单个凭证类别 Schema定义 =====================
U8_DSIGN_GET_SCHEMA = {
    "name": "u8_dsign_get",
    "description": "获取单个凭证类别信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "凭证类别标识（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取凭证类别 Schema定义 =====================
U8_DSIGN_BATCH_GET_SCHEMA = {
    "name": "u8_dsign_batch_get",
    "description": "批量获取凭证类别列表，支持分页、凭证类别标识范围、类别字、名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "id_begin": {"type": "string", "description": "起始凭证类别标识"},
            "id_end": {"type": "string", "description": "结束凭证类别标识"},
            "type": {"type": "string", "description": "凭证类别字关键字"},
            "type_name": {"type": "string", "description": "凭证类别名称关键字"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}


# endregion

# region 发运方式

# # ===================== 批量获取发运方式 数据模型 =====================
# class BatchGetShippingchoiceInput(BaseModel):
#     """批量获取发运方式输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     page_index: Optional[str] = Field(None, description="页号")
#     rows_per_page: Optional[str] = Field(None, description="每页行数")
#     code_begin: Optional[str] = Field(None, description="起始编码")
#     code_end: Optional[str] = Field(None, description="结束编码")
#     name: Optional[str] = Field(None, description="名称关键字")




# # ===================== 批量获取发运方式 Tool函数 =====================
# def u8_shippingchoice_batch_get_tool(input_data: BatchGetShippingchoiceInput, client: U8OpenAPIClient) -> str:
#     """
#     发运方式批量查询，支持分页、编码范围、名称等条件筛选。
#     """
#     # 构造接口请求参数（仅传递非None的参数）
#     params = input_data.model_dump(exclude_none=True)

#     # 发运方式接口路径
#     api_path = "/api/shippingchoice/batch_get"

#     try:
#         # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
#         result = client.request_api("GET", api_path, inparams=params)

#         # 统一返回格式
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取发运方式失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取发运方式成功",
#             "data": {
#                 "page_index": result.get("page_index"),
#                 "rows_per_page": result.get("rows_per_page"),
#                 "row_count": result.get("row_count"),
#                 "page_count": result.get("page_count"),
#                 "shippingchoice": result.get("shippingchoice")
#             },
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)




# # ===================== 批量获取发运方式 Schema定义 =====================
# U8_SHIPPINGCHOICE_BATCH_GET_SCHEMA = {
#     "name": "u8_shippingchoice_batch_get",
#     "description": "发运方式批量查询，支持分页、编码范围、名称等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "page_index": {"type": "string", "description": "页号"},
#             "rows_per_page": {"type": "string", "description": "每页行数"},
#             "code_begin": {"type": "string", "description": "起始编码"},
#             "code_end": {"type": "string", "description": "结束编码"},
#             "name": {"type": "string", "description": "名称关键字"}
#         },
#         "required": []  # 所有参数均为可选，无必填项
#     }
# }

# endregion

# region 地区分类
# ===================== 获取单个地区分类 数据模型 =====================
class GetDistrictclassInput(BaseModel):
    """获取单个地区分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="地区编码（必填）")


# ===================== 批量获取地区分类 数据模型 =====================
class BatchGetDistrictclassInput(BaseModel):
    """批量获取地区分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始地区编码")
    code_end: Optional[str] = Field(None, description="结束地区编码")
    name: Optional[str] = Field(None, description="地区名称关键字")
    endflag: Optional[bool] = Field(None, description="是否末级")




# ===================== 获取单个地区分类 Tool函数 =====================
def u8_districtclass_get_tool(input_data: GetDistrictclassInput, client: U8OpenAPIClient) -> str:
    """
    获取单个地区分类信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 地区分类接口路径
    api_path = "/api/districtclass/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取地区分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取地区分类成功",
            "data": {
                "districtclass": result.get("districtclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取地区分类 Tool函数 =====================
def u8_districtclass_batch_get_tool(input_data: BatchGetDistrictclassInput, client: U8OpenAPIClient) -> str:
    """
    批量获取地区分类列表，支持按地区编码范围、名称、是否末级等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 地区分类接口路径
    api_path = "/api/districtclass/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取地区分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取地区分类成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "districtclass": result.get("districtclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取单个地区分类 Schema定义 =====================
U8_DISTRICTCLASS_GET_SCHEMA = {
    "name": "u8_districtclass_get",
    "description": "获取单个地区分类信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "地区编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取地区分类 Schema定义 =====================
U8_DISTRICTCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_districtclass_batch_get",
    "description": "批量获取地区分类列表，支持按地区编码范围、名称、是否末级等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始地区编码"},
            "code_end": {"type": "string", "description": "结束地区编码"},
            "name": {"type": "string", "description": "地区名称关键字"},
            "endflag": {"type": "boolean", "description": "是否末级"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}


# endregion

# region 存货分类

# ===================== 获取单个存货分类 数据模型 =====================
class GetInventoryclassInput(BaseModel):
    """获取单个存货分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="存货分类编码（必填）")


# ===================== 批量获取存货分类 数据模型 =====================
class BatchGetInventoryclassInput(BaseModel):
    """批量获取存货分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")


# ===================== 新增存货分类 数据模型 =====================
class AddInventoryclassInput(BaseModel):
    """新增存货分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code: str = Field(..., description="存货分类编码（必填）")
    name: str = Field(..., description="存货分类名称（必填）")
    rank: Optional[int] = Field(None, description="存货分类编码级次")
    end_rank_flag: Optional[bool] = Field(None, description="末级标志")
    biz_id: str = Field(..., description="上游id，需要保证biz_id与ERP主键唯一对应关系（必填）")
    sync: Optional[int] = Field(None, description="0=异步新增（默认）;1=同步新增")




# ===================== 获取单个存货分类 Tool函数 =====================
def u8_inventoryclass_get_tool(input_data: GetInventoryclassInput, client: U8OpenAPIClient) -> str:
    """
    获取单个存货分类信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 存货分类接口路径
    api_path = "/api/inventoryclass/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取存货分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取存货分类成功",
            "data": {
                "inventoryclass": result.get("inventoryclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取存货分类 Tool函数 =====================
def u8_inventoryclass_batch_get_tool(input_data: BatchGetInventoryclassInput, client: U8OpenAPIClient) -> str:
    """
    批量获取存货分类列表，支持分页、编码范围、名称等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 存货分类接口路径
    api_path = "/api/inventoryclass/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取存货分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取存货分类成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "inventoryclass": result.get("inventoryclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增存货分类 Tool函数 =====================
def u8_inventoryclass_add_tool(input_data: AddInventoryclassInput, client: U8OpenAPIClient) -> str:
    """
    新增一张存货分类。
    """
    # 提取biz_id和sync参数
    params_data = input_data.model_dump(exclude_none=True)
    biz_id = params_data.pop("biz_id", None)
    sync = params_data.pop("sync", None)
    ds_sequence = params_data.pop("ds_sequence", None)

    # 构造接口要求的标准 JSON 结构（外层包一层 inventoryclass）
    request_body: dict = {
        "inventoryclass": params_data
    }

    # 构造URL参数
    inparams = {"biz_id": biz_id}
    if sync is not None:
        inparams["sync"] = sync
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    # 存货分类接口路径
    api_path = "/api/inventoryclass/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增存货分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "新增存货分类成功",
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




# ===================== 获取单个存货分类 Schema定义 =====================
U8_INVENTORYCLASS_GET_SCHEMA = {
    "name": "u8_inventoryclass_get",
    "description": "获取单个存货分类信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "存货分类编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取存货分类 Schema定义 =====================
U8_INVENTORYCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_inventoryclass_batch_get",
    "description": "批量获取存货分类列表，支持分页、编码范围、名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"}
        },
        "required": []  # 所有参数均为可选，无必填项
    }
}


# ===================== 新增存货分类 Schema定义 =====================
U8_INVENTORYCLASS_ADD_SCHEMA = {
    "name": "u8_inventoryclass_add",
    "description": "新增一张存货分类",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code": {"type": "string", "description": "存货分类编码（必填）"},
            "name": {"type": "string", "description": "存货分类名称（必填）"},
            "rank": {"type": "number", "description": "存货分类编码级次"},
            "end_rank_flag": {"type": "boolean", "description": "末级标志"},
            "biz_id": {"type": "string", "description": "上游id，需要保证biz_id与ERP主键唯一对应关系（必填）"},
            "sync": {"type": "number", "description": "0=异步新增（默认）;1=同步新增"}
        },
        "required": ["code", "name", "biz_id"]
    }
}


# endregion

# region 存货档案

# ===================== 获取单个存货档案 数据模型 =====================
class GetInventoryInput(BaseModel):
    """获取单个存货档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="存货编码（必填）")


# ===================== 批量获取存货档案 数据模型 =====================
class BatchGetInventoryInput(BaseModel):
    """批量获取存货档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始存货编码")
    code_end: Optional[str] = Field(None, description="结束存货编码")
    name: Optional[str] = Field(None, description="存货名称关键字")
    invaddcode: Optional[str] = Field(None, description="存货代码")
    specs: Optional[str] = Field(None, description="规格型号关键字")
    sort_code: Optional[str] = Field(None, description="所属分类码")
    sort_name: Optional[str] = Field(None, description="所属分类")
    start_date_begin: Optional[str] = Field(None, description="起始启用日期，格式：yyyy-MM-dd")
    start_date_end: Optional[str] = Field(None, description="结束启用日期，格式：yyyy-MM-dd")
    modifydate_begin: Optional[str] = Field(None, description="起始变更日期，格式：yyyy-MM-dd")
    modifydate_end: Optional[str] = Field(None, description="结束变更日期，格式：yyyy-MM-dd")
    sale_flag: Optional[int] = Field(None, description="是否内销（0否1是）")
    bexpsale: Optional[int] = Field(None, description="是否外销（0否1是）")


# ===================== 修改存货档案 数据模型 =====================
class EditInventoryInput(BaseModel):
    """修改存货档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    # 基本信息
    code: str = Field(..., description="存货编码（必填）")
    name: str = Field(..., description="存货名称（必填）")
    invaddcode: Optional[str] = Field(None, description="存货代码")
    specs: Optional[str] = Field(None, description="规格型号")
    sort_code: Optional[str] = Field(None, description="所属分类码")
    # 计量单位
    main_measure: Optional[str] = Field(None, description="主计量单位")
    puunit_code: Optional[str] = Field(None, description="采购默认单位编码")
    puunit_name: Optional[str] = Field(None, description="采购默认单位名称")
    puunit_ichangrate: Optional[float] = Field(None, description="采购默认单位换算率")
    saunit_code: Optional[str] = Field(None, description="销售默认单位编码")
    saunit_name: Optional[str] = Field(None, description="销售默认单位名称")
    saunit_ichangrate: Optional[float] = Field(None, description="销售默认单位换算率")
    stunit_code: Optional[str] = Field(None, description="库存默认单位编码")
    stunit_name: Optional[str] = Field(None, description="库存默认单位名称")
    stunit_ichangrate: Optional[float] = Field(None, description="库存默认单位换算率")
    unitgroup_code: Optional[str] = Field(None, description="计量单位组编码")
    unitgroup_type: Optional[int] = Field(None, description="计量单位组类型(0=无换算;1=固定;2=浮动)")
    # 条形码
    bbarcode: Optional[bool] = Field(None, description="条形码管理")
    barcode: Optional[str] = Field(None, description="条形码")
    # 价格
    ref_sale_price: Optional[float] = Field(None, description="参考售价")
    bottom_sale_price: Optional[float] = Field(None, description="最低售价")
    bsuitretail: Optional[bool] = Field(None, description="适用零售(0:否 1：是)")
    # 日期
    start_date: Optional[str] = Field(None, description="启用日期")
    end_date: Optional[str] = Field(None, description="停用日期")
    # 仓库
    defwarehouse: Optional[str] = Field(None, description="默认仓库")
    defwarehousename: Optional[str] = Field(None, description="默认仓库名称")
    # 业务属性
    iSupplyType: Optional[str] = Field(None, description="供应类型")
    drawtype: Optional[str] = Field(None, description="领料方式")
    iimptaxrate: Optional[float] = Field(None, description="进项税率")
    tax_rate: Optional[float] = Field(None, description="销项税率")
    # 自定义项
    self_define1: Optional[str] = Field(None, description="自定义项1")
    self_define2: Optional[str] = Field(None, description="自定义项2")
    self_define3: Optional[str] = Field(None, description="自定义项3")
    self_define4: Optional[str] = Field(None, description="自定义项4")
    self_define5: Optional[str] = Field(None, description="自定义项5")
    self_define6: Optional[str] = Field(None, description="自定义项6")
    self_define7: Optional[str] = Field(None, description="自定义项7")
    self_define8: Optional[str] = Field(None, description="自定义项8")
    self_define9: Optional[str] = Field(None, description="自定义项9")
    self_define10: Optional[str] = Field(None, description="自定义项10")
    self_define11: Optional[str] = Field(None, description="自定义项11")
    self_define12: Optional[str] = Field(None, description="自定义项12")
    self_define13: Optional[str] = Field(None, description="自定义项13")
    self_define14: Optional[str] = Field(None, description="自定义项14")
    self_define15: Optional[str] = Field(None, description="自定义项15")
    self_define16: Optional[str] = Field(None, description="自定义项16")


# ===================== 新增存货档案 数据模型 =====================
class AddInventoryInput(BaseModel):
    """新增存货档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    # 基本信息
    code: str = Field(..., description="存货编码（必填）")
    name: str = Field(..., description="存货名称（必填）")
    invaddcode: Optional[str] = Field(None, description="存货代码")
    specs: Optional[str] = Field(None, description="规格型号")
    sort_code: str = Field(..., description="所属分类码（必填）")
    # 计量单位
    unitgroup_code: str = Field(..., description="计量单位组编码（必填）")
    unitgroup_type: int = Field(..., description="计量单位组类型(0=无换算;1=固定;2=浮动)（必填）")
    main_measure: str = Field(..., description="主计量单位（必填）")
    puunit_code: Optional[str] = Field(None, description="采购默认单位编码")
    puunit_name: Optional[str] = Field(None, description="采购默认单位名称")
    puunit_ichangrate: Optional[float] = Field(None, description="采购默认单位换算率")
    saunit_code: Optional[str] = Field(None, description="销售默认单位编码")
    saunit_name: Optional[str] = Field(None, description="销售默认单位名称")
    saunit_ichangrate: Optional[float] = Field(None, description="销售默认单位换算率")
    stunit_code: Optional[str] = Field(None, description="库存默认单位编码")
    stunit_name: Optional[str] = Field(None, description="库存默认单位名称")
    stunit_ichangrate: Optional[float] = Field(None, description="库存默认单位换算率")
    # 条形码
    bbarcode: Optional[bool] = Field(None, description="条形码管理")
    barcode: Optional[str] = Field(None, description="条形码")
    # 价格
    ref_sale_price: Optional[float] = Field(None, description="参考售价")
    bottom_sale_price: Optional[float] = Field(None, description="最低售价")
    retailprice: Optional[float] = Field(None, description="零售单价")
    fRetailPrice: Optional[float] = Field(None, description="零售价格")
    bsuitretail: Optional[bool] = Field(None, description="适用零售(0:否 1：是)")
    # 日期
    start_date: Optional[str] = Field(None, description="启用日期")
    end_date: Optional[str] = Field(None, description="停用日期")
    # 仓库
    defwarehouse: Optional[str] = Field(None, description="默认仓库")
    defwarehousename: Optional[str] = Field(None, description="默认仓库名称")
    # 业务属性
    iSupplyType: Optional[str] = Field(None, description="供应类型")
    drawtype: Optional[str] = Field(None, description="领料方式")
    iimptaxrate: Optional[float] = Field(None, description="进项税率")
    tax_rate: Optional[float] = Field(None, description="销项税率")
    # 其他标志
    purchase_flag: Optional[bool] = Field(None, description="是否采购")
    sale_flag: Optional[bool] = Field(None, description="是否内销")
    bexpsale: Optional[bool] = Field(None, description="是否外销")
    prod_consu_flag: Optional[bool] = Field(None, description="是否生产耗用")
    selfmake_flag: Optional[bool] = Field(None, description="是否自制")
    bProxyForeign: Optional[bool] = Field(None, description="是否委外")
    # 自定义项
    self_define1: Optional[str] = Field(None, description="自定义项1")
    self_define2: Optional[str] = Field(None, description="自定义项2")
    self_define3: Optional[str] = Field(None, description="自定义项3")
    self_define4: Optional[str] = Field(None, description="自定义项4")
    self_define5: Optional[str] = Field(None, description="自定义项5")
    self_define6: Optional[str] = Field(None, description="自定义项6")
    self_define7: Optional[str] = Field(None, description="自定义项7")
    self_define8: Optional[str] = Field(None, description="自定义项8")
    self_define9: Optional[str] = Field(None, description="自定义项9")
    self_define10: Optional[str] = Field(None, description="自定义项10")
    self_define11: Optional[str] = Field(None, description="自定义项11")
    self_define12: Optional[str] = Field(None, description="自定义项12")
    self_define13: Optional[str] = Field(None, description="自定义项13")
    self_define14: Optional[str] = Field(None, description="自定义项14")
    self_define15: Optional[str] = Field(None, description="自定义项15")
    self_define16: Optional[str] = Field(None, description="自定义项16")
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
    # 交易参数
    biz_id: str = Field(..., description="上游id，需要保证biz_id与ERP主键唯一对应关系（必填）")
    sync: Optional[int] = Field(None, description="0=异步新增（默认）;1=同步新增")




# ===================== 获取单个存货档案 Tool函数 =====================
def u8_inventory_get_tool(input_data: GetInventoryInput, client: U8OpenAPIClient) -> str:
    """
    获取单个存货信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 存货档案接口路径
    api_path = "/api/inventory/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取存货档案失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取存货档案成功",
            "data": {
                "inventory": result.get("inventory")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取存货档案 Tool函数 =====================
def u8_inventory_batch_get_tool(input_data: BatchGetInventoryInput, client: U8OpenAPIClient) -> str:
    """
    批量获取存货档案列表，支持分页、编码范围、名称、分类等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 存货档案接口路径
    api_path = "/api/inventory/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取存货档案失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取存货档案成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "inventory": result.get("inventory")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 修改存货档案 Tool函数 =====================
def u8_inventory_edit_tool(input_data: EditInventoryInput, client: U8OpenAPIClient) -> str:
    """
    修改一个存货档案。
    """
    # 提取ds_sequence参数
    params_data = input_data.model_dump(exclude_none=True)
    ds_sequence = params_data.pop("ds_sequence", None)

    # 构造接口要求的标准 JSON 结构（外层包一层 inventory）
    request_body: dict = {
        "inventory": params_data
    }

    # 构造URL参数
    inparams = {}
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    # 存货档案接口路径
    api_path = "/api/inventory/edit"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "修改存货档案失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "修改存货档案成功",
            "data": {},
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 新增存货档案 Tool函数 =====================
def u8_inventory_add_tool(input_data: AddInventoryInput, client: U8OpenAPIClient) -> str:
    """
    添加一个新存货档案。
    """
    # 提取biz_id和sync参数
    params_data = input_data.model_dump(exclude_none=True)
    biz_id = params_data.pop("biz_id", None)
    sync = params_data.pop("sync", None)
    ds_sequence = params_data.pop("ds_sequence", None)

    # 构造自由项列表
    free_items = []
    for i in range(1, 11):
        free_key = f"free{i}"
        if free_key in params_data:
            free_items.append({"invcode": params_data.get("code"), free_key: params_data.pop(free_key)})

    # 移除自由项字段，合并到entry中
    inventory_data = {}
    for k, v in params_data.items():
        if k.startswith("free"):
            continue
        inventory_data[k] = v

    if free_items:
        inventory_data["entry"] = free_items

    # 构造接口要求的标准 JSON 结构（外层包一层 inventory）
    request_body: dict = {
        "inventory": inventory_data
    }

    # 构造URL参数
    inparams = {"biz_id": biz_id}
    if sync is not None:
        inparams["sync"] = sync
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    # 存货档案接口路径
    api_path = "/api/inventory/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增存货档案失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "新增存货档案成功",
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




# ===================== 获取单个存货档案 Schema定义 =====================
U8_INVENTORY_GET_SCHEMA = {
    "name": "u8_inventory_get",
    "description": "获取单个存货信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "存货编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取存货档案 Schema定义 =====================
U8_INVENTORY_BATCH_GET_SCHEMA = {
    "name": "u8_inventory_batch_get",
    "description": "批量获取存货档案列表，支持分页、编码范围、名称、分类等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始存货编码"},
            "code_end": {"type": "string", "description": "结束存货编码"},
            "name": {"type": "string", "description": "存货名称关键字"},
            "invaddcode": {"type": "string", "description": "存货代码"},
            "specs": {"type": "string", "description": "规格型号关键字"},
            "sort_code": {"type": "string", "description": "所属分类码"},
            "sort_name": {"type": "string", "description": "所属分类"},
            "start_date_begin": {"type": "string", "description": "起始启用日期，格式：yyyy-MM-dd"},
            "start_date_end": {"type": "string", "description": "结束启用日期，格式：yyyy-MM-dd"},
            "modifydate_begin": {"type": "string", "description": "起始变更日期，格式：yyyy-MM-dd"},
            "modifydate_end": {"type": "string", "description": "结束变更日期，格式：yyyy-MM-dd"},
            "sale_flag": {"type": "number", "description": "是否内销（0否1是）"},
            "bexpsale": {"type": "number", "description": "是否外销（0否1是）"}
        },
        "required": []
    }
}


# ===================== 修改存货档案 Schema定义 =====================
U8_INVENTORY_EDIT_SCHEMA = {
    "name": "u8_inventory_edit",
    "description": "修改一个存货档案",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code": {"type": "string", "description": "存货编码（必填）"},
            "name": {"type": "string", "description": "存货名称（必填）"},
            "invaddcode": {"type": "string", "description": "存货代码"},
            "specs": {"type": "string", "description": "规格型号"},
            "sort_code": {"type": "string", "description": "所属分类码"},
            "main_measure": {"type": "string", "description": "主计量单位"},
            "puunit_code": {"type": "string", "description": "采购默认单位编码"},
            "puunit_name": {"type": "string", "description": "采购默认单位名称"},
            "puunit_ichangrate": {"type": "number", "description": "采购默认单位换算率"},
            "saunit_code": {"type": "string", "description": "销售默认单位编码"},
            "saunit_name": {"type": "string", "description": "销售默认单位名称"},
            "saunit_ichangrate": {"type": "number", "description": "销售默认单位换算率"},
            "stunit_code": {"type": "string", "description": "库存默认单位编码"},
            "stunit_name": {"type": "string", "description": "库存默认单位名称"},
            "stunit_ichangrate": {"type": "number", "description": "库存默认单位换算率"},
            "unitgroup_code": {"type": "string", "description": "计量单位组编码"},
            "unitgroup_type": {"type": "number", "description": "计量单位组类型(0=无换算;1=固定;2=浮动)"},
            "bbarcode": {"type": "boolean", "description": "条形码管理"},
            "barcode": {"type": "string", "description": "条形码"},
            "ref_sale_price": {"type": "number", "description": "参考售价"},
            "bottom_sale_price": {"type": "number", "description": "最低售价"},
            "bsuitretail": {"type": "boolean", "description": "适用零售(0:否 1：是)"},
            "start_date": {"type": "string", "description": "启用日期"},
            "end_date": {"type": "string", "description": "停用日期"},
            "defwarehouse": {"type": "string", "description": "默认仓库"},
            "defwarehousename": {"type": "string", "description": "默认仓库名称"},
            "iSupplyType": {"type": "string", "description": "供应类型"},
            "drawtype": {"type": "string", "description": "领料方式"},
            "iimptaxrate": {"type": "number", "description": "进项税率"},
            "tax_rate": {"type": "number", "description": "销项税率"},
            "self_define1": {"type": "string", "description": "自定义项1"},
            "self_define2": {"type": "string", "description": "自定义项2"},
            "self_define3": {"type": "string", "description": "自定义项3"},
            "self_define4": {"type": "string", "description": "自定义项4"},
            "self_define5": {"type": "string", "description": "自定义项5"},
            "self_define6": {"type": "string", "description": "自定义项6"},
            "self_define7": {"type": "string", "description": "自定义项7"},
            "self_define8": {"type": "string", "description": "自定义项8"},
            "self_define9": {"type": "string", "description": "自定义项9"},
            "self_define10": {"type": "string", "description": "自定义项10"},
            "self_define11": {"type": "string", "description": "自定义项11"},
            "self_define12": {"type": "string", "description": "自定义项12"},
            "self_define13": {"type": "string", "description": "自定义项13"},
            "self_define14": {"type": "string", "description": "自定义项14"},
            "self_define15": {"type": "string", "description": "自定义项15"},
            "self_define16": {"type": "string", "description": "自定义项16"}
        },
        "required": ["code", "name"]
    }
}


# ===================== 新增存货档案 Schema定义 =====================
U8_INVENTORY_ADD_SCHEMA = {
    "name": "u8_inventory_add",
    "description": "添加一个新存货档案",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code": {"type": "string", "description": "存货编码（必填）"},
            "name": {"type": "string", "description": "存货名称（必填）"},
            "invaddcode": {"type": "string", "description": "存货代码"},
            "specs": {"type": "string", "description": "规格型号"},
            "sort_code": {"type": "string", "description": "所属分类码（必填）"},
            "unitgroup_code": {"type": "string", "description": "计量单位组编码（必填）"},
            "unitgroup_type": {"type": "number", "description": "计量单位组类型(0=无换算;1=固定;2=浮动)（必填）"},
            "main_measure": {"type": "string", "description": "主计量单位（必填）"},
            "puunit_code": {"type": "string", "description": "采购默认单位编码"},
            "puunit_name": {"type": "string", "description": "采购默认单位名称"},
            "puunit_ichangrate": {"type": "number", "description": "采购默认单位换算率"},
            "saunit_code": {"type": "string", "description": "销售默认单位编码"},
            "saunit_name": {"type": "string", "description": "销售默认单位名称"},
            "saunit_ichangrate": {"type": "number", "description": "销售默认单位换算率"},
            "stunit_code": {"type": "string", "description": "库存默认单位编码"},
            "stunit_name": {"type": "string", "description": "库存默认单位名称"},
            "stunit_ichangrate": {"type": "number", "description": "库存默认单位换算率"},
            "bbarcode": {"type": "boolean", "description": "条形码管理"},
            "barcode": {"type": "string", "description": "条形码"},
            "ref_sale_price": {"type": "number", "description": "参考售价"},
            "bottom_sale_price": {"type": "number", "description": "最低售价"},
            "retailprice": {"type": "number", "description": "零售单价"},
            "fRetailPrice": {"type": "number", "description": "零售价格"},
            "bsuitretail": {"type": "boolean", "description": "适用零售(0:否 1：是)"},
            "start_date": {"type": "string", "description": "启用日期"},
            "end_date": {"type": "string", "description": "停用日期"},
            "defwarehouse": {"type": "string", "description": "默认仓库"},
            "defwarehousename": {"type": "string", "description": "默认仓库名称"},
            "iSupplyType": {"type": "string", "description": "供应类型"},
            "drawtype": {"type": "string", "description": "领料方式"},
            "iimptaxrate": {"type": "number", "description": "进项税率"},
            "tax_rate": {"type": "number", "description": "销项税率"},
            "purchase_flag": {"type": "boolean", "description": "是否采购"},
            "sale_flag": {"type": "boolean", "description": "是否内销"},
            "bexpsale": {"type": "boolean", "description": "是否外销"},
            "prod_consu_flag": {"type": "boolean", "description": "是否生产耗用"},
            "selfmake_flag": {"type": "boolean", "description": "是否自制"},
            "bProxyForeign": {"type": "boolean", "description": "是否委外"},
            "self_define1": {"type": "string", "description": "自定义项1"},
            "self_define2": {"type": "string", "description": "自定义项2"},
            "self_define3": {"type": "string", "description": "自定义项3"},
            "self_define4": {"type": "string", "description": "自定义项4"},
            "self_define5": {"type": "string", "description": "自定义项5"},
            "self_define6": {"type": "string", "description": "自定义项6"},
            "self_define7": {"type": "string", "description": "自定义项7"},
            "self_define8": {"type": "string", "description": "自定义项8"},
            "self_define9": {"type": "string", "description": "自定义项9"},
            "self_define10": {"type": "string", "description": "自定义项10"},
            "self_define11": {"type": "string", "description": "自定义项11"},
            "self_define12": {"type": "string", "description": "自定义项12"},
            "self_define13": {"type": "string", "description": "自定义项13"},
            "self_define14": {"type": "string", "description": "自定义项14"},
            "self_define15": {"type": "string", "description": "自定义项15"},
            "self_define16": {"type": "string", "description": "自定义项16"},
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
            "biz_id": {"type": "string", "description": "上游id，需要保证biz_id与ERP主键唯一对应关系（必填）"},
            "sync": {"type": "number", "description": "0=异步新增（默认）;1=同步新增"}
        },
        "required": ["code", "name", "sort_code", "unitgroup_code", "unitgroup_type", "main_measure", "biz_id"]
    }
}


# endregion

# region 客户



# ===================== 添加一个新客户 子数据模型 =====================
class Addresses(BaseModel):
    ccuscode: Optional[str] = Field(None, description="客户编码")
    caddcode: Optional[str] = Field(None, description="收货地址编码")
    bdefault: Optional[bool] = Field(None, description="默认值（1/0）")
    cdeliveradd: Optional[str] = Field(None, description="收货地址")
    cenglishadd: Optional[str] = Field(None, description="英文地址")
    cenglishadd2: Optional[str] = Field(None, description="英文地址2")
    cenglishadd3: Optional[str] = Field(None, description="英文地址3")
    cenglishadd4: Optional[str] = Field(None, description="英文地址4")
    cdeliverunit: Optional[str] = Field(None, description="收货单位")
    clinkperson: Optional[str] = Field(None, description="联系人")

class Banks(BaseModel):
    ccuscode: Optional[str] = Field(None, description="客户编码")
    caccountnum: Optional[str] = Field(None, description="银行账号")
    bdefault: Optional[bool] = Field(None, description="默认值（1/0）")
    cbank: Optional[str] = Field(None, description="所属银行编码")
    cbranch: Optional[str] = Field(None, description="开户银行")
    caccountname: Optional[str] = Field(None, description="账户名称")
    cCusPrinvince: Optional[str] = Field(None, description="省")
    cCusCity: Optional[str] = Field(None, description="市")
    cCusCBBDepId: Optional[str] = Field(None, description="机构号")
    cCusBranchId: Optional[str] = Field(None, description="联行号")
    cCusBranchIdSec: Optional[str] = Field(None, description="联行号II")

class Invoicecustomers(BaseModel):
    ccuscode: Optional[str] = Field(None, description="客户编码")
    cinvoicecompany: Optional[str] = Field(None, description="开票单位编码")
    bdefault: Optional[bool] = Field(None, description="默认值（1/0）")

class Users(BaseModel):
    ccuscode: Optional[str] = Field(None, description="客户编码")
    user_id: Optional[str] = Field(None, description="操作员编码")
    is_self: Optional[bool] = Field(None, description="相关或负责员工(true/false)")

class Auths(BaseModel):
    ccuscode: Optional[str] = Field(None, description="客户编码")
    privilege_type: Optional[str] = Field(None, description="管理维度类型编码")
    privilege_id: Optional[str] = Field(None, description="管理维度编码")

# ===================== 获取单个客户信息 数据模型 =====================
class GetCustomerInfoInput(BaseModel):
    id: str = Field(..., description="客户编号，用于查询客户详细信息")

# ===================== 批量获取客户信息 数据模型 =====================
class BatchGetCustomerInfoInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")
    timestamp_begin: Optional[str] = Field(None, description="起始时间戳")
    timestamp_end: Optional[str] = Field(None, description="结束时间戳")
    modifydate_begin: Optional[str] = Field(None, description="起始修改日期")
    modifydate_end: Optional[str] = Field(None, description="结束修改日期")
    seed_date_begin: Optional[str] = Field(None, description="起始发展日期")
    seed_date_end: Optional[str] = Field(None, description="结束发展日期")

# ===================== 修改客户信息 数据模型 =====================
class EditCustomerInfoInput(BaseModel):
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code: str = Field(..., description="客户编码")
    name: Optional[str] = Field(None, description="客户名称")
    abbrname: Optional[str] = Field(None, description="客户简称")
    sort_code: Optional[str] = Field(None, description="所属分类码")
    domain_code: Optional[str] = Field(None, description="所属地区码")
    industry: Optional[str] = Field(None, description="所属行业")
    address: Optional[str] = Field(None, description="地址")
    bank_open: Optional[str] = Field(None, description="开户银行")
    bank_acc_number: Optional[str] = Field(None, description="银行账号")
    contact: Optional[str] = Field(None, description="联系人")
    phone: Optional[str] = Field(None, description="电话")
    fax: Optional[str] = Field(None, description="传真")
    mobile: Optional[str] = Field(None, description="手机")
    devliver_site: Optional[str] = Field(None, description="发货地址")
    end_date: Optional[str] = Field(None, description="停用日期")
    ccusexch_name: Optional[str] = Field(None, description="币种")
    bcusdomestic: Optional[str] = Field(None, description="国内")
    bcusoverseas: Optional[str] = Field(None, description="国外")
    bserviceattribute: Optional[str] = Field(None, description="服务")
    ccusmngtypecode: Optional[str] = Field(None, description="客户管理类型")
    ccusmngtypename: Optional[str] = Field(None, description="客户管理类型名称")
    spec_operator: Optional[str] = Field(None, description="专管业务员编码")
    spec_operator_name: Optional[str] = Field(None, description="专管业务员名称")
    memo: Optional[str] = Field(None, description="备注")
    self_define1: Optional[str] = Field(None, description="自定义项1")
    self_define2: Optional[str] = Field(None, description="自定义项2")
    self_define3: Optional[str] = Field(None, description="自定义项3")
    self_define4: Optional[str] = Field(None, description="自定义项4")
    self_define5: Optional[str] = Field(None, description="自定义项5")
    self_define6: Optional[str] = Field(None, description="自定义项6")
    self_define7: Optional[str] = Field(None, description="自定义项7")
    self_define8: Optional[str] = Field(None, description="自定义项8")
    self_define9: Optional[str] = Field(None, description="自定义项9")
    self_define10: Optional[str] = Field(None, description="自定义项10")
    self_define11: Optional[str] = Field(None, description="自定义项11")
    self_define12: Optional[str] = Field(None, description="自定义项12")
    self_define13: Optional[str] = Field(None, description="自定义项13")
    self_define14: Optional[str] = Field(None, description="自定义项14")
    self_define15: Optional[str] = Field(None, description="自定义项15")
    self_define16: Optional[str] = Field(None, description="自定义项16")

# ===================== 添加一个新客户 主数据模型 =====================
class AddCustomerInfoInput(BaseModel):
    code: str = Field(..., description="客户编码")
    name: str = Field(..., description="客户名称")
    abbrname: Optional[str] = Field(None, description="客户简称")
    sort_code: Optional[str] = Field(None, description="所属分类码")
    domain_code: Optional[str] = Field(None, description="所属地区码")
    industry: Optional[str] = Field(None, description="所属行业")
    contact: Optional[str] = Field(None, description="联系人")
    phone: Optional[str] = Field(None, description="电话")
    fax: Optional[str] = Field(None, description="传真")
    mobile: Optional[str] = Field(None, description="手机")
    devliver_site: Optional[str] = Field(None, description="发货地址")
    end_date: Optional[str] = Field(None, description="停用日期")
    memo: Optional[str] = Field(None, description="备注")
    ccusexch_name: Optional[str] = Field(None, description="币种")
    bcusdomestic: Optional[str] = Field(None, description="国内")
    bcusoverseas: Optional[str] = Field(None, description="国外")
    bserviceattribute: Optional[str] = Field(None, description="服务")
    self_define1: Optional[str] = Field(None, description="自定义项1")
    self_define2: Optional[str] = Field(None, description="自定义项2")
    self_define3: Optional[str] = Field(None, description="自定义项3")
    self_define4: Optional[str] = Field(None, description="自定义项4")
    self_define5: Optional[str] = Field(None, description="自定义项5")
    self_define6: Optional[str] = Field(None, description="自定义项6")
    self_define7: Optional[str] = Field(None, description="自定义项7")
    self_define8: Optional[str] = Field(None, description="自定义项8")
    self_define9: Optional[str] = Field(None, description="自定义项9")
    self_define10: Optional[str] = Field(None, description="自定义项10")
    self_define11: Optional[str] = Field(None, description="自定义项11")
    self_define12: Optional[str] = Field(None, description="自定义项12")
    self_define13: Optional[str] = Field(None, description="自定义项13")
    self_define14: Optional[str] = Field(None, description="自定义项14")
    self_define15: Optional[str] = Field(None, description="自定义项15")
    self_define16: Optional[str] = Field(None, description="自定义项16")
    addresses: Optional[List[Addresses]] = Field(None, description="地址信息列表")
    banks: Optional[List[Banks]] = Field(None, description="银行信息列表")
    invoicecustomers: Optional[List[Invoicecustomers]] = Field(None, description="开票客户信息列表")
    users: Optional[List[Users]] = Field(None, description="用户信息列表")
    auths: Optional[List[Auths]] = Field(None, description="权限信息列表")

# ===================== 获取单个客户信息 Tool函数 =====================
def u8_customer_get_tool(input_data: GetCustomerInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过客户编号获取用友U8中的客户基本信息。
    """
    params = {
        "id": input_data.id 
    }
    api_path = "/api/customer/get" 
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取客户信息 Tool函数 =====================
def u8_customer_batch_get_tool(input_data: BatchGetCustomerInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取客户列表，支持分页、编码范围、名称、时间等条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)
    api_path = "/api/customer/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取客户失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取客户成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customer": result.get("customer")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 修改客户信息 Tool函数 =====================
def u8_customer_edit_tool(input_data: EditCustomerInfoInput, client: U8OpenAPIClient) -> str:
    """
    修改一个客户信息。
    """
    params_data = input_data.model_dump(exclude_none=True)
    ds_sequence = params_data.pop("ds_sequence", None)

    request_body: Dict[str, Any] = {
        "customer": params_data
    }

    inparams = {}
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    api_path = "/api/customer/edit"

    try:
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "修改客户失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "修改客户成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 添加新客户 Tool函数 =====================
def u8_customer_add_tool(input_data: AddCustomerInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中添加新的客户信息。
    """
    request_body: Dict[str, Any] = {
        "customer": input_data.model_dump(exclude_none=True)
    }
    api_path = "/api/customer/add" 

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
            "message": "客户新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个客户信息 Schema定义 =====================
U8_CUSTOMER_GET_SCHEMA = {
    "name": "u8_customer_get",
    "description": "通过客户编码获得客户信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "客户编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取客户信息 Schema定义 =====================
U8_CUSTOMER_BATCH_GET_SCHEMA = {
    "name": "u8_customer_batch_get",
    "description": "批量获取客户列表，支持分页、编码范围、名称、时间等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"},
            "timestamp_begin": {"type": "string", "description": "起始时间戳"},
            "timestamp_end": {"type": "string", "description": "结束时间戳"},
            "modifydate_begin": {"type": "string", "description": "起始修改日期"},
            "modifydate_end": {"type": "string", "description": "结束修改日期"},
            "seed_date_begin": {"type": "string", "description": "起始发展日期"},
            "seed_date_end": {"type": "string", "description": "结束发展日期"}
        },
        "required": []
    }
}

# ===================== 修改客户信息 Schema定义 =====================
U8_CUSTOMER_EDIT_SCHEMA = {
    "name": "u8_customer_edit",
    "description": "修改一个客户信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code": {"type": "string", "description": "客户编码（必填）"},
            "name": {"type": "string", "description": "客户名称"},
            "abbrname": {"type": "string", "description": "客户简称"},
            "sort_code": {"type": "string", "description": "所属分类码"},
            "domain_code": {"type": "string", "description": "所属地区码"},
            "industry": {"type": "string", "description": "所属行业"},
            "address": {"type": "string", "description": "地址"},
            "bank_open": {"type": "string", "description": "开户银行"},
            "bank_acc_number": {"type": "string", "description": "银行账号"},
            "contact": {"type": "string", "description": "联系人"},
            "phone": {"type": "string", "description": "电话"},
            "fax": {"type": "string", "description": "传真"},
            "mobile": {"type": "string", "description": "手机"},
            "devliver_site": {"type": "string", "description": "发货地址"},
            "end_date": {"type": "string", "description": "停用日期"},
            "ccusexch_name": {"type": "string", "description": "币种"},
            "bcusdomestic": {"type": "string", "description": "国内"},
            "bcusoverseas": {"type": "string", "description": "国外"},
            "bserviceattribute": {"type": "string", "description": "服务"},
            "ccusmngtypecode": {"type": "string", "description": "客户管理类型"},
            "ccusmngtypename": {"type": "string", "description": "客户管理类型名称"},
            "spec_operator": {"type": "string", "description": "专管业务员编码"},
            "spec_operator_name": {"type": "string", "description": "专管业务员名称"},
            "memo": {"type": "string", "description": "备注"},
            "self_define1": {"type": "string", "description": "自定义项1"},
            "self_define2": {"type": "string", "description": "自定义项2"},
            "self_define3": {"type": "string", "description": "自定义项3"},
            "self_define4": {"type": "string", "description": "自定义项4"},
            "self_define5": {"type": "string", "description": "自定义项5"},
            "self_define6": {"type": "string", "description": "自定义项6"},
            "self_define7": {"type": "string", "description": "自定义项7"},
            "self_define8": {"type": "string", "description": "自定义项8"},
            "self_define9": {"type": "string", "description": "自定义项9"},
            "self_define10": {"type": "string", "description": "自定义项10"},
            "self_define11": {"type": "string", "description": "自定义项11"},
            "self_define12": {"type": "string", "description": "自定义项12"},
            "self_define13": {"type": "string", "description": "自定义项13"},
            "self_define14": {"type": "string", "description": "自定义项14"},
            "self_define15": {"type": "string", "description": "自定义项15"},
            "self_define16": {"type": "string", "description": "自定义项16"}
        },
        "required": ["code"]
    }
}

# ===================== 添加新客户 Schema定义 =====================
U8_CUSTOMER_ADD_SCHEMA = {
    "name": "u8_customer_add",
    "description": "在用友U8 OpenAPI中新增客户，支持客户主信息、地址、银行账户、开票单位、负责员工、数据权限等完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "客户编码（必填）"},
            "name": {"type": "string", "description": "客户名称（必填）"},
            "abbrname": {"type": "string", "description": "客户简称"},
            "sort_code": {"type": "string", "description": "所属分类编码"},
            "domain_code": {"type": "string", "description": "地区编码"},
            "industry": {"type": "string", "description": "所属行业"},
            "contact": {"type": "string", "description": "联系人"},
            "phone": {"type": "string", "description": "固定电话"},
            "fax": {"type": "string", "description": "传真"},
            "mobile": {"type": "string", "description": "手机号码"},
            "devliver_site": {"type": "string", "description": "发货地址"},
            "end_date": {"type": "string", "description": "停用日期，格式：yyyy-MM-dd"},
            "memo": {"type": "string", "description": "备注信息"},
            "ccusexch_name": {"type": "string", "description": "币种"},
            "bcusdomestic": {"type": "string", "description": "是否国内客户"},
            "bcusoverseas": {"type": "string", "description": "是否国外客户"},
            "bserviceattribute": {"type": "string", "description": "服务属性"},
            "self_define1": {"type": "string", "description": "自定义项1"},
            "self_define2": {"type": "string", "description": "自定义项2"},
            "self_define3": {"type": "string", "description": "自定义项3"},
            "self_define4": {"type": "string", "description": "自定义项4"},
            "self_define5": {"type": "string", "description": "自定义项5"},
            "self_define6": {"type": "string", "description": "自定义项6"},
            "self_define7": {"type": "string", "description": "自定义项7"},
            "self_define8": {"type": "string", "description": "自定义项8"},
            "self_define9": {"type": "string", "description": "自定义项9"},
            "self_define10": {"type": "string", "description": "自定义项10"},
            "self_define11": {"type": "string", "description": "自定义项11"},
            "self_define12": {"type": "string", "description": "自定义项12"},
            "self_define13": {"type": "string", "description": "自定义项13"},
            "self_define14": {"type": "string", "description": "自定义项14"},
            "self_define15": {"type": "string", "description": "自定义项15"},
            "self_define16": {"type": "string", "description": "自定义项16"},
            "addresses": {
                "type": "array",
                "description": "收货地址列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "caddcode": {"type": "string", "description": "地址编码"},
                        "bdefault": {"type": "boolean", "description": "是否默认地址"},
                        "cdeliveradd": {"type": "string", "description": "收货地址"},
                        "cenglishadd": {"type": "string", "description": "英文地址"},
                        "cdeliverunit": {"type": "string", "description": "收货单位"},
                        "clinkperson": {"type": "string", "description": "联系人"}
                    }
                }
            },
            "banks": {
                "type": "array",
                "description": "银行账户信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "caccountnum": {"type": "string", "description": "银行账号"},
                        "bdefault": {"type": "boolean", "description": "是否默认账户"},
                        "cbank": {"type": "string", "description": "银行编码"},
                        "cbranch": {"type": "string", "description": "开户支行"},
                        "caccountname": {"type": "string", "description": "账户名称"}
                    }
                }
            },
            "invoicecustomers": {
                "type": "array",
                "description": "开票单位列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "cinvoicecompany": {"type": "string", "description": "开票单位编码"},
                        "bdefault": {"type": "boolean", "description": "是否默认开票单位"}
                    }
                }
            },
            "users": {
                "type": "array",
                "description": "负责员工列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "user_id": {"type": "string", "description": "操作员编码"},
                        "is_self": {"type": "boolean", "description": "是否负责员工"}
                    }
                }
            },
            "auths": {
                "type": "array",
                "description": "数据权限列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "privilege_type": {"type": "string", "description": "管理维度类型编码"},
                        "privilege_id": {"type": "string", "description": "管理维度编码"}
                    }
                }
            }
        },
        "required": ["code", "name"]
    }
}




# region 备份

# # ===================== 添加一个新客户 子数据模型 =====================
# class Addresses(BaseModel):
#     ccuscode: Optional[str] = Field(None, description="客户编码")
#     caddcode: Optional[str] = Field(None, description="收货地址编码")
#     bdefault: Optional[bool] = Field(None, description="默认值（1/0）")
#     cdeliveradd: Optional[str] = Field(None, description="收货地址")
#     cenglishadd: Optional[str] = Field(None, description="英文地址")
#     cenglishadd2: Optional[str] = Field(None, description="英文地址2")
#     cenglishadd3: Optional[str] = Field(None, description="英文地址3")
#     cenglishadd4: Optional[str] = Field(None, description="英文地址4")
#     cdeliverunit: Optional[str] = Field(None, description="收货单位")
#     clinkperson: Optional[str] = Field(None, description="联系人")

# class Banks(BaseModel):
#     ccuscode: Optional[str] = Field(None, description="客户编码")
#     caccountnum: Optional[str] = Field(None, description="银行账号")
#     bdefault: Optional[bool] = Field(None, description="默认值（1/0）")
#     cbank: Optional[str] = Field(None, description="所属银行编码")
#     cbranch: Optional[str] = Field(None, description="开户银行")
#     caccountname: Optional[str] = Field(None, description="账户名称")
#     cCusPrinvince: Optional[str] = Field(None, description="省")
#     cCusCity: Optional[str] = Field(None, description="市")
#     cCusCBBDepId: Optional[str] = Field(None, description="机构号")
#     cCusBranchId: Optional[str] = Field(None, description="联行号")
#     cCusBranchIdSec: Optional[str] = Field(None, description="联行号II")

# class Invoicecustomers(BaseModel):
#     ccuscode: Optional[str] = Field(None, description="客户编码")
#     cinvoicecompany: Optional[str] = Field(None, description="开票单位编码")
#     bdefault: Optional[bool] = Field(None, description="默认值（1/0）")

# class Users(BaseModel):
#     ccuscode: Optional[str] = Field(None, description="客户编码")
#     user_id: Optional[str] = Field(None, description="操作员编码")
#     is_self: Optional[bool] = Field(None, description="相关或负责员工(true/false)")

# class Auths(BaseModel):
#     ccuscode: Optional[str] = Field(None, description="客户编码")
#     privilege_type: Optional[str] = Field(None, description="管理维度类型编码")
#     privilege_id: Optional[str] = Field(None, description="管理维度编码")

# # ===================== 添加一个新客户 主数据模型 =====================
# class AddCustomerInfoInput(BaseModel):
#     code: str = Field(..., description="客户编码")
#     name: str = Field(..., description="客户名称")
#     abbrname: Optional[str] = Field(None, description="客户简称")
#     sort_code: Optional[str] = Field(None, description="所属分类码")
#     domain_code: Optional[str] = Field(None, description="所属地区码")
#     industry: Optional[str] = Field(None, description="所属行业")
#     contact: Optional[str] = Field(None, description="联系人")
#     phone: Optional[str] = Field(None, description="电话")
#     fax: Optional[str] = Field(None, description="传真")
#     mobile: Optional[str] = Field(None, description="手机")
#     devliver_site: Optional[str] = Field(None, description="发货地址")
#     end_date: Optional[str] = Field(None, description="停用日期")
#     memo: Optional[str] = Field(None, description="备注")
#     ccusexch_name: Optional[str] = Field(None, description="币种")
#     bcusdomestic: Optional[str] = Field(None, description="国内")
#     bcusoverseas: Optional[str] = Field(None, description="国外")
#     bserviceattribute: Optional[str] = Field(None, description="服务")
#     self_define1: Optional[str] = Field(None, description="自定义项1")
#     self_define2: Optional[str] = Field(None, description="自定义项2")
#     self_define3: Optional[str] = Field(None, description="自定义项3")
#     self_define4: Optional[str] = Field(None, description="自定义项4")
#     self_define5: Optional[str] = Field(None, description="自定义项5")
#     self_define6: Optional[str] = Field(None, description="自定义项6")
#     self_define7: Optional[str] = Field(None, description="自定义项7")
#     self_define8: Optional[str] = Field(None, description="自定义项8")
#     self_define9: Optional[str] = Field(None, description="自定义项9")
#     self_define10: Optional[str] = Field(None, description="自定义项10")
#     self_define11: Optional[str] = Field(None, description="自定义项11")
#     self_define12: Optional[str] = Field(None, description="自定义项12")
#     self_define13: Optional[str] = Field(None, description="自定义项13")
#     self_define14: Optional[str] = Field(None, description="自定义项14")
#     self_define15: Optional[str] = Field(None, description="自定义项15")
#     self_define16: Optional[str] = Field(None, description="自定义项16")
#     addresses: Optional[List[Addresses]] = Field(None, description="地址信息列表")
#     banks: Optional[List[Banks]] = Field(None, description="银行信息列表")
#     invoicecustomers: Optional[List[Invoicecustomers]] = Field(None, description="开票客户信息列表")
#     users: Optional[List[Users]] = Field(None, description="用户信息列表")
#     auths: Optional[List[Auths]] = Field(None, description="权限信息列表")

# # ===================== 获取单个客户信息 数据模型 =====================
# class GetCustomerInfoInput(BaseModel):
#     id: str = Field(..., description="客户编号，用于查询客户详细信息")


# # ===================== 添加一个新客户 Tool函数 =====================
# def u8_customer_add_tool(input_data: AddCustomerInfoInput, client: U8OpenAPIClient) -> str:
#     """
#     向用友U8系统中添加新的客户信息。
#     """
#     # 1. 构造接口要求的标准 JSON 结构（外层必须包一层 customer）
#     request_body: Dict[str, Any] = {
#         "customer": input_data.model_dump(exclude_none=True)
#     }

#     # 2. 固定接口路径
#     api_path = "/api/customer/add" 
    
#     try:
#         # 3. 核心：POST 请求
#         # inparams = None（公共参数由 U8OpenAPIClient 自动拼接）
#         # json_body = 完整的请求体（必须带外层 customer）

#         result = client.request_api("POST", api_path, inparams=None, json_body=request_body,is_tradeid=True)
        
#         # 4. 统一返回格式
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "接口调用失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "客户新增成功",
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# # ===================== 获取单个客户信息 Tool函数 =====================
# def u8_customer_get_tool(input_data: GetCustomerInfoInput, client: U8OpenAPIClient) -> str:
#     """
#     通过客户编号获取用友U8中的客户基本信息。
#     """
#     params = {
#         "id": input_data.id 
#     }
    
#     # 修改点3: 更新API路径
#     api_path = "/api/customer/get" 
    
#     try:
#         # 使用 GET 或 POST 取决于具体接口要求，这里假设是 GET 请求带参数
#         result = client.request_api("GET", api_path, inparams=params)
        
#         # 检查业务错误码 (可选，根据实际返回结构调整)
#         if result.get("errcode") != "0":
#             return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
#         return json.dumps(result, ensure_ascii=False)
#     except Exception as e:
#         return json.dumps({"error": str(e)}, ensure_ascii=False)
 






# # ===================== 添加一个新客户 Schema定义 =====================
# U8_CUSTOMER_ADD_SCHEMA = {
#     "name": "u8_customer_add",
#     "description": "在用友U8 OpenAPI中新增客户，支持客户主信息、地址、银行账户、开票单位、负责员工、数据权限等完整信息录入",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             # 客户主信息
#             "code": {
#                 "type": "string",
#                 "description": "客户编码（必填）"
#             },
#             "name": {
#                 "type": "string",
#                 "description": "客户名称（必填）"
#             },
#             "abbrname": {
#                 "type": "string",
#                 "description": "客户简称"
#             },
#             "sort_code": {
#                 "type": "string",
#                 "description": "所属分类编码"
#             },
#             "domain_code": {
#                 "type": "string",
#                 "description": "地区编码"
#             },
#             "industry": {
#                 "type": "string",
#                 "description": "所属行业"
#             },
#             "contact": {
#                 "type": "string",
#                 "description": "联系人"
#             },
#             "phone": {
#                 "type": "string",
#                 "description": "固定电话"
#             },
#             "fax": {
#                 "type": "string",
#                 "description": "传真"
#             },
#             "mobile": {
#                 "type": "string",
#                 "description": "手机号码"
#             },
#             "devliver_site": {
#                 "type": "string",
#                 "description": "发货地址"
#             },
#             "end_date": {
#                 "type": "string",
#                 "description": "停用日期，格式：yyyy-MM-dd"
#             },
#             "memo": {
#                 "type": "string",
#                 "description": "备注信息"
#             },
#             "ccusexch_name": {
#                 "type": "string",
#                 "description": "币种"
#             },
#             "bcusdomestic": {
#                 "type": "string",
#                 "description": "是否国内客户"
#             },
#             "bcusoverseas": {
#                 "type": "string",
#                 "description": "是否国外客户"
#             },
#             "bserviceattribute": {
#                 "type": "string",
#                 "description": "服务属性"
#             },

#             # 自定义项 1~16
#             "self_define1": {"type": "string", "description": "自定义项1"},
#             "self_define2": {"type": "string", "description": "自定义项2"},
#             "self_define3": {"type": "string", "description": "自定义项3"},
#             "self_define4": {"type": "string", "description": "自定义项4"},
#             "self_define5": {"type": "string", "description": "自定义项5"},
#             "self_define6": {"type": "string", "description": "自定义项6"},
#             "self_define7": {"type": "string", "description": "自定义项7"},
#             "self_define8": {"type": "string", "description": "自定义项8"},
#             "self_define9": {"type": "string", "description": "自定义项9"},
#             "self_define10": {"type": "string", "description": "自定义项10"},
#             "self_define11": {"type": "string", "description": "自定义项11"},
#             "self_define12": {"type": "string", "description": "自定义项12"},
#             "self_define13": {"type": "string", "description": "自定义项13"},
#             "self_define14": {"type": "string", "description": "自定义项14"},
#             "self_define15": {"type": "string", "description": "自定义项15"},
#             "self_define16": {"type": "string", "description": "自定义项16"},

#             # 子表：收货地址（数组）
#             "addresses": {
#                 "type": "array",
#                 "description": "收货地址列表",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "ccuscode": {"type": "string", "description": "客户编码"},
#                         "caddcode": {"type": "string", "description": "地址编码"},
#                         "bdefault": {"type": "boolean", "description": "是否默认地址：true=是，false=否"},
#                         "cdeliveradd": {"type": "string", "description": "收货地址"},
#                         "cenglishadd": {"type": "string", "description": "英文地址"},
#                         "cenglishadd2": {"type": "string", "description": "英文地址2"},
#                         "cenglishadd3": {"type": "string", "description": "英文地址3"},
#                         "cenglishadd4": {"type": "string", "description": "英文地址4"},
#                         "cdeliverunit": {"type": "string", "description": "收货单位"},
#                         "clinkperson": {"type": "string", "description": "联系人"}
#                     }
#                 }
#             },

#             # 子表：银行账户（数组）
#             "banks": {
#                 "type": "array",
#                 "description": "银行账户信息列表",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "ccuscode": {"type": "string", "description": "客户编码"},
#                         "caccountnum": {"type": "string", "description": "银行账号"},
#                         "bdefault": {"type": "boolean", "description": "是否默认账户：true=是，false=否"},
#                         "cbank": {"type": "string", "description": "银行编码"},
#                         "cbranch": {"type": "string", "description": "开户支行"},
#                         "caccountname": {"type": "string", "description": "账户名称"},
#                         "cCusPrinvince": {"type": "string", "description": "省"},
#                         "cCusCity": {"type": "string", "description": "市"},
#                         "cCusCBBDepId": {"type": "string", "description": "机构号"},
#                         "cCusBranchId": {"type": "string", "description": "联行号"},
#                         "cCusBranchIdSec": {"type": "string", "description": "联行号II"}
#                     }
#                 }
#             },

#             # 子表：开票单位（数组）
#             "invoicecustomers": {
#                 "type": "array",
#                 "description": "开票单位列表",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "ccuscode": {"type": "string", "description": "客户编码"},
#                         "cinvoicecompany": {"type": "string", "description": "开票单位编码"},
#                         "bdefault": {"type": "boolean", "description": "是否默认开票单位"}
#                     }
#                 }
#             },

#             # 子表：负责员工（数组）
#             "users": {
#                 "type": "array",
#                 "description": "负责员工列表",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "ccuscode": {"type": "string", "description": "客户编码"},
#                         "user_id": {"type": "string", "description": "操作员编码"},
#                         "is_self": {"type": "boolean", "description": "是否负责员工：true=是，false=否"}
#                     }
#                 }
#             },

#             # 子表：数据权限（数组）
#             "auths": {
#                 "type": "array",
#                 "description": "数据权限列表",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "ccuscode": {"type": "string", "description": "客户编码"},
#                         "privilege_type": {"type": "string", "description": "管理维度类型编码"},
#                         "privilege_id": {"type": "string", "description": "管理维度编码"}
#                     }
#                 }
#             }
#         },
#         "required": [
#             "code",
#             "name"
#         ]
#     }
# }

# # ===================== 获取单个客户信息 Schema定义 =====================
# U8_CUSTOMER_GET_SCHEMA = {
#     "name": "u8_customer_get",
#     "description": "通过客户编码获得客户信息",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "id": {
#                 "type": "string",
#                 "description": "客户编码"
#             }
#         },
#         "required": ["id"]
#     }
# }

# endregion

# endregion

# region 客户分类


# ===================== 获取单个客户分类 数据模型 =====================
class GetCustomerclassInput(BaseModel):
    """获取单个客户分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="客户分类编码（必填）")


# ===================== 批量获取客户分类 数据模型 =====================
class BatchGetCustomerclassInput(BaseModel):
    """批量获取客户分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")
    timestamp_begin: Optional[int] = Field(None, description="起始时间戳")
    timestamp_end: Optional[int] = Field(None, description="结束时间戳")


# ===================== 添加客户分类 数据模型 =====================
class AddCustomerclassInput(BaseModel):
    """添加客户分类输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code: str = Field(..., description="客户分类编码（必填）")
    name: str = Field(..., description="客户分类名称（必填）")
    rank: Optional[int] = Field(None, description="客户分类编码级次")
    end_rank_flag: Optional[bool] = Field(None, description="末级标志")
    biz_id: str = Field(..., description="上游id，需要保证biz_id与ERP主键唯一对应关系（必填）")
    sync: Optional[int] = Field(None, description="0=异步新增（默认）;1=同步新增")




# ===================== 获取单个客户分类 Tool函数 =====================
def u8_customerclass_get_tool(input_data: GetCustomerclassInput, client: U8OpenAPIClient) -> str:
    """
    获取单个客户分类信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户分类接口路径
    api_path = "/api/customerclass/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取客户分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取客户分类成功",
            "data": {
                "customerclass": result.get("customerclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取客户分类 Tool函数 =====================
def u8_customerclass_batch_get_tool(input_data: BatchGetCustomerclassInput, client: U8OpenAPIClient) -> str:
    """
    批量获取客户分类列表，支持分页、编码范围、名称等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户分类接口路径
    api_path = "/api/customerclass/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取客户分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取客户分类成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customerclass": result.get("customerclass")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 添加客户分类 Tool函数 =====================
def u8_customerclass_add_tool(input_data: AddCustomerclassInput, client: U8OpenAPIClient) -> str:
    """
    添加一个新客户分类。
    """
    # 提取biz_id和sync参数
    params_data = input_data.model_dump(exclude_none=True)
    biz_id = params_data.pop("biz_id", None)
    sync = params_data.pop("sync", None)
    ds_sequence = params_data.pop("ds_sequence", None)

    # 构造接口要求的标准 JSON 结构（外层包一层 customerclass）
    request_body: dict = {
        "customerclass": params_data
    }

    # 构造URL参数
    inparams = {"biz_id": biz_id}
    if sync is not None:
        inparams["sync"] = sync
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence

    # 客户分类接口路径
    api_path = "/api/customerclass/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "添加客户分类失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "添加客户分类成功",
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




# ===================== 获取单个客户分类 Schema定义 =====================
U8_CUSTOMERCLASS_GET_SCHEMA = {
    "name": "u8_customerclass_get",
    "description": "获取单个客户分类信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "客户分类编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取客户分类 Schema定义 =====================
U8_CUSTOMERCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_customerclass_batch_get",
    "description": "批量获取客户分类列表，支持分页、编码范围、名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"},
            "timestamp_begin": {"type": "number", "description": "起始时间戳"},
            "timestamp_end": {"type": "number", "description": "结束时间戳"}
        },
        "required": []
    }
}


# ===================== 添加客户分类 Schema定义 =====================
U8_CUSTOMERCLASS_ADD_SCHEMA = {
    "name": "u8_customerclass_add",
    "description": "添加一个新客户分类",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code": {"type": "string", "description": "客户分类编码（必填）"},
            "name": {"type": "string", "description": "客户分类名称（必填）"},
            "rank": {"type": "number", "description": "客户分类编码级次"},
            "end_rank_flag": {"type": "boolean", "description": "末级标志"},
            "biz_id": {"type": "string", "description": "上游id，需要保证biz_id与ERP主键唯一对应关系（必填）"},
            "sync": {"type": "number", "description": "0=异步新增（默认）;1=同步新增"}
        },
        "required": ["code", "name", "biz_id"]
    }
}


# endregion

# region 客户地址


# ===================== 获取单个客户地址 数据模型 =====================
class GetCustomeraddressInput(BaseModel):
    """获取单个客户地址输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="主键（必填）")


# ===================== 批量获取客户地址 数据模型 =====================
class BatchGetCustomeraddressInput(BaseModel):
    """批量获取客户地址输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    caddcode_begin: Optional[str] = Field(None, description="起始收货地址编码")
    caddcode_end: Optional[str] = Field(None, description="结束收货地址编码")
    cdeliveradd: Optional[str] = Field(None, description="收货地址")
    cenglishadd: Optional[str] = Field(None, description="英文地址")
    cenglishadd2: Optional[str] = Field(None, description="英文地址2")
    cenglishadd3: Optional[str] = Field(None, description="英文地址3")
    cenglishadd4: Optional[str] = Field(None, description="英文地址4")
    cdeliverunit: Optional[str] = Field(None, description="收货单位")
    clinkpersoncode: Optional[str] = Field(None, description="联系人编码")
    clinkpersonname: Optional[str] = Field(None, description="联系人名称关键字")
    bdefault: Optional[bool] = Field(None, description="默认地址")
    caddcode: Optional[str] = Field(None, description="收货地址编码")
    ccuscode: Optional[str] = Field(None, description="客户编码")
    ccusname: Optional[str] = Field(None, description="客户名称关键字")




# ===================== 获取单个客户地址 Tool函数 =====================
def u8_customeraddress_get_tool(input_data: GetCustomeraddressInput, client: U8OpenAPIClient) -> str:
    """
    获取单个客户地址信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户地址接口路径
    api_path = "/api/customeraddress/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取客户地址失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取客户地址成功",
            "data": {
                "customeraddress": result.get("customeraddress")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取客户地址 Tool函数 =====================
def u8_customeraddress_batch_get_tool(input_data: BatchGetCustomeraddressInput, client: U8OpenAPIClient) -> str:
    """
    批量获取客户地址列表，支持多条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户地址接口路径
    api_path = "/api/customeraddress/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取客户地址失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取客户地址成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customeraddress": result.get("customeraddress")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取单个客户地址 Schema定义 =====================
U8_CUSTOMERADDRESS_GET_SCHEMA = {
    "name": "u8_customeraddress_get",
    "description": "获取单个客户地址信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "主键（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取客户地址 Schema定义 =====================
U8_CUSTOMERADDRESS_BATCH_GET_SCHEMA = {
    "name": "u8_customeraddress_batch_get",
    "description": "批量获取客户地址列表，支持多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "caddcode_begin": {"type": "string", "description": "起始收货地址编码"},
            "caddcode_end": {"type": "string", "description": "结束收货地址编码"},
            "cdeliveradd": {"type": "string", "description": "收货地址"},
            "cenglishadd": {"type": "string", "description": "英文地址"},
            "cenglishadd2": {"type": "string", "description": "英文地址2"},
            "cenglishadd3": {"type": "string", "description": "英文地址3"},
            "cenglishadd4": {"type": "string", "description": "英文地址4"},
            "cdeliverunit": {"type": "string", "description": "收货单位"},
            "clinkpersoncode": {"type": "string", "description": "联系人编码"},
            "clinkpersonname": {"type": "string", "description": "联系人名称关键字"},
            "bdefault": {"type": "boolean", "description": "默认地址"},
            "caddcode": {"type": "string", "description": "收货地址编码"},
            "ccuscode": {"type": "string", "description": "客户编码"},
            "ccusname": {"type": "string", "description": "客户名称关键字"}
        },
        "required": []
    }
}


# endregion

# region 客户级别

# # ===================== 批量获取客户级别 数据模型 =====================
# class BatchGetCustomerrankInput(BaseModel):
#     """批量获取客户级别输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")




# # ===================== 批量获取客户级别 Tool函数 =====================
# def u8_customerrank_batch_get_tool(input_data: BatchGetCustomerrankInput, client: U8OpenAPIClient) -> str:
#     """
#     客户级别批量查询。
#     """
#     # 构造接口请求参数（仅传递非None的参数）
#     params = input_data.model_dump(exclude_none=True)

#     # 客户级别接口路径
#     api_path = "/api/customerrank/batch_get"

#     try:
#         # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
#         result = client.request_api("GET", api_path, inparams=params)

#         # 统一返回格式
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取客户级别失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取客户级别成功",
#             "data": {
#                 "customerrank": result.get("customerrank")
#             },
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)




# # ===================== 批量获取客户级别 Schema定义 =====================
# U8_CUSTOMERRANK_BATCH_GET_SCHEMA = {
#     "name": "u8_customerrank_batch_get",
#     "description": "客户级别批量查询",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"}
#         },
#         "required": []
#     }
# }


# endregion

# region 客户联系人


# ===================== 获取单个客户联系人 数据模型 =====================
class GetCustomercontactsInput(BaseModel):
    """获取单个客户联系人输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="联系人编码（必填）")


# ===================== 批量获取客户联系人 数据模型 =====================
class BatchGetCustomercontactsInput(BaseModel):
    """批量获取客户联系人输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始联系人编码")
    code_end: Optional[str] = Field(None, description="结束联系人编码")
    name: Optional[str] = Field(None, description="联系人名称关键字")
    ccuscode: Optional[str] = Field(None, description="所属客户编码")
    ccusname: Optional[str] = Field(None, description="所属客户名称关键字")
    direct_leader_code: Optional[str] = Field(None, description="直接上级编码")
    direct_leader: Optional[str] = Field(None, description="直接上级关键字")
    charge_person_code: Optional[str] = Field(None, description="负责人编码")
    charge_person: Optional[str] = Field(None, description="负责人关键字")
    be_main_linker: Optional[bool] = Field(None, description="主要联系人")
    title: Optional[str] = Field(None, description="称呼")
    sex: Optional[str] = Field(None, description="性别")
    birthday: Optional[str] = Field(None, description="生日")
    position: Optional[str] = Field(None, description="职位")
    mobile_phone: Optional[str] = Field(None, description="手机号")
    office_phone: Optional[str] = Field(None, description="办公电话")
    email: Optional[str] = Field(None, description="email关键字")
    favorite: Optional[str] = Field(None, description="个人爱好关键字")
    wechat: Optional[str] = Field(None, description="微信号关键字")
    qq: Optional[str] = Field(None, description="QQ号")
    default: Optional[bool] = Field(None, description="默认联系人")




# ===================== 获取单个客户联系人 Tool函数 =====================
def u8_customercontacts_get_tool(input_data: GetCustomercontactsInput, client: U8OpenAPIClient) -> str:
    """
    获取单个客户联系人信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户联系人接口路径
    api_path = "/api/customercontacts/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取客户联系人失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取客户联系人成功",
            "data": {
                "customercontacts": result.get("customercontacts")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取客户联系人 Tool函数 =====================
def u8_customercontacts_batch_get_tool(input_data: BatchGetCustomercontactsInput, client: U8OpenAPIClient) -> str:
    """
    批量获取客户联系人列表，支持多条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户联系人接口路径
    api_path = "/api/customercontacts/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取客户联系人失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取客户联系人成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customercontacts": result.get("customercontacts")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取单个客户联系人 Schema定义 =====================
U8_CUSTOMERCONTACTS_GET_SCHEMA = {
    "name": "u8_customercontacts_get",
    "description": "获取单个客户联系人信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "联系人编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取客户联系人 Schema定义 =====================
U8_CUSTOMERCONTACTS_BATCH_GET_SCHEMA = {
    "name": "u8_customercontacts_batch_get",
    "description": "批量获取客户联系人列表，支持多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始联系人编码"},
            "code_end": {"type": "string", "description": "结束联系人编码"},
            "name": {"type": "string", "description": "联系人名称关键字"},
            "ccuscode": {"type": "string", "description": "所属客户编码"},
            "ccusname": {"type": "string", "description": "所属客户名称关键字"},
            "direct_leader_code": {"type": "string", "description": "直接上级编码"},
            "direct_leader": {"type": "string", "description": "直接上级关键字"},
            "charge_person_code": {"type": "string", "description": "负责人编码"},
            "charge_person": {"type": "string", "description": "负责人关键字"},
            "be_main_linker": {"type": "boolean", "description": "主要联系人"},
            "title": {"type": "string", "description": "称呼"},
            "sex": {"type": "string", "description": "性别"},
            "birthday": {"type": "string", "description": "生日"},
            "position": {"type": "string", "description": "职位"},
            "mobile_phone": {"type": "string", "description": "手机号"},
            "office_phone": {"type": "string", "description": "办公电话"},
            "email": {"type": "string", "description": "email关键字"},
            "favorite": {"type": "string", "description": "个人爱好关键字"},
            "wechat": {"type": "string", "description": "微信号关键字"},
            "qq": {"type": "string", "description": "QQ号"},
            "default": {"type": "boolean", "description": "默认联系人"}
        },
        "required": []
    }
}


# endregion

# region 客户银行


# ===================== 获取单个客户银行 数据模型 =====================
class GetCustomerBankInput(BaseModel):
    """获取单个客户银行输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="客户编码（必填）")


# ===================== 批量获取客户银行 数据模型 =====================
class BatchGetCustomerBankInput(BaseModel):
    """批量获取客户银行输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始客户编码")
    code_end: Optional[str] = Field(None, description="结束客户编码")
    cbank: Optional[str] = Field(None, description="所属银行")
    cbranch: Optional[str] = Field(None, description="开户银行")
    caccountnum: Optional[str] = Field(None, description="银行账号")




# ===================== 获取单个客户银行 Tool函数 =====================
def u8_customer_bank_get_tool(input_data: GetCustomerBankInput, client: U8OpenAPIClient) -> str:
    """
    获取单个客户银行信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户银行接口路径
    api_path = "/api/customer_bank/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取客户银行失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取客户银行成功",
            "data": {
                "customer_bank": result.get("customer_bank")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取客户银行 Tool函数 =====================
def u8_customer_bank_batch_get_tool(input_data: BatchGetCustomerBankInput, client: U8OpenAPIClient) -> str:
    """
    批量获取客户银行列表，支持编码范围、银行等条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 客户银行接口路径
    api_path = "/api/customer_bank/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取客户银行失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取客户银行成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "customer_bank": result.get("customer_bank")
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)




# ===================== 获取单个客户银行 Schema定义 =====================
U8_CUSTOMER_BANK_GET_SCHEMA = {
    "name": "u8_customer_bank_get",
    "description": "获取单个客户银行信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "客户编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取客户银行 Schema定义 =====================
U8_CUSTOMER_BANK_BATCH_GET_SCHEMA = {
    "name": "u8_customer_bank_batch_get",
    "description": "批量获取客户银行列表，支持编码范围、银行等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始客户编码"},
            "code_end": {"type": "string", "description": "结束客户编码"},
            "cbank": {"type": "string", "description": "所属银行"},
            "cbranch": {"type": "string", "description": "开户银行"},
            "caccountnum": {"type": "string", "description": "银行账号"}
        },
        "required": []
    }
}


# endregion

# region 币种
# ===================== 获取单个币种信息 数据模型 =====================
class GetCurrencyInfoInput(BaseModel):
    """获取单个币种信息输入模型"""
    id: str = Field(..., description="币种编码")

# ===================== 批量获取币种信息 数据模型 =====================
class BatchGetCurrencyInfoInput(BaseModel):
    """批量获取币种信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cexch_name: Optional[str] = Field(None, description="币种名称关键字")
    cexch_code: Optional[str] = Field(None, description="币种编码")

# ===================== 获取单个币种信息 Tool函数 =====================
def u8_currency_get_tool(input_data: GetCurrencyInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过币种编码获取用友U8中的币种信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/currency/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取币种信息 Tool函数 =====================
def u8_currency_batch_get_tool(input_data: BatchGetCurrencyInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的币种信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/currency/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取币种信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取币种信息成功",
            "data": result.get("currencys", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个币种信息 Schema定义 =====================
U8_CURRENCY_GET_SCHEMA = {
    "name": "u8_currency_get",
    "description": "通过币种编码获取币种信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "币种编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取币种信息 Schema定义 =====================
U8_CURRENCY_BATCH_GET_SCHEMA = {
    "name": "u8_currency_batch_get",
    "description": "批量获取币种信息列表，支持按币种名称、币种编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cexch_name": {"type": "string", "description": "币种名称关键字"},
            "cexch_code": {"type": "string", "description": "币种编码"}
        },
        "required": []
    }
}

# endregion

# region 常用摘要

# ===================== 获取单个常用摘要信息 数据模型 =====================
class GetDigestInfoInput(BaseModel):
    """获取单个常用摘要信息输入模型"""
    id: str = Field(..., description="常用摘要编码")

# ===================== 批量获取常用摘要信息 数据模型 =====================
class BatchGetDigestInfoInput(BaseModel):
    """批量获取常用摘要信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="常用摘要编码")
    cname: Optional[str] = Field(None, description="常用摘要名称关键字")

# ===================== 获取单个常用摘要信息 Tool函数 =====================
def u8_digest_get_tool(input_data: GetDigestInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过常用摘要编码获取用友U8中的常用摘要信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/digest/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取常用摘要信息 Tool函数 =====================
def u8_digest_batch_get_tool(input_data: BatchGetDigestInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的常用摘要信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/digest/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取常用摘要信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取常用摘要信息成功",
            "data": result.get("digests", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个常用摘要信息 Schema定义 =====================
U8_DIGEST_GET_SCHEMA = {
    "name": "u8_digest_get",
    "description": "通过常用摘要编码获取常用摘要信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "常用摘要编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取常用摘要信息 Schema定义 =====================
U8_DIGEST_BATCH_GET_SCHEMA = {
    "name": "u8_digest_batch_get",
    "description": "批量获取常用摘要信息列表，支持按摘要编码、摘要名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "常用摘要编码"},
            "cname": {"type": "string", "description": "常用摘要名称关键字"}
        },
        "required": []
    }
}

# endregion

# region 批次档案

# ===================== 批量获取批次档案信息 数据模型 =====================
class BatchGetBatchPropertyInfoInput(BaseModel):
    """批量获取批次档案信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cinvcode: Optional[str] = Field(None, description="存货编码")
    cbatch: Optional[str] = Field(None, description="批次号")
    cwhcode: Optional[str] = Field(None, description="仓库编码")

# ===================== 批量获取批次档案信息 Tool函数 =====================
def u8_batchproperty_batch_get_tool(input_data: BatchGetBatchPropertyInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的批次档案信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/batchproperty/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取批次档案信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取批次档案信息成功",
            "data": result.get("batchpropertys", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 批量获取批次档案信息 Schema定义 =====================
U8_BATCHPROPERTY_BATCH_GET_SCHEMA = {
    "name": "u8_batchproperty_batch_get",
    "description": "批量获取批次档案信息列表，支持按存货编码、批次号、仓库编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cinvcode": {"type": "string", "description": "存货编码"},
            "cbatch": {"type": "string", "description": "批次号"},
            "cwhcode": {"type": "string", "description": "仓库编码"}
        },
        "required": []
    }
}

# endregion

# region 收付款协议档案

# ===================== 获取单个收付款协议信息 数据模型 =====================
class GetAgreementInfoInput(BaseModel):
    """获取单个收付款协议信息输入模型"""
    id: str = Field(..., description="收付款协议编码")

# ===================== 批量获取收付款协议信息 数据模型 =====================
class BatchGetAgreementInfoInput(BaseModel):
    """批量获取收付款协议信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="收付款协议编码")
    cname: Optional[str] = Field(None, description="收付款协议名称关键字")

# ===================== 获取单个收付款协议信息 Tool函数 =====================
def u8_agreement_get_tool(input_data: GetAgreementInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过收付款协议编码获取用友U8中的收付款协议信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/agreement/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取收付款协议信息 Tool函数 =====================
def u8_agreement_batch_get_tool(input_data: BatchGetAgreementInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的收付款协议信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/agreement/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取收付款协议信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取收付款协议信息成功",
            "data": result.get("agreements", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个收付款协议信息 Schema定义 =====================
U8_AGREEMENT_GET_SCHEMA = {
    "name": "u8_agreement_get",
    "description": "通过收付款协议编码获取收付款协议信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "收付款协议编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取收付款协议信息 Schema定义 =====================
U8_AGREEMENT_BATCH_GET_SCHEMA = {
    "name": "u8_agreement_batch_get",
    "description": "批量获取收付款协议信息列表，支持按协议编码、协议名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "收付款协议编码"},
            "cname": {"type": "string", "description": "收付款协议名称关键字"}
        },
        "required": []
    }
}


# endregion

# region 收发类别

# ===================== 获取单个收发类别信息 数据模型 =====================
class GetReceiveSendTypeInfoInput(BaseModel):
    """获取单个收发类别信息输入模型"""
    id: str = Field(..., description="收发类别编码")

# ===================== 批量获取收发类别信息 数据模型 =====================
class BatchGetReceiveSendTypeInfoInput(BaseModel):
    """批量获取收发类别信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="收发类别编码")
    cname: Optional[str] = Field(None, description="收发类别名称关键字")
    rd_style: Optional[str] = Field(None, description="收发标志（收/发）")

# ===================== 获取单个收发类别信息 Tool函数 =====================
def u8_receivesendtype_get_tool(input_data: GetReceiveSendTypeInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过收发类别编码获取用友U8中的收发类别信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/receivesendtype/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取收发类别信息 Tool函数 =====================
def u8_receivesendtype_batch_get_tool(input_data: BatchGetReceiveSendTypeInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的收发类别信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/receivesendtype/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取收发类别信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取收发类别信息成功",
            "data": result.get("receivesendtypes", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个收发类别信息 Schema定义 =====================
U8_RECEIVESENDTYPE_GET_SCHEMA = {
    "name": "u8_receivesendtype_get",
    "description": "通过收发类别编码获取收发类别信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "收发类别编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取收发类别信息 Schema定义 =====================
U8_RECEIVESENDTYPE_BATCH_GET_SCHEMA = {
    "name": "u8_receivesendtype_batch_get",
    "description": "批量获取收发类别信息列表，支持按类别编码、类别名称、收发标志等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "收发类别编码"},
            "cname": {"type": "string", "description": "收发类别名称关键字"},
            "rd_style": {"type": "string", "description": "收发标志（收/发）"}
        },
        "required": []
    }
}

# endregion

# region 本单位开户银行

# ===================== 获取单个本单位开户银行信息 数据模型 =====================
class GetAccountingBankInfoInput(BaseModel):
    """获取单个本单位开户银行信息输入模型"""
    id: str = Field(..., description="本单位开户银行编码")

# ===================== 批量获取本单位开户银行信息 数据模型 =====================
class BatchGetAccountingBankInfoInput(BaseModel):
    """批量获取本单位开户银行信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="本单位开户银行编码")
    cname: Optional[str] = Field(None, description="本单位开户银行名称关键字")

# ===================== 添加本单位开户银行信息 数据模型 =====================
class AddAccountingBankInfoInput(BaseModel):
    """添加本单位开户银行信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: str = Field(..., description="本单位开户银行编码（必填）")
    cname: str = Field(..., description="本单位开户银行名称（必填）")
    caccountnum: Optional[str] = Field(None, description="银行账号")
    cbank: Optional[str] = Field(None, description="所属银行编码")
    cbranch: Optional[str] = Field(None, description="开户银行")
    caccountname: Optional[str] = Field(None, description="账户名称")
    cCusPrinvince: Optional[str] = Field(None, description="省")
    cCusCity: Optional[str] = Field(None, description="市")
    cmemo: Optional[str] = Field(None, description="备注")

# ===================== 获取单个本单位开户银行信息 Tool函数 =====================
def u8_accountingbank_get_tool(input_data: GetAccountingBankInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过本单位开户银行编码获取用友U8中的本单位开户银行信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/accountingbank/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取本单位开户银行信息 Tool函数 =====================
def u8_accountingbank_batch_get_tool(input_data: BatchGetAccountingBankInfoInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的本单位开户银行信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/accountingbank/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取本单位开户银行信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取本单位开户银行信息成功",
            "data": result.get("accountingbanks", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 添加本单位开户银行信息 Tool函数 =====================
def u8_accountingbank_add_tool(input_data: AddAccountingBankInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中添加新的本单位开户银行信息。
    """
    params_data = input_data.model_dump(exclude_none=True)
    ds_sequence = params_data.pop("ds_sequence", None)
    
    request_body = {
        "accountingbank": params_data
    }
    
    inparams = {}
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence
    
    api_path = "/api/accountingbank/add"
    
    try:
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body, is_tradeid=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "本单位开户银行新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个本单位开户银行信息 Schema定义 =====================
U8_ACCOUNTINGBANK_GET_SCHEMA = {
    "name": "u8_accountingbank_get",
    "description": "通过本单位开户银行编码获取本单位开户银行信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "本单位开户银行编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取本单位开户银行信息 Schema定义 =====================
U8_ACCOUNTINGBANK_BATCH_GET_SCHEMA = {
    "name": "u8_accountingbank_batch_get",
    "description": "批量获取本单位开户银行信息列表，支持按银行编码、银行名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "本单位开户银行编码"},
            "cname": {"type": "string", "description": "本单位开户银行名称关键字"}
        },
        "required": []
    }
}

# ===================== 添加本单位开户银行信息 Schema定义 =====================
U8_ACCOUNTINGBANK_ADD_SCHEMA = {
    "name": "u8_accountingbank_add",
    "description": "在用友U8 OpenAPI中新增本单位开户银行信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "本单位开户银行编码（必填）"},
            "cname": {"type": "string", "description": "本单位开户银行名称（必填）"},
            "caccountnum": {"type": "string", "description": "银行账号"},
            "cbank": {"type": "string", "description": "所属银行编码"},
            "cbranch": {"type": "string", "description": "开户银行"},
            "caccountname": {"type": "string", "description": "账户名称"},
            "cCusPrinvince": {"type": "string", "description": "省"},
            "cCusCity": {"type": "string", "description": "市"},
            "cmemo": {"type": "string", "description": "备注"}
        },
        "required": ["ccode", "cname"]
    }
}

# endregion

# region 汇率

# ===================== 获取单个汇率信息 数据模型 =====================
class GetExchangeRateInput(BaseModel):
    """获取单个汇率信息输入模型"""
    id: str = Field(..., description="汇率ID")

# ===================== 批量获取汇率信息 数据模型 =====================
class BatchGetExchangeRateInput(BaseModel):
    """批量获取汇率信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    i_id: Optional[int] = Field(None, description="序号")
    iyear: Optional[int] = Field(None, description="年份")
    imonth: Optional[str] = Field(None, description="月份")
    cexch_name: Optional[str] = Field(None, description="币种名称")
    iperiod: Optional[int] = Field(None, description="会计期间")
    itype: Optional[int] = Field(None, description="汇率类型")


# ===================== 获取单个汇率信息 Tool函数 =====================
def u8_exchangerate_get_tool(input_data: GetExchangeRateInput, client: U8OpenAPIClient) -> str:
    """
    通过汇率ID获取用友U8中的汇率信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/exchangerate/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取汇率信息 Tool函数 =====================
def u8_exchangerate_batch_get_tool(input_data: BatchGetExchangeRateInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的汇率信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/exchangerate/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取汇率信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取汇率信息成功",
            "data": result.get("exchangerates", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个汇率信息 Schema定义 =====================
U8_EXCHANGERATE_GET_SCHEMA = {
    "name": "u8_exchangerate_get",
    "description": "通过汇率ID获取汇率信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "汇率ID"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取汇率信息 Schema定义 =====================
U8_EXCHANGERATE_BATCH_GET_SCHEMA = {
    "name": "u8_exchangerate_batch_get",
    "description": "批量获取汇率信息列表，支持按年份、月份、币种名称、会计期间、汇率类型等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "i_id": {"type": "integer", "description": "序号"},
            "iyear": {"type": "integer", "description": "年份"},
            "imonth": {"type": "string", "description": "月份"},
            "cexch_name": {"type": "string", "description": "币种名称"},
            "iperiod": {"type": "integer", "description": "会计期间"},
            "itype": {"type": "integer", "description": "汇率类型"}
        },
        "required": []
    }
}




# endregion

# region 现金流量列表

# # ===================== 现金流量列表子数据模型 =====================
# class CashFlowItem(BaseModel):
#     """现金流量项目明细"""
#     ccode: Optional[str] = Field(None, description="科目编码")
#     ccode_name: Optional[str] = Field(None, description="科目名称")
#     iamount: Optional[float] = Field(None, description="借方金额")
#     md: Optional[float] = Field(None, description="借方本币金额")
#     mc: Optional[float] = Field(None, description="贷方本币金额")

# # ===================== 批量获取现金流量列表 数据模型 =====================
# class BatchGetCashFlowListInput(BaseModel):
#     """批量获取现金流量列表输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     ccode: Optional[str] = Field(None, description="科目编码")
#     ccode_name: Optional[str] = Field(None, description="科目名称关键字")
#     citem_code: Optional[str] = Field(None, description="现金流量项目编码")
#     citem_name: Optional[str] = Field(None, description="现金流量项目名称")
#     cdirection: Optional[str] = Field(None, description="方向（借/贷）")
#     iyear: Optional[int] = Field(None, description="年份")
#     iperiod: Optional[int] = Field(None, description="会计期间")
#     dbill_date_begin: Optional[str] = Field(None, description="起始单据日期，格式：yyyy-MM-dd")
#     dbill_date_end: Optional[str] = Field(None, description="结束单据日期，格式：yyyy-MM-dd")

# # ===================== 批量获取现金流量列表 Tool函数 =====================
# def u8_cashflowlist_batch_get_tool(input_data: BatchGetCashFlowListInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的现金流量列表。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/cashflowlist/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取现金流量列表失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取现金流量列表成功",
#             "data": result.get("cashflowlists", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# ===================== 批量获取现金流量列表 Schema定义 =====================
U8_CASHFLOWLIST_BATCH_GET_SCHEMA = {
    "name": "u8_cashflowlist_batch_get",
    "description": "批量获取现金流量列表，支持按科目、现金流量项目、方向、年份、会计期间、单据日期等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "科目编码"},
            "ccode_name": {"type": "string", "description": "科目名称关键字"},
            "citem_code": {"type": "string", "description": "现金流量项目编码"},
            "citem_name": {"type": "string", "description": "现金流量项目名称"},
            "cdirection": {"type": "string", "description": "方向（借/贷）"},
            "iyear": {"type": "integer", "description": "年份"},
            "iperiod": {"type": "integer", "description": "会计期间"},
            "dbill_date_begin": {"type": "string", "description": "起始单据日期，格式：yyyy-MM-dd"},
            "dbill_date_end": {"type": "string", "description": "结束单据日期，格式：yyyy-MM-dd"}
        },
        "required": []
    }
}

# endregion

# region 现金流量项目
# ===================== 现金流量项目子数据模型 =====================
class CashFlowItemEntry(BaseModel):
    """现金流量项目子表-科目明细"""
    iid: Optional[int] = Field(None, description="序号")
    citemcode: Optional[str] = Field(None, description="项目编码")
    bdatastyle: Optional[str] = Field(None, description="数据类型")
    cdatasource: Optional[str] = Field(None, description="数据源")
    bdir: Optional[str] = Field(None, description="取数方式")
    ccode_name: Optional[str] = Field(None, description="科目名称")

# ===================== 获取单个现金流量项目信息 数据模型 =====================
class GetCashFlowItemInput(BaseModel):
    """获取单个现金流量项目信息输入模型"""
    id: str = Field(..., description="现金流量项目编码")

# ===================== 批量获取现金流量项目信息 数据模型 =====================
class BatchGetCashFlowItemInput(BaseModel):
    """批量获取现金流量项目信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    code_begin: Optional[str] = Field(None, description="起始项目编码")
    code_end: Optional[str] = Field(None, description="结束项目编码")
    citemname: Optional[str] = Field(None, description="项目名称关键字")
    bclose: Optional[bool] = Field(None, description="是否结算")
    citemccode: Optional[str] = Field(None, description="项目大类编码")
    citemcname: Optional[str] = Field(None, description="项目大类名称关键字")
    cdirection: Optional[str] = Field(None, description="方向")
    bdatastyle: Optional[bool] = Field(None, description="数据类型")
    cdatasource: Optional[str] = Field(None, description="数据源")

# ===================== 获取单个现金流量项目信息 Tool函数 =====================
def u8_cashflowitem_get_tool(input_data: GetCashFlowItemInput, client: U8OpenAPIClient) -> str:
    """
    通过现金流量项目编码获取用友U8中的现金流量项目信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/cashflowitem/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取现金流量项目信息 Tool函数 =====================
def u8_cashflowitem_batch_get_tool(input_data: BatchGetCashFlowItemInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的现金流量项目信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/cashflowitem/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取现金流量项目信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取现金流量项目信息成功",
            "data": result.get("cashflowitems", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个现金流量项目信息 Schema定义 =====================
U8_CASHFLOWITEM_GET_SCHEMA = {
    "name": "u8_cashflowitem_get",
    "description": "通过现金流量项目编码获取现金流量项目信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "现金流量项目编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取现金流量项目信息 Schema定义 =====================
U8_CASHFLOWITEM_BATCH_GET_SCHEMA = {
    "name": "u8_cashflowitem_batch_get",
    "description": "批量获取现金流量项目信息列表，支持按项目编码、项目名称、项目大类、方向、是否结算等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "code_begin": {"type": "string", "description": "起始项目编码"},
            "code_end": {"type": "string", "description": "结束项目编码"},
            "citemname": {"type": "string", "description": "项目名称关键字"},
            "bclose": {"type": "boolean", "description": "是否结算"},
            "citemccode": {"type": "string", "description": "项目大类编码"},
            "citemcname": {"type": "string", "description": "项目大类名称关键字"},
            "cdirection": {"type": "string", "description": "方向"},
            "bdatastyle": {"type": "boolean", "description": "数据类型"},
            "cdatasource": {"type": "string", "description": "数据源"}
        },
        "required": []
    }
}

# endregion

# region 种汇率
# # ===================== 批量获取种汇率信息 数据模型 =====================
# class BatchGetExchangeRateExtInput(BaseModel):
#     """批量获取种汇率信息输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     cexch_code: Optional[str] = Field(None, description="币种编码")
#     cexch_name: Optional[str] = Field(None, description="币种名称")
#     iyear: Optional[int] = Field(None, description="年份")
#     imonth: Optional[str] = Field(None, description="月份")
#     iperiod: Optional[int] = Field(None, description="会计期间")

# # ===================== 批量获取种汇率信息 Tool函数 =====================
# def u8_exchangerateext_batch_get_tool(input_data: BatchGetExchangeRateExtInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的种汇率（外币汇率扩展）信息列表。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/exchangerateext/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取种汇率信息失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取种汇率信息成功",
#             "data": result.get("exchangerateexts", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)


# # ===================== 批量获取种汇率信息 Schema定义 =====================
# U8_EXCHANGERATEEXT_BATCH_GET_SCHEMA = {
#     "name": "u8_exchangerateext_batch_get",
#     "description": "批量获取种汇率（外币汇率扩展）信息列表，支持按币种编码、币种名称、年份、月份、会计期间等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "cexch_code": {"type": "string", "description": "币种编码"},
#             "cexch_name": {"type": "string", "description": "币种名称"},
#             "iyear": {"type": "integer", "description": "年份"},
#             "imonth": {"type": "string", "description": "月份"},
#             "iperiod": {"type": "integer", "description": "会计期间"}
#         },
#         "required": []
#     }
# }

# endregion

# region 科目
# ===================== 获取单个科目信息 数据模型 =====================
class GetCodeInput(BaseModel):
    """获取单个科目信息输入模型"""
    id: str = Field(..., description="科目编码")

# ===================== 批量获取科目信息 数据模型 =====================
class BatchGetCodeInput(BaseModel):
    """批量获取科目信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始科目编码")
    code_end: Optional[str] = Field(None, description="结束科目编码")
    name: Optional[str] = Field(None, description="科目名称关键字")
    year: Optional[str] = Field(None, description="年度")

# # ===================== 批量获取科目扩展信息 数据模型 =====================
# class BatchGetCodeExtInput(BaseModel):
#     """批量获取科目扩展信息输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     page_index: Optional[str] = Field(None, description="页号")
#     rows_per_page: Optional[str] = Field(None, description="每页行数")
#     code_begin: Optional[str] = Field(None, description="起始科目编码")
#     code_end: Optional[str] = Field(None, description="结束科目编码")
#     name: Optional[str] = Field(None, description="科目名称关键字")
#     year: Optional[str] = Field(None, description="年度")

# ===================== 获取单个科目信息 Tool函数 =====================
def u8_code_get_tool(input_data: GetCodeInput, client: U8OpenAPIClient) -> str:
    """
    通过科目编码获取用友U8中的科目信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/code/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取科目信息 Tool函数 =====================
def u8_code_batch_get_tool(input_data: BatchGetCodeInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的科目信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/code/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取科目信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取科目信息成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "codes": result.get("codes", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# # ===================== 批量获取科目扩展信息 Tool函数 =====================
# def u8_codeext_batch_get_tool(input_data: BatchGetCodeExtInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的科目扩展信息列表。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/codeext/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取科目扩展信息失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取科目扩展信息成功",
#             "data": {
#                 "page_index": result.get("page_index"),
#                 "rows_per_page": result.get("rows_per_page"),
#                 "row_count": result.get("row_count"),
#                 "page_count": result.get("page_count"),
#                 "codeexts": result.get("codeexts", [])
#             },
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# ===================== 获取单个科目信息 Schema定义 =====================
U8_CODE_GET_SCHEMA = {
    "name": "u8_code_get",
    "description": "通过科目编码获取科目信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "科目编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取科目信息 Schema定义 =====================
U8_CODE_BATCH_GET_SCHEMA = {
    "name": "u8_code_batch_get",
    "description": "批量获取科目信息列表，支持按科目编码范围、科目名称、年度等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始科目编码"},
            "code_end": {"type": "string", "description": "结束科目编码"},
            "name": {"type": "string", "description": "科目名称关键字"},
            "year": {"type": "string", "description": "年度"}
        },
        "required": []
    }
}

# # ===================== 批量获取科目扩展信息 Schema定义 =====================
# U8_CODEEXT_BATCH_GET_SCHEMA = {
#     "name": "u8_codeext_batch_get",
#     "description": "批量获取科目扩展信息列表，支持按科目编码范围、科目名称、年度等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "page_index": {"type": "string", "description": "页号"},
#             "rows_per_page": {"type": "string", "description": "每页行数"},
#             "code_begin": {"type": "string", "description": "起始科目编码"},
#             "code_end": {"type": "string", "description": "结束科目编码"},
#             "name": {"type": "string", "description": "科目名称关键字"},
#             "year": {"type": "string", "description": "年度"}
#         },
#         "required": []
#     }
# }

# endregion

# region 科目与辅助核算关系（ys专用）

# # ===================== 批量获取科目与辅助核算关系 数据模型 =====================
# class BatchGetCodeAuxiliaryRelationInput(BaseModel):
#     """批量获取科目与辅助核算关系输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     ccode: Optional[str] = Field(None, description="科目编码")
#     ccode_name: Optional[str] = Field(None, description="科目名称关键字")
#     cacc_id: Optional[str] = Field(None, description="辅助核算ID")
#     cacc_name: Optional[str] = Field(None, description="辅助核算名称")
#     cacc_code: Optional[str] = Field(None, description="辅助核算编码")
#     cacc_standard_name: Optional[str] = Field(None, description="辅助核算标准名称")

# # ===================== 批量获取科目与辅助核算关系 Tool函数 =====================
# def u8_code_auxiliary_relation_ys_batch_get_tool(input_data: BatchGetCodeAuxiliaryRelationInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的科目与辅助核算关系信息列表（ys专用）。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/code_auxiliary_relation_ys/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取科目与辅助核算关系失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取科目与辅助核算关系成功",
#             "data": result.get("code_auxiliary_relation", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# # ===================== 批量获取科目与辅助核算关系 Schema定义 =====================
# U8_CODE_AUXILIARY_RELATION_YS_BATCH_GET_SCHEMA = {
#     "name": "u8_code_auxiliary_relation_ys_batch_get",
#     "description": "批量获取科目与辅助核算关系信息列表（ys专用），支持按科目编码、科目名称、辅助核算ID、辅助核算名称等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "ccode": {"type": "string", "description": "科目编码"},
#             "ccode_name": {"type": "string", "description": "科目名称关键字"},
#             "cacc_id": {"type": "string", "description": "辅助核算ID"},
#             "cacc_name": {"type": "string", "description": "辅助核算名称"},
#             "cacc_code": {"type": "string", "description": "辅助核算编码"},
#             "cacc_standard_name": {"type": "string", "description": "辅助核算标准名称"}
#         },
#         "required": []
#     }
# }

# endregion

# region 科目分类

# # ===================== 批量获取科目分类信息 数据模型 =====================
# class BatchGetCodeClassInput(BaseModel):
#     """批量获取科目分类信息输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     cclass_code: Optional[str] = Field(None, description="科目分类编码")
#     cclass_name: Optional[str] = Field(None, description="科目分类名称关键字")
#     igrade: Optional[int] = Field(None, description="级次")
#     bclass_end: Optional[bool] = Field(None, description="是否末级分类")

# # ===================== 批量获取科目分类信息 Tool函数 =====================
# def u8_codeclass_batch_get_tool(input_data: BatchGetCodeClassInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的科目分类信息列表。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/codeclass/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取科目分类信息失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取科目分类信息成功",
#             "data": result.get("codeclasss", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# # ===================== 批量获取科目分类信息 Schema定义 =====================
# U8_CODECLASS_BATCH_GET_SCHEMA = {
#     "name": "u8_codeclass_batch_get",
#     "description": "批量获取科目分类信息列表，支持按分类编码、分类名称、级次、是否末级分类等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "cclass_code": {"type": "string", "description": "科目分类编码"},
#             "cclass_name": {"type": "string", "description": "科目分类名称关键字"},
#             "igrade": {"type": "integer", "description": "级次"},
#             "bclass_end": {"type": "boolean", "description": "是否末级分类"}
#         },
#         "required": []
#     }
# }

# endregion

# region 科目表（ys专用）

# # ===================== 科目表子数据模型 =====================
# class GlAccountEntry(BaseModel):
#     """科目表明细"""
#     ccode: Optional[str] = Field(None, description="科目编码")
#     ccode_name: Optional[str] = Field(None, description="科目名称")
#     igrade: Optional[int] = Field(None, description="级次")
#     bend: Optional[bool] = Field(None, description="是否末级")
#     bneedacc: Optional[bool] = Field(None, description="是否辅助核算")
#     bcash: Optional[bool] = Field(None, description="是否现金科目")
#     bbank: Optional[bool] = Field(None, description="是否银行科目")
#     bport: Optional[bool] = Field(None, description="是否Portfolio")

# # ===================== 批量获取科目表信息 数据模型 =====================
# class BatchGetGlAccountYsInput(BaseModel):
#     """批量获取科目表信息输入模型（ys专用）"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     ccode: Optional[str] = Field(None, description="科目编码")
#     ccode_name: Optional[str] = Field(None, description="科目名称关键字")
#     igrade: Optional[int] = Field(None, description="级次")
#     bend: Optional[bool] = Field(None, description="是否末级")
#     bneedacc: Optional[bool] = Field(None, description="是否辅助核算")
#     bcash: Optional[bool] = Field(None, description="是否现金科目")
#     bbank: Optional[bool] = Field(None, description="是否银行科目")

# # ===================== 批量获取科目表信息 Tool函数 =====================
# def u8_glaccount_ys_batch_get_tool(input_data: BatchGetGlAccountYsInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的科目表信息列表（ys专用）。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/glaccount_ys/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取科目表信息失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取科目表信息成功",
#             "data": result.get("glaccounts", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# # ===================== 批量获取科目表信息 Schema定义 =====================
# U8_GLACCOUNT_YS_BATCH_GET_SCHEMA = {
#     "name": "u8_glaccount_ys_batch_get",
#     "description": "批量获取科目表信息列表（ys专用），支持按科目编码、科目名称、级次、是否末级、是否辅助核算、是否现金科目、是否银行科目等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "ccode": {"type": "string", "description": "科目编码"},
#             "ccode_name": {"type": "string", "description": "科目名称关键字"},
#             "igrade": {"type": "integer", "description": "级次"},
#             "bend": {"type": "boolean", "description": "是否末级"},
#             "bneedacc": {"type": "boolean", "description": "是否辅助核算"},
#             "bcash": {"type": "boolean", "description": "是否现金科目"},
#             "bbank": {"type": "boolean", "description": "是否银行科目"}
#         },
#         "required": []
#     }
# }


# endregion

# region 结算方式
# ===================== 获取单个结算方式信息 数据模型 =====================
class GetSettleStyleInput(BaseModel):
    """获取单个结算方式信息输入模型"""
    id: str = Field(..., description="结算方式编码")

# ===================== 批量获取结算方式信息 数据模型 =====================
class BatchGetSettleStyleInput(BaseModel):
    """批量获取结算方式信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cscode: Optional[str] = Field(None, description="结算方式编码")
    csname: Optional[str] = Field(None, description="结算方式名称关键字")
    bsdefault: Optional[bool] = Field(None, description="是否默认")

# ===================== 获取单个结算方式信息 Tool函数 =====================
def u8_settlestyle_get_tool(input_data: GetSettleStyleInput, client: U8OpenAPIClient) -> str:
    """
    通过结算方式编码获取用友U8中的结算方式信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/settlestyle/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取结算方式信息 Tool函数 =====================
def u8_settlestyle_batch_get_tool(input_data: BatchGetSettleStyleInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的结算方式信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/settlestyle/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取结算方式信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取结算方式信息成功",
            "data": result.get("settlestyles", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个结算方式信息 Schema定义 =====================
U8_SETTLESTYLE_GET_SCHEMA = {
    "name": "u8_settlestyle_get",
    "description": "通过结算方式编码获取结算方式信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "结算方式编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取结算方式信息 Schema定义 =====================
U8_SETTLESTYLE_BATCH_GET_SCHEMA = {
    "name": "u8_settlestyle_batch_get",
    "description": "批量获取结算方式信息列表，支持按结算方式编码、结算方式名称、是否默认等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cscode": {"type": "string", "description": "结算方式编码"},
            "csname": {"type": "string", "description": "结算方式名称关键字"},
            "bsdefault": {"type": "boolean", "description": "是否默认"}
        },
        "required": []
    }
}

# endregion

# region 编码方案

# ===================== 获取单个编码方案信息 数据模型 =====================
class GetCodeSchemeInput(BaseModel):
    """获取单个编码方案信息输入模型"""
    id: str = Field(..., description="编码方案编号")

# ===================== 批量获取编码方案信息 数据模型 =====================
class BatchGetCodeSchemeInput(BaseModel):
    """批量获取编码方案信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    codeschm_code: Optional[str] = Field(None, description="编码方案编码")
    codeschm_name: Optional[str] = Field(None, description="编码方案名称关键字")
    cobject: Optional[str] = Field(None, description="对象")

# ===================== 获取单个编码方案信息 Tool函数 =====================
def u8_codescheme_get_tool(input_data: GetCodeSchemeInput, client: U8OpenAPIClient) -> str:
    """
    通过编码方案编号获取用友U8中的编码方案信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/codescheme/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取编码方案信息 Tool函数 =====================
def u8_codescheme_batch_get_tool(input_data: BatchGetCodeSchemeInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的编码方案信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/codescheme/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取编码方案信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取编码方案信息成功",
            "data": result.get("codeschemes", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个编码方案信息 Schema定义 =====================
U8_CODESCHEME_GET_SCHEMA = {
    "name": "u8_codescheme_get",
    "description": "通过编码方案编号获取编码方案信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "编码方案编号"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取编码方案信息 Schema定义 =====================
U8_CODESCHEME_BATCH_GET_SCHEMA = {
    "name": "u8_codescheme_batch_get",
    "description": "批量获取编码方案信息列表，支持按编码方案编码、编码方案名称、对象等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "codeschm_code": {"type": "string", "description": "编码方案编码"},
            "codeschm_name": {"type": "string", "description": "编码方案名称关键字"},
            "cobject": {"type": "string", "description": "对象"}
        },
        "required": []
    }
}

# endregion

# region 职位档案

# ===================== 获取单个职位信息 数据模型 =====================
class GetJobInput(BaseModel):
    """获取单个职位信息输入模型"""
    id: str = Field(..., description="职位编码")

# ===================== 批量获取职位信息 数据模型 =====================
class BatchGetJobInput(BaseModel):
    """批量获取职位信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cjobcode: Optional[str] = Field(None, description="职位编码")
    cjobname: Optional[str] = Field(None, description="职位名称关键字")
    bjobstate: Optional[bool] = Field(None, description="是否停用")

# ===================== 获取单个职位信息 Tool函数 =====================
def u8_job_get_tool(input_data: GetJobInput, client: U8OpenAPIClient) -> str:
    """
    通过职位编码获取用友U8中的职位信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/job/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取职位信息 Tool函数 =====================
def u8_job_batch_get_tool(input_data: BatchGetJobInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的职位信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/job/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取职位信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取职位信息成功",
            "data": result.get("jobs", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个职位信息 Schema定义 =====================
U8_JOB_GET_SCHEMA = {
    "name": "u8_job_get",
    "description": "通过职位编码获取职位信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "职位编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取职位信息 Schema定义 =====================
U8_JOB_BATCH_GET_SCHEMA = {
    "name": "u8_job_batch_get",
    "description": "批量获取职位信息列表，支持按职位编码、职位名称、是否停用等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cjobcode": {"type": "string", "description": "职位编码"},
            "cjobname": {"type": "string", "description": "职位名称关键字"},
            "bjobstate": {"type": "boolean", "description": "是否停用"}
        },
        "required": []
    }
}

# endregion

# region 职务档案

# ===================== 获取单个职务信息 数据模型 =====================
class GetDutyInput(BaseModel):
    """获取单个职务信息输入模型"""
    id: str = Field(..., description="职务编码")

# ===================== 批量获取职务信息 数据模型 =====================
class BatchGetDutyInput(BaseModel):
    """批量获取职务信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cdutycode: Optional[str] = Field(None, description="职务编码")
    cdutyname: Optional[str] = Field(None, description="职务名称关键字")
    bdutystate: Optional[bool] = Field(None, description="是否停用")

# ===================== 获取单个职务信息 Tool函数 =====================
def u8_duty_get_tool(input_data: GetDutyInput, client: U8OpenAPIClient) -> str:
    """
    通过职务编码获取用友U8中的职务信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/duty/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取职务信息 Tool函数 =====================
def u8_duty_batch_get_tool(input_data: BatchGetDutyInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的职务信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/duty/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取职务信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取职务信息成功",
            "data": result.get("dutys", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个职务信息 Schema定义 =====================
U8_DUTY_GET_SCHEMA = {
    "name": "u8_duty_get",
    "description": "通过职务编码获取职务信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "职务编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取职务信息 Schema定义 =====================
U8_DUTY_BATCH_GET_SCHEMA = {
    "name": "u8_duty_batch_get",
    "description": "批量获取职务信息列表，支持按职务编码、职务名称、是否停用等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cdutycode": {"type": "string", "description": "职务编码"},
            "cdutyname": {"type": "string", "description": "职务名称关键字"},
            "bdutystate": {"type": "boolean", "description": "是否停用"}
        },
        "required": []
    }
}

# endregion

# region 职务类别

# ===================== 获取单个职务类别信息 数据模型 =====================
class GetDutyTypeInput(BaseModel):
    """获取单个职务类别信息输入模型"""
    id: str = Field(..., description="职务类别编码")

# ===================== 批量获取职务类别信息 数据模型 =====================
class BatchGetDutyTypeInput(BaseModel):
    """批量获取职务类别信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cdutytypecode: Optional[str] = Field(None, description="职务类别编码")
    cdutytypename: Optional[str] = Field(None, description="职务类别名称关键字")

# ===================== 获取单个职务类别信息 Tool函数 =====================
def u8_dutytype_get_tool(input_data: GetDutyTypeInput, client: U8OpenAPIClient) -> str:
    """
    通过职务类别编码获取用友U8中的职务类别信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/dutytype/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取职务类别信息 Tool函数 =====================
def u8_dutytype_batch_get_tool(input_data: BatchGetDutyTypeInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的职务类别信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/dutytype/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取职务类别信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取职务类别信息成功",
            "data": result.get("dutytypes", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个职务类别信息 Schema定义 =====================
U8_DUTYTYPE_GET_SCHEMA = {
    "name": "u8_dutytype_get",
    "description": "通过职务类别编码获取职务类别信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "职务类别编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取职务类别信息 Schema定义 =====================
U8_DUTYTYPE_BATCH_GET_SCHEMA = {
    "name": "u8_dutytype_batch_get",
    "description": "批量获取职务类别信息列表，支持按职务类别编码、职务类别名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cdutytypecode": {"type": "string", "description": "职务类别编码"},
            "cdutytypename": {"type": "string", "description": "职务类别名称关键字"}
        },
        "required": []
    }
}

# endregion

# region 自定义档案

# # ===================== 批量获取自定义项档案信息 数据模型 =====================
# class BatchGetUserDefBaseInput(BaseModel):
#     """批量获取自定义项档案信息输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     cuserdefname: Optional[str] = Field(None, description="自定义项名称关键字")
#     cuserdefcode: Optional[str] = Field(None, description="自定义项编码")
#     cuserdeftype: Optional[str] = Field(None, description="自定义项类型")

# # ===================== 批量获取自定义项档案信息 Tool函数 =====================
# def u8_userdefbase_batch_get_tool(input_data: BatchGetUserDefBaseInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的自定义项档案信息列表。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/userdefbase/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取自定义项档案信息失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取自定义项档案信息成功",
#             "data": result.get("userdefbases", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# # ===================== 批量获取自定义项档案信息 Schema定义 =====================
# U8_USERDEFBASE_BATCH_GET_SCHEMA = {
#     "name": "u8_userdefbase_batch_get",
#     "description": "批量获取自定义项档案信息列表，支持按自定义项名称、自定义项编码、自定义项类型等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "cuserdefname": {"type": "string", "description": "自定义项名称关键字"},
#             "cuserdefcode": {"type": "string", "description": "自定义项编码"},
#             "cuserdeftype": {"type": "string", "description": "自定义项类型"}
#         },
#         "required": []
#     }
# }

# endregion

# region 自定义档案设置

# # ===================== 批量获取自定义档案设置信息 数据模型 =====================
# class BatchGetUserDefBaseInput(BaseModel):
#     """批量获取自定义档案设置信息输入模型"""
#     ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
#     cuserdefname: Optional[str] = Field(None, description="自定义项名称关键字")
#     cuserdefcode: Optional[str] = Field(None, description="自定义项编码")
#     cuserdeftype: Optional[str] = Field(None, description="自定义项类型")

# # ===================== 批量获取自定义档案设置信息 Tool函数 =====================
# def u8_userdefbase_batch_get_tool(input_data: BatchGetUserDefBaseInput, client: U8OpenAPIClient) -> str:
#     """
#     批量获取用友U8中的自定义档案设置信息列表。
#     """
#     params = input_data.model_dump(exclude_none=True)
    
#     api_path = "/api/userdefbase/batch_get"
    
#     try:
#         result = client.request_api("GET", api_path, inparams=params)
        
#         if str(result.get("errcode", "")) != "0":
#             return json.dumps({
#                 "success": False,
#                 "error": result.get("errmsg", "批量获取自定义档案设置信息失败"),
#                 "raw_response": result
#             }, ensure_ascii=False, indent=2)

#         return json.dumps({
#             "success": True,
#             "message": "批量获取自定义档案设置信息成功",
#             "data": result.get("userdefbases", []),
#             "raw_response": result
#         }, ensure_ascii=False, indent=2)

#     except Exception as e:
#         return json.dumps({
#             "success": False,
#             "error": f"程序异常：{str(e)}"
#         }, ensure_ascii=False, indent=2)

# # ===================== 批量获取自定义档案设置信息 Schema定义 =====================
# U8_USERDEFBASE_BATCH_GET_SCHEMA = {
#     "name": "u8_userdefbase_batch_get",
#     "description": "批量获取自定义档案设置信息列表，支持按自定义项名称、自定义项编码、自定义项类型等条件筛选",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
#             "cuserdefname": {"type": "string", "description": "自定义项名称关键字"},
#             "cuserdefcode": {"type": "string", "description": "自定义项编码"},
#             "cuserdeftype": {"type": "string", "description": "自定义项类型"}
#         },
#         "required": []
#     }
# }

# endregion

# region 自定义项档案
# ===================== 批量获取自定义项档案信息 数据模型 =====================
class BatchGetDefineInput(BaseModel):
    """批量获取自定义项档案信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cdef_name: Optional[str] = Field(None, description="自定义项名称关键字")
    cdef_code: Optional[str] = Field(None, description="自定义项编码")

# ===================== 批量获取自定义项档案信息 Tool函数 =====================
def u8_define_batch_get_tool(input_data: BatchGetDefineInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的自定义项档案信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/define/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取自定义项档案信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取自定义项档案信息成功",
            "data": result.get("defines", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 批量获取自定义项档案信息 Schema定义 =====================
U8_DEFINE_BATCH_GET_SCHEMA = {
    "name": "u8_define_batch_get",
    "description": "批量获取自定义项档案信息列表，支持按自定义项名称、自定义项编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cdef_name": {"type": "string", "description": "自定义项名称关键字"},
            "cdef_code": {"type": "string", "description": "自定义项编码"}
        },
        "required": []
    }
}

# endregion

# region 自由项
# ===================== 获取单个自由项档案信息 数据模型 =====================
class GetFreeArchInput(BaseModel):
    """获取单个自由项档案信息输入模型"""
    id: str = Field(..., description="自由项档案编码")

# ===================== 批量获取自由项档案信息 数据模型 =====================
class BatchGetFreeArchInput(BaseModel):
    """批量获取自由项档案信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cfearchname: Optional[str] = Field(None, description="自由项档案名称关键字")
    cfeaccodes: Optional[str] = Field(None, description="自由项档案编码（多个用逗号分隔）")
    cfeaclasscode: Optional[str] = Field(None, description="自由项档案类别编码")

# ===================== 获取单个自由项档案信息 Tool函数 =====================
def u8_freearch_get_tool(input_data: GetFreeArchInput, client: U8OpenAPIClient) -> str:
    """
    通过自由项档案编码获取用友U8中的自由项档案信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/freearch/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取自由项档案信息 Tool函数 =====================
def u8_freearch_batch_get_tool(input_data: BatchGetFreeArchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的自由项档案信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/freearch/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取自由项档案信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取自由项档案信息成功",
            "data": result.get("freearchs", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个自由项档案信息 Schema定义 =====================
U8_FREEARCH_GET_SCHEMA = {
    "name": "u8_freearch_get",
    "description": "通过自由项档案编码获取自由项档案信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "自由项档案编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取自由项档案信息 Schema定义 =====================
U8_FREEARCH_BATCH_GET_SCHEMA = {
    "name": "u8_freearch_batch_get",
    "description": "批量获取自由项档案信息列表，支持按档案名称、档案编码、档案类别编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cfearchname": {"type": "string", "description": "自由项档案名称关键字"},
            "cfeaccodes": {"type": "string", "description": "自由项档案编码（多个用逗号分隔）"},
            "cfeaclasscode": {"type": "string", "description": "自由项档案类别编码"}
        },
        "required": []
    }
}

# endregion

# region 自由项类型

# endregion

# region 行业

# endregion

# region 计量单位

# ===================== 获取单个计量单位信息 数据模型 =====================
class GetUnitInput(BaseModel):
    """获取单个计量单位信息输入模型"""
    id: str = Field(..., description="计量单位编码")

# ===================== 批量获取计量单位信息 数据模型 =====================
class BatchGetUnitInput(BaseModel):
    """批量获取计量单位信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="计量单位编码")
    cname: Optional[str] = Field(None, description="计量单位名称关键字")
    cgroupcode: Optional[str] = Field(None, description="计量单位组编码")

# ===================== 获取单个计量单位信息 Tool函数 =====================
def u8_unit_get_tool(input_data: GetUnitInput, client: U8OpenAPIClient) -> str:
    """
    通过计量单位编码获取用友U8中的计量单位信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/unit/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取计量单位信息 Tool函数 =====================
def u8_unit_batch_get_tool(input_data: BatchGetUnitInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的计量单位信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/unit/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取计量单位信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取计量单位信息成功",
            "data": result.get("units", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个计量单位信息 Schema定义 =====================
U8_UNIT_GET_SCHEMA = {
    "name": "u8_unit_get",
    "description": "通过计量单位编码获取计量单位信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "计量单位编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取计量单位信息 Schema定义 =====================
U8_UNIT_BATCH_GET_SCHEMA = {
    "name": "u8_unit_batch_get",
    "description": "批量获取计量单位信息列表，支持按计量单位编码、计量单位名称、计量单位组编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "计量单位编码"},
            "cname": {"type": "string", "description": "计量单位名称关键字"},
            "cgroupcode": {"type": "string", "description": "计量单位组编码"}
        },
        "required": []
    }
}

# endregion

# region 计量单位组

# endregion

# region 账套

# endregion

# region 货位

# endregion

# region 费用项目
# ===================== 获取单个费用项目信息 数据模型 =====================
class GetExpenseItemInput(BaseModel):
    """获取单个费用项目信息输入模型"""
    id: str = Field(..., description="费用项目编码")

# ===================== 批量获取费用项目信息 数据模型 =====================
class BatchGetExpenseItemInput(BaseModel):
    """批量获取费用项目信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="费用项目编码")
    cname: Optional[str] = Field(None, description="费用项目名称关键字")
    cclasscode: Optional[str] = Field(None, description="费用项目分类编码")

# ===================== 获取单个费用项目信息 Tool函数 =====================
def u8_expenseitem_get_tool(input_data: GetExpenseItemInput, client: U8OpenAPIClient) -> str:
    """
    通过费用项目编码获取用友U8中的费用项目信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/expenseitem/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取费用项目信息 Tool函数 =====================
def u8_expenseitem_batch_get_tool(input_data: BatchGetExpenseItemInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的费用项目信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/expenseitem/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取费用项目信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取费用项目信息成功",
            "data": result.get("expenseitems", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个费用项目信息 Schema定义 =====================
U8_EXPENSEITEM_GET_SCHEMA = {
    "name": "u8_expenseitem_get",
    "description": "通过费用项目编码获取费用项目信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "费用项目编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取费用项目信息 Schema定义 =====================
U8_EXPENSEITEM_BATCH_GET_SCHEMA = {
    "name": "u8_expenseitem_batch_get",
    "description": "批量获取费用项目信息列表，支持按费用项目编码、费用项目名称、费用项目分类编码等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "费用项目编码"},
            "cname": {"type": "string", "description": "费用项目名称关键字"},
            "cclasscode": {"type": "string", "description": "费用项目分类编码"}
        },
        "required": []
    }
}

# endregion

# region 费用项目分类

# ===================== 获取单个费用项目分类信息 数据模型 =====================
class GetExpItemClassInput(BaseModel):
    """获取单个费用项目分类信息输入模型"""
    id: str = Field(..., description="费用项目分类编码")

# ===================== 批量获取费用项目分类信息 数据模型 =====================
class BatchGetExpItemClassInput(BaseModel):
    """批量获取费用项目分类信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cclasscode: Optional[str] = Field(None, description="费用项目分类编码")
    cclassname: Optional[str] = Field(None, description="费用项目分类名称关键字")

# ===================== 获取单个费用项目分类信息 Tool函数 =====================
def u8_expitemclass_get_tool(input_data: GetExpItemClassInput, client: U8OpenAPIClient) -> str:
    """
    通过费用项目分类编码获取用友U8中的费用项目分类信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/expitemclass/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取费用项目分类信息 Tool函数 =====================
def u8_expitemclass_batch_get_tool(input_data: BatchGetExpItemClassInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的费用项目分类信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/expitemclass/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取费用项目分类信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取费用项目分类信息成功",
            "data": result.get("expitemclasss", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个费用项目分类信息 Schema定义 =====================
U8_EXPITEMCLASS_GET_SCHEMA = {
    "name": "u8_expitemclass_get",
    "description": "通过费用项目分类编码获取费用项目分类信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "费用项目分类编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取费用项目分类信息 Schema定义 =====================
U8_EXPITEMCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_expitemclass_batch_get",
    "description": "批量获取费用项目分类信息列表，支持按费用项目分类编码、费用项目分类名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cclasscode": {"type": "string", "description": "费用项目分类编码"},
            "cclassname": {"type": "string", "description": "费用项目分类名称关键字"}
        },
        "required": []
    }
}

# endregion

# region 辅助核算启用（ys专用）

# endregion

# region 部门

# ===================== 获取单个部门信息 数据模型 =====================
class GetDepartmentInfoInput(BaseModel):
    """获取单个部门信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="部门编码（必填）")


# ===================== 添加一个新部门 数据模型 =====================
class AddDepartmentInfoInput(BaseModel):
    """添加一个新部门输入模型"""
    code: str = Field(..., description="部门编码（必填）")
    name: str = Field(..., description="部门名称（必填）")
    endflag: Optional[bool] = Field(None, description="是否末级")
    rank: Optional[int] = Field(None, description="编码级次")
    manager: Optional[str] = Field(None, description="负责人编码")
    managername: Optional[str] = Field(None, description="负责人名称")
    cdepleader: Optional[str] = Field(None, description="分管领导编码")
    cdepleadername: Optional[str] = Field(None, description="分管领导名称")
    remark: Optional[str] = Field(None, description="备注")
    ddependdate: Optional[str] = Field(None, description="撤销日期")
    ds_sequence: Optional[int] = Field(None, description="数据源序号")
    biz_id: Optional[str] = Field(None, description="上游id，需要保证biz_id与ERP主键唯一对应关系")
    sync: Optional[int] = Field(None, description="0=异步新增（默认）;1=同步新增")


# ===================== 修改部门 数据模型 =====================
class EditDepartmentInfoInput(BaseModel):
    """修改部门输入模型"""
    code: str = Field(..., description="部门编码（必填）")
    name: str = Field(..., description="部门名称（必填）")
    endflag: Optional[bool] = Field(None, description="是否末级")
    rank: Optional[int] = Field(None, description="编码级次")
    manager: Optional[str] = Field(None, description="负责人编码")
    managername: Optional[str] = Field(None, description="负责人名称")
    cdepleader: Optional[str] = Field(None, description="分管领导编码")
    cdepleadername: Optional[str] = Field(None, description="分管领导名称")
    remark: Optional[str] = Field(None, description="备注")
    ddependdate: Optional[str] = Field(None, description="撤销日期")
    ds_sequence: Optional[int] = Field(None, description="数据源序号")


# ===================== 批量获取部门信息 数据模型 =====================
class GetDepartmentListInput(BaseModel):
    """批量获取部门信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")
    timestamp_begin: Optional[str] = Field(None, description="起始时间戳")
    timestamp_end: Optional[str] = Field(None, description="结束时间戳")


# ===================== 获取单个部门信息 Tool函数 =====================
def u8_department_get_tool(input_data: GetDepartmentInfoInput, client: U8OpenAPIClient) -> str:
    """
    获取单个部门信息。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 部门获取接口路径
    api_path = "/api/department/get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取部门信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取部门信息成功",
            "data": {
                "department": result.get("department", {})
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 添加一个新部门 Tool函数 =====================
def u8_department_add_tool(input_data: AddDepartmentInfoInput, client: U8OpenAPIClient) -> str:
    """
    添加一个新部门。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 department）
    request_body: dict = {
        "department": {
            "code": input_data.code,
            "name": input_data.name,
            "endflag": input_data.endflag,
            "rank": input_data.rank,
            "manager": input_data.manager,
            "managername": input_data.managername,
            "cdepleader": input_data.cdepleader,
            "cdepleadername": input_data.cdepleadername,
            "remark": input_data.remark,
            "ddependdate": input_data.ddependdate
        }
    }

    # 移除None值
    request_body["department"] = {k: v for k, v in request_body["department"].items() if v is not None}

    # URL参数
    url_params = {}
    if input_data.ds_sequence is not None:
        url_params["ds_sequence"] = input_data.ds_sequence
    if input_data.biz_id is not None:
        url_params["biz_id"] = input_data.biz_id
    if input_data.sync is not None:
        url_params["sync"] = input_data.sync

    # 部门新增接口路径
    api_path = "/api/department/add"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=url_params if url_params else None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "新增部门失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "新增部门成功",
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


# ===================== 修改部门 Tool函数 =====================
def u8_department_edit_tool(input_data: EditDepartmentInfoInput, client: U8OpenAPIClient) -> str:
    """
    修改一个部门信息。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 department）
    request_body: dict = {
        "department": {
            "code": input_data.code,
            "name": input_data.name,
            "endflag": input_data.endflag,
            "rank": input_data.rank,
            "manager": input_data.manager,
            "managername": input_data.managername,
            "cdepleader": input_data.cdepleader,
            "cdepleadername": input_data.cdepleadername,
            "remark": input_data.remark,
            "ddependdate": input_data.ddependdate
        }
    }

    # 移除None值
    request_body["department"] = {k: v for k, v in request_body["department"].items() if v is not None}

    # URL参数
    url_params = {}
    if input_data.ds_sequence is not None:
        url_params["ds_sequence"] = input_data.ds_sequence

    # 部门修改接口路径
    api_path = "/api/department/edit"

    try:
        # 发送 POST 请求
        result = client.request_api("POST", api_path, inparams=url_params if url_params else None, json_body=request_body, is_tradeid=True)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "修改部门失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "修改部门成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取部门信息 Tool函数 =====================
def u8_department_list_tool(input_data: GetDepartmentListInput, client: U8OpenAPIClient) -> str:
    """
    批量获取部门信息，支持分页和多条件筛选。
    """
    # 构造接口请求参数（仅传递非None的参数）
    params = input_data.model_dump(exclude_none=True)

    # 部门批量获取接口路径
    api_path = "/api/department/batch_get"

    try:
        # 发送 GET 请求（公共参数由 U8OpenAPIClient 自动拼接）
        result = client.request_api("GET", api_path, inparams=params)

        # 统一返回格式
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "获取部门列表失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取部门列表成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "department": result.get("department", [])
            },
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个部门信息 Schema定义 =====================
U8_DEPARTMENT_GET_SCHEMA = {
    "name": "u8_department_get",
    "description": "获取单个部门信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "部门编码（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 添加一个新部门 Schema定义 =====================
U8_DEPARTMENT_ADD_SCHEMA = {
    "name": "u8_department_add",
    "description": "添加一个新部门",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "部门编码（必填）"},
            "name": {"type": "string", "description": "部门名称（必填）"},
            "endflag": {"type": "boolean", "description": "是否末级"},
            "rank": {"type": "integer", "description": "编码级次"},
            "manager": {"type": "string", "description": "负责人编码"},
            "managername": {"type": "string", "description": "负责人名称"},
            "cdepleader": {"type": "string", "description": "分管领导编码"},
            "cdepleadername": {"type": "string", "description": "分管领导名称"},
            "remark": {"type": "string", "description": "备注"},
            "ddependdate": {"type": "string", "description": "撤销日期"},
            "ds_sequence": {"type": "integer", "description": "数据源序号"},
            "biz_id": {"type": "string", "description": "上游id，需要保证biz_id与ERP主键唯一对应关系"},
            "sync": {"type": "integer", "description": "0=异步新增（默认）;1=同步新增"}
        },
        "required": ["code", "name"]
    }
}

# ===================== 修改部门 Schema定义 =====================
U8_DEPARTMENT_EDIT_SCHEMA = {
    "name": "u8_department_edit",
    "description": "修改一个部门信息",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "部门编码（必填）"},
            "name": {"type": "string", "description": "部门名称（必填）"},
            "endflag": {"type": "boolean", "description": "是否末级"},
            "rank": {"type": "integer", "description": "编码级次"},
            "manager": {"type": "string", "description": "负责人编码"},
            "managername": {"type": "string", "description": "负责人名称"},
            "cdepleader": {"type": "string", "description": "分管领导编码"},
            "cdepleadername": {"type": "string", "description": "分管领导名称"},
            "remark": {"type": "string", "description": "备注"},
            "ddependdate": {"type": "string", "description": "撤销日期"},
            "ds_sequence": {"type": "integer", "description": "数据源序号"}
        },
        "required": ["code", "name"]
    }
}

# ===================== 批量获取部门信息 Schema定义 =====================
U8_DEPARTMENT_LIST_SCHEMA = {
    "name": "u8_department_list",
    "description": "批量获取部门信息，支持分页、编码范围、名称关键字、时间戳等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "name": {"type": "string", "description": "名称关键字"},
            "timestamp_begin": {"type": "string", "description": "起始时间戳"},
            "timestamp_end": {"type": "string", "description": "结束时间戳"}
        },
        "required": []
    }
}

# endregion

# region 银行
# ===================== 获取单个银行信息 数据模型 =====================
class GetBankInput(BaseModel):
    """获取单个银行信息输入模型"""
    id: str = Field(..., description="银行编码")

# ===================== 批量获取银行信息 数据模型 =====================
class BatchGetBankInput(BaseModel):
    """批量获取银行信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="银行编码")
    cname: Optional[str] = Field(None, description="银行名称关键字")

# ===================== 获取单个银行信息 Tool函数 =====================
def u8_bank_get_tool(input_data: GetBankInput, client: U8OpenAPIClient) -> str:
    """
    通过银行编码获取用友U8中的银行信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/bank/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取银行信息 Tool函数 =====================
def u8_bank_batch_get_tool(input_data: BatchGetBankInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的银行信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/bank/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取银行信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取银行信息成功",
            "data": result.get("banks", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个银行信息 Schema定义 =====================
U8_BANK_GET_SCHEMA = {
    "name": "u8_bank_get",
    "description": "通过银行编码获取银行信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "银行编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取银行信息 Schema定义 =====================
U8_BANK_BATCH_GET_SCHEMA = {
    "name": "u8_bank_batch_get",
    "description": "批量获取银行信息列表，支持按银行编码、银行名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "银行编码"},
            "cname": {"type": "string", "description": "银行名称关键字"}
        },
        "required": []
    }
}

# endregion

# region 销售类型
# ===================== 获取单个销售类型信息 数据模型 =====================
class GetSaleTypeInput(BaseModel):
    """获取单个销售类型信息输入模型"""
    id: str = Field(..., description="销售类型编码")

# ===================== 批量获取销售类型信息 数据模型 =====================
class BatchGetSaleTypeInput(BaseModel):
    """批量获取销售类型信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="销售类型编码")
    cname: Optional[str] = Field(None, description="销售类型名称关键字")

# ===================== 获取单个销售类型信息 Tool函数 =====================
def u8_saletype_get_tool(input_data: GetSaleTypeInput, client: U8OpenAPIClient) -> str:
    """
    通过销售类型编码获取用友U8中的销售类型信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/saletype/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取销售类型信息 Tool函数 =====================
def u8_saletype_batch_get_tool(input_data: BatchGetSaleTypeInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的销售类型信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/saletype/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取销售类型信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取销售类型信息成功",
            "data": result.get("saletypes", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个销售类型信息 Schema定义 =====================
U8_SALETYPE_GET_SCHEMA = {
    "name": "u8_saletype_get",
    "description": "通过销售类型编码获取销售类型信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "销售类型编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取销售类型信息 Schema定义 =====================
U8_SALETYPE_BATCH_GET_SCHEMA = {
    "name": "u8_saletype_batch_get",
    "description": "批量获取销售类型信息列表，支持按销售类型编码、销售类型名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "销售类型编码"},
            "cname": {"type": "string", "description": "销售类型名称关键字"}
        },
        "required": []
    }
}

# endregion

# region 项目
# ===================== 批量获取项目档案信息 数据模型 =====================
class BatchGetFitemInput(BaseModel):
    """批量获取项目档案信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cfitem_code: Optional[str] = Field(None, description="项目编码")
    cfitem_name: Optional[str] = Field(None, description="项目名称关键字")
    cfitem_classcode: Optional[str] = Field(None, description="项目大类编码")
    bend: Optional[bool] = Field(None, description="是否末级")

# ===================== 添加项目档案信息 数据模型 =====================
class AddFitemInput(BaseModel):
    """添加项目档案信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cfitem_code: str = Field(..., description="项目编码（必填）")
    cfitem_name: str = Field(..., description="项目名称（必填）")
    cfitem_classcode: Optional[str] = Field(None, description="项目大类编码")
    cenglishname: Optional[str] = Field(None, description="英文名称")
    cmemo: Optional[str] = Field(None, description="备注")

# ===================== 批量获取项目档案信息 Tool函数 =====================
def u8_fitem_batch_get_tool(input_data: BatchGetFitemInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的项目档案信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/fitem/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取项目档案信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取项目档案信息成功",
            "data": result.get("fitems", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 添加项目档案信息 Tool函数 =====================
def u8_fitem_add_tool(input_data: AddFitemInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中添加新的项目档案信息。
    """
    params_data = input_data.model_dump(exclude_none=True)
    ds_sequence = params_data.pop("ds_sequence", None)
    
    request_body: Dict[str, Any] = {
        "fitem": params_data
    }
    
    inparams = {}
    if ds_sequence is not None:
        inparams["ds_sequence"] = ds_sequence
    
    api_path = "/api/fitem/add"
    
    try:
        result = client.request_api("POST", api_path, inparams=inparams, json_body=request_body, is_tradeid=True)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "接口调用失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "项目档案新增成功",
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 批量获取项目档案信息 Schema定义 =====================
U8_FITEM_BATCH_GET_SCHEMA = {
    "name": "u8_fitem_batch_get",
    "description": "批量获取项目档案信息列表，支持按项目编码、项目名称、项目大类编码、是否末级等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cfitem_code": {"type": "string", "description": "项目编码"},
            "cfitem_name": {"type": "string", "description": "项目名称关键字"},
            "cfitem_classcode": {"type": "string", "description": "项目大类编码"},
            "bend": {"type": "boolean", "description": "是否末级"}
        },
        "required": []
    }
}

# ===================== 添加项目档案信息 Schema定义 =====================
U8_FITEM_ADD_SCHEMA = {
    "name": "u8_fitem_add",
    "description": "在用友U8 OpenAPI中新增项目档案信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cfitem_code": {"type": "string", "description": "项目编码（必填）"},
            "cfitem_name": {"type": "string", "description": "项目名称（必填）"},
            "cfitem_classcode": {"type": "string", "description": "项目大类编码"},
            "cenglishname": {"type": "string", "description": "英文名称"},
            "cmemo": {"type": "string", "description": "备注"}
        },
        "required": ["cfitem_code", "cfitem_name"]
    }
}

# endregion

# region 项目分类

# ===================== 批量获取项目分类信息 数据模型 =====================
class BatchGetFitemClassInput(BaseModel):
    """批量获取项目分类信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cfitem_classcode: Optional[str] = Field(None, description="项目分类编码")
    cfitem_classname: Optional[str] = Field(None, description="项目分类名称关键字")

# ===================== 批量获取项目分类信息 Tool函数 =====================
def u8_fitemclass_batch_get_tool(input_data: BatchGetFitemClassInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的项目分类信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/fitemclass/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取项目分类信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取项目分类信息成功",
            "data": result.get("fitemclasss", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 批量获取项目分类信息 Schema定义 =====================
U8_FITEMCLASS_BATCH_GET_SCHEMA = {
    "name": "u8_fitemclass_batch_get",
    "description": "批量获取项目分类信息列表，支持按项目分类编码、项目分类名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cfitem_classcode": {"type": "string", "description": "项目分类编码"},
            "cfitem_classname": {"type": "string", "description": "项目分类名称关键字"}
        },
        "required": []
    }
}

# endregion

# region 项目大类
# ===================== 获取单个项目大类信息 数据模型 =====================
class GetFitemCategoryInput(BaseModel):
    """获取单个项目大类信息输入模型"""
    id: str = Field(..., description="项目大类编码")

# ===================== 批量获取项目大类信息 数据模型 =====================
class BatchGetFitemCategoryInput(BaseModel):
    """批量获取项目大类信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cfitem_classcode: Optional[str] = Field(None, description="项目大类编码")
    cfitem_classname: Optional[str] = Field(None, description="项目大类名称关键字")
    bclassend: Optional[bool] = Field(None, description="是否末级")

# ===================== 获取单个项目大类信息 Tool函数 =====================
def u8_fitemcategory_get_tool(input_data: GetFitemCategoryInput, client: U8OpenAPIClient) -> str:
    """
    通过项目大类编码获取用友U8中的项目大类信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/fitemcategory/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取项目大类信息 Tool函数 =====================
def u8_fitemcategory_batch_get_tool(input_data: BatchGetFitemCategoryInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的项目大类信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/fitemcategory/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取项目大类信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取项目大类信息成功",
            "data": result.get("fitemcategorys", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个项目大类信息 Schema定义 =====================
U8_FITEMCATEGORY_GET_SCHEMA = {
    "name": "u8_fitemcategory_get",
    "description": "通过项目大类编码获取项目大类信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "项目大类编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取项目大类信息 Schema定义 =====================
U8_FITEMCATEGORY_BATCH_GET_SCHEMA = {
    "name": "u8_fitemcategory_batch_get",
    "description": "批量获取项目大类信息列表，支持按项目大类编码、项目大类名称、是否末级等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cfitem_classcode": {"type": "string", "description": "项目大类编码"},
            "cfitem_classname": {"type": "string", "description": "项目大类名称关键字"},
            "bclassend": {"type": "boolean", "description": "是否末级"}
        },
        "required": []
    }
}

# endregion

# region 预算口径
# ===================== 获取单个预算口径信息 数据模型 =====================
class GetBudgetCaliberInput(BaseModel):
    """获取单个预算口径信息输入模型"""
    id: str = Field(..., description="预算口径编码")

# ===================== 批量获取预算口径信息 数据模型 =====================
class BatchGetBudgetCaliberInput(BaseModel):
    """批量获取预算口径信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    ccode: Optional[str] = Field(None, description="预算口径编码")
    cname: Optional[str] = Field(None, description="预算口径名称关键字")

# ===================== 获取单个预算口径信息 Tool函数 =====================
def u8_budgetcaliber_get_tool(input_data: GetBudgetCaliberInput, client: U8OpenAPIClient) -> str:
    """
    通过预算口径编码获取用友U8中的预算口径信息。
    """
    params = {
        "id": input_data.id
    }
    
    api_path = "/api/budgetcaliber/get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ===================== 批量获取预算口径信息 Tool函数 =====================
def u8_budgetcaliber_batch_get_tool(input_data: BatchGetBudgetCaliberInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的预算口径信息列表。
    """
    params = input_data.model_dump(exclude_none=True)
    
    api_path = "/api/budgetcaliber/batch_get"
    
    try:
        result = client.request_api("GET", api_path, inparams=params)
        
        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "error": result.get("errmsg", "批量获取预算口径信息失败"),
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取预算口径信息成功",
            "data": result.get("budgetcalibers", []),
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"程序异常：{str(e)}"
        }, ensure_ascii=False, indent=2)

# ===================== 获取单个预算口径信息 Schema定义 =====================
U8_BUDGETCALIBER_GET_SCHEMA = {
    "name": "u8_budgetcaliber_get",
    "description": "通过预算口径编码获取预算口径信息",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "预算口径编码"
            }
        },
        "required": ["id"]
    }
}

# ===================== 批量获取预算口径信息 Schema定义 =====================
U8_BUDGETCALIBER_BATCH_GET_SCHEMA = {
    "name": "u8_budgetcaliber_batch_get",
    "description": "批量获取预算口径信息列表，支持按预算口径编码、预算口径名称等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "ccode": {"type": "string", "description": "预算口径编码"},
            "cname": {"type": "string", "description": "预算口径名称关键字"}
        },
        "required": []
    }
}

# endregion