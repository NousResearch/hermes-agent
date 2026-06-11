import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)



# ===================== 获取U8模块启用状态 数据模型 =====================
class GetSystemstateBatchInput(BaseModel):
    """获取U8模块启用状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    sub_id: Optional[str] = Field(None, description="模块标识")


# ===================== 获取单个操作员 数据模型 =====================
class GetOperatorInput(BaseModel):
    """获取单个操作员输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="操作员编码（必填）")


# ===================== 批量获取操作员 数据模型 =====================
class GetOperatorBatchInput(BaseModel):
    """批量获取操作员输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    name: Optional[str] = Field(None, description="操作员姓名")
    user_id_begin: Optional[str] = Field(None, description="起始操作员编码")
    user_id_end: Optional[str] = Field(None, description="结束操作员编码")
    user_name: Optional[str] = Field(None, description="操作员姓名关键字")
    admin: Optional[bool] = Field(None, description="是否账套主管")
    department: Optional[str] = Field(None, description="部门编码")
    state: Optional[int] = Field(None, description="是否停用")


# ===================== 用户登录 数据模型 =====================
class UserLoginInput(BaseModel):
    """用户登录输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: str = Field(..., description="操作员编码（必填）")
    password: str = Field(..., description="密码（必填）")


# ===================== 用户登录v2 数据模型 =====================
class UserLoginV2Input(BaseModel):
    """用户登录v2输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    user_id: str = Field(..., description="操作员编码（必填）")
    password: str = Field(..., description="密码（必填）")


# ===================== 获取U8模块启用状态 Tool函数 =====================
def u8_systemstate_batch_get_tool(input_data: GetSystemstateBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中获取U8某模块的启用状态，支持分页查询。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/systemstate/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "U8模块启用状态获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "U8模块启用状态获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "systemstate": result.get("systemstate", {})
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


# ===================== 获取单个操作员 Tool函数 =====================
def u8_operator_get_tool(input_data: GetOperatorInput, client: U8OpenAPIClient) -> str:
    """
    通过操作员编码获取用友U8中的操作员基本信息。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/operator/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "操作员信息获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "操作员信息获取成功",
            "data": {
                "operator": result.get("operator", {})
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


# ===================== 批量获取操作员 Tool函数 =====================
def u8_operator_batch_get_tool(input_data: GetOperatorBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8系统中批量获取操作员信息，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/operator/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "操作员列表获取失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "操作员列表获取成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "operator": result.get("operator", {})
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


# ===================== 用户登录 Tool函数 =====================
def u8_user_login_tool(input_data: UserLoginInput, client: U8OpenAPIClient) -> str:
    """
    用户登录，进行ERP系统身份验证。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 user）
    request_body: dict = {
        "user": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/user/login"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "用户登录失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "用户登录成功",
            "data": {
                "user": result.get("user", {})
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


# ===================== 用户登录v2 Tool函数 =====================
def u8_user_login_v2_tool(input_data: UserLoginV2Input, client: U8OpenAPIClient) -> str:
    """
    用户登录v2，登录日期采用当前系统日期，避免审核等操作的日期错误问题。
    """
    # 构造接口要求的标准 JSON 结构（外层包一层 user）
    request_body: dict = {
        "user": input_data.model_dump(exclude_none=True)
    }

    api_path = "/api/user/login_v2"

    try:
        result = client.request_api("POST", api_path, inparams=None, json_body=request_body, is_tradeid=True)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "用户登录失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "用户登录成功",
            "data": {
                "user": result.get("user", {})
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


# ===================== 获取U8模块启用状态 Schema定义 =====================
U8_SYSTEMSTATE_BATCH_GET_SCHEMA = {
    "name": "u8_systemstate_batch_get",
    "description": "在用友U8 OpenAPI中获取U8某模块的启用状态，返回模块是否启用、启用时间等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "sub_id": {"type": "string", "description": "模块标识"}
        },
        "required": []
    }
}


# ===================== 获取单个操作员 Schema定义 =====================
U8_OPERATOR_GET_SCHEMA = {
    "name": "u8_operator_get",
    "description": "在用友U8 OpenAPI中通过操作员编码获取操作员信息，返回操作员编码、姓名、是否账套主管、部门、状态等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "操作员编码（必填）"}
        },
        "required": ["id"]
    }
}


# ===================== 批量获取操作员 Schema定义 =====================
U8_OPERATOR_BATCH_GET_SCHEMA = {
    "name": "u8_operator_batch_get",
    "description": "在用友U8 OpenAPI中批量获取操作员信息，支持按姓名、编码范围、是否账套主管、部门、状态等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "name": {"type": "string", "description": "操作员姓名"},
            "user_id_begin": {"type": "string", "description": "起始操作员编码"},
            "user_id_end": {"type": "string", "description": "结束操作员编码"},
            "user_name": {"type": "string", "description": "操作员姓名关键字"},
            "admin": {"type": "boolean", "description": "是否账套主管"},
            "department": {"type": "string", "description": "部门编码"},
            "state": {"type": "integer", "description": "是否停用"}
        },
        "required": []
    }
}


# ===================== 用户登录 Schema定义 =====================
U8_USER_LOGIN_SCHEMA = {
    "name": "u8_user_login",
    "description": "在用友U8 OpenAPI中进行用户登录，进行ERP系统身份验证",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "user_id": {"type": "string", "description": "操作员编码（必填）"},
            "password": {"type": "string", "description": "密码（必填）"}
        },
        "required": ["user_id", "password"]
    }
}


# ===================== 用户登录v2 Schema定义 =====================
U8_USER_LOGIN_V2_SCHEMA = {
    "name": "u8_user_login_v2",
    "description": "在用友U8 OpenAPI中进行用户登录v2，登录日期采用当前系统日期，避免审核等操作的日期错误问题",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "user_id": {"type": "string", "description": "操作员编码（必填）"},
            "password": {"type": "string", "description": "密码（必填）"}
        },
        "required": ["user_id", "password"]
    }
}

