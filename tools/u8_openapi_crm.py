import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 批量查询CRM客户档案 数据模型 =====================
class GetCrmaccountBatchInput(BaseModel):
    """批量查询CRM客户档案输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    person_code: Optional[str] = Field(None, description="人员编码，需要先调用 api/user_login 进行用户登录")
    user_id: Optional[str] = Field(None, description="审批人(用户编码)，user_id与person_code输入一个参数即可")
    AccountNumber: Optional[str] = Field(None, description="客户编码关键字")
    Name: Optional[str] = Field(None, description="客户名称关键字")
    CCusAbbName: Optional[str] = Field(None, description="客户简称关键字")

# ===================== 批量查询CRM客户档案 Tool函数 =====================
def u8_crmaccount_batch_get_tool(input_data: GetCrmaccountBatchInput, client: U8OpenAPIClient) -> str:
    """
    从用友U8 CRM系统中批量查询客户档案信息，支持按客户编码、客户名称、客户简称等条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/crmaccount/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "CRM客户档案查询失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "CRM客户档案查询成功",
            "data": {
                "page_index": result.get("page_index"),
                "rows_per_page": result.get("rows_per_page"),
                "row_count": result.get("row_count"),
                "page_count": result.get("page_count"),
                "crmaccount": result.get("crmaccount", {})
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


# ===================== 批量查询CRM客户档案 Schema定义 =====================
U8_CRMACCOUNT_BATCH_GET_SCHEMA = {
    "name": "u8_crmaccount_batch_get",
    "description": "在用友U8 CRM系统中批量查询客户档案信息，支持按客户编码、客户名称、客户简称等条件筛选，返回客户基本信息、信用信息、联系方式、关联部门/业务员等完整档案数据",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "person_code": {"type": "string", "description": "人员编码，需要先调用 api/user_login 进行用户登录"},
            "user_id": {"type": "string", "description": "审批人(用户编码)，user_id与person_code输入一个参数即可"},
            "AccountNumber": {"type": "string", "description": "客户编码关键字"},
            "Name": {"type": "string", "description": "客户名称关键字"},
            "CCusAbbName": {"type": "string", "description": "客户简称关键字"}
        },
        "required": []
    }
}

