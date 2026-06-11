import json
import logging
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field
from .u8_openapi_client import U8OpenAPIClient

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)


# ===================== 获取单个员工任职信息 数据模型 =====================
class GetJobInfoInput(BaseModel):
    """获取单个员工任职信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="主键（必填）")


# ===================== 批量获取员工任职信息 数据模型 =====================
class GetJobInfoBatchInput(BaseModel):
    """批量获取员工任职信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    cpsn_num: Optional[str] = Field(None, description="人员编码")
    cdepcode: Optional[str] = Field(None, description="部门")
    cdeptname: Optional[str] = Field(None, description="部门名称关键字")
    cjobcode: Optional[str] = Field(None, description="岗位")
    cjobname: Optional[str] = Field(None, description="岗位名称关键字")
    cdutyname: Optional[str] = Field(None, description="职务名称关键字")
    csupperson: Optional[str] = Field(None, description="上级主管")
    bpartjob: Optional[bool] = Field(None, description="是否兼职")
    pk_hr_hi_jobinfo: Optional[str] = Field(None, description="主键")
    irecordid: Optional[int] = Field(None, description="记录序号")
    blastflag: Optional[bool] = Field(None, description="当前记录标识")
    cjobrankcode: Optional[str] = Field(None, description="职级")
    cjobgradecode: Optional[str] = Field(None, description="职等")
    cdutycode: Optional[str] = Field(None, description="职务")
    dbegindate: Optional[str] = Field(None, description="任职开始时间")
    denddate: Optional[str] = Field(None, description="任职结束时间")
    rdchgtype: Optional[str] = Field(None, description="任职变动类别")
    rdutylev: Optional[str] = Field(None, description="职务级别")
    rholdpostway: Optional[str] = Field(None, description="任职方式")
    rhpreason: Optional[str] = Field(None, description="任职原因")
    vhpauthunit: Optional[str] = Field(None, description="任职批准单位")
    vauthorizedoc: Optional[str] = Field(None, description="任职文号")
    dremovdate: Optional[str] = Field(None, description="免职时间")
    rremovmode: Optional[str] = Field(None, description="免职方式")
    rremovreason: Optional[str] = Field(None, description="免职原因")
    vrmauthunit: Optional[str] = Field(None, description="免职批准单位")
    vrmauthdoc: Optional[str] = Field(None, description="免职文号")
    bmaxjobrankcode: Optional[bool] = Field(None, description="是否最高职级")
    timestamp: Optional[int] = Field(None, description="时间戳")


# ===================== 获取某个员工工资记录 数据模型 =====================
class GetSalaryInput(BaseModel):
    """获取某个员工工资记录输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="员工编码，可以通过api/person获取（必填）")


# ===================== 批量获取员工工资记录 数据模型 =====================
class GetSalaryBatchInput(BaseModel):
    """批量获取员工工资记录输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    departmentno: Optional[str] = Field(None, description="员工所在部门的编码，可以通过api/department获取")
    month: Optional[int] = Field(None, description="月份")
    month_begin: Optional[int] = Field(None, description="起始月份")
    month_end: Optional[int] = Field(None, description="结束月份")
    name: Optional[str] = Field(None, description="名称关键字")
    year: Optional[int] = Field(None, description="年份")


# ===================== 获取单个工资项目 数据模型 =====================
class GetSalaryItemInput(BaseModel):
    """获取单个工资项目输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    id: str = Field(..., description="工资项目标识（必填）")


# ===================== 批量获取工资项目 数据模型 =====================
class GetSalaryItemBatchInput(BaseModel):
    """批量获取工资项目输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    code_begin: Optional[str] = Field(None, description="起始编码")
    code_end: Optional[str] = Field(None, description="结束编码")
    name: Optional[str] = Field(None, description="名称关键字")


# ===================== 批量获取考勤信息 数据模型 =====================
class GetAttendanceBatchInput(BaseModel):
    """批量获取考勤信息输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    page_index: Optional[str] = Field(None, description="页号")
    rows_per_page: Optional[str] = Field(None, description="每页行数")
    date_begin: Optional[str] = Field(None, description="起始日期，格式：yyyy-MM-dd")
    date_end: Optional[str] = Field(None, description="结束日期，格式：yyyy-MM-dd")
    dept_code: Optional[str] = Field(None, description="员工所在部门的编码，可以通过api/department获取")
    dept_name: Optional[str] = Field(None, description="员工所在部门的名称关键字，可以通过api/department获取")
    person_code: Optional[str] = Field(None, description="员工编码，可以通过api/person获取")


# ===================== 批量获取薪资结账状态 数据模型 =====================
class GetMendwaBatchInput(BaseModel):
    """批量获取薪资结账状态输入模型"""
    ds_sequence: Optional[int] = Field(None, description="数据源序号（默认取应用的第一个数据源）")
    iyear: Optional[int] = Field(None, description="会计年度")
    iperiod_begin: Optional[int] = Field(None, description="起始会计期间")
    iperiod_end: Optional[int] = Field(None, description="结束会计期间")


# ===================== 获取单个员工任职信息 Tool函数 =====================
def u8_jobinfo_get_tool(input_data: GetJobInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过主键获取用友U8中的单个员工任职信息。
    """
    params = {
        "id": input_data.id
    }
    if input_data.ds_sequence is not None:
        params["ds_sequence"] = input_data.ds_sequence

    api_path = "/api/jobinfo/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "获取员工任职信息失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取员工任职信息成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取员工任职信息 Tool函数 =====================
def u8_jobinfo_batch_get_tool(input_data: GetJobInfoBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的员工任职信息，支持多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/jobinfo/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "批量获取员工任职信息失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取员工任职信息成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 获取某个员工工资记录 Tool函数 =====================
def u8_salary_get_tool(input_data: GetSalaryInput, client: U8OpenAPIClient) -> str:
    """
    通过员工编码获取用友U8中的某个员工工资记录。
    """
    params = {
        "id": input_data.id
    }
    if input_data.ds_sequence is not None:
        params["ds_sequence"] = input_data.ds_sequence

    api_path = "/api/salary/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "获取员工工资记录失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取员工工资记录成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取员工工资记录 Tool函数 =====================
def u8_salary_batch_get_tool(input_data: GetSalaryBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的员工工资记录，支持分页和多条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/salary/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "批量获取员工工资记录失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取员工工资记录成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个工资项目 Tool函数 =====================
def u8_salaryitem_get_tool(input_data: GetSalaryItemInput, client: U8OpenAPIClient) -> str:
    """
    通过工资项目标识获取用友U8中的单个工资项目。
    """
    params = {
        "id": input_data.id
    }
    if input_data.ds_sequence is not None:
        params["ds_sequence"] = input_data.ds_sequence

    api_path = "/api/salaryitem/get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "获取工资项目失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "获取工资项目成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取工资项目 Tool函数 =====================
def u8_salaryitem_batch_get_tool(input_data: GetSalaryItemBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的工资项目，支持分页和编码范围筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/salaryitem/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "批量获取工资项目失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取工资项目成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取考勤信息 Tool函数 =====================
def u8_attendance_batch_get_tool(input_data: GetAttendanceBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的考勤信息，支持日期范围、部门、员工等条件筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/attendance/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "批量获取考勤信息失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取考勤信息成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 批量获取薪资结账状态 Tool函数 =====================
def u8_mendwa_batch_get_tool(input_data: GetMendwaBatchInput, client: U8OpenAPIClient) -> str:
    """
    批量获取用友U8中的薪资结账状态，支持会计年度和期间范围筛选。
    """
    params = input_data.model_dump(exclude_none=True)

    api_path = "/api/mendwa/batch_get"

    try:
        result = client.request_api("GET", api_path, inparams=params)

        if str(result.get("errcode", "")) != "0":
            return json.dumps({
                "success": False,
                "message": result.get("errmsg", "批量获取薪资结账状态失败"),
                "data": None,
                "raw_response": result
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "success": True,
            "message": "批量获取薪资结账状态成功",
            "data": result,
            "raw_response": result
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": f"程序异常：{str(e)}",
            "data": None,
            "raw_response": None
        }, ensure_ascii=False, indent=2)


# ===================== 获取单个员工任职信息 Schema定义 =====================
U8_JOBINFO_GET_SCHEMA = {
    "name": "u8_jobinfo_get",
    "description": "通过主键获取用友U8中的单个员工任职信息，包含部门、岗位、职级、职务、任职时间等详细信息",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "主键（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取员工任职信息 Schema定义 =====================
U8_JOBINFO_BATCH_GET_SCHEMA = {
    "name": "u8_jobinfo_batch_get",
    "description": "批量获取用友U8中的员工任职信息，支持按人员编码、部门、岗位、职务等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "cpsn_num": {"type": "string", "description": "人员编码"},
            "cdepcode": {"type": "string", "description": "部门"},
            "cdeptname": {"type": "string", "description": "部门名称关键字"},
            "cjobcode": {"type": "string", "description": "岗位"},
            "cjobname": {"type": "string", "description": "岗位名称关键字"},
            "cdutyname": {"type": "string", "description": "职务名称关键字"},
            "csupperson": {"type": "string", "description": "上级主管"},
            "bpartjob": {"type": "boolean", "description": "是否兼职"},
            "pk_hr_hi_jobinfo": {"type": "string", "description": "主键"},
            "irecordid": {"type": "integer", "description": "记录序号"},
            "blastflag": {"type": "boolean", "description": "当前记录标识"},
            "cjobrankcode": {"type": "string", "description": "职级"},
            "cjobgradecode": {"type": "string", "description": "职等"},
            "cdutycode": {"type": "string", "description": "职务"},
            "dbegindate": {"type": "string", "description": "任职开始时间"},
            "denddate": {"type": "string", "description": "任职结束时间"},
            "rdchgtype": {"type": "string", "description": "任职变动类别"},
            "rdutylev": {"type": "string", "description": "职务级别"},
            "rholdpostway": {"type": "string", "description": "任职方式"},
            "rhpreason": {"type": "string", "description": "任职原因"},
            "vhpauthunit": {"type": "string", "description": "任职批准单位"},
            "vauthorizedoc": {"type": "string", "description": "任职文号"},
            "dremovdate": {"type": "string", "description": "免职时间"},
            "rremovmode": {"type": "string", "description": "免职方式"},
            "rremovreason": {"type": "string", "description": "免职原因"},
            "vrmauthunit": {"type": "string", "description": "免职批准单位"},
            "vrmauthdoc": {"type": "string", "description": "免职文号"},
            "bmaxjobrankcode": {"type": "boolean", "description": "是否最高职级"},
            "timestamp": {"type": "integer", "description": "时间戳"}
        },
        "required": []
    }
}

# ===================== 获取某个员工工资记录 Schema定义 =====================
U8_SALARY_GET_SCHEMA = {
    "name": "u8_salary_get",
    "description": "通过员工编码获取用友U8中的某个员工工资记录，包含工资类别、工资项目及金额等",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "员工编码，可以通过api/person获取（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取员工工资记录 Schema定义 =====================
U8_SALARY_BATCH_GET_SCHEMA = {
    "name": "u8_salary_batch_get",
    "description": "批量获取用友U8中的员工工资记录，支持分页、部门、年份、月份等多条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "code_begin": {"type": "string", "description": "起始编码"},
            "code_end": {"type": "string", "description": "结束编码"},
            "departmentno": {"type": "string", "description": "员工所在部门的编码，可以通过api/department获取"},
            "month": {"type": "integer", "description": "月份"},
            "month_begin": {"type": "integer", "description": "起始月份"},
            "month_end": {"type": "integer", "description": "结束月份"},
            "name": {"type": "string", "description": "名称关键字"},
            "year": {"type": "integer", "description": "年份"}
        },
        "required": []
    }
}

# ===================== 获取单个工资项目 Schema定义 =====================
U8_SALARYITEM_GET_SCHEMA = {
    "name": "u8_salaryitem_get",
    "description": "通过工资项目标识获取用友U8中的单个工资项目，包含编码、名称、增减项属性",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "id": {"type": "string", "description": "工资项目标识（必填）"}
        },
        "required": ["id"]
    }
}

# ===================== 批量获取工资项目 Schema定义 =====================
U8_SALARYITEM_BATCH_GET_SCHEMA = {
    "name": "u8_salaryitem_batch_get",
    "description": "批量获取用友U8中的工资项目，支持分页、编码范围、名称关键字筛选",
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

# ===================== 批量获取考勤信息 Schema定义 =====================
U8_ATTENDANCE_BATCH_GET_SCHEMA = {
    "name": "u8_attendance_batch_get",
    "description": "批量获取用友U8中的考勤信息，支持日期范围、部门、员工等条件筛选",
    "parameters": {
        "type": "object",
        "properties": {
            "ds_sequence": {"type": "integer", "description": "数据源序号（默认取应用的第一个数据源）"},
            "page_index": {"type": "string", "description": "页号"},
            "rows_per_page": {"type": "string", "description": "每页行数"},
            "date_begin": {"type": "string", "description": "起始日期，格式：yyyy-MM-dd"},
            "date_end": {"type": "string", "description": "结束日期，格式：yyyy-MM-dd"},
            "dept_code": {"type": "string", "description": "员工所在部门的编码，可以通过api/department获取"},
            "dept_name": {"type": "string", "description": "员工所在部门的名称关键字，可以通过api/department获取"},
            "person_code": {"type": "string", "description": "员工编码，可以通过api/person获取"}
        },
        "required": []
    }
}

# ===================== 批量获取薪资结账状态 Schema定义 =====================
U8_MENDWA_BATCH_GET_SCHEMA = {
    "name": "u8_mendwa_batch_get",
    "description": "批量获取用友U8中的薪资结账状态，支持会计年度和期间范围筛选",
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
