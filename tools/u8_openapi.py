
import os
import json
import time
import hashlib
import logging
import requests
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field

# ===================== 日志配置 =====================
logger = logging.getLogger(__name__)

# ===================== 全局客户端单例 =====================
_u8_client_instance = None

def get_u8_client() -> "U8OpenAPIClient":
    """获取单例 U8 客户端，以复用 Token"""
    global _u8_client_instance
    if _u8_client_instance is None:
        _u8_client_instance = U8OpenAPIClient()
    return _u8_client_instance


# ===================== 检查环境变量是否配置 =====================
def check_u8_openapi_requirements() -> bool:
    """检查U8环境变量是否配置"""
    required_vars = [
        "U8_OPENAPI_APPKEY", 
        "U8_OPENAPI_APPSECRET", 
        "U8_OPENAPI_FROM_ACCOUNT", 
        "U8_OPENAPI_TO_ACCOUNT"
    ]
    return all(os.getenv(var) for var in required_vars)


# ===================== U8 接口基础配置 =====================
U8_OPENAPI_URL = "https://api.yonyouup.com"

class U8OpenAPIClient:
    def __init__(self):
        self.app_key = os.getenv("U8_OPENAPI_APPKEY")
        self.app_secret = os.getenv("U8_OPENAPI_APPSECRET")
        self.from_account = os.getenv("U8_OPENAPI_FROM_ACCOUNT")
        self.to_account = os.getenv("U8_OPENAPI_TO_ACCOUNT")
        self.token_expire_time = 0
        self.token = None
    def get_access_token(self) -> str:

        if self.token and time.time() < self.token_expire_time:
            return self.token
        
        url = f"{U8_OPENAPI_URL}/system/token"
        payload = {
            "from_account": self.from_account,
            "app_key": self.app_key,
            "app_secret": self.app_secret
        }

        response = requests.get(url, params=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("errcode") == "0":
            self.token = data["token"]["id"]
            expires_in = data.get("expiresIn", 7200)
            self.token_expire_time = time.time() + expires_in - 60  # 提前1分钟过期
        
        return self.token

    def get_access_tradeid(self) -> str:
        """获取交易ID"""
        token = self.get_access_token()
        
        url = f"{U8_OPENAPI_URL}/system/tradeid"
        payload = {
            "from_account": self.from_account,
            "app_key": self.app_key,
            "token": token
        }

        response = requests.get(url, params=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("errcode") == "0":
            tradeid = data["trade"]["id"]
        
        return tradeid
    
    def user_login_v2(self,token:str, tradeid:str)->str:
        """用户登录"""

        url = f"{U8_OPENAPI_URL}/api/user/login_v2"

        inparams = {
            "from_account": self.from_account,
            "to_account": self.to_account,
            "app_key": self.app_key,
            "token": token,
            "tradeid": tradeid,
            "user_id": "demo",
            "password": "DEMO"
        }

        json_body = {
            "user":{
            "user_id": "demo",
            "password": "DEMO"
            }
        }

        resp = requests.post(url, params=inparams, json=json_body)
        resp.raise_for_status()
        data = resp.json()
        return data

    
    def request_api(self, method: str, path: str, inparams: Optional[Dict] = None, json_body: Optional[Dict] = None, is_tradeid: bool = False,is_user_login_v2=False) -> Dict[str, Any]:
        """通用API请求方法"""
        token = self.get_access_token()
        url = f"{U8_OPENAPI_URL}{path}"
        
        headers = {
            "Content-Type": "application/json"
        }

        try:
            if method.upper() == "GET":

                # 公共参数
                public_params = {
                    "from_account": self.from_account,
                    "to_account": self.to_account,
                    "app_key": self.app_key,
                    "token": token,
                }
        
                # 合并参数：公共参数在前，用户参数在后（用户参数优先级高）
                if inparams:
                    params = {**public_params, **inparams}
                else:
                    params = public_params
        
                resp = requests.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp.json()
            
            elif method.upper() == "POST":
                tradeid=self.get_access_tradeid()
                if is_tradeid:
                    # 公共参数
                    public_params = {
                        "from_account": self.from_account,
                        "to_account": self.to_account,
                        "app_key": self.app_key,
                        "token": token,
                        "tradeid": tradeid,
                    }
                else:
                    public_params = {
                        "from_account": self.from_account,
                        "to_account": self.to_account,
                        "app_key": self.app_key,
                        "token": token,
                    }

                if is_user_login_v2:
                    userinfo=self.user_login_v2(token=token,tradeid=tradeid)
        
                # 合并参数：公共参数在前，用户参数在后（用户参数优先级高）
                if inparams:
                    params = {**public_params, **inparams}
                else:
                    params = public_params
        
                resp = requests.post(url, params=params, json=json_body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                result_url=data.get("url")
                if result_url:
                    time.sleep(3)
                    result_response = requests.get(result_url)
                    result_response.raise_for_status()
                    result_data = result_response.json()
                else:
                    result_data = data
                
                if result_data.get("errcode") == "0":
                    # 如果 errmsg 缺失或为空字符串，则设置为“成功”
                    if not result_data.get("errmsg"):
                        result_data["errmsg"] = "成功"

                return result_data

            else:
                raise ValueError(f"Unsupported method: {method}")

        except Exception as e:
            return {"error": str(e), "status_code": getattr(resp, 'status_code', None)}
        
















# 定义嵌套模型
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

# 主模型
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


class GetCustomerInfoInput(BaseModel):
    id: str = Field(..., description="客户编号，用于查询客户详细信息")



def u8_openapi_customer_add_tool(input_data: AddCustomerInfoInput, client: U8OpenAPIClient) -> str:
    """
    向用友U8系统中添加新的客户信息。
    """
    # 1. 构造接口要求的标准 JSON 结构（外层必须包一层 customer）
    request_body: Dict[str, Any] = {
        "customer": input_data.model_dump(exclude_none=True)
    }

    # 2. 固定接口路径
    api_path = "/api/customer/add" 
    
    try:
        # 3. 核心：POST 请求
        # inparams = None（公共参数由 U8OpenAPIClient 自动拼接）
        # json_body = 完整的请求体（必须带外层 customer）

        result = client.request_api("POST", api_path, inparams=None, json_body=request_body,is_tradeid=True)
        
        # 4. 统一返回格式
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
   
def u8_openapi_customer_get_tool(input_data: GetCustomerInfoInput, client: U8OpenAPIClient) -> str:
    """
    通过客户编号获取用友U8中的客户基本信息。
    """
    params = {
        "id": input_data.id 
    }
    
    # 修改点3: 更新API路径
    api_path = "/api/customer/get" 
    
    try:
        # 使用 GET 或 POST 取决于具体接口要求，这里假设是 GET 请求带参数
        result = client.request_api("GET", api_path, inparams=params)
        
        # 检查业务错误码 (可选，根据实际返回结构调整)
        if result.get("errcode") != "0":
            return json.dumps({"error": result.get("errmsg", "Unknown error"), "data": result}, ensure_ascii=False)
            
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
 














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

# ===================== 新增：收款单 数据模型 =====================
class AddAcceptInfoInput(AcceptInfo):
    """新增收款单输入模型，继承自通用模型，可在此处强化必填项校验"""
    vouchtype: str = Field(..., description="单据类型(48=收款单;49=付款单)（必填）")
    customercode: str = Field(..., description="客商编码（必填）")
    amount: float = Field(..., description="本币金额（必填）")
    digest: str = Field(..., description="摘要（必填）")
    entry: List[AcceptEntry] = Field(..., description="收款单体列表（必填）")

# ===================== 查询：单张收款单 数据模型 =====================
class GetAcceptInfoInput(BaseModel):
    id: str = Field(..., description="单据编码，用于查询收款单单据信息")

# ===================== 查询：收款单列表 数据模型 =====================
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

# ===================== 审批：收款单 数据模型 =====================
class VerifyAcceptInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    voucher_type: str = Field(..., description="单据类型（48=收款；49=付款）（必填）")
    person_code: Optional[str] = Field(None, description="审核人(人员编码)，需要先调用api/user_login进行用户登录，方可传入此参数，不传入则走EAI的默认登录用户审核。审核人编码可以通过api/person获取")
    user_id: Optional[str] = Field(None, description="审批人用户编码，同person_code参数，且与person_code二选一传入")

# ===================== 弃审：收款单 数据模型 =====================
class UnVerifyAcceptInfoInput(BaseModel):
    voucher_code: str = Field(..., description="单据编号（必填）")
    voucher_type: str = Field(..., description="单据类型（48=收款；49=付款）（必填）")




# ===================== 新增：收款单 Tool 函数 =====================
def u8_openapi_accept_add_tool(input_data: AddAcceptInfoInput, client: U8OpenAPIClient) -> str:
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

# ===================== 查询：单张收款单 Tool 函数 =====================
def u8_openapi_accept_get_tool(input_data: GetAcceptInfoInput, client: U8OpenAPIClient) -> str:
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

# ===================== 查询：收款单列表 Tool 函数 =====================
def u8_openapi_accept_list_get_tool(input_data: GetAcceptListInfoInput, client: U8OpenAPIClient) -> str:
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

# ===================== 审批：收款单 Tool 函数 =====================
def u8_openapi_accept_verify_tool(input_data: VerifyAcceptInfoInput, client: U8OpenAPIClient) -> str:
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

# ===================== 弃审：收款单 Tool 函数 =====================
def u8_openapi_accept_unverify_tool(input_data: VerifyAcceptInfoInput, client: U8OpenAPIClient) -> str:
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


















# ===================== Schema 定义：弃审收款单 =====================
U8_OPENAPI_ACCEPT_UNVERIFY_SCHEMA = {
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


# ===================== Schema 定义：审批收款单 =====================
U8_OPENAPI_ACCEPT_VERIFY_SCHEMA = {
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


# ===================== Schema 定义：查询收款单列表  =====================
U8_OPENAPI_ACCEPT_LIST_GET_SCHEMA = {
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

# ===================== Schema 定义：查询客户 =====================
U8_OPENAPI_ACCEPT_GET_SCHEMA = {
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

# ===================== Schema 定义：新增收款单  =====================
U8_OPENAPI_ACCEPT_ADD_SCHEMA = {
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

# ===================== Schema 定义：添加客户 =====================
U8_OPENAPI_CUSTOMER_ADD_SCHEMA = {
    "name": "u8_customer_add",
    "description": "在用友U8 OpenAPI中新增客户，支持客户主信息、地址、银行账户、开票单位、负责员工、数据权限等完整信息录入",
    "parameters": {
        "type": "object",
        "properties": {
            # 客户主信息
            "code": {
                "type": "string",
                "description": "客户编码（必填）"
            },
            "name": {
                "type": "string",
                "description": "客户名称（必填）"
            },
            "abbrname": {
                "type": "string",
                "description": "客户简称"
            },
            "sort_code": {
                "type": "string",
                "description": "所属分类编码"
            },
            "domain_code": {
                "type": "string",
                "description": "地区编码"
            },
            "industry": {
                "type": "string",
                "description": "所属行业"
            },
            "contact": {
                "type": "string",
                "description": "联系人"
            },
            "phone": {
                "type": "string",
                "description": "固定电话"
            },
            "fax": {
                "type": "string",
                "description": "传真"
            },
            "mobile": {
                "type": "string",
                "description": "手机号码"
            },
            "devliver_site": {
                "type": "string",
                "description": "发货地址"
            },
            "end_date": {
                "type": "string",
                "description": "停用日期，格式：yyyy-MM-dd"
            },
            "memo": {
                "type": "string",
                "description": "备注信息"
            },
            "ccusexch_name": {
                "type": "string",
                "description": "币种"
            },
            "bcusdomestic": {
                "type": "string",
                "description": "是否国内客户"
            },
            "bcusoverseas": {
                "type": "string",
                "description": "是否国外客户"
            },
            "bserviceattribute": {
                "type": "string",
                "description": "服务属性"
            },

            # 自定义项 1~16
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

            # 子表：收货地址（数组）
            "addresses": {
                "type": "array",
                "description": "收货地址列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "caddcode": {"type": "string", "description": "地址编码"},
                        "bdefault": {"type": "boolean", "description": "是否默认地址：true=是，false=否"},
                        "cdeliveradd": {"type": "string", "description": "收货地址"},
                        "cenglishadd": {"type": "string", "description": "英文地址"},
                        "cenglishadd2": {"type": "string", "description": "英文地址2"},
                        "cenglishadd3": {"type": "string", "description": "英文地址3"},
                        "cenglishadd4": {"type": "string", "description": "英文地址4"},
                        "cdeliverunit": {"type": "string", "description": "收货单位"},
                        "clinkperson": {"type": "string", "description": "联系人"}
                    }
                }
            },

            # 子表：银行账户（数组）
            "banks": {
                "type": "array",
                "description": "银行账户信息列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "caccountnum": {"type": "string", "description": "银行账号"},
                        "bdefault": {"type": "boolean", "description": "是否默认账户：true=是，false=否"},
                        "cbank": {"type": "string", "description": "银行编码"},
                        "cbranch": {"type": "string", "description": "开户支行"},
                        "caccountname": {"type": "string", "description": "账户名称"},
                        "cCusPrinvince": {"type": "string", "description": "省"},
                        "cCusCity": {"type": "string", "description": "市"},
                        "cCusCBBDepId": {"type": "string", "description": "机构号"},
                        "cCusBranchId": {"type": "string", "description": "联行号"},
                        "cCusBranchIdSec": {"type": "string", "description": "联行号II"}
                    }
                }
            },

            # 子表：开票单位（数组）
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

            # 子表：负责员工（数组）
            "users": {
                "type": "array",
                "description": "负责员工列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "ccuscode": {"type": "string", "description": "客户编码"},
                        "user_id": {"type": "string", "description": "操作员编码"},
                        "is_self": {"type": "boolean", "description": "是否负责员工：true=是，false=否"}
                    }
                }
            },

            # 子表：数据权限（数组）
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
        "required": [
            "code",
            "name"
        ]
    }
}

# ===================== Schema 定义：查询客户 =====================
U8_OPENAPI_CUSTOMER_GET_SCHEMA = {
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






















# ===================== 工具注册 =====================
from tools.registry import registry

# ===================== 查询：客户信息 工具注册 =====================
# u8_openapi_customer_get
registry.register(
    name="u8_openapi_customer_get",
    toolset="u8_openapi",
    schema=U8_OPENAPI_CUSTOMER_GET_SCHEMA,
    handler=lambda args, **kw: u8_openapi_customer_get_tool(
        input_data=GetCustomerInfoInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增：客户 工具注册 =====================
# u8_openapi_customer_add
registry.register(
    name="u8_openapi_customer_add",
    toolset="u8_openapi",
    schema=U8_OPENAPI_CUSTOMER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_openapi_customer_add_tool(
        input_data=AddCustomerInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增：收款单 工具注册 =====================
# u8_openapi_accept_add
registry.register(
    name="u8_openapi_accept_add",
    toolset="u8_openapi",
    schema=U8_OPENAPI_ACCEPT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_openapi_accept_add_tool(
        input_data=AddAcceptInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询：单张收款单 工具注册 =====================
# u8_openapi_accept_get
registry.register(
    name="u8_openapi_accept_get",
    toolset="u8_openapi",
    schema=U8_OPENAPI_ACCEPT_GET_SCHEMA,
    handler=lambda args, **kw: u8_openapi_accept_get_tool(
        input_data=GetAcceptInfoInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询：收款单列表 工具注册 =====================
# u8_openapi_accept_list_get
registry.register(
    name="u8_openapi_accept_list_get",
    toolset="u8_openapi",
    schema=U8_OPENAPI_ACCEPT_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_openapi_accept_list_get_tool(
        input_data=GetAcceptListInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审批：收款单 工具注册 =====================
# u8_openapi_accept_verify
registry.register(
    name="u8_openapi_accept_verify",
    toolset="u8_openapi",
    schema=U8_OPENAPI_ACCEPT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_openapi_accept_verify_tool(
        input_data=VerifyAcceptInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审：收款单 工具注册 =====================
# u8_openapi_accept_unverify
registry.register(
    name="u8_openapi_accept_unverify",
    toolset="u8_openapi",
    schema=U8_OPENAPI_ACCEPT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_openapi_accept_unverify_tool(
        input_data=UnVerifyAcceptInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)