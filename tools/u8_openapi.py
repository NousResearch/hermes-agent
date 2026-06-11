
import os
import json
import time
import logging
import requests
from typing import Dict, Any, Optional,List
from pydantic import BaseModel, Field

from .u8_openapi_client import U8OpenAPIClient

from .u8_openapi_ba import *
from. u8_openapi_gl import *
from .u8_openapi_ne import *
from .u8_openapi_sa import *
from .u8_openapi_pu import *
from .u8_openapi_st import *
from .u8_openapi_hr import *
from .u8_openapi_ar import *
from .u8_openapi_bi import *
from .u8_openapi_dp import *
from .u8_openapi_bm import *
from .u8_openapi_fb import *
from .u8_openapi_eb import *
from .u8_openapi_ex import *
from .u8_openapi_kc import *
from .u8_openapi_3u import *
from .u8_openapi_crm import *
from .u8_openapi_cl import *
from .u8_openapi_gsp import *
from .u8_openapi_mm import *
from .u8_openapi_nb import *

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




# ===================== 工具注册 =====================
from tools.registry import registry
# 21个文件 359个接口

# region u8_openapi_ba 基础档案类 101个接口
## 23
# region U8帐套 2个接口

# ===================== 获取单个U8帐套信息 工具注册 =====================
# u8_account_get
registry.register(
    name="u8_account_get",
    toolset="u8",
    schema=U8_ACCOUNT_GET_SCHEMA,
    handler=lambda args, **kw: u8_account_get_tool(
        input_data=GetAccountInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取U8帐套信息 工具注册 =====================
# u8_account_batch_get
registry.register(
    name="u8_account_batch_get",
    toolset="u8",
    schema=U8_ACCOUNT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_account_batch_get_tool(
        input_data=GetAccountBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8扩展接口(废弃) 缺1个接口
# endregion

# region 交易分类 2个接口

# ===================== 获取单个交易分类信息 工具注册 =====================
# u8_payunitclass_get
registry.register(
    name="u8_payunitclass_get",
    toolset="u8",
    schema=U8_PAYUNITCLASS_GET_SCHEMA,
    handler=lambda args, **kw: u8_payunitclass_get_tool(
        input_data=GetPayunitclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取交易分类信息 工具注册 =====================
# u8_payunitclass_batch_get
registry.register(
    name="u8_payunitclass_batch_get",
    toolset="u8",
    schema=U8_PAYUNITCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_payunitclass_batch_get_tool(
        input_data=GetPayunitclassBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 交易单位 2个接口

# ===================== 获取单个交易单位信息 工具注册 =====================
# u8_payunit_get
registry.register(
    name="u8_payunit_get",
    toolset="u8",
    schema=U8_PAYUNIT_GET_SCHEMA,
    handler=lambda args, **kw: u8_payunit_get_tool(
        input_data=GetPayunitInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取交易单位信息 工具注册 =====================
# u8_payunit_batch_get
registry.register(
    name="u8_payunit_batch_get",
    toolset="u8",
    schema=U8_PAYUNIT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_payunit_batch_get_tool(
        input_data=GetPayunitBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 人员 4个接口

# ===================== 获取单个人员信息 工具注册 =====================
# u8_person_get
registry.register(
    name="u8_person_get",
    toolset="u8",
    schema=U8_PERSON_GET_SCHEMA,
    handler=lambda args, **kw: u8_person_get_tool(
        input_data=GetPersonInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取人员信息 工具注册 =====================
# u8_person_batch_get
registry.register(
    name="u8_person_batch_get",
    toolset="u8",
    schema=U8_PERSON_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_person_batch_get_tool(
        input_data=GetPersonBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 修改人员 工具注册 =====================
# u8_person_edit
registry.register(
    name="u8_person_edit",
    toolset="u8",
    schema=U8_PERSON_EDIT_SCHEMA,
    handler=lambda args, **kw: u8_person_edit_tool(
        input_data=EditPersonInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增人员 工具注册 =====================
# u8_person_add
registry.register(
    name="u8_person_add",
    toolset="u8",
    schema=U8_PERSON_ADD_SCHEMA,
    handler=lambda args, **kw: u8_person_add_tool(
        input_data=AddPersonInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 人员类别 2个接口

# ===================== 获取单个人员类别 工具注册 =====================
registry.register(
    name="u8_persontype_get",
    toolset="u8",
    schema=U8_PERSONTYPE_GET_SCHEMA,
    handler=lambda args, **kw: u8_persontype_get_tool(
        input_data=GetPersontypeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取人员类别 工具注册 =====================
registry.register(
    name="u8_persontype_batch_get",
    toolset="u8",
    schema=U8_PERSONTYPE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_persontype_batch_get_tool(
        input_data=GetPersontypeBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# endregion

# region 仓库 3个接口

# ===================== 获取单个仓库信息 工具注册 =====================
# u8_warehouse_get
registry.register(
    name="u8_warehouse_get",
    toolset="u8",
    schema=U8_WAREHOUSE_GET_SCHEMA,
    handler=lambda args, **kw: u8_warehouse_get_tool(
        input_data=GetWarehouseInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取仓库信息 工具注册 =====================
# u8_warehouse_batch_get
registry.register(
    name="u8_warehouse_batch_get",
    toolset="u8",
    schema=U8_WAREHOUSE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_warehouse_batch_get_tool(
        input_data=GetWarehouseBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增仓库 工具注册 =====================
# u8_warehouse_add
registry.register(
    name="u8_warehouse_add",
    toolset="u8",
    schema=U8_WAREHOUSE_ADD_SCHEMA,
    handler=lambda args, **kw: u8_warehouse_add_tool(
        input_data=AddWarehouseInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 会计期间 1个接口

# ===================== 批量获取会计期间 工具注册 =====================
# u8_period_batch_get
registry.register(
    name="u8_period_batch_get",
    toolset="u8",
    schema=U8_PERIOD_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_period_batch_get_tool(
        input_data=GetPeriodBatchInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 会计科目（ys专用） 缺1个接口

# endregion

# region 供应商  4个接口

# ===================== 获取单个供应商信息 工具注册 =====================
# u8_vendor_get
registry.register(
    name="u8_vendor_get",
    toolset="u8",
    schema=U8_VENDOR_GET_SCHEMA,
    handler=lambda args, **kw: u8_vendor_get_tool(
        input_data=GetVendorInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取供应商信息 工具注册 =====================
# u8_vendor_batch_get
registry.register(
    name="u8_vendor_batch_get",
    toolset="u8",
    schema=U8_VENDOR_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_vendor_batch_get_tool(
        input_data=GetVendorBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 修改供应商 工具注册 =====================
# u8_vendor_edit
registry.register(
    name="u8_vendor_edit",
    toolset="u8",
    schema=U8_VENDOR_EDIT_SCHEMA,
    handler=lambda args, **kw: u8_vendor_edit_tool(
        input_data=EditVendorInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增供应商 工具注册 =====================
# u8_vendor_add
registry.register(
    name="u8_vendor_add",
    toolset="u8",
    schema=U8_VENDOR_ADD_SCHEMA,
    handler=lambda args, **kw: u8_vendor_add_tool(
        input_data=AddVendorInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 供应商分类 3个接口

# ===================== 获取单个供应商分类 工具注册 =====================
# u8_vendorclass_get
registry.register(
    name="u8_vendorclass_get",
    toolset="u8",
    schema=U8_VENDORCLASS_GET_SCHEMA,
    handler=lambda args, **kw: u8_vendorclass_get_tool(
        input_data=GetVendorclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取供应商分类 工具注册 =====================
# u8_vendorclass_batch_get
registry.register(
    name="u8_vendorclass_batch_get",
    toolset="u8",
    schema=U8_VENDORCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_vendorclass_batch_get_tool(
        input_data=BatchGetVendorclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 添加供应商分类 工具注册 =====================
# u8_vendorclass_add
registry.register(
    name="u8_vendorclass_add",
    toolset="u8",
    schema=U8_VENDORCLASS_ADD_SCHEMA,
    handler=lambda args, **kw: u8_vendorclass_add_tool(
        input_data=AddVendorclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion
## 20
# region 供应商银行 2个接口


# ===================== 获取单个供应商银行 工具注册 =====================
# u8_vendor_bank_get
registry.register(
    name="u8_vendor_bank_get",
    toolset="u8",
    schema=U8_VENDOR_BANK_GET_SCHEMA,
    handler=lambda args, **kw: u8_vendor_bank_get_tool(
        input_data=GetVendorBankInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取供应商银行 工具注册 =====================
# u8_vendor_bank_batch_get
registry.register(
    name="u8_vendor_bank_batch_get",
    toolset="u8",
    schema=U8_VENDOR_BANK_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_vendor_bank_batch_get_tool(
        input_data=BatchGetVendorBankInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 凭证类别 2个接口

# ===================== 获取单个凭证类别 工具注册 =====================
# u8_dsign_get
registry.register(
    name="u8_dsign_get",
    toolset="u8",
    schema=U8_DSIGN_GET_SCHEMA,
    handler=lambda args, **kw: u8_dsign_get_tool(
        input_data=GetDsignInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取凭证类别 工具注册 =====================
# u8_dsign_batch_get
registry.register(
    name="u8_dsign_batch_get",
    toolset="u8",
    schema=U8_DSIGN_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_dsign_batch_get_tool(
        input_data=BatchGetDsignInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 发运方式 缺1个接口

# # ===================== 批量获取发运方式 工具注册 =====================
# # u8_shippingchoice_batch_get
# registry.register(
#     name="u8_shippingchoice_batch_get",
#     toolset="u8",
#     schema=U8_SHIPPINGCHOICE_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_shippingchoice_batch_get_tool(
#         input_data=BatchGetShippingchoiceInput(**args),
#         client=get_u8_client()  # 使用单例客户端
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 地区分类 2个接口

# ===================== 获取单个地区分类 工具注册 =====================
# u8_districtclass_get
registry.register(
    name="u8_districtclass_get",
    toolset="u8",
    schema=U8_DISTRICTCLASS_GET_SCHEMA,
    handler=lambda args, **kw: u8_districtclass_get_tool(
        input_data=GetDistrictclassInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取地区分类 工具注册 =====================
# u8_districtclass_batch_get
registry.register(
    name="u8_districtclass_batch_get",
    toolset="u8",
    schema=U8_DISTRICTCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_districtclass_batch_get_tool(
        input_data=BatchGetDistrictclassInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 存货分类 3个接口

# ===================== 获取单个存货分类 工具注册 =====================
# u8_inventoryclass_get
registry.register(
    name="u8_inventoryclass_get",
    toolset="u8",
    schema=U8_INVENTORYCLASS_GET_SCHEMA,
    handler=lambda args, **kw: u8_inventoryclass_get_tool(
        input_data=GetInventoryclassInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取存货分类 工具注册 =====================
# u8_inventoryclass_batch_get
registry.register(
    name="u8_inventoryclass_batch_get",
    toolset="u8",
    schema=U8_INVENTORYCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_inventoryclass_batch_get_tool(
        input_data=BatchGetInventoryclassInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增存货分类 工具注册 =====================
# u8_inventoryclass_add
registry.register(
    name="u8_inventoryclass_add",
    toolset="u8",
    schema=U8_INVENTORYCLASS_ADD_SCHEMA,
    handler=lambda args, **kw: u8_inventoryclass_add_tool(
        input_data=AddInventoryclassInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 存货档案 4个接口

# ===================== 获取单个存货档案 工具注册 =====================
# u8_inventory_get
registry.register(
    name="u8_inventory_get",
    toolset="u8",
    schema=U8_INVENTORY_GET_SCHEMA,
    handler=lambda args, **kw: u8_inventory_get_tool(
        input_data=GetInventoryInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取存货档案 工具注册 =====================
# u8_inventory_batch_get
registry.register(
    name="u8_inventory_batch_get",
    toolset="u8",
    schema=U8_INVENTORY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_inventory_batch_get_tool(
        input_data=BatchGetInventoryInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 修改存货档案 工具注册 =====================
# u8_inventory_edit
registry.register(
    name="u8_inventory_edit",
    toolset="u8",
    schema=U8_INVENTORY_EDIT_SCHEMA,
    handler=lambda args, **kw: u8_inventory_edit_tool(
        input_data=EditInventoryInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增存货档案 工具注册 =====================
# u8_inventory_add
registry.register(
    name="u8_inventory_add",
    toolset="u8",
    schema=U8_INVENTORY_ADD_SCHEMA,
    handler=lambda args, **kw: u8_inventory_add_tool(
        input_data=AddInventoryInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 客户 4个接口

# ===================== 获取单个客户信息 工具注册 =====================
# u8_customer_get
registry.register(
    name="u8_customer_get",
    toolset="u8",
    schema=U8_CUSTOMER_GET_SCHEMA,
    handler=lambda args, **kw: u8_customer_get_tool(
        input_data=GetCustomerInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户信息 工具注册 =====================
# u8_customer_batch_get
registry.register(
    name="u8_customer_batch_get",
    toolset="u8",
    schema=U8_CUSTOMER_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customer_batch_get_tool(
        input_data=BatchGetCustomerInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 修改客户信息 工具注册 =====================
# u8_customer_edit
registry.register(
    name="u8_customer_edit",
    toolset="u8",
    schema=U8_CUSTOMER_EDIT_SCHEMA,
    handler=lambda args, **kw: u8_customer_edit_tool(
        input_data=EditCustomerInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 添加新客户 工具注册 =====================
# u8_customer_add
registry.register(
    name="u8_customer_add",
    toolset="u8",
    schema=U8_CUSTOMER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_customer_add_tool(
        input_data=AddCustomerInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# region 备份

# # ===================== 获取单个客户信息 工具注册 =====================
# # u8_customer_get
# registry.register(
#     name="u8_customer_get",
#     toolset="u8",
#     schema=U8_CUSTOMER_GET_SCHEMA,
#     handler=lambda args, **kw: u8_customer_get_tool(
#         input_data=GetCustomerInfoInput(id=args.get("id", "")),
#         client=get_u8_client()  # 使用单例客户端
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# # ===================== 添加一个新客户 工具注册 =====================
# # u8_customer_add
# registry.register(
#     name="u8_customer_add",
#     toolset="u8",
#     schema=U8_CUSTOMER_ADD_SCHEMA,
#     handler=lambda args, **kw: u8_customer_add_tool(
#         input_data=AddCustomerInfoInput(**args),
#         client=get_u8_client()  # 使用单例客户端
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# endregion

# region 客户分类 3个接口

# ===================== 获取单个客户分类 工具注册 =====================
# u8_customerclass_get
registry.register(
    name="u8_customerclass_get",
    toolset="u8",
    schema=U8_CUSTOMERCLASS_GET_SCHEMA,
    handler=lambda args, **kw: u8_customerclass_get_tool(
        input_data=GetCustomerclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户分类 工具注册 =====================
# u8_customerclass_batch_get
registry.register(
    name="u8_customerclass_batch_get",
    toolset="u8",
    schema=U8_CUSTOMERCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customerclass_batch_get_tool(
        input_data=BatchGetCustomerclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 添加客户分类 工具注册 =====================
# u8_customerclass_add
registry.register(
    name="u8_customerclass_add",
    toolset="u8",
    schema=U8_CUSTOMERCLASS_ADD_SCHEMA,
    handler=lambda args, **kw: u8_customerclass_add_tool(
        input_data=AddCustomerclassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# endregion
## 15
# region 客户地址 2个接口

# ===================== 获取单个客户地址 工具注册 =====================
# u8_customeraddress_get
registry.register(
    name="u8_customeraddress_get",
    toolset="u8",
    schema=U8_CUSTOMERADDRESS_GET_SCHEMA,
    handler=lambda args, **kw: u8_customeraddress_get_tool(
        input_data=GetCustomeraddressInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户地址 工具注册 =====================
# u8_customeraddress_batch_get
registry.register(
    name="u8_customeraddress_batch_get",
    toolset="u8",
    schema=U8_CUSTOMERADDRESS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customeraddress_batch_get_tool(
        input_data=BatchGetCustomeraddressInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 客户级别 缺1个接口

# # ===================== 批量获取客户级别 工具注册 =====================
# # u8_customerrank_batch_get
# registry.register(
#     name="u8_customerrank_batch_get",
#     toolset="u8",
#     schema=U8_CUSTOMERRANK_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_customerrank_batch_get_tool(
#         input_data=BatchGetCustomerrankInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 客户联系人 2个接口

# ===================== 获取单个客户联系人 工具注册 =====================
# u8_customercontacts_get
registry.register(
    name="u8_customercontacts_get",
    toolset="u8",
    schema=U8_CUSTOMERCONTACTS_GET_SCHEMA,
    handler=lambda args, **kw: u8_customercontacts_get_tool(
        input_data=GetCustomercontactsInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户联系人 工具注册 =====================
# u8_customercontacts_batch_get
registry.register(
    name="u8_customercontacts_batch_get",
    toolset="u8",
    schema=U8_CUSTOMERCONTACTS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customercontacts_batch_get_tool(
        input_data=BatchGetCustomercontactsInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 客户银行 2个接口

# ===================== 获取单个客户银行 工具注册 =====================
# u8_customer_bank_get
registry.register(
    name="u8_customer_bank_get",
    toolset="u8",
    schema=U8_CUSTOMER_BANK_GET_SCHEMA,
    handler=lambda args, **kw: u8_customer_bank_get_tool(
        input_data=GetCustomerBankInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户银行 工具注册 =====================
# u8_customer_bank_batch_get
registry.register(
    name="u8_customer_bank_batch_get",
    toolset="u8",
    schema=U8_CUSTOMER_BANK_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customer_bank_batch_get_tool(
        input_data=BatchGetCustomerBankInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 币种 2个接口

# ===================== 获取单个币种信息 工具注册 =====================
# u8_currency_get
registry.register(
    name="u8_currency_get",
    toolset="u8",
    schema=U8_CURRENCY_GET_SCHEMA,
    handler=lambda args, **kw: u8_currency_get_tool(
        input_data=GetCurrencyInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取币种信息 工具注册 =====================
# u8_currency_batch_get
registry.register(
    name="u8_currency_batch_get",
    toolset="u8",
    schema=U8_CURRENCY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_currency_batch_get_tool(
        input_data=BatchGetCurrencyInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 常用摘要 2个接口

# ===================== 获取单个常用摘要信息 工具注册 =====================
# u8_digest_get
registry.register(
    name="u8_digest_get",
    toolset="u8",
    schema=U8_DIGEST_GET_SCHEMA,
    handler=lambda args, **kw: u8_digest_get_tool(
        input_data=GetDigestInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取常用摘要信息 工具注册 =====================
# u8_digest_batch_get
registry.register(
    name="u8_digest_batch_get",
    toolset="u8",
    schema=U8_DIGEST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_digest_batch_get_tool(
        input_data=BatchGetDigestInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 批次档案 1个接口

# ===================== 批量获取批次档案信息 工具注册 =====================
# u8_batchproperty_batch_get
registry.register(
    name="u8_batchproperty_batch_get",
    toolset="u8",
    schema=U8_BATCHPROPERTY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_batchproperty_batch_get_tool(
        input_data=BatchGetBatchPropertyInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 收付款协议档案 2个接口

# ===================== 获取单个收付款协议信息 工具注册 =====================
# u8_agreement_get
registry.register(
    name="u8_agreement_get",
    toolset="u8",
    schema=U8_AGREEMENT_GET_SCHEMA,
    handler=lambda args, **kw: u8_agreement_get_tool(
        input_data=GetAgreementInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取收付款协议信息 工具注册 =====================
# u8_agreement_batch_get
registry.register(
    name="u8_agreement_batch_get",
    toolset="u8",
    schema=U8_AGREEMENT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_agreement_batch_get_tool(
        input_data=BatchGetAgreementInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 收发类别 2个接口

# ===================== 获取单个收发类别信息 工具注册 =====================
# u8_receivesendtype_get
registry.register(
    name="u8_receivesendtype_get",
    toolset="u8",
    schema=U8_RECEIVESENDTYPE_GET_SCHEMA,
    handler=lambda args, **kw: u8_receivesendtype_get_tool(
        input_data=GetReceiveSendTypeInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取收发类别信息 工具注册 =====================
# u8_receivesendtype_batch_get
registry.register(
    name="u8_receivesendtype_batch_get",
    toolset="u8",
    schema=U8_RECEIVESENDTYPE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_receivesendtype_batch_get_tool(
        input_data=BatchGetReceiveSendTypeInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion
## 9
# region 本单位开户银行 3个接口
# ===================== 获取单个本单位开户银行信息 工具注册 =====================
# u8_accountingbank_get
registry.register(
    name="u8_accountingbank_get",
    toolset="u8",
    schema=U8_ACCOUNTINGBANK_GET_SCHEMA,
    handler=lambda args, **kw: u8_accountingbank_get_tool(
        input_data=GetAccountingBankInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取本单位开户银行信息 工具注册 =====================
# u8_accountingbank_batch_get
registry.register(
    name="u8_accountingbank_batch_get",
    toolset="u8",
    schema=U8_ACCOUNTINGBANK_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_accountingbank_batch_get_tool(
        input_data=BatchGetAccountingBankInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 添加本单位开户银行信息 工具注册 =====================
# u8_accountingbank_add
registry.register(
    name="u8_accountingbank_add",
    toolset="u8",
    schema=U8_ACCOUNTINGBANK_ADD_SCHEMA,
    handler=lambda args, **kw: u8_accountingbank_add_tool(
        input_data=AddAccountingBankInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 汇率 2个接口

# ===================== 获取单个汇率信息 工具注册 =====================
# u8_exchangerate_get
registry.register(
    name="u8_exchangerate_get",
    toolset="u8",
    schema=U8_EXCHANGERATE_GET_SCHEMA,
    handler=lambda args, **kw: u8_exchangerate_get_tool(
        input_data=GetExchangeRateInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取汇率信息 工具注册 =====================
# u8_exchangerate_batch_get
registry.register(
    name="u8_exchangerate_batch_get",
    toolset="u8",
    schema=U8_EXCHANGERATE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_exchangerate_batch_get_tool(
        input_data=BatchGetExchangeRateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 现金流量列表 缺1个接口

# # ===================== 批量获取现金流量列表 工具注册 =====================
# # u8_cashflowlist_batch_get
# registry.register(
#     name="u8_cashflowlist_batch_get",
#     toolset="u8",
#     schema=U8_CASHFLOWLIST_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_cashflowlist_batch_get_tool(
#         input_data=BatchGetCashFlowListInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 现金流量项目 2个接口
# ===================== 获取单个现金流量项目信息 工具注册 =====================
# u8_cashflowitem_get
registry.register(
    name="u8_cashflowitem_get",
    toolset="u8",
    schema=U8_CASHFLOWITEM_GET_SCHEMA,
    handler=lambda args, **kw: u8_cashflowitem_get_tool(
        input_data=GetCashFlowItemInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取现金流量项目信息 工具注册 =====================
# u8_cashflowitem_batch_get
registry.register(
    name="u8_cashflowitem_batch_get",
    toolset="u8",
    schema=U8_CASHFLOWITEM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_cashflowitem_batch_get_tool(
        input_data=BatchGetCashFlowItemInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)
# endregion

# region 种汇率 缺1个接口

# # ===================== 批量获取种汇率信息 工具注册 =====================
# # u8_exchangerateext_batch_get
# registry.register(
#     name="u8_exchangerateext_batch_get",
#     toolset="u8",
#     schema=U8_EXCHANGERATEEXT_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_exchangerateext_batch_get_tool(
#         input_data=BatchGetExchangeRateExtInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 科目 2个接口；缺1个接口
# ===================== 获取单个科目信息 工具注册 =====================
# u8_code_get
registry.register(
    name="u8_code_get",
    toolset="u8",
    schema=U8_CODE_GET_SCHEMA,
    handler=lambda args, **kw: u8_code_get_tool(
        input_data=GetCodeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取科目信息 工具注册 =====================
# u8_code_batch_get
registry.register(
    name="u8_code_batch_get",
    toolset="u8",
    schema=U8_CODE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_code_batch_get_tool(
        input_data=BatchGetCodeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# # ===================== 批量获取科目扩展信息 工具注册 =====================
# # u8_codeext_batch_get
# registry.register(
#     name="u8_codeext_batch_get",
#     toolset="u8",
#     schema=U8_CODEEXT_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_codeext_batch_get_tool(
#         input_data=BatchGetCodeExtInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 科目与辅助核算关系（ys专用） 缺1个接口

# # ===================== 批量获取科目与辅助核算关系 工具注册 =====================
# # u8_code_auxiliary_relation_ys_batch_get
# registry.register(
#     name="u8_code_auxiliary_relation_ys_batch_get",
#     toolset="u8",
#     schema=U8_CODE_AUXILIARY_RELATION_YS_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_code_auxiliary_relation_ys_batch_get_tool(
#         input_data=BatchGetCodeAuxiliaryRelationInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 科目分类 缺1个接口

# # ===================== 批量获取科目分类信息 工具注册 =====================
# # u8_codeclass_batch_get
# registry.register(
#     name="u8_codeclass_batch_get",
#     toolset="u8",
#     schema=U8_CODECLASS_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_codeclass_batch_get_tool(
#         input_data=BatchGetCodeClassInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 科目表（ys专用） 缺1个接口


# # ===================== 批量获取科目表信息 工具注册 =====================
# # u8_glaccount_ys_batch_get
# registry.register(
#     name="u8_glaccount_ys_batch_get",
#     toolset="u8",
#     schema=U8_GLACCOUNT_YS_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_glaccount_ys_batch_get_tool(
#         input_data=BatchGetGlAccountYsInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )


# endregion
# 11
# region 结算方式 2个接口

# ===================== 获取单个结算方式信息 工具注册 =====================
# u8_settlestyle_get
registry.register(
    name="u8_settlestyle_get",
    toolset="u8",
    schema=U8_SETTLESTYLE_GET_SCHEMA,
    handler=lambda args, **kw: u8_settlestyle_get_tool(
        input_data=GetSettleStyleInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取结算方式信息 工具注册 =====================
# u8_settlestyle_batch_get
registry.register(
    name="u8_settlestyle_batch_get",
    toolset="u8",
    schema=U8_SETTLESTYLE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_settlestyle_batch_get_tool(
        input_data=BatchGetSettleStyleInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 编码方案 2个接口
# ===================== 获取单个编码方案信息 工具注册 =====================
# u8_codescheme_get
registry.register(
    name="u8_codescheme_get",
    toolset="u8",
    schema=U8_CODESCHEME_GET_SCHEMA,
    handler=lambda args, **kw: u8_codescheme_get_tool(
        input_data=GetCodeSchemeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取编码方案信息 工具注册 =====================
# u8_codescheme_batch_get
registry.register(
    name="u8_codescheme_batch_get",
    toolset="u8",
    schema=U8_CODESCHEME_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_codescheme_batch_get_tool(
        input_data=BatchGetCodeSchemeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 职位档案 2个接口

# ===================== 获取单个职位信息 工具注册 =====================
# u8_job_get
registry.register(
    name="u8_job_get",
    toolset="u8",
    schema=U8_JOB_GET_SCHEMA,
    handler=lambda args, **kw: u8_job_get_tool(
        input_data=GetJobInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取职位信息 工具注册 =====================
# u8_job_batch_get
registry.register(
    name="u8_job_batch_get",
    toolset="u8",
    schema=U8_JOB_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_job_batch_get_tool(
        input_data=BatchGetJobInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 职务档案 2个接口

# ===================== 获取单个职务信息 工具注册 =====================
# u8_duty_get
registry.register(
    name="u8_duty_get",
    toolset="u8",
    schema=U8_DUTY_GET_SCHEMA,
    handler=lambda args, **kw: u8_duty_get_tool(
        input_data=GetDutyInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取职务信息 工具注册 =====================
# u8_duty_batch_get
registry.register(
    name="u8_duty_batch_get",
    toolset="u8",
    schema=U8_DUTY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_duty_batch_get_tool(
        input_data=BatchGetDutyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 职务类别 2个接口

# ===================== 获取单个职务类别信息 工具注册 =====================
# u8_dutytype_get
registry.register(
    name="u8_dutytype_get",
    toolset="u8",
    schema=U8_DUTYTYPE_GET_SCHEMA,
    handler=lambda args, **kw: u8_dutytype_get_tool(
        input_data=GetDutyTypeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取职务类别信息 工具注册 =====================
# u8_dutytype_batch_get
registry.register(
    name="u8_dutytype_batch_get",
    toolset="u8",
    schema=U8_DUTYTYPE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_dutytype_batch_get_tool(
        input_data=BatchGetDutyTypeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 自定义档案 缺1个接口

# # ===================== 批量获取自定义项档案信息 工具注册 =====================
# # u8_userdefbase_batch_get
# registry.register(
#     name="u8_userdefbase_batch_get",
#     toolset="u8",
#     schema=U8_USERDEFBASE_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_userdefbase_batch_get_tool(
#         input_data=BatchGetUserDefBaseInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 自定义档案设置 缺1个接口

# # ===================== 批量获取自定义档案设置信息 工具注册 =====================
# # u8_userdefbase_batch_get
# registry.register(
#     name="u8_userdefbase_batch_get",
#     toolset="u8",
#     schema=U8_USERDEFBASE_BATCH_GET_SCHEMA,
#     handler=lambda args, **kw: u8_userdefbase_batch_get_tool(
#         input_data=BatchGetUserDefBaseInput(**args),
#         client=get_u8_client()
#     ),
#     check_fn=check_u8_openapi_requirements,
#     requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
# )

# endregion

# region 自定义项档案 1个接口
# ===================== 批量获取自定义项档案信息 工具注册 =====================
# u8_define_batch_get
registry.register(
    name="u8_define_batch_get",
    toolset="u8",
    schema=U8_DEFINE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_define_batch_get_tool(
        input_data=BatchGetDefineInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion
## 8
# region 自由项 2个接口
# ===================== 获取单个自由项档案信息 工具注册 =====================
# u8_freearch_get
registry.register(
    name="u8_freearch_get",
    toolset="u8",
    schema=U8_FREEARCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_freearch_get_tool(
        input_data=GetFreeArchInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取自由项档案信息 工具注册 =====================
# u8_freearch_batch_get
registry.register(
    name="u8_freearch_batch_get",
    toolset="u8",
    schema=U8_FREEARCH_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_freearch_batch_get_tool(
        input_data=BatchGetFreeArchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 自由项类型  缺1个接口

# endregion

# region 行业 缺1个接口

# endregion

# region 计量单位 2个接口
# ===================== 获取单个计量单位信息 工具注册 =====================
# u8_unit_get_tool
registry.register(
    name="u8_unit_get",
    toolset="u8",
    schema=U8_UNIT_GET_SCHEMA,
    handler=lambda args, **kw: u8_unit_get_tool(
        input_data=GetUnitInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取计量单位信息 工具注册 =====================
# u8_unit_batch_get
registry.register(
    name="u8_unit_batch_get",
    toolset="u8",
    schema=U8_UNIT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_unit_batch_get_tool(
        input_data=BatchGetUnitInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 计量单位组 缺1个接口

# endregion

# region 账套 缺2个接口

# endregion

# region 货位 缺1个接口

# endregion

# region 费用项目 2个接口

# ===================== 获取单个费用项目信息 工具注册 =====================
# u8_expenseitem_get
registry.register(
    name="u8_expenseitem_get",
    toolset="u8",
    schema=U8_EXPENSEITEM_GET_SCHEMA,
    handler=lambda args, **kw: u8_expenseitem_get_tool(
        input_data=GetExpenseItemInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取费用项目信息 工具注册 =====================
# u8_expenseitem_batch_get
registry.register(
    name="u8_expenseitem_batch_get",
    toolset="u8",
    schema=U8_EXPENSEITEM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_expenseitem_batch_get_tool(
        input_data=BatchGetExpenseItemInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 费用项目分类 2个接口

# ===================== 获取单个费用项目分类信息 工具注册 =====================
# u8_expitemclass_get
registry.register(
    name="u8_expitemclass_get",
    toolset="u8",
    schema=U8_EXPITEMCLASS_GET_SCHEMA,
    handler=lambda args, **kw: u8_expitemclass_get_tool(
        input_data=GetExpItemClassInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取费用项目分类信息 工具注册 =====================
# u8_expitemclass_batch_get
registry.register(
    name="u8_expitemclass_batch_get",
    toolset="u8",
    schema=U8_EXPITEMCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_expitemclass_batch_get_tool(
        input_data=BatchGetExpItemClassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 辅助核算启用（ys专用） 缺1个接口

# endregion
## 15
# region 部门 4个接口

# ===================== 获取单个部门信息 工具注册 =====================
# u8_department_get
registry.register(
    name="u8_department_get",
    toolset="u8",
    schema=U8_DEPARTMENT_GET_SCHEMA,
    handler=lambda args, **kw: u8_department_get_tool(
        input_data=GetDepartmentInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 添加一个新部门 工具注册 =====================
# u8_department_add
registry.register(
    name="u8_department_add",
    toolset="u8",
    schema=U8_DEPARTMENT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_department_add_tool(
        input_data=AddDepartmentInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 修改部门 工具注册 =====================
# u8_department_edit
registry.register(
    name="u8_department_edit",
    toolset="u8",
    schema=U8_DEPARTMENT_EDIT_SCHEMA,
    handler=lambda args, **kw: u8_department_edit_tool(
        input_data=EditDepartmentInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取部门信息 工具注册 =====================
# u8_department_list
registry.register(
    name="u8_department_list",
    toolset="u8",
    schema=U8_DEPARTMENT_LIST_SCHEMA,
    handler=lambda args, **kw: u8_department_list_tool(
        input_data=GetDepartmentListInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 银行 2个接口；缺1个接口

# ===================== 获取单个银行信息 工具注册 =====================
# u8_bank_get
registry.register(
    name="u8_bank_get",
    toolset="u8",
    schema=U8_BANK_GET_SCHEMA,
    handler=lambda args, **kw: u8_bank_get_tool(
        input_data=GetBankInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取银行信息 工具注册 =====================
# u8_bank_batch_get
registry.register(
    name="u8_bank_batch_get",
    toolset="u8",
    schema=U8_BANK_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_bank_batch_get_tool(
        input_data=BatchGetBankInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 销售类型 2个接口

# ===================== 获取单个销售类型信息 工具注册 =====================
# u8_saletype_get
registry.register(
    name="u8_saletype_get",
    toolset="u8",
    schema=U8_SALETYPE_GET_SCHEMA,
    handler=lambda args, **kw: u8_saletype_get_tool(
        input_data=GetSaleTypeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取销售类型信息 工具注册 =====================
# u8_saletype_batch_get
registry.register(
    name="u8_saletype_batch_get",
    toolset="u8",
    schema=U8_SALETYPE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_saletype_batch_get_tool(
        input_data=BatchGetSaleTypeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 项目 2个接口；缺1个接口

# ===================== 批量获取项目档案信息 工具注册 =====================
# u8_fitem_batch_get
registry.register(
    name="u8_fitem_batch_get",
    toolset="u8",
    schema=U8_FITEM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_fitem_batch_get_tool(
        input_data=BatchGetFitemInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 添加项目档案信息 工具注册 =====================
# u8_fitem_add
registry.register(
    name="u8_fitem_add",
    toolset="u8",
    schema=U8_FITEM_ADD_SCHEMA,
    handler=lambda args, **kw: u8_fitem_add_tool(
        input_data=AddFitemInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 项目分类 1个接口

# ===================== 批量获取项目分类信息 工具注册 =====================
# u8_fitemclass_batch_get
registry.register(
    name="u8_fitemclass_batch_get",
    toolset="u8",
    schema=U8_FITEMCLASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_fitemclass_batch_get_tool(
        input_data=BatchGetFitemClassInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 项目大类 2个接口；缺1个接口
# ===================== 获取单个项目大类信息 工具注册 =====================
# u8_fitemcategory_get 
registry.register(
    name="u8_fitemcategory_get",
    toolset="u8",
    schema=U8_FITEMCATEGORY_GET_SCHEMA,
    handler=lambda args, **kw: u8_fitemcategory_get_tool(
        input_data=GetFitemCategoryInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取项目大类信息 工具注册 =====================
registry.register(
    name="u8_fitemcategory_batch_get",
    toolset="u8",
    schema=U8_FITEMCATEGORY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_fitemcategory_batch_get_tool(
        input_data=BatchGetFitemCategoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 预算口径 2个接口
# ===================== 获取单个预算口径信息 工具注册 =====================
# u8_budgetcaliber_get
registry.register(
    name="u8_budgetcaliber_get",
    toolset="u8",
    schema=U8_BUDGETCALIBER_GET_SCHEMA,
    handler=lambda args, **kw: u8_budgetcaliber_get_tool(
        input_data=GetBudgetCaliberInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取预算口径信息 工具注册 =====================
# u8_budgetcaliber_batch_get
registry.register(
    name="u8_budgetcaliber_batch_get",
    toolset="u8",
    schema=U8_BUDGETCALIBER_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_budgetcaliber_batch_get_tool(
        input_data=BatchGetBudgetCaliberInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# endregion

# region u8_openapi_gl 总账类 9个接口；缺2个接口

# region 凭证 5个接口

# ===================== 凭证列表批量查询 工具注册 =====================
# u8_voucherlist_batch_get
registry.register(
    name="u8_voucherlist_batch_get",
    toolset="u8",
    schema=U8_VOUCHERLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_voucherlist_batch_get_tool(
        input_data=GetVoucherlistBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 凭证详情列表批量查询 工具注册 =====================
# u8_voucherdetaillist_batch_get
registry.register(
    name="u8_voucherdetaillist_batch_get",
    toolset="u8",
    schema=U8_VOUCHERDETAILLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_voucherdetaillist_batch_get_tool(
        input_data=GetVoucherdetaillistBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 凭证作废 工具注册 =====================
# u8_voucher_cancel
registry.register(
    name="u8_voucher_cancel",
    toolset="u8",
    schema=U8_VOUCHER_CANCEL_SCHEMA,
    handler=lambda args, **kw: u8_voucher_cancel_tool(
        input_data=VoucherCancelInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 凭证新增(扩展版) 工具注册 =====================
# u8_voucher_ex_add
registry.register(
    name="u8_voucher_ex_add",
    toolset="u8",
    schema=U8_VOUCHER_EX_ADD_SCHEMA,
    handler=lambda args, **kw: u8_voucher_ex_add_tool(
        input_data=AddVoucherExInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 凭证新增 工具注册 =====================
# u8_voucher_add
registry.register(
    name="u8_voucher_add",
    toolset="u8",
    schema=U8_VOUCHER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_voucher_add_tool(
        input_data=AddVoucherInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 其他 4个接口；缺2个接口

# ===================== 启用期间批量查询 工具注册 =====================
# u8_startperiod_batch_get
registry.register(
    name="u8_startperiod_batch_get",
    toolset="u8",
    schema=U8_STARTPERIOD_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_startperiod_batch_get_tool(
        input_data=GetStartperiodBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 总账结账状态批量查询 工具注册 =====================
# u8_mendglgz_batch_get
registry.register(
    name="u8_mendglgz_batch_get",
    toolset="u8",
    schema=U8_MENDGLGZ_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_mendglgz_batch_get_tool(
        input_data=GetMendglgzBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 科目总账批量查询 工具注册 =====================
# u8_accountsum_batch_get
registry.register(
    name="u8_accountsum_batch_get",
    toolset="u8",
    schema=U8_ACCOUNTSUM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_accountsum_batch_get_tool(
        input_data=GetAccountsumBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 辅助总账批量查询 工具注册 =====================
# u8_accountass_batch_get
registry.register(
    name="u8_accountass_batch_get",
    toolset="u8",
    schema=U8_ACCOUNTASS_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_accountass_batch_get_tool(
        input_data=GetAccountassBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion
# endregion

# region u8_openapi_ne 网上报销类 10个接口

# region 商旅订单 2个接口

# ===================== 获取单张商旅订单 工具注册 =====================
# u8_businesstravelorder_get
registry.register(
    name="u8_businesstravelorder_get",
    toolset="u8",
    schema=U8_BUSINESSTRAVELORDER_GET_SCHEMA,
    handler=lambda args, **kw: u8_businesstravelorder_get_tool(
        input_data=GetBusinessTravelOrderInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增商旅订单 工具注册 =====================
# u8_businesstravelorder_add
registry.register(
    name="u8_businesstravelorder_add",
    toolset="u8",
    schema=U8_BUSINESSTRAVELORDER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_businesstravelorder_add_tool(
        input_data=AddBusinessTravelOrderInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 费用报销单 8个接口

# ===================== 获取费用报销单列表 工具注册 =====================
# u8_expensesclaimlist_batch_get
registry.register(
    name="u8_expensesclaimlist_batch_get",
    toolset="u8",
    schema=U8_EXPENSESCLAIMLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaimlist_batch_get_tool(
        input_data=GetExpensesClaimListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取费用报销单待办任务 工具注册 =====================
# u8_expensesclaim_tasks
registry.register(
    name="u8_expensesclaim_tasks",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_tasks_tool(
        input_data=GetExpensesClaimTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查看费用报销单审批进程 工具注册 =====================
# u8_expensesclaim_history
registry.register(
    name="u8_expensesclaim_history",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_history_tool(
        input_data=GetExpensesClaimHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张费用报销单 工具注册 =====================
# u8_expensesclaim_get
registry.register(
    name="u8_expensesclaim_get",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_GET_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_get_tool(
        input_data=GetExpensesClaimInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取费用报销单工作流按钮状态 工具注册 =====================
# u8_expensesclaim_buttonstate
registry.register(
    name="u8_expensesclaim_buttonstate",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_buttonstate_tool(
        input_data=GetExpensesClaimButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核费用报销单 工具注册 =====================
# u8_expensesclaim_audit    
registry.register(
    name="u8_expensesclaim_audit",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_audit_tool(
        input_data=AuditExpensesClaimInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增费用报销单 工具注册 =====================
# u8_expensesclaim_add
registry.register(
    name="u8_expensesclaim_add",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_ADD_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_add_tool(
        input_data=AddExpensesClaimInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审费用报销单 工具注册 =====================
# u8_expensesclaim_abandon
registry.register(
    name="u8_expensesclaim_abandon",
    toolset="u8",
    schema=U8_EXPENSESCLAIM_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_expensesclaim_abandon_tool(
        input_data=AbandonExpensesClaimInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# endregion

# region u8_openapi_sa 销售管理类 33个接口

# region 发货单 10个接口

# ===================== 获取单张发货单 工具注册 =====================
# u8_consignment_get
registry.register(
    name="u8_consignment_get",
    toolset="u8",
    schema=U8_CONSIGNMENT_GET_SCHEMA,
    handler=lambda args, **kw: u8_consignment_get_tool(
        input_data=ConsignmentGetInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增发货单 工具注册 =====================
# u8_consignment_add
registry.register(
    name="u8_consignment_add",
    toolset="u8",
    schema=U8_CONSIGNMENT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_consignment_add_tool(
        input_data=ConsignmentAddInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取发货单列表 工具注册 =====================
# u8_consignment_list
registry.register(
    name="u8_consignment_list",
    toolset="u8",
    schema=U8_CONSIGNMENT_LIST_SCHEMA,
    handler=lambda args, **kw: u8_consignment_list_tool(
        input_data=ConsignmentListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核发货单 工具注册 =====================
# u8_consignment_verify
registry.register(
    name="u8_consignment_verify",
    toolset="u8",
    schema=U8_CONSIGNMENT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_consignment_verify_tool(
        input_data=ConsignmentVerifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审发货单 工具注册 =====================
# u8_consignment_unverify
registry.register(
    name="u8_consignment_unverify",
    toolset="u8",
    schema=U8_CONSIGNMENT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_consignment_unverify_tool(
        input_data=ConsignmentUnverifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审批发货单(工作流) 工具注册 =====================
# u8_consignment_audit
registry.register(
    name="u8_consignment_audit",
    toolset="u8",
    schema=U8_CONSIGNMENT_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_consignment_audit_tool(
        input_data=ConsignmentAuditInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审发货单(工作流) 工具注册 =====================
# u8_consignment_abandon
registry.register(
    name="u8_consignment_abandon",
    toolset="u8",
    schema=U8_CONSIGNMENT_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_consignment_abandon_tool(
        input_data=ConsignmentAbandonInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取发货单工作流按钮状态 工具注册 =====================
# u8_consignment_buttonstate
registry.register(
    name="u8_consignment_buttonstate",
    toolset="u8",
    schema=U8_CONSIGNMENT_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_consignment_buttonstate_tool(
        input_data=ConsignmentButtonStateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取发货单待办任务 工具注册 =====================
# u8_consignment_tasks
registry.register(
    name="u8_consignment_tasks",
    toolset="u8",
    schema=U8_CONSIGNMENT_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_consignment_tasks_tool(
        input_data=ConsignmentTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查看发货单审批进程 工具注册 =====================
# u8_consignment_history
registry.register(
    name="u8_consignment_history",
    toolset="u8",
    schema=U8_CONSIGNMENT_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_consignment_history_tool(
        input_data=ConsignmentHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 客户调价单 2个接口

# ===================== 获取单个客户调价单 工具注册 =====================
# u8_cuspricejust_get
registry.register(
    name="u8_cuspricejust_get",
    toolset="u8",
    schema=U8_CUSPRICEJUST_GET_SCHEMA,
    handler=lambda args, **kw: u8_cuspricejust_get_tool(
        input_data=CuspricejustGetInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取客户调价单 工具注册 =====================
# u8_cuspricejust_list
registry.register(
    name="u8_cuspricejust_list",
    toolset="u8",
    schema=U8_CUSPRICEJUST_LIST_SCHEMA,
    handler=lambda args, **kw: u8_cuspricejust_list_tool(
        input_data=CuspricejustListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 销售发票 3个接口

# ===================== 新增销售发票 工具注册 =====================
# u8_saleinvoice_add
registry.register(
    name="u8_saleinvoice_add",
    toolset="u8",
    schema=U8_SALEINVOICE_ADD_SCHEMA,
    handler=lambda args, **kw: u8_saleinvoice_add_tool(
        input_data=SaleinvoiceAddInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个销售发票 工具注册 =====================
# u8_saleinvoice_get
registry.register(
    name="u8_saleinvoice_get",
    toolset="u8",
    schema=U8_SALEINVOICE_GET_SCHEMA,
    handler=lambda args, **kw: u8_saleinvoice_get_tool(
        input_data=SaleinvoiceGetInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取销售发票列表 工具注册 =====================
# u8_saleinvoice_list
registry.register(
    name="u8_saleinvoice_list",
    toolset="u8",
    schema=U8_SALEINVOICE_LIST_SCHEMA,
    handler=lambda args, **kw: u8_saleinvoice_list_tool(
        input_data=SaleinvoiceListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 销售结账状态 1个接口
# ===================== 批量获取销售结账状态 工具注册 =====================
# u8_mendsa_list
registry.register(
    name="u8_mendsa_list",
    toolset="u8",
    schema=U8_MENDSA_LIST_SCHEMA,
    handler=lambda args, **kw: u8_mendsa_list_tool(
        input_data=MendsaListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 销售订单 12个接口

# ===================== 新增销售订单 工具注册 =====================
# u8_saleorder_add
registry.register(
    name="u8_saleorder_add",
    toolset="u8",
    schema=U8_SALEORDER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_add_tool(
        input_data=SaleorderAddInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个销售订单 工具注册 =====================
# u8_saleorder_get
registry.register(
    name="u8_saleorder_get",
    toolset="u8",
    schema=U8_SALEORDER_GET_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_get_tool(
        input_data=SaleorderGetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询销售订单列表 工具注册 =====================
# u8_saleorder_list
registry.register(
    name="u8_saleorder_list",
    toolset="u8",
    schema=U8_SALEORDER_LIST_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_list_tool(
        input_data=SaleorderListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核销售订单(verify) 工具注册 =====================
# u8_saleorder_verify
registry.register(
    name="u8_saleorder_verify",
    toolset="u8",
    schema=U8_SALEORDER_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_verify_tool(
        input_data=SaleorderVerifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审销售订单(unverify) 工具注册 =====================
# u8_saleorder_unverify
registry.register(
    name="u8_saleorder_unverify",
    toolset="u8",
    schema=U8_SALEORDER_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_unverify_tool(
        input_data=SaleorderUnverifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 销售订单审批(audit) 工具注册 =====================
# u8_saleorder_audit
registry.register(
    name="u8_saleorder_audit",
    toolset="u8",
    schema=U8_SALEORDER_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_audit_tool(
        input_data=SaleorderAuditInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 销售订单弃审(abandon) 工具注册 =====================
# u8_saleorder_abandon
registry.register(
    name="u8_saleorder_abandon",
    toolset="u8",
    schema=U8_SALEORDER_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_abandon_tool(
        input_data=SaleorderAbandonInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取销售订单按钮状态 工具注册 =====================
# u8_saleorder_buttonstate
registry.register(
    name="u8_saleorder_buttonstate",
    toolset="u8",
    schema=U8_SALEORDER_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_buttonstate_tool(
        input_data=SaleorderButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取销售订单待办任务 工具注册 =====================
# u8_saleorder_tasks
registry.register(
    name="u8_saleorder_tasks",
    toolset="u8",
    schema=U8_SALEORDER_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_tasks_tool(
        input_data=SaleorderTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取销售订单审批历史 工具注册 =====================
# u8_saleorder_history
registry.register(
    name="u8_saleorder_history",
    toolset="u8",
    schema=U8_SALEORDER_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_history_tool(
        input_data=SaleorderHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 打开销售订单 工具注册 =====================
# u8_saleorder_open
registry.register(
    name="u8_saleorder_open",
    toolset="u8",
    schema=U8_SALEORDER_OPEN_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_open_tool(
        input_data=SaleorderOpenInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 关闭销售订单 工具注册 =====================
# u8_saleorder_close
registry.register(
    name="u8_saleorder_close",
    toolset="u8",
    schema=U8_SALEORDER_CLOSE_SCHEMA,
    handler=lambda args, **kw: u8_saleorder_close_tool(
        input_data=SaleorderCloseInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 销售退货单 5个接口

# ===================== 新增销售退货单 工具注册 =====================
# u8_returnorder_add
registry.register(
    name="u8_returnorder_add",
    toolset="u8",
    schema=U8_RETURNORDER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_returnorder_add_tool(
        input_data=ReturnorderAddInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个销售退货单 工具注册 =====================
# u8_returnorder_get
registry.register(
    name="u8_returnorder_get",
    toolset="u8",
    schema=U8_RETURNORDER_GET_SCHEMA,
    handler=lambda args, **kw: u8_returnorder_get_tool(
        input_data=ReturnorderGetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询销售退货单列表 工具注册 =====================
# u8_returnorder_list
registry.register(
    name="u8_returnorder_list",
    toolset="u8",
    schema=U8_RETURNORDER_LIST_SCHEMA,
    handler=lambda args, **kw: u8_returnorder_list_tool(
        input_data=ReturnorderListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核销售退货单 工具注册 =====================
# u8_returnorder_verify
registry.register(
    name="u8_returnorder_verify",
    toolset="u8",
    schema=U8_RETURNORDER_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_returnorder_verify_tool(
        input_data=ReturnorderVerifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审销售退货单 工具注册 =====================
# u8_returnorder_unverify
registry.register(
    name="u8_returnorder_unverify",
    toolset="u8",
    schema=U8_RETURNORDER_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_returnorder_unverify_tool(
        input_data=ReturnorderUnverifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# endregion

# region u8_openapi_pu 采购管理类 59个接口

# region 供应商存货价格表 1个接口

# ===================== 批量获取供应商存货价格表 工具注册 =====================
# u8_veninvprice_batch_get
registry.register(
    name="u8_veninvprice_batch_get",
    toolset="u8",
    schema=U8_VENINVPRICE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_veninvprice_batch_get_tool(
        input_data=GetVenInvPriceInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 供应商存货调价单 8个接口

# ===================== 获取调价单列表 工具注册 =====================
# u8_venpriceadjustlist_batch_get
registry.register(
    name="u8_venpriceadjustlist_batch_get",
    toolset="u8",
    schema=U8_VENPRICEADJUSTLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjustlist_batch_get_tool(
        input_data=GetVenPriceAdjustListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张调价单 工具注册 =====================
# u8_venpriceadjust_get
registry.register(
    name="u8_venpriceadjust_get",
    toolset="u8",
    schema=U8_VENPRICEADJUST_GET_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_get_tool(
        input_data=GetVenPriceAdjustInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取调价单待办任务 工具注册 =====================
# u8_venpriceadjust_tasks
registry.register(
    name="u8_venpriceadjust_tasks",
    toolset="u8",
    schema=U8_VENPRICEADJUST_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_tasks_tool(
        input_data=GetVenPriceAdjustTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查看调价单审批进程 工具注册 =====================
# u8_venpriceadjust_history
registry.register(
    name="u8_venpriceadjust_history",
    toolset="u8",
    schema=U8_VENPRICEADJUST_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_history_tool(
        input_data=GetVenPriceAdjustHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取调价单工作流按钮状态 工具注册 =====================
# u8_venpriceadjust_buttonstate
registry.register(
    name="u8_venpriceadjust_buttonstate",
    toolset="u8",
    schema=U8_VENPRICEADJUST_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_buttonstate_tool(
        input_data=GetVenPriceAdjustButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核调价单 工具注册 =====================
# u8_venpriceadjust_audit
registry.register(
    name="u8_venpriceadjust_audit",
    toolset="u8",
    schema=U8_VENPRICEADJUST_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_audit_tool(
        input_data=AuditVenPriceAdjustInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增调价单 工具注册 =====================
# u8_venpriceadjust_add
registry.register(
    name="u8_venpriceadjust_add",
    toolset="u8",
    schema=U8_VENPRICEADJUST_ADD_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_add_tool(
        input_data=AddVenPriceAdjustInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审调价单 工具注册 =====================
# u8_venpriceadjust_abandon
registry.register(
    name="u8_venpriceadjust_abandon",
    toolset="u8",
    schema=U8_VENPRICEADJUST_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_venpriceadjust_abandon_tool(
        input_data=AbandonVenPriceAdjustInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购到货单 11个接口

# ===================== 获取到货单列表 工具注册 =====================
# u8_purchasereceiptlist_batch_get
registry.register(
    name="u8_purchasereceiptlist_batch_get",
    toolset="u8",
    schema=U8_PURCHASERECEIPTLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceiptlist_batch_get_tool(
        input_data=GetPurchaseReceiptListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个到货单 工具注册 =====================
# u8_purchasereceipt_get
registry.register(
    name="u8_purchasereceipt_get",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_get_tool(
        input_data=GetPurchaseReceiptInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核到货单 工具注册 =====================
# u8_purchasereceipt_verify
registry.register(
    name="u8_purchasereceipt_verify",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_verify_tool(
        input_data=VerifyPurchaseReceiptInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审到货单 工具注册 =====================
# u8_purchasereceipt_unverify
registry.register(
    name="u8_purchasereceipt_unverify",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_unverify_tool(
        input_data=UnVerifyPurchaseReceiptInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取到货单待办任务 工具注册 =====================
# u8_purchasereceipt_tasks
registry.register(
    name="u8_purchasereceipt_tasks",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_tasks_tool(
        input_data=GetPurchaseReceiptTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查看到货单审批进程 工具注册 =====================
# u8_purchasereceipt_history
registry.register(
    name="u8_purchasereceipt_history",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_history_tool(
        input_data=GetPurchaseReceiptHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取到货单是否启用工作流 工具注册 =====================
# u8_purchasereceipt_flowenabled
registry.register(
    name="u8_purchasereceipt_flowenabled",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_flowenabled_tool(
        input_data=GetPurchaseReceiptFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取到货单工作流按钮状态 工具注册 =====================
# u8_purchasereceipt_buttonstate
registry.register(
    name="u8_purchasereceipt_buttonstate",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_buttonstate_tool(
        input_data=GetPurchaseReceiptButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核到货单(工作流) 工具注册 =====================
# u8_purchasereceipt_audit
registry.register(
    name="u8_purchasereceipt_audit",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_audit_tool(
        input_data=AuditPurchaseReceiptInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增到货单 工具注册 =====================
# u8_purchasereceipt_add
registry.register(
    name="u8_purchasereceipt_add",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_add_tool(
        input_data=AddPurchaseReceiptInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审到货单(工作流) 工具注册 =====================
# u8_purchasereceipt_abandon
registry.register(
    name="u8_purchasereceipt_abandon",
    toolset="u8",
    schema=U8_PURCHASERECEIPT_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_purchasereceipt_abandon_tool(
        input_data=AbandonPurchaseReceiptInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购发票 3个接口

# ===================== 新增采购发票 工具注册 =====================
# u8_purchaseinvoice_add
registry.register(
    name="u8_purchaseinvoice_add",
    toolset="u8",
    schema=U8_PURCHASEINVOICE_ADD_SCHEMA,
    handler=lambda args, **kw: u8_purchaseinvoice_add_tool(
        input_data=PurchaseinvoiceAddInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个采购发票 工具注册 =====================
# u8_purinvoice_get
registry.register(
    name="u8_purinvoice_get",
    toolset="u8",
    schema=U8_PURINVOICE_GET_SCHEMA,
    handler=lambda args, **kw: u8_purinvoice_get_tool(
        input_data=PurinvoiceGetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询采购发票列表 工具注册 =====================
# u8_purinvoice_list
registry.register(
    name="u8_purinvoice_list",
    toolset="u8",
    schema=U8_PURINVOICE_LIST_SCHEMA,
    handler=lambda args, **kw: u8_purinvoice_list_tool(
        input_data=PurinvoiceListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购结账状态 1个接口

# ===================== 采购结账状态 工具注册 =====================
# u8_mendpu_batch_get
registry.register(
    name="u8_mendpu_batch_get",
    toolset="u8",
    schema=U8_MENDPU_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_mendpu_batch_get_tool(
        input_data=MendpuBatchGetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购订单 11个接口

# ===================== 新增采购订单 工具注册 =====================
# u8_purchaseorder_add
registry.register(
    name="u8_purchaseorder_add",
    toolset="u8",
    schema=U8_PURCHASEORDER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_add_tool(
        input_data=PurchaseorderAddInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个采购订单 工具注册 =====================
# u8_purchaseorder_get
registry.register(
    name="u8_purchaseorder_get",
    toolset="u8",
    schema=U8_PURCHASEORDER_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_get_tool(
        input_data=PurchaseorderGetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询采购订单列表 工具注册 =====================
# u8_purchaseorder_list
registry.register(
    name="u8_purchaseorder_list",
    toolset="u8",
    schema=U8_PURCHASEORDER_LIST_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_list_tool(
        input_data=PurchaseorderListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查询采购订单列表2 工具注册 =====================
# u8_purchaseorder_list2
registry.register(
    name="u8_purchaseorder_list2",
    toolset="u8",
    schema=U8_PURCHASEORDER_LIST2_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_list2_tool(
        input_data=PurchaseorderList2Input(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核采购订单(verify) 工具注册 =====================
# u8_purchaseorder_verify
registry.register(
    name="u8_purchaseorder_verify",
    toolset="u8",
    schema=U8_PURCHASEORDER_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_verify_tool(
        input_data=PurchaseorderVerifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审采购订单(unverify) 工具注册 =====================
# u8_purchaseorder_unverify
registry.register(
    name="u8_purchaseorder_unverify",
    toolset="u8",
    schema=U8_PURCHASEORDER_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_unverify_tool(
        input_data=PurchaseorderUnverifyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 采购订单审批(audit) 工具注册 =====================
# u8_purchaseorder_audit
registry.register(
    name="u8_purchaseorder_audit",
    toolset="u8",
    schema=U8_PURCHASEORDER_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_audit_tool(
        input_data=PurchaseorderAuditInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 采购订单弃审(abandon) 工具注册 =====================
# u8_purchaseorder_abandon
registry.register(
    name="u8_purchaseorder_abandon",
    toolset="u8",
    schema=U8_PURCHASEORDER_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_abandon_tool(
        input_data=PurchaseorderAbandonInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购订单按钮状态 工具注册 =====================
# u8_purchaseorder_buttonstate
registry.register(
    name="u8_purchaseorder_buttonstate",
    toolset="u8",
    schema=U8_PURCHASEORDER_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_buttonstate_tool(
        input_data=PurchaseorderButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购订单待办任务 工具注册 =====================
# u8_purchaseorder_tasks
registry.register(
    name="u8_purchaseorder_tasks",
    toolset="u8",
    schema=U8_PURCHASEORDER_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_tasks_tool(
        input_data=PurchaseorderTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购订单审批历史 工具注册 =====================
# u8_purchaseorder_history
registry.register(
    name="u8_purchaseorder_history",
    toolset="u8",
    schema=U8_PURCHASEORDER_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_purchaseorder_history_tool(
        input_data=PurchaseorderHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购请购单 11个接口

# ===================== 新增一张采购请购单 工具注册 =====================
# u8_purchaserequisition_add
registry.register(
    name="u8_purchaserequisition_add",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_ADD_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_add_tool(
        input_data=AddPurchaseRequisitionInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张采购请购单 工具注册 =====================
# u8_purchaserequisition_get
registry.register(
    name="u8_purchaserequisition_get",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_get_tool(
        input_data=GetPurchaseRequisitionInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购请购单列表信息 工具注册 =====================
# u8_purchaserequisition_list_get
registry.register(
    name="u8_purchaserequisition_list_get",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_list_get_tool(
        input_data=GetPurchaseRequisitionListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核一张采购请购单 工具注册 =====================
# u8_purchaserequisition_verify
registry.register(
    name="u8_purchaserequisition_verify",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_verify_tool(
        input_data=VerifyPurchaseRequisitionInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审一张采购请购单 工具注册 =====================
# u8_purchaserequisition_unverify
registry.register(
    name="u8_purchaserequisition_unverify",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_unverify_tool(
        input_data=UnVerifyPurchaseRequisitionInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购请购单待办任务 工具注册 =====================
# u8_purchaserequisition_tasks
registry.register(
    name="u8_purchaserequisition_tasks",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_tasks_tool(
        input_data=GetPurchaseRequisitionTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购请购单审批进程 工具注册 =====================
# u8_purchaserequisition_history
registry.register(
    name="u8_purchaserequisition_history",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_history_tool(
        input_data=GetPurchaseRequisitionHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购请购单是否启用工作流 工具注册 =====================
# u8_purchaserequisition_flowenabled
registry.register(
    name="u8_purchaserequisition_flowenabled",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_flowenabled_tool(
        input_data=GetPurchaseRequisitionFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购请购单工作流按钮是否可用状态 工具注册 =====================
# u8_purchaserequisition_buttonstate
registry.register(
    name="u8_purchaserequisition_buttonstate",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_buttonstate_tool(
        input_data=GetPurchaseRequisitionButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核采购请购单（工作流） 工具注册 =====================
# u8_purchaserequisition_audit
registry.register(
    name="u8_purchaserequisition_audit",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_audit_tool(
        input_data=AuditPurchaseRequisitionInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审采购请购单 工具注册 =====================
# u8_purchaserequisition_abandon
registry.register(
    name="u8_purchaserequisition_abandon",
    toolset="u8",
    schema=U8_PURCHASEREQUISITION_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_purchaserequisition_abandon_tool(
        input_data=AbandonPurchaseRequisitionInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购退货单 11个接口

# ===================== 新增一张采购退货单 工具注册 =====================
# u8_purchasereturn_add
registry.register(
    name="u8_purchasereturn_add",
    toolset="u8",
    schema=U8_PURCHASERETURN_ADD_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_add_tool(
        input_data=AddPurchaseReturnInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张采购退货单 工具注册 =====================
# u8_purchasereturn_get
registry.register(
    name="u8_purchasereturn_get",
    toolset="u8",
    schema=U8_PURCHASERETURN_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_get_tool(
        input_data=GetPurchaseReturnInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购退货单列表信息 工具注册 =====================
# u8_purchasereturn_list_get
registry.register(
    name="u8_purchasereturn_list_get",
    toolset="u8",
    schema=U8_PURCHASERETURN_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_list_get_tool(
        input_data=GetPurchaseReturnListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核一张采购退货单 工具注册 =====================
# u8_purchasereturn_verify
registry.register(
    name="u8_purchasereturn_verify",
    toolset="u8",
    schema=U8_PURCHASERETURN_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_verify_tool(
        input_data=VerifyPurchaseReturnInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审一张采购退货单 工具注册 =====================
# u8_purchasereturn_unverify
registry.register(
    name="u8_purchasereturn_unverify",
    toolset="u8",
    schema=U8_PURCHASERETURN_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_unverify_tool(
        input_data=UnVerifyPurchaseReturnInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购退货单待办任务 工具注册 =====================
# u8_purchasereturn_tasks
registry.register(
    name="u8_purchasereturn_tasks",
    toolset="u8",
    schema=U8_PURCHASERETURN_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_tasks_tool(
        input_data=GetPurchaseReturnTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购退货单审批进程 工具注册 =====================
# u8_purchasereturn_history
registry.register(
    name="u8_purchasereturn_history",
    toolset="u8",
    schema=U8_PURCHASERETURN_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_history_tool(
        input_data=GetPurchaseReturnHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购退货单是否启用工作流 工具注册 =====================
# u8_purchasereturn_flowenabled
registry.register(
    name="u8_purchasereturn_flowenabled",
    toolset="u8",
    schema=U8_PURCHASERETURN_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_flowenabled_tool(
        input_data=GetPurchaseReturnFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取采购退货单工作流按钮是否可用状态 工具注册 =====================
# u8_purchasereturn_buttonstate
registry.register(
    name="u8_purchasereturn_buttonstate",
    toolset="u8",
    schema=U8_PURCHASERETURN_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_buttonstate_tool(
        input_data=GetPurchaseReturnButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核采购退货单（工作流） 工具注册 =====================
# u8_purchasereturn_audit
registry.register(
    name="u8_purchasereturn_audit",
    toolset="u8",
    schema=U8_PURCHASERETURN_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_audit_tool(
        input_data=AuditPurchaseReturnInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审采购退货单 工具注册 =====================
# u8_purchasereturn_abandon
registry.register(
    name="u8_purchasereturn_abandon",
    toolset="u8",
    schema=U8_PURCHASERETURN_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_purchasereturn_abandon_tool(
        input_data=AbandonPurchaseReturnInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 预算 2个接口

# ===================== 获取单个预算信息 工具注册 =====================
# u8_budget_get
registry.register(
    name="u8_budget_get",
    toolset="u8",
    schema=U8_BUDGET_GET_SCHEMA,
    handler=lambda args, **kw: u8_budget_get_tool(
        input_data=GetBudgetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取预算信息 工具注册 =====================
# u8_budget_batch_get
registry.register(
    name="u8_budget_batch_get",
    toolset="u8",
    schema=U8_BUDGET_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_budget_batch_get_tool(
        input_data=BatchGetBudgetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# endregion

# region u8_openapi_st 库存管理类 57个接口

# region 产成品入库单 5个接口

# ===================== 产成品入库单列表查询 工具注册 =====================
# u8_productinlist_batch_get
registry.register(
    name="u8_productinlist_batch_get",
    toolset="u8",
    schema=U8_PRODUCTINLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_productinlist_batch_get_tool(
        input_data=GetProductinListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核产成品入库单 工具注册 =====================
# u8_productin_verify
registry.register(
    name="u8_productin_verify",
    toolset="u8",
    schema=U8_PRODUCTIN_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_productin_verify_tool(
        input_data=VerifyProductinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审产成品入库单 工具注册 =====================
# u8_productin_unverify
registry.register(
    name="u8_productin_unverify",
    toolset="u8",
    schema=U8_PRODUCTIN_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_productin_unverify_tool(
        input_data=UnverifyProductinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个产成品入库单 工具注册 =====================
# u8_productin_get
registry.register(
    name="u8_productin_get",
    toolset="u8",
    schema=U8_PRODUCTIN_GET_SCHEMA,
    handler=lambda args, **kw: u8_productin_get_tool(
        input_data=GetProductinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增产成品入库单 工具注册 =====================
# u8_productin_add
registry.register(
    name="u8_productin_add",
    toolset="u8",
    schema=U8_PRODUCTIN_ADD_SCHEMA,
    handler=lambda args, **kw: u8_productin_add_tool(
        input_data=AddProductinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 其他入库单 11个接口

# ===================== 其他入库单列表查询 工具注册 =====================
# u8_otherinlist_batch_get
registry.register(
    name="u8_otherinlist_batch_get",
    toolset="u8",
    schema=U8_OTHERINLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_otherinlist_batch_get_tool(
        input_data=GetOtherinListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核其他入库单 工具注册 =====================
# u8_otherin_verify
registry.register(
    name="u8_otherin_verify",
    toolset="u8",
    schema=U8_OTHERIN_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_otherin_verify_tool(
        input_data=VerifyOtherinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审其他入库单 工具注册 =====================
# u8_otherin_unverify
registry.register(
    name="u8_otherin_unverify",
    toolset="u8",
    schema=U8_OTHERIN_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_otherin_unverify_tool(
        input_data=UnverifyOtherinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取其他入库单待办任务 工具注册 =====================
# u8_otherin_tasks
registry.register(
    name="u8_otherin_tasks",
    toolset="u8",
    schema=U8_OTHERIN_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_otherin_tasks_tool(
        input_data=GetOtherinTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查看其他入库单审批进程 工具注册 =====================
# u8_otherin_history
registry.register(
    name="u8_otherin_history",
    toolset="u8",
    schema=U8_OTHERIN_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_otherin_history_tool(
        input_data=GetOtherinHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个其它入库单 工具注册 =====================
# u8_otherin_get
registry.register(
    name="u8_otherin_get",
    toolset="u8",
    schema=U8_OTHERIN_GET_SCHEMA,
    handler=lambda args, **kw: u8_otherin_get_tool(
        input_data=GetOtherinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取其他入库单是否启用工作流 工具注册 =====================
# u8_otherin_flowenabled
registry.register(
    name="u8_otherin_flowenabled",
    toolset="u8",
    schema=U8_OTHERIN_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_otherin_flowenabled_tool(
        input_data=GetOtherinFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取其他入库单工作流按钮是否可用状态 工具注册 =====================
# u8_otherin_buttonstate
registry.register(
    name="u8_otherin_buttonstate",
    toolset="u8",
    schema=U8_OTHERIN_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_otherin_buttonstate_tool(
        input_data=GetOtherinButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核其他入库单(工作流) 工具注册 =====================
# u8_otherin_audit
registry.register(
    name="u8_otherin_audit",
    toolset="u8",
    schema=U8_OTHERIN_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_otherin_audit_tool(
        input_data=AuditOtherinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增其它入库单 工具注册 =====================
# u8_otherin_add
registry.register(
    name="u8_otherin_add",
    toolset="u8",
    schema=U8_OTHERIN_ADD_SCHEMA,
    handler=lambda args, **kw: u8_otherin_add_tool(
        input_data=AddOtherinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审其他入库单(工作流) 工具注册 =====================
# u8_otherin_abandon
registry.register(
    name="u8_otherin_abandon",
    toolset="u8",
    schema=U8_OTHERIN_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_otherin_abandon_tool(
        input_data=AbandonOtherinInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 其他出库单 11个接口

# ===================== 其他出库单列表查询 工具注册 =====================
# u8_otheroutlist_batch_get
registry.register(
    name="u8_otheroutlist_batch_get",
    toolset="u8",
    schema=U8_OTHEROUTLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_otheroutlist_batch_get_tool(
        input_data=GetOtheroutListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核其他出库单 工具注册 =====================
# u8_otherout_verify
registry.register(
    name="u8_otherout_verify",
    toolset="u8",
    schema=U8_OTHEROUT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_otherout_verify_tool(
        input_data=VerifyOtheroutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审其他出库单 工具注册 =====================
# u8_otherout_unverify
registry.register(
    name="u8_otherout_unverify",
    toolset="u8",
    schema=U8_OTHEROUT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_otherout_unverify_tool(
        input_data=UnverifyOtheroutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取其他出库单待办任务 工具注册 =====================
# u8_otherout_tasks
registry.register(
    name="u8_otherout_tasks",
    toolset="u8",
    schema=U8_OTHEROUT_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_otherout_tasks_tool(
        input_data=GetOtheroutTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 查看其他出库单审批进程 工具注册 =====================
# u8_otherout_history
registry.register(
    name="u8_otherout_history",
    toolset="u8",
    schema=U8_OTHEROUT_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_otherout_history_tool(
        input_data=GetOtheroutHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个其它出库单 工具注册 =====================
# u8_otherout_get
registry.register(
    name="u8_otherout_get",
    toolset="u8",
    schema=U8_OTHEROUT_GET_SCHEMA,
    handler=lambda args, **kw: u8_otherout_get_tool(
        input_data=GetOtheroutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取其他出库单是否启用工作流 工具注册 =====================
# u8_otherout_flowenabled
registry.register(
    name="u8_otherout_flowenabled",
    toolset="u8",
    schema=U8_OTHEROUT_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_otherout_flowenabled_tool(
        input_data=GetOtheroutFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取其他出库单工作流按钮是否可用状态 工具注册 =====================
# u8_otherout_buttonstate
registry.register(
    name="u8_otherout_buttonstate",
    toolset="u8",
    schema=U8_OTHEROUT_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_otherout_buttonstate_tool(
        input_data=GetOtheroutButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核其他出库单(工作流) 工具注册 =====================
# u8_otherout_audit
registry.register(
    name="u8_otherout_audit",
    toolset="u8",
    schema=U8_OTHEROUT_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_otherout_audit_tool(
        input_data=AuditOtheroutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增其它出库单 工具注册 =====================
# u8_otherout_add
registry.register(
    name="u8_otherout_add",
    toolset="u8",
    schema=U8_OTHEROUT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_otherout_add_tool(
        input_data=AddOtheroutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审其他出库单(工作流) 工具注册 =====================
# u8_otherout_abandon
registry.register(
    name="u8_otherout_abandon",
    toolset="u8",
    schema=U8_OTHEROUT_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_otherout_abandon_tool(
        input_data=AbandonOtheroutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 库存结账状态 1个接口

# ===================== 批量获取库存结账状态 工具注册 =====================
# u8_mendst_batch_get
registry.register(
    name="u8_mendst_batch_get",
    toolset="u8",
    schema=U8_MENDST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_mendst_batch_get_tool(
        input_data=GetMendstListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 材料出库单 11个接口

# ===================== 获取材料出库单列表 工具注册 =====================
# u8_materialout_list
registry.register(
    name="u8_materialout_list",
    toolset="u8",
    schema=U8_MATERIALOUT_LIST_SCHEMA,
    handler=lambda args, **kw: u8_materialout_list_tool(
        input_data=GetMaterialoutListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核材料出库单 工具注册 =====================
# u8_materialout_verify
registry.register(
    name="u8_materialout_verify",
    toolset="u8",
    schema=U8_MATERIALOUT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_materialout_verify_tool(
        input_data=VerifyMaterialoutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审材料出库单 工具注册 =====================
# u8_materialout_unverify
registry.register(
    name="u8_materialout_unverify",
    toolset="u8",
    schema=U8_MATERIALOUT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_materialout_unverify_tool(
        input_data=UnverifyMaterialoutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取材料出库单待办任务 工具注册 =====================
# u8_materialout_tasks
registry.register(
    name="u8_materialout_tasks",
    toolset="u8",
    schema=U8_MATERIALOUT_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_materialout_tasks_tool(
        input_data=GetMaterialoutTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取材料出库单审批进程 工具注册 =====================
# u8_materialout_history
registry.register(
    name="u8_materialout_history",
    toolset="u8",
    schema=U8_MATERIALOUT_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_materialout_history_tool(
        input_data=GetMaterialoutHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个材料出库单 工具注册 =====================
# u8_materialout_get
registry.register(
    name="u8_materialout_get",
    toolset="u8",
    schema=U8_MATERIALOUT_GET_SCHEMA,
    handler=lambda args, **kw: u8_materialout_get_tool(
        input_data=GetMaterialoutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取材料出库单是否启用工作流 工具注册 =====================
# u8_materialout_flowenabled
registry.register(
    name="u8_materialout_flowenabled",
    toolset="u8",
    schema=U8_MATERIALOUT_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_materialout_flowenabled_tool(
        input_data=GetMaterialoutFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取材料出库单工作流按钮是否可用状态 工具注册 =====================
# u8_materialout_buttonstate
registry.register(
    name="u8_materialout_buttonstate",
    toolset="u8",
    schema=U8_MATERIALOUT_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_materialout_buttonstate_tool(
        input_data=GetMaterialoutButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核材料出库单（工作流） 工具注册 =====================
# u8_materialout_audit
registry.register(
    name="u8_materialout_audit",
    toolset="u8",
    schema=U8_MATERIALOUT_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_materialout_audit_tool(
        input_data=AuditMaterialoutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增材料出库单 工具注册 =====================
# u8_materialout_add
registry.register(
    name="u8_materialout_add",
    toolset="u8",
    schema=U8_MATERIALOUT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_materialout_add_tool(
        input_data=AddMaterialoutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审材料出库单（工作流） 工具注册 =====================
# u8_materialout_abandon
registry.register(
    name="u8_materialout_abandon",
    toolset="u8",
    schema=U8_MATERIALOUT_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_materialout_abandon_tool(
        input_data=AbandonMaterialoutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 现存量 1个接口

# ===================== 现存量查询 工具注册 =====================
# u8_currentstock_batch_get
registry.register(
    name="u8_currentstock_batch_get",
    toolset="u8",
    schema=U8_CURRENTSTOCK_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_currentstock_batch_get_tool(
        input_data=GetCurrentstockInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 调拨单 5个接口

# ===================== 获取调拨单列表 工具注册 =====================
# u8_transvouch_list
registry.register(
    name="u8_transvouch_list",
    toolset="u8",
    schema=U8_TRANSVOUCH_LIST_SCHEMA,
    handler=lambda args, **kw: u8_transvouch_list_tool(
        input_data=GetTransvouchListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核调拨单 工具注册 =====================
# u8_transvouch_verify
registry.register(
    name="u8_transvouch_verify",
    toolset="u8",
    schema=U8_TRANSVOUCH_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_transvouch_verify_tool(
        input_data=VerifyTransvouchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审调拨单 工具注册 =====================
# u8_transvouch_unverify
registry.register(
    name="u8_transvouch_unverify",
    toolset="u8",
    schema=U8_TRANSVOUCH_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_transvouch_unverify_tool(
        input_data=UnverifyTransvouchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个调拨单 工具注册 =====================
# u8_transvouch_get
registry.register(
    name="u8_transvouch_get",
    toolset="u8",
    schema=U8_TRANSVOUCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_transvouch_get_tool(
        input_data=GetTransvouchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增调拨单 工具注册 =====================
# u8_transvouch_add
registry.register(
    name="u8_transvouch_add",
    toolset="u8",
    schema=U8_TRANSVOUCH_ADD_SCHEMA,
    handler=lambda args, **kw: u8_transvouch_add_tool(
        input_data=AddTransvouchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 调拨申请单 2个接口

# ===================== 获取调拨申请单列表 工具注册 =====================
# u8_transvouchapply_list
registry.register(
    name="u8_transvouchapply_list",
    toolset="u8",
    schema=U8_TRANSVOUCHAPPLY_LIST_SCHEMA,
    handler=lambda args, **kw: u8_transvouchapply_list_tool(
        input_data=GetTransvouchapplyListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个调拨申请单 工具注册 =====================
# u8_transvouchapply_get
registry.register(
    name="u8_transvouchapply_get",
    toolset="u8",
    schema=U8_TRANSVOUCHAPPLY_GET_SCHEMA,
    handler=lambda args, **kw: u8_transvouchapply_get_tool(
        input_data=GetTransvouchapplyInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 采购入库单 5个接口

# ===================== 获取采购入库单列表 工具注册 =====================
# u8_purchaseinlist_batch_get
registry.register(
    name="u8_purchaseinlist_batch_get",
    toolset="u8",
    schema=U8_PURCHASEINLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchaseinlist_batch_get_tool(
        input_data=GetPurchaseInListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核采购入库单 工具注册 =====================
# u8_purchasein_verify
registry.register(
    name="u8_purchasein_verify",
    toolset="u8",
    schema=U8_PURCHASEIN_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchasein_verify_tool(
        input_data=VerifyPurchaseInInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审采购入库单 工具注册 =====================
# u8_purchasein_unverify
registry.register(
    name="u8_purchasein_unverify",
    toolset="u8",
    schema=U8_PURCHASEIN_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_purchasein_unverify_tool(
        input_data=UnverifyPurchaseInInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个采购入库单 工具注册 =====================
# u8_purchasein_get
registry.register(
    name="u8_purchasein_get",
    toolset="u8",
    schema=U8_PURCHASEIN_GET_SCHEMA,
    handler=lambda args, **kw: u8_purchasein_get_tool(
        input_data=GetPurchaseInInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增采购入库单 工具注册 =====================
# u8_purchasein_add
registry.register(
    name="u8_purchasein_add",
    toolset="u8",
    schema=U8_PURCHASEIN_ADD_SCHEMA,
    handler=lambda args, **kw: u8_purchasein_add_tool(
        input_data=AddPurchaseInInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region 销售出库单 5个接口

# ===================== 获取销售出库单列表 工具注册 =====================
# u8_purchaseinlist_batch_get
registry.register(
    name="u8_saleoutlist_batch_get",
    toolset="u8",
    schema=U8_SALEOUTLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_saleoutlist_batch_get_tool(
        input_data=GetSaleOutListInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核销售出库单 工具注册 =====================
# u8_saleout_verify
registry.register(
    name="u8_saleout_verify",
    toolset="u8",
    schema=U8_SALEOUT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_saleout_verify_tool(
        input_data=VerifySaleOutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审销售出库单 工具注册 =====================
# u8_saleout_unverify
registry.register(
    name="u8_saleout_unverify",
    toolset="u8",
    schema=U8_SALEOUT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_saleout_unverify_tool(
        input_data=UnverifySaleOutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个销售出库单 工具注册 =====================
# u8_saleout_get
registry.register(
    name="u8_saleout_get",
    toolset="u8",
    schema=U8_SALEOUT_GET_SCHEMA,
    handler=lambda args, **kw: u8_saleout_get_tool(
        input_data=GetSaleOutInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 新增销售出库单 工具注册 =====================
# u8_saleout_add
registry.register(
    name="u8_saleout_add",
    toolset="u8",
    schema=U8_SALEOUT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_saleout_add_tool(
        input_data=AddSaleOutInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# endregion

# region u8_openapi_hr 人力资源类 8个接口；缺1个接口

# ===================== 获取单个员工任职信息 工具注册 =====================
# u8_jobinfo_get
registry.register(
    name="u8_jobinfo_get",
    toolset="u8",
    schema=U8_JOBINFO_GET_SCHEMA,
    handler=lambda args, **kw: u8_jobinfo_get_tool(
        input_data=GetJobInfoInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取员工任职信息 工具注册 =====================
# u8_jobinfo_batch_get
registry.register(
    name="u8_jobinfo_batch_get",
    toolset="u8",
    schema=U8_JOBINFO_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_jobinfo_batch_get_tool(
        input_data=GetJobInfoBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取某个员工工资记录 工具注册 =====================
# u8_salary_get
registry.register(
    name="u8_salary_get",
    toolset="u8",
    schema=U8_SALARY_GET_SCHEMA,
    handler=lambda args, **kw: u8_salary_get_tool(
        input_data=GetSalaryInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取员工工资记录 工具注册 =====================
# u8_salary_batch_get
registry.register(
    name="u8_salary_batch_get",
    toolset="u8",
    schema=U8_SALARY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_salary_batch_get_tool(
        input_data=GetSalaryBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个工资项目 工具注册 =====================
# u8_salaryitem_get
registry.register(
    name="u8_salaryitem_get",
    toolset="u8",
    schema=U8_SALARYITEM_GET_SCHEMA,
    handler=lambda args, **kw: u8_salaryitem_get_tool(
        input_data=GetSalaryItemInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取工资项目 工具注册 =====================
# u8_salaryitem_batch_get
registry.register(
    name="u8_salaryitem_batch_get",
    toolset="u8",
    schema=U8_SALARYITEM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_salaryitem_batch_get_tool(
        input_data=GetSalaryItemBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取考勤信息 工具注册 =====================
# u8_attendance_batch_get
registry.register(
    name="u8_attendance_batch_get",
    toolset="u8",
    schema=U8_ATTENDANCE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_attendance_batch_get_tool(
        input_data=GetAttendanceBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 批量获取薪资结账状态 工具注册 =====================
# u8_mendwa_batch_get
registry.register(
    name="u8_mendwa_batch_get",
    toolset="u8",
    schema=U8_MENDWA_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_mendwa_batch_get_tool(
        input_data=GetMendwaBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_ar 应收应付类 35个接口

# region 付款单 11个接口

# ===================== 新增一张付款单 工具注册 =====================
# u8_pay_add
registry.register(
    name="u8_pay_add",
    toolset="u8",
    schema=U8_PAY_ADD_SCHEMA,
    handler=lambda args, **kw: u8_pay_add_tool(
        input_data=AddPayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张付款单 工具注册 =====================
# u8_pay_get
registry.register(
    name="u8_pay_get",
    toolset="u8",
    schema=U8_PAY_GET_SCHEMA,
    handler=lambda args, **kw: u8_pay_get_tool(
        input_data=GetPayInfoInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款单列表信息 工具注册 =====================
# u8_pay_list_get
registry.register(
    name="u8_pay_list_get",
    toolset="u8",
    schema=U8_PAY_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_pay_list_get_tool(
        input_data=GetPayListInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审批一张付款单 工具注册 =====================
# u8_pay_verify
registry.register(
    name="u8_pay_verify",
    toolset="u8",
    schema=U8_PAY_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_pay_verify_tool(
        input_data=VerifyPayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审一张付款单 工具注册 =====================
# u8_pay_unverify
registry.register(
    name="u8_pay_unverify",
    toolset="u8",
    schema=U8_PAY_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_pay_unverify_tool(
        input_data=UnVerifyPayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)




# ===================== 获取付款单待办任务 工具注册 =====================
# u8_pay_tasks
registry.register(
    name="u8_pay_tasks",
    toolset="u8",
    schema=U8_PAY_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_pay_tasks_tool(
        input_data=GetPayTasksInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款单审批进程 工具注册 =====================
# u8_pay_history
registry.register(
    name="u8_pay_history",
    toolset="u8",
    schema=U8_PAY_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_pay_history_tool(
        input_data=GetPayHistoryInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款单是否启用工作流 工具注册 =====================
# u8_pay_flowenabled
registry.register(
    name="u8_pay_flowenabled",
    toolset="u8",
    schema=U8_PAY_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_pay_flowenabled_tool(
        input_data=GetPayFlowenabledInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款单工作流按钮是否可用状态 工具注册 =====================
# u8_pay_buttonstate
registry.register(
    name="u8_pay_buttonstate",
    toolset="u8",
    schema=U8_PAY_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_pay_buttonstate_tool(
        input_data=GetPayButtonstateInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核付款单 工具注册 =====================
# u8_pay_audit
registry.register(
    name="u8_pay_audit",
    toolset="u8",
    schema=U8_PAY_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_pay_audit_tool(
        input_data=AuditPayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审付款单 工具注册 =====================
# u8_pay_abandon
registry.register(
    name="u8_pay_abandon",
    toolset="u8",
    schema=U8_PAY_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_pay_abandon_tool(
        input_data=AbandonPayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion


# region 付款申请单 9个接口

# ===================== 获取付款申请单列表 工具注册 =====================
# u8_payrequest_list_get
registry.register(
    name="u8_payrequest_list_get",
    toolset="u8",
    schema=U8_PAYREQUEST_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_list_get_tool(
        input_data=GetPayrequestListInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款申请单待办任务 工具注册 =====================
# u8_payrequest_tasks
registry.register(
    name="u8_payrequest_tasks",
    toolset="u8",
    schema=U8_PAYREQUEST_TASKS_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_tasks_tool(
        input_data=GetPayrequestTasksInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 撤销付款申请单 工具注册 =====================
# u8_payrequest_return
registry.register(
    name="u8_payrequest_return",
    toolset="u8",
    schema=U8_PAYREQUEST_RETURN_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_return_tool(
        input_data=ReturnPayrequestInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款申请单审批进程 工具注册 =====================
# u8_payrequest_history
registry.register(
    name="u8_payrequest_history",
    toolset="u8",
    schema=U8_PAYREQUEST_HISTORY_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_history_tool(
        input_data=GetPayrequestHistoryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单个付款申请单 工具注册 =====================
# u8_payrequest_get
registry.register(
    name="u8_payrequest_get",
    toolset="u8",
    schema=U8_PAYREQUEST_GET_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_get_tool(
        input_data=GetPayrequestInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款申请单是否启用工作流 工具注册 =====================
# u8_payrequest_flowenabled
registry.register(
    name="u8_payrequest_flowenabled",
    toolset="u8",
    schema=U8_PAYREQUEST_FLOWENABLED_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_flowenabled_tool(
        input_data=GetPayrequestFlowenabledInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取付款申请单工作流按钮是否可用状态 工具注册 =====================
# u8_payrequest_buttonstate
registry.register(
    name="u8_payrequest_buttonstate",
    toolset="u8",
    schema=U8_PAYREQUEST_BUTTONSTATE_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_buttonstate_tool(
        input_data=GetPayrequestButtonstateInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审核付款申请单（工作流） 工具注册 =====================
# u8_payrequest_audit
registry.register(
    name="u8_payrequest_audit",
    toolset="u8",
    schema=U8_PAYREQUEST_AUDIT_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_audit_tool(
        input_data=AuditPayrequestInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审付款申请单 工具注册 =====================
# u8_payrequest_abandon
registry.register(
    name="u8_payrequest_abandon",
    toolset="u8",
    schema=U8_PAYREQUEST_ABANDON_SCHEMA,
    handler=lambda args, **kw: u8_payrequest_abandon_tool(
        input_data=AbandonPayrequestInfoInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion


# region 应付单 5个接口

# ===================== 新增一张应付单 工具注册 =====================
# u8_oughtpay_add
registry.register(
    name="u8_oughtpay_add",
    toolset="u8",
    schema=U8_OUGHTPAY_ADD_SCHEMA,
    handler=lambda args, **kw: u8_oughtpay_add_tool(
        input_data=AddOughtpayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张应付单 工具注册 =====================
# u8_oughtpay_get
registry.register(
    name="u8_oughtpay_get",
    toolset="u8",
    schema=U8_OUGHTPAY_GET_SCHEMA,
    handler=lambda args, **kw: u8_oughtpay_get_tool(
        input_data=GetOughtpayInfoInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取应付单列表信息 工具注册 =====================
# u8_oughtpay_list_get
registry.register(
    name="u8_oughtpay_list_get",
    toolset="u8",
    schema=U8_OUGHTPAY_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_oughtpay_list_get_tool(
        input_data=GetOughtpayListInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审批一张应付单 工具注册 =====================
# u8_oughtpay_verify
registry.register(
    name="u8_oughtpay_verify",
    toolset="u8",
    schema=U8_OUGHTPAY_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_oughtpay_verify_tool(
        input_data=VerifyOughtpayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审一张应付单 工具注册 =====================
# u8_oughtpay_unverify
registry.register(
    name="u8_oughtpay_unverify",
    toolset="u8",
    schema=U8_OUGHTPAY_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_oughtpay_unverify_tool(
        input_data=UnVerifyOughtpayInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion


# region 应收单 5个接口

# ===================== 新增一张应收单 工具注册 =====================
# u8_oughtreceive_add
registry.register(
    name="u8_oughtreceive_add",
    toolset="u8",
    schema=U8_OUGHTRECEIVE_ADD_SCHEMA,
    handler=lambda args, **kw: u8_oughtreceive_add_tool(
        input_data=AddOughtreceiveInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张应收单 工具注册 =====================
# u8_oughtreceive_get
registry.register(
    name="u8_oughtreceive_get",
    toolset="u8",
    schema=U8_OUGHTRECEIVE_GET_SCHEMA,
    handler=lambda args, **kw: u8_oughtreceive_get_tool(
        input_data=GetOughtreceiveInfoInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取应收单列表信息 工具注册 =====================
# u8_oughtreceive_list_get
registry.register(
    name="u8_oughtreceive_list_get",
    toolset="u8",
    schema=U8_OUGHTRECEIVE_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_oughtreceive_list_get_tool(
        input_data=GetOughtreceiveListInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审批一张应收单 工具注册 =====================
# u8_oughtreceive_verify
registry.register(
    name="u8_oughtreceive_verify",
    toolset="u8",
    schema=U8_OUGHTRECEIVE_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_oughtreceive_verify_tool(
        input_data=VerifyOughtreceiveInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审一张应收单 工具注册 =====================
# u8_oughtreceive_unverify
registry.register(
    name="u8_oughtreceive_unverify",
    toolset="u8",
    schema=U8_OUGHTRECEIVE_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_oughtreceive_unverify_tool(
        input_data=UnVerifyOughtreceiveInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion


# region 收款单 5个接口

# ===================== 新增一张收款单 工具注册 =====================
# u8_accept_add
registry.register(
    name="u8_accept_add",
    toolset="u8",
    schema=U8_ACCEPT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_accept_add_tool(
        input_data=AddAcceptInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取单张收款单 工具注册 =====================
# u8_accept_get
registry.register(
    name="u8_accept_get",
    toolset="u8",
    schema=U8_ACCEPT_GET_SCHEMA,
    handler=lambda args, **kw: u8_accept_get_tool(
        input_data=GetAcceptInfoInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 获取收款单列表信息 工具注册 =====================
# u8_accept_list_get
registry.register(
    name="u8_accept_list_get",
    toolset="u8",
    schema=U8_ACCEPT_LIST_GET_SCHEMA,
    handler=lambda args, **kw: u8_accept_list_get_tool(
        input_data=GetAcceptListInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 审批一张收款单 工具注册 =====================
# u8_accept_verify
registry.register(
    name="u8_accept_verify",
    toolset="u8",
    schema=U8_ACCEPT_VERIFY_SCHEMA,
    handler=lambda args, **kw: u8_accept_verify_tool(
        input_data=VerifyAcceptInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# ===================== 弃审一张收款单 工具注册 =====================
# u8_accept_unverify
registry.register(
    name="u8_accept_unverify",
    toolset="u8",
    schema=U8_ACCEPT_UNVERIFY_SCHEMA,
    handler=lambda args, **kw: u8_accept_unverify_tool(
        input_data=UnVerifyAcceptInfoInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion



# endregion

# region u8_openapi_bi 商业智能类 3个接口

# ===================== 获取EVA体检模型信息 工具注册 =====================
registry.register(
    name="u8_eva_batch_get",
    toolset="u8",
    schema=U8_EVA_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_eva_batch_get_tool(
        input_data=EvaBatchGetInput(year=args.get("year", ""), month=args.get("month", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取商业盈利状况评价信息 工具注册 =====================
registry.register(
    name="u8_productprofitability_batch_get",
    toolset="u8",
    schema=U8_PRODUCTPROFITABILITY_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_productprofitability_batch_get_tool(
        input_data=ProductProfitabilityBatchGetInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取资金体检模型信息 工具注册 =====================
registry.register(
    name="u8_fund_batch_get",
    toolset="u8",
    schema=U8_FUND_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_fund_batch_get_tool(
        input_data=FundBatchGetInput(year=args.get("year", ""), month=args.get("month", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_dp 用户管理类 5个接口

# ===================== 获取U8模块启用状态 工具注册 =====================
# u8_systemstate_batch_get
registry.register(
    name="u8_systemstate_batch_get",
    toolset="u8",
    schema=U8_SYSTEMSTATE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_systemstate_batch_get_tool(
        input_data=GetSystemstateBatchInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取单个操作员 工具注册 =====================
# u8_operator_get
registry.register(
    name="u8_operator_get",
    toolset="u8",
    schema=U8_OPERATOR_GET_SCHEMA,
    handler=lambda args, **kw: u8_operator_get_tool(
        input_data=GetOperatorInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取操作员 工具注册 =====================
# u8_operator_batch_get
registry.register(
    name="u8_operator_batch_get",
    toolset="u8",
    schema=U8_OPERATOR_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_operator_batch_get_tool(
        input_data=GetOperatorBatchInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 用户登录 工具注册 =====================
# u8_user_login
registry.register(
    name="u8_user_login",
    toolset="u8",
    schema=U8_USER_LOGIN_SCHEMA,
    handler=lambda args, **kw: u8_user_login_tool(
        input_data=UserLoginInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 用户登录v2 工具注册 =====================
# u8_user_login_v2
registry.register(
    name="u8_user_login_v2",
    toolset="u8",
    schema=U8_USER_LOGIN_V2_SCHEMA,
    handler=lambda args, **kw: u8_user_login_v2_tool(
        input_data=UserLoginV2Input(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_bm 预算管理类 2个接口

# ===================== 获取单个预算项目 工具注册 =====================
# u8_bgitem_get
registry.register(
    name="u8_bgitem_get",
    toolset="u8",
    schema=U8_BGITEM_GET_SCHEMA,
    handler=lambda args, **kw: u8_bgitem_get_tool(
        input_data=GetBgitemInput(id=args.get("id", "")),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取预算项目 工具注册 =====================
# u8_bgitem_batch_get
registry.register(
    name="u8_bgitem_batch_get",
    toolset="u8",
    schema=U8_BGITEM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_bgitem_batch_get_tool(
        input_data=GetBgitemBatchInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_fb 费用预算类 4个接口

# ===================== 费用预算查看 工具注册 =====================
# u8_budget_query
registry.register(
    name="u8_budget_query",
    toolset="u8",
    schema=U8_BUDGET_QUERY_SCHEMA,
    handler=lambda args, **kw: u8_budget_query_tool(
        input_data=BudgetQueryInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取预算信息 工具注册 =====================
# u8_budget_batch_get
registry.register(
    name="u8_budget_batch_get",
    toolset="u8",
    schema=U8_BUDGET_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_budget_batch_get_tool(
        input_data=GetBudgetBatchInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取单个预算项目 工具注册 =====================
# u8_budgetitem_get
registry.register(
    name="u8_budgetitem_get",
    toolset="u8",
    schema=U8_BUDGETITEM_GET_SCHEMA,
    handler=lambda args, **kw: u8_budgetitem_get_tool(
        input_data=GetBudgetitemInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取预算项目 工具注册 =====================
# u8_budgetitem_batch_get
registry.register(
    name="u8_budgetitem_batch_get",
    toolset="u8",
    schema=U8_BUDGETITEM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_budgetitem_batch_get_tool(
        input_data=GetBudgetitemBatchInput(**args),
        client=get_u8_client()  # 使用单例客户端
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_eb 电商类 8个接口

# ===================== 批量获取店铺商品档案 工具注册 =====================
# u8_eb_iteminvcontrapose_batch_get
registry.register(
    name="u8_eb_iteminvcontrapose_batch_get",
    toolset="u8",
    schema=U8_EB_ITEMINVCONTRAPOSE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_eb_iteminvcontrapose_batch_get_tool(
        input_data=GetEbIteminvcontraposeBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增店铺商品档案 工具注册 =====================
# u8_eb_iteminvcontrapose_add
registry.register(
    name="u8_eb_iteminvcontrapose_add",
    toolset="u8",
    schema=U8_EB_ITEMINVCONTRAPOSE_ADD_SCHEMA,
    handler=lambda args, **kw: u8_eb_iteminvcontrapose_add_tool(
        input_data=AddEbIteminvcontraposeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取电商订单(v2) 工具注册 =====================
# u8_eb_tradelist_v2_batch_get
registry.register(
    name="u8_eb_tradelist_v2_batch_get",
    toolset="u8",
    schema=U8_EB_TRADELIST_V2_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_eb_tradelist_v2_batch_get_tool(
        input_data=GetEbTradelistV2BatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取电商订单(v2) 工具注册 =====================
# u8_eb_trade_v2_get
registry.register(
    name="u8_eb_trade_v2_get",
    toolset="u8",
    schema=U8_EB_TRADE_V2_GET_SCHEMA,
    handler=lambda args, **kw: u8_eb_trade_v2_get_tool(
        input_data=GetEbTradeV2Input(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取电商订单(v2) 工具注册 =====================
# u8_eb_trade_v2_batch_get
registry.register(
    name="u8_eb_trade_v2_batch_get",
    toolset="u8",
    schema=U8_EB_TRADE_V2_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_eb_trade_v2_batch_get_tool(
        input_data=GetEbTradeV2BatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取电商订单 工具注册 =====================
# u8_eb_tradelist_batch_get
registry.register(
    name="u8_eb_tradelist_batch_get",
    toolset="u8",
    schema=U8_EB_TRADELIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_eb_tradelist_batch_get_tool(
        input_data=GetEbTradelistBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取电商订单 工具注册 =====================
# u8_eb_trade_get
registry.register(
    name="u8_eb_trade_get",
    toolset="u8",
    schema=U8_EB_TRADE_GET_SCHEMA,
    handler=lambda args, **kw: u8_eb_trade_get_tool(
        input_data=GetEbTradeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增电商订单 工具注册 =====================
# u8_eb_trade_add
registry.register(
    name="u8_eb_trade_add",
    toolset="u8",
    schema=U8_EB_TRADE_ADD_SCHEMA,
    handler=lambda args, **kw: u8_eb_trade_add_tool(
        input_data=AddEbTradeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_ex 出口管理类 3个接口；缺1个接口

# ===================== 批量获取出口订单列表 工具注册 =====================
# u8_exportorderlist_batch_get
registry.register(
    name="u8_exportorderlist_batch_get",
    toolset="u8",
    schema=U8_EXPORTORDERLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_exportorderlist_batch_get_tool(
        input_data=GetExportorderlistBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取出口订单 工具注册 =====================
# u8_exportorder_get
registry.register(
    name="u8_exportorder_get",
    toolset="u8",
    schema=U8_EXPORTORDER_GET_SCHEMA,
    handler=lambda args, **kw: u8_exportorder_get_tool(
        input_data=GetExportorderInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增出口订单 工具注册 =====================
# u8_exportorder_add
registry.register(
    name="u8_exportorder_add",
    toolset="u8",
    schema=U8_EXPORTORDER_ADD_SCHEMA,
    handler=lambda args, **kw: u8_exportorder_add_tool(
        input_data=AddExportorderInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# endregion

# region u8_openapi_kc 条码管理类 2个接口

# ===================== 获取单个条码档案 工具注册 =====================
# u8_barcode_get
registry.register(
    name="u8_barcode_get",
    toolset="u8",
    schema=U8_BARCODE_GET_SCHEMA,
    handler=lambda args, **kw: u8_barcode_get_tool(
        input_data=GetBarcodeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取条码档案 工具注册 =====================
# u8_barcode_batch_get
registry.register(
    name="u8_barcode_batch_get",
    toolset="u8",
    schema=U8_BARCODE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_barcode_batch_get_tool(
        input_data=GetBarcodeBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_3u U易联类 U商城类 U订货类 5个接口

# ===================== U易联积分增加 工具注册 =====================
# u8_points_change
registry.register(
    name="u8_points_change",
    toolset="u8",
    schema=U8_POINTS_CHANGE_SCHEMA,
    handler=lambda args, **kw: u8_points_change_tool(
        input_data=PointsChangeInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== U易联积分明细查询 工具注册 =====================
# u8_points_query
registry.register(
    name="u8_points_query",
    toolset="u8",
    schema=U8_POINTS_QUERY_SCHEMA,
    handler=lambda args, **kw: u8_points_query_tool(
        input_data=PointsQueryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== U易联订单交易记录查询 工具注册 =====================
# u8_orders_query
registry.register(
    name="u8_orders_query",
    toolset="u8",
    schema=U8_ORDERS_QUERY_SCHEMA,
    handler=lambda args, **kw: u8_orders_query_tool(
        input_data=OrdersQueryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== U商城商品列表 工具注册 =====================
# u8_productlist_query
registry.register(
    name="u8_productlist_query",
    toolset="u8",
    schema=U8_PRODUCTLIST_QUERY_SCHEMA,
    handler=lambda args, **kw: u8_productlist_query_tool(
        input_data=ProductlistQueryInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== U订货批量获取订单信息 工具注册 =====================
# u8_udh_orderlist_batch_get
registry.register(
    name="u8_udh_orderlist_batch_get",
    toolset="u8",
    schema=U8_UDH_ORDERLIST_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_udh_orderlist_batch_get_tool(
        input_data=UdhOrderlistBatchGetInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_crm U8CRM类 1个接口

# ===================== 批量查询CRM客户档案 工具注册 =====================
# u8_crmaccount_batch_get
registry.register(
    name="u8_crmaccount_batch_get",
    toolset="u8",
    schema=U8_CRMACCOUNT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_crmaccount_batch_get_tool(
        input_data=GetCrmaccountBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_cl 资质管理类 4个接口

# ===================== 获取单个客户资质审批 工具注册 =====================
# u8_customerlicence_get
registry.register(
    name="u8_customerlicence_get",
    toolset="u8",
    schema=U8_CUSTOMERLICENCE_GET_SCHEMA,
    handler=lambda args, **kw: u8_customerlicence_get_tool(
        input_data=GetCustomerlicenceInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户资质审批 工具注册 =====================
# u8_customerlicence_batch_get
registry.register(
    name="u8_customerlicence_batch_get",
    toolset="u8",
    schema=U8_CUSTOMERLICENCE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customerlicence_batch_get_tool(
        input_data=GetCustomerlicenceBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取单个客户资质经营范围审批 工具注册 =====================
# u8_customerlicencebizscope_get
registry.register(
    name="u8_customerlicencebizscope_get",
    toolset="u8",
    schema=U8_CUSTOMERLICENCEBIZSCOPE_GET_SCHEMA,
    handler=lambda args, **kw: u8_customerlicencebizscope_get_tool(
        input_data=GetCustomerlicencebizscopeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取客户资质经营范围审批 工具注册 =====================
# u8_customerlicencebizscope_batch_get
registry.register(
    name="u8_customerlicencebizscope_batch_get",
    toolset="u8",
    schema=U8_CUSTOMERLICENCEBIZSCOPE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_customerlicencebizscope_batch_get_tool(
        input_data=GetCustomerlicencebizscopeBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_gsp GSP管理类 2个接口

# ===================== 获取单个药品停售通知单 工具注册 =====================
# u8_stopsalenotice_get
registry.register(
    name="u8_stopsalenotice_get",
    toolset="u8",
    schema=U8_STOPSALENOTICE_GET_SCHEMA,
    handler=lambda args, **kw: u8_stopsalenotice_get_tool(
        input_data=GetStopsalenoticeInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取药品停售通知单 工具注册 =====================
# u8_stopsalenotice_batch_get
registry.register(
    name="u8_stopsalenotice_batch_get",
    toolset="u8",
    schema=U8_STOPSALENOTICE_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_stopsalenotice_batch_get_tool(
        input_data=GetStopsalenoticeBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_mm 物料清单类 2个接口

# ===================== 获取单个物料清单 工具注册 =====================
# u8_bom_get
registry.register(
    name="u8_bom_get",
    toolset="u8",
    schema=U8_BOM_GET_SCHEMA,
    handler=lambda args, **kw: u8_bom_get_tool(
        input_data=GetBomInput(id=args.get("id", 0)),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取物料清单 工具注册 =====================
# u8_bom_batch_get
registry.register(
    name="u8_bom_batch_get",
    toolset="u8",
    schema=U8_BOM_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_bom_batch_get_tool(
        input_data=GetBomBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion

# region u8_openapi_nb 网上银行类 6个接口

# ===================== 获取单个对私支付单 工具注册 =====================
# u8_privatepayment_get
registry.register(
    name="u8_privatepayment_get",
    toolset="u8",
    schema=U8_PRIVATEPAYMENT_GET_SCHEMA,
    handler=lambda args, **kw: u8_privatepayment_get_tool(
        input_data=GetPrivatepaymentInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取对私支付单 工具注册 =====================
# u8_privatepayment_batch_get
registry.register(
    name="u8_privatepayment_batch_get",
    toolset="u8",
    schema=U8_PRIVATEPAYMENT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_privatepayment_batch_get_tool(
        input_data=GetPrivatepaymentBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增对私支付单 工具注册 =====================
# u8_privatepayment_add
registry.register(
    name="u8_privatepayment_add",
    toolset="u8",
    schema=U8_PRIVATEPAYMENT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_privatepayment_add_tool(
        input_data=AddPrivatepaymentInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 获取单个普通支付单 工具注册 =====================
# u8_payment_get
registry.register(
    name="u8_payment_get",
    toolset="u8",
    schema=U8_PAYMENT_GET_SCHEMA,
    handler=lambda args, **kw: u8_payment_get_tool(
        input_data=GetPaymentInput(id=args.get("id", "")),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 批量获取普通支付单 工具注册 =====================
# u8_payment_batch_get
registry.register(
    name="u8_payment_batch_get",
    toolset="u8",
    schema=U8_PAYMENT_BATCH_GET_SCHEMA,
    handler=lambda args, **kw: u8_payment_batch_get_tool(
        input_data=GetPaymentBatchInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)


# ===================== 新增普通支付单 工具注册 =====================
# u8_payment_add
registry.register(
    name="u8_payment_add",
    toolset="u8",
    schema=U8_PAYMENT_ADD_SCHEMA,
    handler=lambda args, **kw: u8_payment_add_tool(
        input_data=AddPaymentInput(**args),
        client=get_u8_client()
    ),
    check_fn=check_u8_openapi_requirements,
    requires_env=["U8_OPENAPI_APPKEY", "U8_OPENAPI_APPSECRET", "U8_OPENAPI_FROM_ACCOUNT", "U8_OPENAPI_TO_ACCOUNT"],
)

# endregion