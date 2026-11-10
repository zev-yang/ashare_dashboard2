"""全局配置。板块成分、ETF 代码、阈值都在这里改，不需要动其它文件。"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
# 演示模式：读 data_demo/ 里的模拟数据。只有 ASHARE_DEMO=1 且没有指定真实日线目录时才生效，
# 避免"之前试过演示、环境变量还留在终端里"导致真实数据被演示模式覆盖。
DEMO = os.environ.get("ASHARE_DEMO") == "1" and "ASHARE_DAILY_DIR" not in os.environ
DATA_ROOT = BASE / ("data_demo" if DEMO else "data")
# 本地日线目录：把你下载好的指数/ETF/个股日线 csv 放这里（可有子目录），或用环境变量指定
DAILY_DIR = Path(os.environ.get("ASHARE_DAILY_DIR", DATA_ROOT / "daily"))
CACHE_DIR = DATA_ROOT / "cache"
for _p in (DAILY_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# 本地日线"成交额"列换算成"元"的乘数（本地是元就填 1，万元填 10000，千元填 1000）
AMOUNT_UNIT = float(os.environ.get("ASHARE_AMOUNT_UNIT", "1"))

# 四个宽基：指数代码、对应股指期货、合约乘数（元/点）
INDEXES = {
    "沪深300": dict(code="000300", fut="IF", mult=300),
    "上证50": dict(code="000016", fut="IH", mult=300),
    "中证500": dict(code="000905", fut="IC", mult=200),
    "中证1000": dict(code="000852", fut="IM", mult=200),
}
BENCH = "沪深300"

# 宽基 ETF（份额数据用交易所口径；深市 159xxx 的历史份额不易批量获取，先只放沪市，需要再加）
ETF_GROUPS = {
    "沪深300": ["510300", "510310", "510330"],
    "上证50": ["510050", "510100", "510710"],
    "中证500": ["510500", "512500", "510510"],
    "中证1000": ["512100", "560010"],
}

# 避险板块。代码凭记忆整理，首次使用请在侧边栏"代码核对"里确认本地都有数据，并自行校对龙头名单。
# role: defensive=纳入"防御综合"；cyclical=只做对照，不进综合；optional=默认不进综合
SECTORS = {
    "银行": dict(role="defensive", stocks=[
        ("601398", "工商银行"), ("601939", "建设银行"), ("601288", "农业银行"), ("601988", "中国银行"),
        ("601328", "交通银行"), ("601658", "邮储银行"), ("600036", "招商银行")]),
    "电力(水电/核电)": dict(role="defensive", stocks=[
        ("600900", "长江电力"), ("600025", "华能水电"), ("600886", "国投电力"),
        ("600674", "川投能源"), ("601985", "中国核电"), ("003816", "中国广核")]),
    "铁路公路": dict(role="defensive", stocks=[
        ("601006", "大秦铁路"), ("601816", "京沪高铁"), ("600350", "山东高速"), ("600377", "宁沪高速"),
        ("600548", "深高速"), ("600012", "皖通高速"), ("000429", "粤高速A"), ("601107", "四川成渝")]),
    "港口": dict(role="defensive", stocks=[
        ("600018", "上港集团"), ("001872", "招商港口"), ("601000", "唐山港"),
        ("601018", "宁波港"), ("601298", "青岛港")]),
    "煤炭": dict(role="defensive", stocks=[
        ("601088", "中国神华"), ("601225", "陕西煤业"), ("601898", "中煤能源")]),
    "石油石化": dict(role="defensive", stocks=[
        ("601857", "中国石油"), ("600028", "中国石化"), ("600938", "中国海油")]),
    "通信运营商": dict(role="defensive", stocks=[
        ("600941", "中国移动"), ("601728", "中国电信"), ("600050", "中国联通")]),
    "央企基建": dict(role="optional", stocks=[
        ("601390", "中国中铁"), ("601186", "中国铁建"), ("601668", "中国建筑"), ("601800", "中国交建")]),
    "消费防御": dict(role="optional", stocks=[
        ("600519", "贵州茅台"), ("600887", "伊利股份"), ("603288", "海天味业")]),
    "黄金": dict(role="optional", stocks=[
        ("601899", "紫金矿业"), ("600547", "山东黄金"), ("600489", "中金黄金")]),
    "航运(周期对照)": dict(role="cyclical", stocks=[
        ("601919", "中远海控"), ("601872", "招商轮船"), ("600026", "中远海能")]),
}
# 参考 ETF（本地有日线就显示）
REF_ETFS = {"红利ETF": "510880", "红利低波ETF": "512890", "黄金ETF": "518880", "国债ETF": "511010"}

# 前20会员里重点看的席位
WATCH_MEMBERS = ["中信期货", "国泰君安", "海通期货", "华泰期货", "东证期货", "银河期货", "中信建投", "广发期货"]

# "空单性质"判断阈值（金额单位：亿元）。这些数是启发式的，需要用回测页检验后再调。
HEDGE = dict(window=5, min_short_add=5.0, ratio_lo=0.5, ratio_hi=1.5, ratio_spec=2.0)

# 自动下载的起始日期（日线、外资代理）；首次回补从这里开始
PX_START = "20190101"
# 外资代理指标：A50 在新浪外盘期货里的代码，若接口改名可在此调整
A50_SYMBOL = "CHA50CFD"   # 实测：新浪 XIN9 不可用，CHA50CFD 可用

# 数据源：auto=有 Tushare token 就优先用它、失败再用免费源；free=只用免费源；tushare=只用 Tushare
SOURCE = os.environ.get("ASHARE_SOURCE", "auto")
# Tushare 每分钟最多调用次数（你的账号 daily 接口上限是 50 次/分，留余量用 42）
TUSHARE_CALLS_PER_MIN = int(os.environ.get("TUSHARE_CALLS_PER_MIN", "42"))
# token 优先读环境变量 TUSHARE_TOKEN，其次读这个文件（已在 .gitignore，不要提交到 GitHub）
TOKEN_FILE = BASE / "tushare_token.txt"

# 自动更新：收盘后（默认 17:30 后）在交易日首次打开网页、或由计划任务调用 auto_update.py 时，自动补齐最新数据。
# 环境变量 ASHARE_AUTO_UPDATE=0 可整体关闭默认值（网页「数据」页里的开关优先）。
AUTO_UPDATE_DEFAULT = os.environ.get("ASHARE_AUTO_UPDATE", "1") != "0"
READY_CUTOFF = "17:30"
