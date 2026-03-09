"""Flat version of code/main.py with major steps inlined into one file."""
import os
import getpass
from dataclasses import dataclass
from pathlib import Path

import keyring
import polars as pl

from aux_functions import *  # noqa: F403,F401

########## PIPELINE ROADMAP ##########
# This file is intentionally procedural and linear.
# Each large block corresponds to one original pipeline step from code/main.py.
# Section banners below help you scan setup -> raw data -> transforms -> factors -> outputs.

os.environ["PYTHON_KEYRING_BACKEND"] = "keyrings.alt.file.PlaintextKeyring"
SERVICE_NAME = "WRDS"
LAST_USER_FILE = Path.home() / ".wrds_user"

@dataclass(frozen=True)
class Credentials:
    username: str
    password: str

end_date = pl.datetime(2024, 12, 31)


########## GET_WRDS_CREDENTIALS ##########
# Inlined from get_wrds_credentials

"""
Automatically retrieves credentials for wrds.
- On first run: asks for username and password/token, stores them.
- On later runs: loads both silently from the system keyring.
"""
# Try to remember the last-used username
if LAST_USER_FILE.exists():
    username = LAST_USER_FILE.read_text().strip()
else:
    username = input(f"Username for {SERVICE_NAME}: ").strip()
    LAST_USER_FILE.write_text(username)

# Try to retrieve the stored password for this username
password = keyring.get_password(SERVICE_NAME, username)

# If not found, prompt once and store it securely
if not password:
    password = getpass.getpass(f"Password or token for {username} at {SERVICE_NAME}: ")
    keyring.set_password(SERVICE_NAME, username, password)
    print(f"Stored credentials for '{username}' in keyring under '{SERVICE_NAME}'")

creds = Credentials(username, password)


########## AUX_FUNCTIONS.SETUP_FOLDER_STRUCTURE() ##########
# Inlined from aux_functions.setup_folder_structure()
"""
Description:
    Create the project’s folder structure if missing.

Steps:
    1) Make directories: raw_tables, raw_data_dfs, characteristics, return_data, accounting_data, other_output.

Output:
    Folders created on disk (no return value).
"""
os.chdir(os.path.join(os.path.dirname(__file__), "..", "data/interim"))
os.system(
    "mkdir -p raw_data_dfs ../raw/raw_tables ../processed/characteristics ../processed/return_data ../processed/accounting_data ../processed/other_output"
)
os.system("mkdir -p ../processed/return_data/daily_rets_by_country")


########## AUX_FUNCTIONS.DOWNLOAD_RAW_DATA_TABLES(USERNAME=CREDS.USERNAME, PASSWORD=CREDS.PASSWORD) ##########
# Inlined from aux_functions.download_raw_data_tables(username=creds.username, password=creds.password)
username = creds.username
password = creds.password
"""
Description:
    Bulk-download core WRDS tables to raw_tables and a few curated variants with column subsets.

Steps:
    1) Connect to WRDS; iterate through a fixed list of library.tables.
    2) For each table: download to raw_tables/lib_table.parquet; periodically reset connection.
    3) Additionally fetch comp.secd and comp.g_secd with curated columns.
    4) Disconnect.

Output:
    Parquet files under raw_tables/ (Compustat, CRSP, FF, etc.).
"""
table_names = [
    "comp.exrt_dly",
    "ff.factors_monthly",
    "comp.g_security",
    "comp.security",
    "comp.r_ex_codes",
    "comp.g_sec_history",
    "comp.sec_history",
    "comp.company",
    "comp.g_company",
    "crsp.stksecurityinfohist",
    "crsp.stkissuerinfohist",
    "crsp.ccmxpf_lnkhist",
    "comp.funda",
    "comp.fundq",
    "crsp.stkdelists",
    "comp.secm",
    "crsp.indmthseriesdata_ind",
    "crsp.indseriesinfohdr_ind",
    "crsp.msf_v2",
    "comp.g_co_hgic",
    "crsp.dsf_v2",
    "comp.g_funda",
    "comp.co_hgic",
    "comp.g_fundq",
    "comp.secd",
    "comp.g_secd",
]

wrds_session_data = gen_wrds_connection_info(username, password)
con = duckdb.connect(":memory:")
con.execute("INSTALL postgres; LOAD postgres;")

for table in table_names:
    # print(f"Downloading WRDS table: {table}", flush=True)
    download_wrds_table(
        wrds_session_data,
        con,
        table,
        "../raw/raw_tables/" + table.replace(".", "_") + ".parquet",
    )

con.close()


########## AUX_FUNCTIONS.AUG_MSF_V2() ##########
# Inlined from aux_functions.aug_msf_v2()
"""
Description:
    Add month-level high/low transaction-price fields to the CRSP CIZ monthly file (msf_v2)
    using the CRSP CIZ daily file (dsf_v2). Keep all msf_v2 rows; set the new fields to
    missing for non-TR monthly rows (e.g., BA).

Steps:
    1) Read msf_v2 (monthly) and dsf_v2 (daily) from parquet.
    2) Filter dsf_v2 to dlyprcflg == "TR", construct yyyymm from dlycaldt, and keep (permno, yyyymm, dlyprc).
    3) For each (permno, yyyymm), compute:
       - mthaskhi = max(dlyprc)
       - mthbidlo = min(dlyprc)
    4) Left-join these two fields onto msf_v2 by (permno, yyyymm).
    5) Set mthaskhi/mthbidlo to NULL when mthprcflg != "TR".
    6) Overwrite crsp_msf_v2.parquet (via a temp file).

Output:
    Overwrites ../raw/raw_tables/crsp_msf_v2.parquet with two new columns:
    mthaskhi and mthbidlo.
"""
con = ibis.duckdb.connect(threads=os.cpu_count())
msf = con.read_parquet("../raw/raw_tables/crsp_msf_v2.parquet")
dsf = con.read_parquet("../raw/raw_tables/crsp_dsf_v2.parquet")

dt = dsf.dlycaldt.cast("date")

d = (
    dsf.filter(dsf.dlyprcflg == "TR")
    .mutate(
        yyyymm=(dt.year() * 100 + dt.month()).cast("int32"),
        dlyprc=dsf.dlyprc.cast("double"),
    )
    .select(["permno", "yyyymm", "dlyprc"])
)

m = d.group_by(["permno", "yyyymm"]).aggregate(
    mthaskhi=d.dlyprc.max(),
    mthbidlo=d.dlyprc.min(),
)

msf_joined = msf.join(
    m,
    how="left",
    predicates=[msf.permno == m.permno, msf.yyyymm == m.yyyymm],
).select([msf] + [m.mthaskhi, m.mthbidlo])

msf_aug = msf_joined.mutate(
    mthaskhi=ibis.cases(
        (msf_joined.mthprcflg == "TR", msf_joined.mthaskhi),
        else_=ibis.null(),
    ),
    mthbidlo=ibis.cases(
        (msf_joined.mthprcflg == "TR", msf_joined.mthbidlo),
        else_=ibis.null(),
    ),
)

msf_aug.to_parquet("../raw/raw_tables/crsp_msf_v2.parquet.tmp")
os.replace("../raw/raw_tables/crsp_msf_v2.parquet.tmp", "../raw/raw_tables/crsp_msf_v2.parquet")


########## AUX_FUNCTIONS.BUILD_MCTI() ##########
# Inlined from aux_functions.build_mcti()
"""
Description:
    Build monthly t30 return raw data.

Steps:
    1) Read indmthseriesdata and indseriesinfohdr from parquet.
    2) Inner join indmthseriesdata with indseriesinfohdr on "indno".
    4) Filter for CRSP 30-Year Treasury Returns (indno == 1000708).
    5) Write parquet to ../raw/raw_tables/crsp_mcti.parquet

Output:
    Polars DataFrame (also written to parquet).
"""

a = pl.read_parquet("../raw/raw_tables/crsp_indmthseriesdata_ind.parquet")
b = pl.read_parquet("../raw/raw_tables/crsp_indseriesinfohdr_ind.parquet")

ab = a.join(b, on="indno", how="inner")

out = (
    ab.filter(pl.col("indno") == 1000708)
    .select(["indno", "indnm", "mthcaldt", "mthtotret", "mthtotind"])
    .rename({"mthcaldt": "caldt", "mthtotret": "t30ret"})
)

os.makedirs("../raw/raw_tables", exist_ok=True)
out.write_parquet("../raw/raw_tables/crsp_mcti.parquet")


########## AUX_FUNCTIONS.GEN_RAW_DATA_DFS() ##########
# Inlined from aux_functions.gen_raw_data_dfs()
"""
Description:
    Generate a suite of “raw data” helper Parquet files from Compustat/CRSP sources.

Steps:
    1) Generate firm-share links, then derive related raw helper datasets.

    2) Derive SIC/NAICS (NA & Global), GICS (NA & Global), delist files (CRSP m/d),
    security info (NA & Global), T-bill return, FF monthly factors, exchange code map,
    exchange→country map, company headers (NA+Global), and CRSP security files (m & d).
    3) Standardize types/columns, sort/deduplicate where needed.
    4) Write all to raw_data_dfs/*.parquet.

Output:
    Multiple helper Parquets under raw_data_dfs/ used in later pipelines.
"""
# Inlined body of aux_functions.gen_firmshares()
perm = (
    pl.scan_parquet("../raw/raw_tables/crsp_stksecurityinfohist.parquet")
    .select(["permno", "permco", "secinfostartdt", "secinfoenddt", "sharetype", "securitysubtype"])
    .filter((col("sharetype") == "NS") & (col("securitysubtype") == "COM"))
    .with_columns(
        [
            col("permno").cast(pl.Int64),
            col("permco").cast(pl.Int64),
            col("secinfostartdt").dt.date(),
            col("secinfoenddt").dt.date(),
        ]
    )
    .unique()
)

ccm = (
    pl.scan_parquet("../raw/raw_tables/crsp_ccmxpf_lnkhist.parquet")
    .select(["gvkey", "lpermco", "linkdt", "linkenddt", "linkprim", "liid", "linktype"])
    .filter(col("lpermco").is_not_null())
    .with_columns(
        [
            col("lpermco").cast(pl.Int64).alias("permco"),
            col("gvkey").cast(pl.String),
            col("linkdt").dt.date(),
            col("linkenddt").dt.date(),
        ]
    )
    .select(["gvkey", "permco", "linkdt", "linkenddt", "linkprim", "liid", "linktype"])
    .unique()
)

shs = (
    ccm.join(perm, on="permco", how="inner")
    .filter((col("linkdt") <= col("secinfoenddt")) & (col("linkenddt") >= col("secinfostartdt")))
    .with_columns(
        [
            pl.max_horizontal([col("linkdt"), col("secinfostartdt")]).alias("start_date"),
            pl.min_horizontal([col("linkenddt"), col("secinfoenddt")]).alias("end_date"),
        ]
    )
    .select(["gvkey", "permco", "permno", "start_date", "end_date"])
    .sort(["gvkey", "permco", "permno", "start_date", "end_date"])
)

shs.collect().write_parquet("raw_data_dfs/__firm_shares1.parquet")
sic_naics_na = (
    pl.scan_parquet("../raw/raw_tables/comp_funda.parquet")
    .select(["gvkey", "datadate", col("sich").alias("sic"), col("naicsh").alias("naics")])
    .unique()
)
sic_naics_na.collect().write_parquet("raw_data_dfs/sic_naics_na.parquet")
sic_naics_gl = (
    pl.scan_parquet("../raw/raw_tables/comp_g_funda.parquet")
    .select(["gvkey", "datadate", col("sich").alias("sic"), col("naicsh").alias("naics")])
    .unique()
)
sic_naics_gl.collect().write_parquet("raw_data_dfs/sic_naics_gl.parquet")
permno0 = (
    pl.scan_parquet("../raw/raw_tables/crsp_stksecurityinfohist.parquet")
    .select(
        [
            col("permno").cast(pl.Int64),
            col("permco").cast(pl.Int64),
            "secinfostartdt",
            "secinfoenddt",
            col("siccd").cast(pl.Int64).alias("sic"),
            col("naics").cast(pl.Int64),
        ]
    )
    .unique()
    .sort(["permno", "secinfostartdt", "secinfoenddt"])
)
permno0.collect().write_parquet("raw_data_dfs/permno0.parquet")
comp_hgics_na = (
    pl.scan_parquet("../raw/raw_tables/comp_co_hgic.parquet")
    .filter(col("gvkey").is_not_null())
    .select(["gvkey", "indfrom", "indthru", col("gsubind").alias("gics")])
    .unique()
)
comp_hgics_na.collect().write_parquet("raw_data_dfs/comp_hgics_na.parquet")
comp_hgics_gl = (
    pl.scan_parquet("../raw/raw_tables/comp_g_co_hgic.parquet")
    .filter(col("gvkey").is_not_null())
    .select(["gvkey", "indfrom", "indthru", col("gsubind").alias("gics")])
    .unique()
)
comp_hgics_gl.collect().write_parquet("raw_data_dfs/comp_hgics_gl.parquet")
crsp_dsedelist = pl.scan_parquet("../raw/raw_tables/crsp_stkdelists.parquet").select(
    [
        "delret",
        "delactiontype",
        "delstatustype",
        "delreasontype",
        "delpaymenttype",
        col("permno").cast(pl.Int64),
        "delistingdt",
    ]
)
crsp_dsedelist.collect().write_parquet("raw_data_dfs/crsp_dsedelist.parquet")
crsp_msedelist = pl.scan_parquet("../raw/raw_tables/crsp_stkdelists.parquet").select(
    [
        "delret",
        "delactiontype",
        "delstatustype",
        "delreasontype",
        "delpaymenttype",
        col("permno").cast(pl.Int64),
        "delistingdt",
    ]
)
crsp_msedelist.collect().write_parquet("raw_data_dfs/crsp_msedelist.parquet")
# NOTE: Variables prefixed with '__' are temporary intermediate tables kept local to this stage.
__sec_info = pl.concat(
    [
        pl.scan_parquet("../raw/raw_tables/comp_security.parquet").select(["gvkey", "iid", "secstat", "dlrsni"]),
        pl.scan_parquet("../raw/raw_tables/comp_g_security.parquet").select(["gvkey", "iid", "secstat", "dlrsni"]),
    ]
)
__sec_info.collect().write_parquet("raw_data_dfs/__sec_info.parquet")
crsp_mcti_t30ret = pl.scan_parquet("../raw/raw_tables/crsp_mcti.parquet").select(
    ["caldt", "t30ret"]
)
crsp_mcti_t30ret.collect().write_parquet("raw_data_dfs/crsp_mcti_t30ret.parquet")
ff_factors_monthly = pl.scan_parquet("../raw/raw_tables/ff_factors_monthly.parquet").select(
    ["date", "rf"]
)
ff_factors_monthly.collect().write_parquet("raw_data_dfs/ff_factors_monthly.parquet")
comp_r_ex_codes = pl.scan_parquet("../raw/raw_tables/comp_r_ex_codes.parquet").select(
    ["exchgdesc", "exchgcd"]
)
comp_r_ex_codes.collect().write_parquet("raw_data_dfs/comp_r_ex_codes.parquet")
__ex_country1 = pl.concat(
    [
        pl.scan_parquet("../raw/raw_tables/comp_g_security.parquet").select(["exchg", "excntry"]).unique(),
        pl.scan_parquet("../raw/raw_tables/comp_security.parquet").select(["exchg", "excntry"]).unique(),
    ]
)
__ex_country1.collect().write_parquet("raw_data_dfs/__ex_country1.parquet")
con = ibis.duckdb.connect(threads=os.cpu_count())
comp = con.read_parquet("../raw/raw_tables/comp_company.parquet").select(["gvkey", "prirow", "priusa", "prican"])
g_comp = con.read_parquet("../raw/raw_tables/comp_g_company.parquet").select(["gvkey", "prirow", "priusa", "prican"])
comp.union(g_comp).distinct(on="gvkey", keep="first").to_parquet("raw_data_dfs/__header.parquet")
con.disconnect()
gen_prihist_files()
gen_fx1()
gen_crsp_sf("m").to_parquet("raw_data_dfs/__crsp_sf_m.parquet")
gen_crsp_sf("d").to_parquet("raw_data_dfs/__crsp_sf_d.parquet")


########## AUX_FUNCTIONS.PREPARE_COMP_SF('BOTH') ##########
# Inlined from aux_functions.prepare_comp_sf('both')
freq = 'both'
"""
Description:
    Prepare Compustat security-file derivatives (Comp DSF/SSF equivalents) for daily/monthly runs.

Steps:
    1) Ensure firm-shares table is populated (populate_own), then create Comp DSF (gen_comp_dsf).
    2) Run process_comp_sf1 for requested frequency: 'd', 'm', or 'both'.

Output:
    Intermediate Comp security files written by downstream helpers (no direct return).
"""
populate_own("raw_data_dfs/__firm_shares1.parquet", "gvkey", "datadate", "ddate")
gen_comp_dsf()
if freq == "both":
    process_comp_sf1("d")
    process_comp_sf1("m")
else:
    process_comp_sf1(freq)


########## AUX_FUNCTIONS.PREPARE_CRSP_SF('M') ##########
# Inlined from aux_functions.prepare_crsp_sf('m')
freq = 'm'
"""
Description:
    Clean and finalize the CRSP security-file panel (monthly or daily) produced by gen_crsp_sf.
    This step adds trading-volume diagnostics, dividend totals, delisting-return adjustments,
    excess returns (over T-bill / RF), and company-level market equity, using the CIZ delist
    fields (DelReasonType/DelActionType/DelPaymentType/DelStatusType).

Steps:
    1) Read raw_data_dfs/__crsp_sf_{freq}.parquet; cast key numeric columns; apply NASDAQ volume adjustment.
    2) Compute dollar volume and infer dividend totals from (ret − retx) scaled by lagged price and split factors.
    3) Join CRSP delists (crsp_{freq}sedelist); impute missing delret = −0.30 for “bad delist” buckets defined by CIZ codes;
       set ret=0 when ret is missing but delret exists; compound ret with delret.
    4) Join risk-free proxies (CRSP T-bill and FF RF) and compute excess return ret_exc; compute company ME by summing ME across permnos within permco-date.
    5) If monthly, rescale vol and dolvol for unit alignment.
    6) Drop helper columns, deduplicate by (permno, date), sort, and write crsp_{freq}sf.parquet.

Output:
    Writes crsp_msf.parquet (freq="m") or crsp_dsf.parquet (freq="d") with cleaned returns and ret_exc.
"""
assert freq in ("m", "d")

merge_vars = ["permno", "merge_aux"] if (freq == "m") else ["permno", "date"]

# NOTE: __crsp_sf is the working CRSP security-file frame for the selected frequency.
__crsp_sf = (
    pl.scan_parquet(f"raw_data_dfs/__crsp_sf_{freq}.parquet")
    .with_columns(
        [
            col(var).cast(pl.Float64)
            for var in ["prc", "cfacshr", "ret", "retx", "prc_high", "prc_low"]
        ]
        + [col("vol").cast(pl.Int64)]
    )
    .with_columns(adj_trd_vol_NASDAQ("date", "vol", "exchcd", 3))
    .sort(["permno", "date"])
    .with_columns(
        dolvol=col("prc").abs() * col("vol"),
        div_tot=pl.when(col("cfacshr").shift(1).over("permno") != 0)
        .then(
            (
                (col("ret") - col("retx"))
                * col("prc").shift(1)
                * (col("cfacshr") / col("cfacshr").shift(1))
            ).over("permno")
        )
        .otherwise(fl_none()),
        permno=col("permno").cast(pl.Int64),
        merge_aux=gen_MMYY_column("date"),
    )
)

crsp_sedelist_aux_col = (
    [gen_MMYY_column("delistingdt").alias("merge_aux")]
    if (freq == "m")
    else [col("delistingdt").alias("date")]
)

crsp_sedelist = pl.scan_parquet(f"raw_data_dfs/crsp_{freq}sedelist.parquet").with_columns(
    crsp_sedelist_aux_col
)

crsp_mcti = add_MMYY_column_drop_original(
    pl.scan_parquet("raw_data_dfs/crsp_mcti_t30ret.parquet"), "caldt"
)
ff_factors_monthly = add_MMYY_column_drop_original(
    pl.scan_parquet("raw_data_dfs/ff_factors_monthly.parquet"), "date"
)

c1 = col("delret").is_null()

# CIZ replacement for legacy "dlstcd == 500"
c2 = (
    (col("delreasontype") == "UNAV")
    & (col("delactiontype") == "GDR")
    & (col("delpaymenttype") == "PRCF")
    & (col("delstatustype") == "VCL")
)

# CIZ replacement for legacy "dlstcd between 520 and 584"
c3 = (
    (col("delactiontype") == "GDR")
    & (col("delpaymenttype") == "PRCF")
    & (col("delstatustype") == "VCL")
    & col("delreasontype").is_in(
        [
            "MVOT",  # Move to OTC
            "MTMK",  # Market Makers
            "SHLD",  # Shareholders
            "LP",  # Low Price
            "INSC",  # Insufficient Capital
            "INSF",  # Insufficient Float
            "CORQ",  # Company Request
            "DERE",  # Deregistration
            "BKPY",  # Bankruptcy
            "OFFRE",  # Offer Rescinded
            "DELQ",  # Delinquent
            "FARG",  # Failure to Register
            "EQRQ",  # Equity Requirements
            "DEEX",  # Denied Exception
            "FING",  # Financial Guidelines
        ]
    )
)

c4 = c1 & (c2 | c3)

c5 = col("ret").is_null()
c6 = col("delret").is_not_null()
c7 = c5 & c6

me_company_exp = (
    pl.when(pl.count("me").over(["permco", "date"]) != 0)
    .then(pl.coalesce(["me", 0.0]).sum().over(["permco", "date"]))
    .otherwise(fl_none())
)

scale = 1 if (freq == "m") else 21
ret_exc_exp = col("ret") - pl.coalesce(["t30ret", "rf"]) / scale

__crsp_sf = (
    __crsp_sf.join(crsp_sedelist, how="left", on=merge_vars)
    # impute missing delret to -0.30 for the “bad delist” buckets
    .with_columns(delret=pl.when(c4).then(pl.lit(-0.3)).otherwise(col("delret")))
    # if ret missing but delret exists, set ret=0 so compounding works
    .with_columns(ret=pl.when(c7).then(pl.lit(0.0)).otherwise(col("ret")))
    # compound ret with delret
    .with_columns(ret=((col("ret") + 1) * (pl.coalesce(["delret", 0.0]) + 1) - 1))
    # rf joins
    .join(crsp_mcti, how="left", on="merge_aux")
    .join(ff_factors_monthly, how="left", on="merge_aux")
    .with_columns(ret_exc=ret_exc_exp, me_company=me_company_exp)
)

if freq == "m":
    __crsp_sf = __crsp_sf.with_columns(
        [(col(var) * 100).alias(var) for var in ["vol", "dolvol"]]
    )

__crsp_sf = (
    __crsp_sf.drop(
        [
            "rf",
            "t30ret",
            "merge_aux",
            "delret",
            "delistingdt",
            "delreasontype",
            "delactiontype",
            "delpaymenttype",
            "delstatustype",
        ]
    )
    .unique(["permno", "date"])
    .sort(["permno", "date"])
)

__crsp_sf.collect().write_parquet(f"crsp_{freq}sf.parquet")


########## AUX_FUNCTIONS.PREPARE_CRSP_SF('D') ##########
# Inlined from aux_functions.prepare_crsp_sf('d')
freq = 'd'
"""
Description:
    Clean and finalize the CRSP security-file panel (monthly or daily) produced by gen_crsp_sf.
    This step adds trading-volume diagnostics, dividend totals, delisting-return adjustments,
    excess returns (over T-bill / RF), and company-level market equity, using the CIZ delist
    fields (DelReasonType/DelActionType/DelPaymentType/DelStatusType).

Steps:
    1) Read raw_data_dfs/__crsp_sf_{freq}.parquet; cast key numeric columns; apply NASDAQ volume adjustment.
    2) Compute dollar volume and infer dividend totals from (ret − retx) scaled by lagged price and split factors.
    3) Join CRSP delists (crsp_{freq}sedelist); impute missing delret = −0.30 for “bad delist” buckets defined by CIZ codes;
       set ret=0 when ret is missing but delret exists; compound ret with delret.
    4) Join risk-free proxies (CRSP T-bill and FF RF) and compute excess return ret_exc; compute company ME by summing ME across permnos within permco-date.
    5) If monthly, rescale vol and dolvol for unit alignment.
    6) Drop helper columns, deduplicate by (permno, date), sort, and write crsp_{freq}sf.parquet.

Output:
    Writes crsp_msf.parquet (freq="m") or crsp_dsf.parquet (freq="d") with cleaned returns and ret_exc.
"""
assert freq in ("m", "d")

merge_vars = ["permno", "merge_aux"] if (freq == "m") else ["permno", "date"]

__crsp_sf = (
    pl.scan_parquet(f"raw_data_dfs/__crsp_sf_{freq}.parquet")
    .with_columns(
        [
            col(var).cast(pl.Float64)
            for var in ["prc", "cfacshr", "ret", "retx", "prc_high", "prc_low"]
        ]
        + [col("vol").cast(pl.Int64)]
    )
    .with_columns(adj_trd_vol_NASDAQ("date", "vol", "exchcd", 3))
    .sort(["permno", "date"])
    .with_columns(
        dolvol=col("prc").abs() * col("vol"),
        div_tot=pl.when(col("cfacshr").shift(1).over("permno") != 0)
        .then(
            (
                (col("ret") - col("retx"))
                * col("prc").shift(1)
                * (col("cfacshr") / col("cfacshr").shift(1))
            ).over("permno")
        )
        .otherwise(fl_none()),
        permno=col("permno").cast(pl.Int64),
        merge_aux=gen_MMYY_column("date"),
    )
)

crsp_sedelist_aux_col = (
    [gen_MMYY_column("delistingdt").alias("merge_aux")]
    if (freq == "m")
    else [col("delistingdt").alias("date")]
)

crsp_sedelist = pl.scan_parquet(f"raw_data_dfs/crsp_{freq}sedelist.parquet").with_columns(
    crsp_sedelist_aux_col
)

crsp_mcti = add_MMYY_column_drop_original(
    pl.scan_parquet("raw_data_dfs/crsp_mcti_t30ret.parquet"), "caldt"
)
ff_factors_monthly = add_MMYY_column_drop_original(
    pl.scan_parquet("raw_data_dfs/ff_factors_monthly.parquet"), "date"
)

c1 = col("delret").is_null()

# CIZ replacement for legacy "dlstcd == 500"
c2 = (
    (col("delreasontype") == "UNAV")
    & (col("delactiontype") == "GDR")
    & (col("delpaymenttype") == "PRCF")
    & (col("delstatustype") == "VCL")
)

# CIZ replacement for legacy "dlstcd between 520 and 584"
c3 = (
    (col("delactiontype") == "GDR")
    & (col("delpaymenttype") == "PRCF")
    & (col("delstatustype") == "VCL")
    & col("delreasontype").is_in(
        [
            "MVOT",  # Move to OTC
            "MTMK",  # Market Makers
            "SHLD",  # Shareholders
            "LP",  # Low Price
            "INSC",  # Insufficient Capital
            "INSF",  # Insufficient Float
            "CORQ",  # Company Request
            "DERE",  # Deregistration
            "BKPY",  # Bankruptcy
            "OFFRE",  # Offer Rescinded
            "DELQ",  # Delinquent
            "FARG",  # Failure to Register
            "EQRQ",  # Equity Requirements
            "DEEX",  # Denied Exception
            "FING",  # Financial Guidelines
        ]
    )
)

c4 = c1 & (c2 | c3)

c5 = col("ret").is_null()
c6 = col("delret").is_not_null()
c7 = c5 & c6

me_company_exp = (
    pl.when(pl.count("me").over(["permco", "date"]) != 0)
    .then(pl.coalesce(["me", 0.0]).sum().over(["permco", "date"]))
    .otherwise(fl_none())
)

scale = 1 if (freq == "m") else 21
ret_exc_exp = col("ret") - pl.coalesce(["t30ret", "rf"]) / scale

__crsp_sf = (
    __crsp_sf.join(crsp_sedelist, how="left", on=merge_vars)
    # impute missing delret to -0.30 for the “bad delist” buckets
    .with_columns(delret=pl.when(c4).then(pl.lit(-0.3)).otherwise(col("delret")))
    # if ret missing but delret exists, set ret=0 so compounding works
    .with_columns(ret=pl.when(c7).then(pl.lit(0.0)).otherwise(col("ret")))
    # compound ret with delret
    .with_columns(ret=((col("ret") + 1) * (pl.coalesce(["delret", 0.0]) + 1) - 1))
    # rf joins
    .join(crsp_mcti, how="left", on="merge_aux")
    .join(ff_factors_monthly, how="left", on="merge_aux")
    .with_columns(ret_exc=ret_exc_exp, me_company=me_company_exp)
)

if freq == "m":
    __crsp_sf = __crsp_sf.with_columns(
        [(col(var) * 100).alias(var) for var in ["vol", "dolvol"]]
    )

__crsp_sf = (
    __crsp_sf.drop(
        [
            "rf",
            "t30ret",
            "merge_aux",
            "delret",
            "delistingdt",
            "delreasontype",
            "delactiontype",
            "delpaymenttype",
            "delstatustype",
        ]
    )
    .unique(["permno", "date"])
    .sort(["permno", "date"])
)

__crsp_sf.collect().write_parquet(f"crsp_{freq}sf.parquet")


########## AUX_FUNCTIONS.COMBINE_CRSP_COMP_SF() ##########
# Inlined from aux_functions.combine_crsp_comp_sf()
"""
Description:
    Create unified monthly and daily security datasets by combining CRSP and Compustat,
    determining the main observation per id/eom, and writing outputs.

Steps:
    1) Prepare normalized CRSP and Compustat frames (monthly & daily).
    2) Build __msf_world and __dsf_world via gen_temp_sf.
    3) From __msf_world derive obs_main: if only one obs or multiple but source_crsp==1 → 1 else 0.
    4) Write final monthly and daily world files, injecting obs_main where applicable.

Output:
    '__msf_world.parquet' and 'world_dsf.parquet' ready for downstream processing.
"""
crsp_msf, crsp_dsf = prepare_crsp_sfs_for_merging()
comp_msf, comp_dsf = prepare_comp_sfs_for_merging()
__msf_world = gen_temp_sf("m", crsp_msf, comp_msf)
__dsf_world = gen_temp_sf("d", crsp_dsf, comp_dsf)
obs_main = (
    __msf_world.select(["id", "source_crsp", "gvkey", "iid", "eom"])
    .with_columns(count=pl.count("gvkey").over(["gvkey", "iid", "eom"]))
    .with_columns(
        obs_main=pl.when(
            (col("count").is_in([0, 1])) | ((col("count") > 1) & (col("source_crsp") == 1))
        )
        .then(1)
        .otherwise(0)
    )
    .drop(["count", "iid", "gvkey", "source_crsp"])
)
add_obs_main_to_sf_and_write_file("m", __msf_world, obs_main)
add_obs_main_to_sf_and_write_file("d", __dsf_world, obs_main)


########## AUX_FUNCTIONS.CRSP_INDUSTRY() ##########
# Inlined from aux_functions.crsp_industry()
"""
Description:
    Generate a daily panel of CRSP SIC/NAICS codes per permno based on name-date spans.

Steps:
    1) Read permno0; nullify sic==0; build date ranges from secinfostartdt to secinfoenddt.
    2) Explode to daily rows; keep distinct (permno,date); sort.
    3) Write to crsp_ind.parquet.

Output:
    Parquet crsp_ind.parquet with {permno,permco,date,sic,naics}.
"""
permno0 = pl.scan_parquet("raw_data_dfs/permno0.parquet")
permno0 = (
    permno0.with_columns(
        sic=pl.when(col("sic") == 0).then(pl.lit(None).cast(pl.Int64)).otherwise(col("sic"))
    )
    .with_columns(date=pl.date_ranges("secinfostartdt", "secinfoenddt"))
    .explode("date")
    .select(["permno", "permco", "date", "sic", "naics"])
    .unique(["permno", "date"])
    .sort(["permno", "date"])
)
permno0.collect().write_parquet("crsp_ind.parquet")


########## AUX_FUNCTIONS.COMP_INDUSTRY() ##########
# Inlined from aux_functions.comp_industry()
"""
Description:
    Merge daily GICS and SIC/NAICS into a single daily Compustat industry file,
    filling gaps day-by-day to ensure continuity.

Steps:
    1) Run comp_sic_naics() and hgics_join(); load into DuckDB.
    2) Full-outer-join on (gvkey,date); compute aux_date = next date − 1 day to detect gaps.
    3) Build gap ranges via generate_series and fill from gap_dates; union with continuous rows.
    4) Select distinct first by (gvkey,date); write comp_ind.parquet.

Output:
    Parquet comp_ind.parquet with {gvkey,date,gics,sic,naics} daily.
"""
comp_sic_naics()
hgics_join()
os.system("rm -f aux_comp_ind.ddb")
con = ibis.duckdb.connect("aux_comp_ind.ddb", threads=os.cpu_count())
con.create_table("comp_other", con.read_parquet("comp_other.parquet"))
con.create_table("comp_gics", con.read_parquet("comp_hgics.parquet"))
con.raw_sql("""
            DROP TABLE IF EXISTS join_table;
            CREATE TABLE join_table AS
            SELECT          *,
                            COALESCE( LEAD(date) OVER (PARTITION BY gvkey ORDER BY date) - INTERVAL '1 day', date )::DATE AS aux_date
            FROM            comp_gics
            FULL OUTER JOIN comp_other
            USING           (gvkey, date);

            DROP TABLE IF EXISTS gap_dates;
            CREATE TABLE gap_dates AS
            SELECT *
            FROM join_table
            WHERE date <> aux_date;

            DROP TABLE IF EXISTS gaps;
            CREATE TABLE gaps AS
            WITH full_span AS (
            SELECT
                j.gvkey, gs.gap_date::DATE AS date,
                FROM gap_dates as j
                CROSS JOIN LATERAL
                generate_series(j.date, j.aux_date, INTERVAL '1 day') AS gs(gap_date)
                ORDER BY gvkey, date
            )
            SELECT
            fs.gvkey, fs.date, gd.gics, gd.sic, gd.naics
            FROM full_span fs
            LEFT JOIN gap_dates gd
            ON gd.gvkey = fs.gvkey
            AND gd.date  = fs.date
            ORDER BY fs.gvkey, fs.date;

            DROP TABLE IF EXISTS continuous;
            CREATE TABLE continuous AS
            SELECT *
            FROM join_table
            WHERE date = aux_date;

            DROP TABLE IF EXISTS merged_data;
            CREATE TABLE merged_data AS
            SELECT gvkey, date, gics, sic, naics FROM continuous
            UNION
            SELECT gvkey, date, gics, sic, naics FROM gaps;

            DROP TABLE IF EXISTS comp_industry;
            CREATE TABLE comp_industry AS
            SELECT DISTINCT ON (gvkey, date)
                *
            FROM merged_data
            ORDER BY (gvkey, date);
""")
con.table("comp_industry").to_parquet("comp_ind.parquet")
con.disconnect()


########## AUX_FUNCTIONS.MERGE_INDUSTRY_TO_WORLD_MSF() ##########
# Inlined from aux_functions.merge_industry_to_world_msf()
"""
Description:
    Merge industry codes into world MSF dataset.

Steps:
    1) Load __msf_world, comp_ind, and crsp_ind datasets.
    2) Join compustat and CRSP industry codes on matching keys.
    3) Coalesce SIC/NAICS from both sources.
    4) Drop redundant columns.

Output:
    '__msf_world2.parquet' with industry codes appended.
"""
__msf_world = pl.scan_parquet("__msf_world.parquet")
comp_ind = pl.scan_parquet("comp_ind.parquet")
crsp_ind = pl.scan_parquet("crsp_ind.parquet").rename(
    {"sic": "sic_crsp", "naics": "naics_crsp"}
)
__msf_world = (
    __msf_world.join(comp_ind, how="left", left_on=["gvkey", "eom"], right_on=["gvkey", "date"])
    .join(
        crsp_ind,
        how="left",
        left_on=["permco", "permno", "eom"],
        right_on=["permco", "permno", "date"],
    )
    .with_columns(
        sic=pl.coalesce(["sic", "sic_crsp"]),
        naics=pl.coalesce(["naics", "naics_crsp"]),
    )
    .drop(["sic_crsp", "naics_crsp"])
)
__msf_world.collect().write_parquet("__msf_world2.parquet")


########## AUX_FUNCTIONS.FF_IND_CLASS('__MSF_WORLD2.PARQUET', 49) ##########
# Inlined from aux_functions.ff_ind_class('__msf_world2.parquet', 49)
data_path = '__msf_world2.parquet'
ff_grps = 49
"""
Description:
    Assign Fama–French industry classifications (38-group or 49-group) based on SIC codes.

Steps:
    1) If ff_grps==38: define lower/upper SIC bounds; iterate to map to ff38 groups (2..N),
    with special rule for (100–999) → 1; set null when no match.
    2) Else: build ff49 via explicit SIC enumerations/ranges per FF (Ken French) taxonomy.
    3) Attach the classification column to input data and write __msf_world3.parquet.

Output:
    Parquet __msf_world3.parquet with an added ff38 or ff49 integer column.
"""
data = pl.scan_parquet(data_path)
if ff_grps == 38:
    lower_bounds = [
        1000,
        1300,
        1400,
        1500,
        2000,
        2100,
        2200,
        2300,
        2400,
        2500,
        2600,
        2700,
        2800,
        2900,
        3000,
        3100,
        3200,
        3300,
        3400,
        3500,
        3600,
        3700,
        3800,
        3900,
        4000,
        4800,
        4830,
        4900,
        4950,
        4960,
        4970,
        5000,
        5200,
        6000,
        7000,
        9000,
    ]
    upper_bounds = [
        1299,
        1399,
        1499,
        1799,
        2099,
        2199,
        2299,
        2399,
        2499,
        2599,
        2661,
        2799,
        2899,
        2999,
        3099,
        3199,
        3299,
        3399,
        3499,
        3599,
        3699,
        3799,
        3879,
        3999,
        4799,
        4829,
        4899,
        4949,
        4959,
        4969,
        4979,
        5199,
        5999,
        6999,
        8999,
        9999,
    ]
    classification = [i + 2 for i in range(len(lower_bounds))]
    ff38_exp = pl.when(col("sic").is_between(100, 999)).then(1)
    for i, j, k in zip(lower_bounds, upper_bounds, classification, strict=True):
        ff38_exp = ff38_exp.when(col("sic").is_between(i, j)).then(k)
    ff38_exp = (ff38_exp.otherwise(pl.lit(None))).alias("ff38")
    data = data.with_columns(ff38_exp)
else:
    data = data.with_columns(
        pl.when(
            col("sic").is_in(
                [
                    2048,
                    *range(100, 299 + 1),
                    *range(700, 799 + 1),
                    *range(910, 919 + 1),
                ]
            )
        )
        .then(1)
        .when(
            col("sic").is_in(
                [
                    2095,
                    2098,
                    2099,
                    *range(2000, 2046 + 1),
                    *range(2050, 2063 + 1),
                    *range(2070, 2079 + 1),
                    *range(2090, 2092 + 1),
                ]
            )
        )
        .then(2)
        .when(col("sic").is_in([2086, 2087, 2096, 2097, *range(2064, 2068 + 1)]))
        .then(3)
        .when(col("sic").is_in([2080, *range(2082, 2085 + 1)]))
        .then(4)
        .when(col("sic").is_in([*range(2100, 2199 + 1)]))
        .then(5)
        .when(
            col("sic").is_in(
                [
                    3732,
                    3930,
                    3931,
                    *range(920, 999 + 1),
                    *range(3650, 3652 + 1),
                    *range(3940, 3949 + 1),
                ]
            )
        )
        .then(6)
        .when(
            col("sic").is_in(
                [
                    7840,
                    7841,
                    7900,
                    7910,
                    7911,
                    7980,
                    *range(7800, 7833 + 1),
                    *range(7920, 7933 + 1),
                    *range(7940, 7949 + 1),
                    *range(7990, 7999 + 1),
                ]
            )
        )
        .then(7)
        .when(col("sic").is_in([2770, 2771, *range(2700, 2749 + 1), *range(2780, 2799 + 1)]))
        .then(8)
        .when(
            col("sic").is_in(
                [
                    2047,
                    2391,
                    2392,
                    3160,
                    3161,
                    3229,
                    3260,
                    3262,
                    3263,
                    3269,
                    3230,
                    3231,
                    3750,
                    3751,
                    3800,
                    3860,
                    3861,
                    3910,
                    3911,
                    3914,
                    3915,
                    3991,
                    3995,
                    *range(2510, 2519 + 1),
                    *range(2590, 2599 + 1),
                    *range(2840, 2844 + 1),
                    *range(3170, 3172 + 1),
                    *range(3190, 3199 + 1),
                    *range(3630, 3639 + 1),
                    *range(3870, 3873 + 1),
                    *range(3960, 3962 + 1),
                ]
            )
        )
        .then(9)
        .when(
            col("sic").is_in(
                [
                    3020,
                    3021,
                    3130,
                    3131,
                    3150,
                    3151,
                    *range(2300, 2390 + 1),
                    *range(3100, 3111 + 1),
                    *range(3140, 3149 + 1),
                    *range(3963, 3965 + 1),
                ]
            )
        )
        .then(10)
        .when(col("sic").is_in([*range(8000, 8099 + 1)]))
        .then(11)
        .when(col("sic").is_in([3693, 3850, 3851, *range(3840, 3849 + 1)]))
        .then(12)
        .when(col("sic").is_in([2830, 2831, *range(2833, 2836 + 1)]))
        .then(13)
        .when(
            col("sic").is_in(
                [
                    *range(2800, 2829 + 1),
                    *range(2850, 2879 + 1),
                    *range(2890, 2899 + 1),
                ]
            )
        )
        .then(14)
        .when(col("sic").is_in([3031, 3041, *range(3050, 3053 + 1), *range(3060, 3099 + 1)]))
        .then(15)
        .when(
            col("sic").is_in(
                [
                    *range(2200, 2284 + 1),
                    *range(2290, 2295 + 1),
                    *range(2297, 2299 + 1),
                    *range(2393, 2395 + 1),
                    *range(2397, 2399 + 1),
                ]
            )
        )
        .then(16)
        .when(
            col("sic").is_in(
                [
                    2660,
                    2661,
                    3200,
                    3210,
                    3211,
                    3240,
                    3241,
                    3261,
                    3264,
                    3280,
                    3281,
                    3446,
                    3996,
                    *range(800, 899 + 1),
                    *range(2400, 2439 + 1),
                    *range(2450, 2459 + 1),
                    *range(2490, 2499 + 1),
                    *range(2950, 2952 + 1),
                    *range(3250, 3259 + 1),
                    *range(3270, 3275 + 1),
                    *range(3290, 3293 + 1),
                    *range(3295, 3299 + 1),
                    *range(3420, 3429 + 1),
                    *range(3430, 3433 + 1),
                    *range(3440, 3442 + 1),
                    *range(3448, 3452 + 1),
                    *range(3490, 3499 + 1),
                ]
            )
        )
        .then(17)
        .when(
            col("sic").is_in(
                [
                    *range(1500, 1511 + 1),
                    *range(1520, 1549 + 1),
                    *range(1600, 1799 + 1),
                ]
            )
        )
        .then(18)
        .when(
            col("sic").is_in(
                [
                    3300,
                    *range(3310, 3317 + 1),
                    *range(3320, 3325 + 1),
                    *range(3330, 3341 + 1),
                    *range(3350, 3357 + 1),
                    *range(3360, 3379 + 1),
                    *range(3390, 3399 + 1),
                ]
            )
        )
        .then(19)
        .when(col("sic").is_in([3400, 3443, 3444, *range(3460, 3479 + 1)]))
        .then(20)
        .when(
            col("sic").is_in(
                [
                    3538,
                    3585,
                    3586,
                    *range(3510, 3536 + 1),
                    *range(3540, 3569 + 1),
                    *range(3580, 3582 + 1),
                    *range(3589, 3599 + 1),
                ]
            )
        )
        .then(21)
        .when(
            col("sic").is_in(
                [
                    3600,
                    3620,
                    3621,
                    3648,
                    3649,
                    3660,
                    3699,
                    *range(3610, 3613 + 1),
                    *range(3623, 3629 + 1),
                    *range(3640, 3646 + 1),
                    *range(3690, 3692 + 1),
                ]
            )
        )
        .then(22)
        .when(
            col("sic").is_in(
                [
                    2296,
                    2396,
                    3010,
                    3011,
                    3537,
                    3647,
                    3694,
                    3700,
                    3710,
                    3711,
                    3799,
                    *range(3713, 3716 + 1),
                    *range(3790, 3792 + 1),
                ]
            )
        )
        .then(23)
        .when(col("sic").is_in([3720, 3721, 3728, 3729, *range(3723, 3725 + 1)]))
        .then(24)
        .when(col("sic").is_in([3730, 3731, *range(3740, 3743 + 1)]))
        .then(25)
        .when(col("sic").is_in([3795, *range(3760, 3769 + 1), *range(3480, 3489 + 1)]))
        .then(26)
        .when(col("sic").is_in([*range(1040, 1049 + 1)]))
        .then(27)
        .when(
            col("sic").is_in(
                [
                    *range(1000, 1039 + 1),
                    *range(1050, 1119 + 1),
                    *range(1400, 1499 + 1),
                ]
            )
        )
        .then(28)
        .when(col("sic").is_in([*range(1200, 1299 + 1)]))
        .then(29)
        .when(
            col("sic").is_in(
                [
                    1300,
                    1389,
                    *range(1310, 1339 + 1),
                    *range(1370, 1382 + 1),
                    *range(2900, 2912 + 1),
                    *range(2990, 2999 + 1),
                ]
            )
        )
        .then(30)
        .when(
            col("sic").is_in(
                [
                    4900,
                    4910,
                    4911,
                    4939,
                    *range(4920, 4925 + 1),
                    *range(4930, 4932 + 1),
                    *range(4940, 4942 + 1),
                ]
            )
        )
        .then(31)
        .when(
            col("sic").is_in(
                [
                    4800,
                    4899,
                    *range(4810, 4813 + 1),
                    *range(4820, 4822 + 1),
                    *range(4830, 4841 + 1),
                    *range(4880, 4892 + 1),
                ]
            )
        )
        .then(32)
        .when(
            col("sic").is_in(
                [
                    7020,
                    7021,
                    7200,
                    7230,
                    7231,
                    7240,
                    7241,
                    7250,
                    7251,
                    7395,
                    7500,
                    7600,
                    7620,
                    7622,
                    7623,
                    7640,
                    7641,
                    *range(7030, 7033 + 1),
                    *range(7210, 7212 + 1),
                    *range(7214, 7217 + 1),
                    *range(7219, 7221 + 1),
                    *range(7260, 7299 + 1),
                    *range(7520, 7549 + 1),
                    *range(7629, 7631 + 1),
                    *range(7690, 7699 + 1),
                    *range(8100, 8499 + 1),
                    *range(8600, 8699 + 1),
                    *range(8800, 8899 + 1),
                    *range(7510, 7515 + 1),
                ]
            )
        )
        .then(33)
        .when(
            col("sic").is_in(
                [
                    3993,
                    7218,
                    7300,
                    7374,
                    7396,
                    7397,
                    7399,
                    7519,
                    8700,
                    8720,
                    8721,
                    *range(2750, 2759 + 1),
                    *range(7310, 7342 + 1),
                    *range(7349, 7353 + 1),
                    *range(7359, 7369 + 1),
                    *range(7376, 7385 + 1),
                    *range(7389, 7394 + 1),
                    *range(8710, 8713 + 1),
                    *range(8730, 8734 + 1),
                    *range(8740, 8748 + 1),
                    *range(8900, 8911 + 1),
                    *range(8920, 8999 + 1),
                    *range(4220, 4229 + 1),
                ]
            )
        )
        .then(34)
        .when(col("sic").is_in([3695, *range(3570, 3579 + 1), *range(3680, 3689 + 1)]))
        .then(35)
        .when(col("sic").is_in([7375, *range(7370, 7373 + 1)]))
        .then(36)
        .when(
            col("sic").is_in([3622, 3810, 3812, *range(3661, 3666 + 1), *range(3669, 3679 + 1)])
        )
        .then(37)
        .when(col("sic").is_in([3811, *range(3820, 3827 + 1), *range(3829, 3839 + 1)]))
        .then(38)
        .when(
            col("sic").is_in(
                [
                    2760,
                    2761,
                    *range(2520, 2549 + 1),
                    *range(2600, 2639 + 1),
                    *range(2670, 2699 + 1),
                    *range(3950, 3955 + 1),
                ]
            )
        )
        .then(39)
        .when(
            col("sic").is_in(
                [
                    3220,
                    3221,
                    *range(2440, 2449 + 1),
                    *range(2640, 2659 + 1),
                    *range(3410, 3412 + 1),
                ]
            )
        )
        .then(40)
        .when(
            col("sic").is_in(
                [
                    4100,
                    4130,
                    4131,
                    4150,
                    4151,
                    4230,
                    4231,
                    4780,
                    4789,
                    *range(4000, 4013 + 1),
                    *range(4040, 4049 + 1),
                    *range(4110, 4121 + 1),
                    *range(4140, 4142 + 1),
                    *range(4170, 4173 + 1),
                    *range(4190, 4200 + 1),
                    *range(4210, 4219 + 1),
                    *range(4240, 4249 + 1),
                    *range(4400, 4700 + 1),
                    *range(4710, 4712 + 1),
                    *range(4720, 4749 + 1),
                    *range(4782, 4785 + 1),
                ]
            )
        )
        .then(41)
        .when(
            col("sic").is_in(
                [
                    5000,
                    5099,
                    5100,
                    *range(5010, 5015 + 1),
                    *range(5020, 5023 + 1),
                    *range(5030, 5060 + 1),
                    *range(5063, 5065 + 1),
                    *range(5070, 5078 + 1),
                    *range(5080, 5088 + 1),
                    *range(5090, 5094 + 1),
                    *range(5110, 5113 + 1),
                    *range(5120, 5122 + 1),
                    *range(5130, 5172 + 1),
                    *range(5180, 5182 + 1),
                    *range(5190, 5199 + 1),
                ]
            )
        )
        .then(42)
        .when(
            col("sic").is_in(
                [
                    5200,
                    5250,
                    5251,
                    5260,
                    5261,
                    5270,
                    5271,
                    5300,
                    5310,
                    5311,
                    5320,
                    5330,
                    5331,
                    5334,
                    5900,
                    5999,
                    *range(5210, 5231 + 1),
                    *range(5340, 5349 + 1),
                    *range(5390, 5400 + 1),
                    *range(5410, 5412 + 1),
                    *range(5420, 5469 + 1),
                    *range(5490, 5500 + 1),
                    *range(5510, 5579 + 1),
                    *range(5590, 5700 + 1),
                    *range(5710, 5722 + 1),
                    *range(5730, 5736 + 1),
                    *range(5750, 5799 + 1),
                    *range(5910, 5912 + 1),
                    *range(5920, 5932 + 1),
                    *range(5940, 5990 + 1),
                    *range(5992, 5995 + 1),
                ]
            )
        )
        .then(43)
        .when(
            col("sic").is_in(
                [
                    7000,
                    7213,
                    *range(5800, 5829 + 1),
                    *range(5890, 5899 + 1),
                    *range(7010, 7019 + 1),
                    *range(7040, 7049 + 1),
                ]
            )
        )
        .then(44)
        .when(
            col("sic").is_in(
                [
                    6000,
                    *range(6010, 6036 + 1),
                    *range(6040, 6062 + 1),
                    *range(6080, 6082 + 1),
                    *range(6090, 6100 + 1),
                    *range(6110, 6113 + 1),
                    *range(6120, 6179 + 1),
                    *range(6190, 6199 + 1),
                ]
            )
        )
        .then(45)
        .when(
            col("sic").is_in(
                [
                    6300,
                    6350,
                    6351,
                    6360,
                    6361,
                    *range(6310, 6331 + 1),
                    *range(6370, 6379 + 1),
                    *range(6390, 6411 + 1),
                ]
            )
        )
        .then(46)
        .when(
            col("sic").is_in(
                [
                    6500,
                    6510,
                    6540,
                    6541,
                    6610,
                    6611,
                    *range(6512, 6515 + 1),
                    *range(6517, 6532 + 1),
                    *range(6550, 6553 + 1),
                    *range(6590, 6599 + 1),
                ]
            )
        )
        .then(47)
        .when(
            col("sic").is_in(
                [
                    6700,
                    6798,
                    6799,
                    *range(6200, 6299 + 1),
                    *range(6710, 6726 + 1),
                    *range(6730, 6733 + 1),
                    *range(6740, 6779 + 1),
                    *range(6790, 6795 + 1),
                ]
            )
        )
        .then(48)
        .when(col("sic").is_in([4970, 4971, 4990, 4991, *range(4950, 4961 + 1)]))
        .then(49)
        .otherwise(pl.lit(None))
        .alias("ff49")
    )

data.collect().write_parquet("__msf_world3.parquet")


########## AUX_FUNCTIONS.NYSE_SIZE_CUTOFFS('__MSF_WORLD3.PARQUET') ##########
# Inlined from aux_functions.nyse_size_cutoffs('__msf_world3.parquet')
data_path = '__msf_world3.parquet'
"""
Description:
    Compute NYSE market equity cutoffs (1%,20%,50%,80%) by month.

Steps:
    1) Load parquet lazily, filter to NYSE common stocks.
    2) Group by eom, count obs.
    3) Apply QUANTILE_DISC for cutoffs.
    4) Collect and save.

Output:
    'nyse_cutoffs.parquet' with [eom, n, nyse_p1, nyse_p20, nyse_p50, nyse_p80].
"""
nyse_sf = pl.scan_parquet(data_path).sql("""
        SELECT
            eom,
            COUNT(*)                    AS n,
            QUANTILE_DISC(me, 0.01)     AS nyse_p1,
            QUANTILE_DISC(me, 0.20)     AS nyse_p20,
            QUANTILE_DISC(me, 0.50)     AS nyse_p50,
            QUANTILE_DISC(me, 0.80)     AS nyse_p80
        FROM self
        WHERE  crsp_exchcd = 1
            AND obs_main   = 1
            AND exch_main  = 1
            AND primary_sec= 1
            AND common     = 1
            AND me IS NOT NULL
        GROUP BY eom
        ORDER BY eom
        """)
nyse_sf.sink_parquet("nyse_cutoffs.parquet")


########## AUX_FUNCTIONS.CLASSIFY_STOCKS_SIZE_GROUPS() ##########
# Inlined from aux_functions.classify_stocks_size_groups()
"""
Description:
    Join world MSF with NYSE size cutoffs and classify stocks into size buckets.

Steps:
    1) Read 'nyse_cutoffs.parquet' and '__msf_world3.parquet' lazily.
    2) Left-join on eom; compute size_grp via ME vs NYSE p1/p20/p50/p80 (fallback to 'mega' if cutoffs missing).
    3) Drop cutoff columns; collect and save.

Output:
    Writes 'world_msf.parquet' with size_grp per row.
"""
nyse_cutoffs = pl.scan_parquet("nyse_cutoffs.parquet")
__msf_world = pl.scan_parquet("__msf_world3.parquet")
world_msf = (
    __msf_world.join(nyse_cutoffs, how="left", on="eom")
    .with_columns(
        size_grp=pl.when(col("me").is_null())
        .then(None)
        .when(col("nyse_p80").is_null())
        .then(pl.lit("mega"))  # This is just to match SAS excactly
        .when(col("me") >= col("nyse_p80"))
        .then(pl.lit("mega"))
        .when(col("me") >= col("nyse_p50"))
        .then(pl.lit("large"))
        .when(col("me") >= col("nyse_p20"))
        .then(pl.lit("small"))
        .when(col("me") >= col("nyse_p1"))
        .then(pl.lit("micro"))
        .otherwise(pl.lit("nano"))
    )
    .drop([i for i in nyse_cutoffs.collect_schema().names() if i not in ["eom"]])
)
world_msf.collect().write_parquet("world_msf.parquet")


########## AUX_FUNCTIONS.RETURN_CUTOFFS('M', 0) ##########
# Inlined from aux_functions.return_cutoffs('m', 0)
freq = 'm'
crsp_only = 0
"""
Description:
    Compute return percentile cutoffs by period (monthly or daily), optionally CRSP-only.

Steps:
    1) Select group vars and output path from freq; scan 'world_{freq}sf.parquet'.
    2) Optional CRSP filter; require common/main/primary, exclude ZWE, non-null ret_exc.
    3) Add year/month; group and aggregate counts + percentiles for ret, ret_local, ret_exc.
    4) Sort, collect, save.

Output:
    Writes 'return_cutoffs.parquet' (monthly) or 'return_cutoffs_daily.parquet' (daily).
"""
group_vars = "eom" if freq == "m" else "year, month"
res_path = "return_cutoffs.parquet" if freq == "m" else "return_cutoffs_daily.parquet"
data = (
    pl.scan_parquet(f"world_{freq}sf.parquet")
    .filter(
        (col("common") == 1)
        & (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("primary_sec") == 1)
        & (col("excntry") != "ZWE")
        & (col("ret_exc").is_not_null())
        & ((col("source_crsp") == 1) if crsp_only == 1 else pl.lit(True))
    )
    .with_columns(year=col("date").dt.year(), month=col("date").dt.month())
)
data = data.sql(f"""
        SELECT
            {group_vars},
            COUNT(ret)                                AS n,

            -- ret percentiles
            QUANTILE_DISC(ret,        0.001)          AS ret_0_1,
            QUANTILE_DISC(ret,        0.01)           AS ret_1,
            QUANTILE_DISC(ret,        0.99)           AS ret_99,
            QUANTILE_DISC(ret,        0.999)          AS ret_99_9,

            -- ret_local percentiles
            QUANTILE_DISC(ret_local,  0.001)          AS ret_local_0_1,
            QUANTILE_DISC(ret_local,  0.01)           AS ret_local_1,
            QUANTILE_DISC(ret_local,  0.99)           AS ret_local_99,
            QUANTILE_DISC(ret_local,  0.999)          AS ret_local_99_9,

            -- ret_exc percentiles
            QUANTILE_DISC(ret_exc,    0.001)          AS ret_exc_0_1,
            QUANTILE_DISC(ret_exc,    0.01)           AS ret_exc_1,
            QUANTILE_DISC(ret_exc,    0.99)           AS ret_exc_99,
            QUANTILE_DISC(ret_exc,    0.999)          AS ret_exc_99_9

        FROM self
        GROUP BY {group_vars}
        ORDER BY {group_vars}
        """)
data.sink_parquet(res_path)


########## AUX_FUNCTIONS.RETURN_CUTOFFS('D', 0) ##########
# Inlined from aux_functions.return_cutoffs('d', 0)
freq = 'd'
crsp_only = 0
"""
Description:
    Compute return percentile cutoffs by period (monthly or daily), optionally CRSP-only.

Steps:
    1) Select group vars and output path from freq; scan 'world_{freq}sf.parquet'.
    2) Optional CRSP filter; require common/main/primary, exclude ZWE, non-null ret_exc.
    3) Add year/month; group and aggregate counts + percentiles for ret, ret_local, ret_exc.
    4) Sort, collect, save.

Output:
    Writes 'return_cutoffs.parquet' (monthly) or 'return_cutoffs_daily.parquet' (daily).
"""
group_vars = "eom" if freq == "m" else "year, month"
res_path = "return_cutoffs.parquet" if freq == "m" else "return_cutoffs_daily.parquet"
data = (
    pl.scan_parquet(f"world_{freq}sf.parquet")
    .filter(
        (col("common") == 1)
        & (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("primary_sec") == 1)
        & (col("excntry") != "ZWE")
        & (col("ret_exc").is_not_null())
        & ((col("source_crsp") == 1) if crsp_only == 1 else pl.lit(True))
    )
    .with_columns(year=col("date").dt.year(), month=col("date").dt.month())
)
data = data.sql(f"""
        SELECT
            {group_vars},
            COUNT(ret)                                AS n,

            -- ret percentiles
            QUANTILE_DISC(ret,        0.001)          AS ret_0_1,
            QUANTILE_DISC(ret,        0.01)           AS ret_1,
            QUANTILE_DISC(ret,        0.99)           AS ret_99,
            QUANTILE_DISC(ret,        0.999)          AS ret_99_9,

            -- ret_local percentiles
            QUANTILE_DISC(ret_local,  0.001)          AS ret_local_0_1,
            QUANTILE_DISC(ret_local,  0.01)           AS ret_local_1,
            QUANTILE_DISC(ret_local,  0.99)           AS ret_local_99,
            QUANTILE_DISC(ret_local,  0.999)          AS ret_local_99_9,

            -- ret_exc percentiles
            QUANTILE_DISC(ret_exc,    0.001)          AS ret_exc_0_1,
            QUANTILE_DISC(ret_exc,    0.01)           AS ret_exc_1,
            QUANTILE_DISC(ret_exc,    0.99)           AS ret_exc_99,
            QUANTILE_DISC(ret_exc,    0.999)          AS ret_exc_99_9

        FROM self
        GROUP BY {group_vars}
        ORDER BY {group_vars}
        """)
data.sink_parquet(res_path)


########## AUX_FUNCTIONS.MARKET_RETURNS('WORLD_DSF.PARQUET', 'D', 1, 'RETURN_CUTOFFS_DAILY.PARQUET') ##########
# Inlined from aux_functions.market_returns('world_dsf.parquet', 'd', 1, 'return_cutoffs_daily.parquet')
data_path = 'world_dsf.parquet'
freq = 'd'
wins_comp = 1
wins_data_path = 'return_cutoffs_daily.parquet'
"""
Description:
    Build country-level market returns (daily or monthly), optional winsorization, and save to disk.

Steps:
    1) Load params from freq; scan data; keep common-stock fields; sort by [id, dt].
    2) Add lags me_lag1, dolvol_lag1 per id.
    3) If wins_comp, join cutoffs and winsorize returns.
    4) Apply stock filters & compute VW/EW country returns.
    5) If daily, drop low-coverage trading days.
    6) Sort and write 'market_returns{_daily}.parquet'.

Output:
    Parquet file of country × date market returns.
"""
dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols = load_mkt_returns_params(freq)
__common_stocks = (
    pl.scan_parquet(data_path)
    .select(comm_stocks_cols)
    .unique()
    .sort(["id", dt_col])
    .with_columns(
        me_lag1=col("me").shift(1).over("id"),
        dolvol_lag1=col("dolvol").shift(1).over("id"),
    )
)
if wins_comp == 1:
    __common_stocks = add_cutoffs_and_winsorize(
        __common_stocks, wins_data_path, group_vars, dt_col
    )
__common_stocks = apply_stock_filter_and_compute_indexes(__common_stocks, dt_col, max_date_lag)
if freq == "d":
    __common_stocks = drop_non_trading_days(
        __common_stocks, "stocks", dt_col, ["excntry", "year", "month"], 0.25
    )
__common_stocks.sort(["excntry", dt_col]).collect().write_parquet(
    f"market_returns{path_aux}.parquet"
)


########## AUX_FUNCTIONS.MARKET_RETURNS('WORLD_MSF.PARQUET', 'M', 1, 'RETURN_CUTOFFS.PARQUET') ##########
# Inlined from aux_functions.market_returns('world_msf.parquet', 'm', 1, 'return_cutoffs.parquet')
data_path = 'world_msf.parquet'
freq = 'm'
wins_comp = 1
wins_data_path = 'return_cutoffs.parquet'
"""
Description:
    Build country-level market returns (daily or monthly), optional winsorization, and save to disk.

Steps:
    1) Load params from freq; scan data; keep common-stock fields; sort by [id, dt].
    2) Add lags me_lag1, dolvol_lag1 per id.
    3) If wins_comp, join cutoffs and winsorize returns.
    4) Apply stock filters & compute VW/EW country returns.
    5) If daily, drop low-coverage trading days.
    6) Sort and write 'market_returns{_daily}.parquet'.

Output:
    Parquet file of country × date market returns.
"""
dt_col, max_date_lag, path_aux, group_vars, comm_stocks_cols = load_mkt_returns_params(freq)
__common_stocks = (
    pl.scan_parquet(data_path)
    .select(comm_stocks_cols)
    .unique()
    .sort(["id", dt_col])
    .with_columns(
        me_lag1=col("me").shift(1).over("id"),
        dolvol_lag1=col("dolvol").shift(1).over("id"),
    )
)
if wins_comp == 1:
    __common_stocks = add_cutoffs_and_winsorize(
        __common_stocks, wins_data_path, group_vars, dt_col
    )
__common_stocks = apply_stock_filter_and_compute_indexes(__common_stocks, dt_col, max_date_lag)
if freq == "d":
    __common_stocks = drop_non_trading_days(
        __common_stocks, "stocks", dt_col, ["excntry", "year", "month"], 0.25
    )
__common_stocks.sort(["excntry", dt_col]).collect().write_parquet(
    f"market_returns{path_aux}.parquet"
)


########## AUX_FUNCTIONS.STANDARDIZED_ACCOUNTING_DATA('WORLD', 1, 'WORLD_MSF.PARQUET', 1, PL.DATETIME(1949, 12, 31)) ##########
# Inlined from aux_functions.standardized_accounting_data('world', 1, 'world_msf.parquet', 1, pl.datetime(1949, 12, 31))
coverage = 'world'
convert_to_usd = 1
me_data_path = 'world_msf.parquet'
include_helpers_vars = 1
start_date = pl.datetime(1949, 12, 31)
"""
Description:
    Build standardized annual/quarterly accounting panels (NA, Global, or World), optionally USD-converted; attach ME; write Parquet.

Steps:
    1) Inspect FUNDQ schemas; define target income/CF/BS/other vars; collect quarterly suffix vars (…q/…y).
    2) Load & filter raw GLOBAL and/or NA (annual/quarterly) via helper; add computed fields (e.g., ni, niq, ppegtq); drop vars as needed; apply INDFMT resolver.
    3) If world: concat NA+GLOBAL and break ties per key by preferring NA.
    4) If convert_to_usd: join FX and convert listed vars (annual & quarterly).
    5) Load ME and join to annual/quarterly panels.
    6) Quarterly: quarterize …y → …y_q, coalesce to …q; create ni_qtr/sale_qtr/ocf_qtr; cumulate 4Q flows with continuity checks; normalize currency codes; de-dupe, prefer later/NA rows.
    7) Annual: add empty quarterly helpers; join ME; sort.
    8) Optionally add helper variables.
    9) Unique by (gvkey, datadate), drop row index, sort, and write 'acc_std_ann.parquet' and 'acc_std_qtr.parquet'.

Output:
    Two Parquet files: 'acc_std_ann.parquet' (annual) and 'acc_std_qtr.parquet' (quarterly) standardized accounting data.
"""
g_fundq_cols = (
    pl.scan_parquet("../raw/raw_tables/comp_g_fundq.parquet").collect_schema().names()
)
fundq_cols = pl.scan_parquet("../raw/raw_tables/comp_fundq.parquet").collect_schema().names()
# Compustat Accounting Vars to Extract
avars_inc = [
    "sale",
    "revt",
    "gp",
    "ebitda",
    "oibdp",
    "ebit",
    "oiadp",
    "pi",
    "ib",
    "ni",
    "mii",
    "cogs",
    "xsga",
    "xopr",
    "xrd",
    "xad",
    "xlr",
    "dp",
    "xi",
    "do",
    "xido",
    "xint",
    "spi",
    "nopi",
    "txt",
    "dvt",
]
avars_cf = [
    "oancf",
    "ibc",
    "dpc",
    "xidoc",
    "capx",
    "wcapt",  # Operating
    "fincf",
    "fiao",
    "txbcof",
    "ltdch",
    "dltis",
    "dltr",
    "dlcch",
    "purtshr",
    "prstkc",
    "sstk",
    "dv",
    "dvc",
]  # Financing
avars_bs = [
    "at",
    "act",
    "aco",
    "che",
    "invt",
    "rect",
    "ivao",
    "ivst",
    "ppent",
    "ppegt",
    "intan",
    "ao",
    "gdwl",
    "re",  # Assets
    "lt",
    "lct",
    "dltt",
    "dlc",
    "txditc",
    "txdb",
    "itcb",
    "txp",
    "ap",
    "lco",
    "lo",
    "seq",
    "ceq",
    "pstkrv",
    "pstkl",
    "pstk",
    "mib",
    "icapt",
]  # Liabilities
# Variables in avars_other are not measured in currency units, and only available in annual data
avars_other = ["emp"]
avars = avars_inc + avars_cf + avars_bs
# finding which variables of interest are available in the quarterly data
combined_columns = g_fundq_cols + fundq_cols
qvars_q = list(
    {
        aux_var
        for aux_var in combined_columns
        if aux_var[:-1].lower() in avars and aux_var.endswith("q")
    }
)  # different from above to get only unique values
qvars_y = list(
    {
        aux_var
        for aux_var in combined_columns
        if aux_var[:-1].lower() in avars and aux_var.endswith("y")
    }
)
qvars = qvars_q + qvars_y
if coverage in ["global", "world"]:
    # Annual global data:
    vars_not_in_query = ["gp", "pstkrv", "pstkl", "itcb", "xad", "txbcof", "ni"]
    query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
    g_funda = load_raw_fund_table_and_filter(
        "../raw/raw_tables/comp_g_funda.parquet", start_date, "GLOBAL", 1
    )
    __gfunda = (
        g_funda.with_columns(
            ni=(col("ib") + pl.coalesce("xi", 0) + pl.coalesce("do", 0)).cast(pl.Float64)
        )
        .select(
            ["gvkey", "datadate", "n", "indfmt", "curcd", "source", "ni"]
            + [fl_none().alias(i) for i in ["gp", "pstkrv", "pstkl", "itcb", "xad", "txbcof"]]
            + query_vars
        )
        .pipe(apply_indfmt_filter)
    )
    # Quarterly global data:
    vars_not_in_query = [
        "icaptq",
        "niy",
        "txditcq",
        "txpq",
        "xidoq",
        "xidoy",
        "xrdq",
        "xrdy",
        "txbcofy",
        "niq",
        "ppegtq",
        "doq",
        "doy",
    ]
    query_vars = [var for var in qvars if var not in vars_not_in_query]
    g_fundq = load_raw_fund_table_and_filter(
        "../raw/raw_tables/comp_g_fundq.parquet", start_date, "GLOBAL", 1
    )
    __gfundq = (
        g_fundq.with_columns(
            niq=(col("ibq") + pl.coalesce("xiq", 0.0)).cast(pl.Float64),
            ppegtq=(col("ppentq") + col("dpactq")).cast(pl.Float64),
        )
        .select(
            [
                "gvkey",
                "datadate",
                "n",
                "indfmt",
                "fyr",
                "fyearq",
                "fqtr",
                "curcdq",
                "source",
                "niq",
                "ppegtq",
            ]
            + [
                fl_none().alias(i)
                for i in [
                    "icaptq",
                    "niy",
                    "txditcq",
                    "txpq",
                    "xidoq",
                    "xidoy",
                    "xrdq",
                    "xrdy",
                    "txbcofy",
                ]
            ]
            + query_vars
        )
        .pipe(apply_indfmt_filter)
    )
if coverage in ["na", "world"]:
    # Annual north american data:
    vars_not_in_query = ["wcapt", "ltdch", "purtshr"]
    query_vars = [var for var in (avars + avars_other) if var not in vars_not_in_query]
    funda = load_raw_fund_table_and_filter(
        "../raw/raw_tables/comp_funda.parquet", start_date, "NA", 2
    )
    __funda = funda.select(
        ["gvkey", "datadate", "n", "curcd", "source"]
        + [fl_none().alias(i) for i in ["wcapt", "ltdch", "purtshr"]]
        + query_vars
    )
    # Quarterly north american data:
    vars_not_in_query = [
        "dvtq",
        "gpq",
        "dvty",
        "gpy",
        "ltdchy",
        "purtshry",
        "wcapty",
    ]
    query_vars = [var for var in qvars if var not in vars_not_in_query]
    fundq = load_raw_fund_table_and_filter(
        "../raw/raw_tables/comp_fundq.parquet", start_date, "NA", 2
    )
    __fundq = fundq.select(
        ["gvkey", "datadate", "n", "fyr", "fyearq", "fqtr", "curcdq", "source"]
        + [
            fl_none().alias(i)
            for i in ["dvtq", "gpq", "dvty", "gpy", "ltdchy", "purtshry", "wcapty"]
        ]
        + query_vars
    )
if coverage == "world":
    __wfunda = pl.concat([__gfunda, __funda], how="diagonal_relaxed").filter(
        (pl.len().over(["gvkey", "datadate"]) == 1)
        | ((pl.len().over(["gvkey", "datadate"]) == 2) & (col("source") == "GLOBAL"))
    )
    __wfundq = pl.concat([__gfundq, __fundq], how="diagonal_relaxed").filter(
        (pl.len().over(["gvkey", "fyr", "fyearq", "fqtr"]) == 1)
        | (
            (pl.len().over(["gvkey", "fyr", "fyearq", "fqtr"]) == 2)
            & (col("source") == "GLOBAL")
        )
    )
else:
    pass
if coverage == "na":
    aname, qname = __funda, __fundq
elif coverage == "global":
    aname, qname = __gfunda, __gfundq
else:
    aname, qname = __wfunda, __wfundq

if convert_to_usd == 1:
    fx = compustat_fx().lazy()
    __compa = add_fx_and_convert_vars(aname, fx, avars, "annual")
    __compq = add_fx_and_convert_vars(qname, fx, qvars, "quarterly")
else:
    __compa, __compq = aname, qname

__me_data = load_mkt_equity_data(me_data_path)

yrl_vars = [
    "cogsq",
    "xsgaq",
    "xintq",
    "dpq",
    "txtq",
    "xrdq",
    "dvq",
    "spiq",
    "saleq",
    "revtq",
    "xoprq",
    "oibdpq",
    "oiadpq",
    "ibq",
    "niq",
    "xidoq",
    "nopiq",
    "miiq",
    "piq",
    "xiq",
    "xidocq",
    "capxq",
    "oancfq",
    "ibcq",
    "dpcq",
    "wcaptq",
    "prstkcq",
    "sstkq",
    "purtshrq",
    "dsq",
    "dltrq",
    "ltdchq",
    "dlcchq",
    "fincfq",
    "fiaoq",
    "txbcofq",
    "dvtq",
]
bs_vars = [
    "seqq",
    "ceqq",
    "pstkq",
    "icaptq",
    "mibq",
    "gdwlq",
    "req",
    "atq",
    "actq",
    "invtq",
    "rectq",
    "ppegtq",
    "ppentq",
    "aoq",
    "acoq",
    "intanq",
    "cheq",
    "ivaoq",
    "ivstq",
    "ltq",
    "lctq",
    "dlttq",
    "dlcq",
    "txpq",
    "apq",
    "lcoq",
    "loq",
    "txditcq",
    "txdbq",
]
__compq = __compq.with_columns(
    [
        col(var).cast(pl.Int64)
        for var in ["fyr", "fyearq", "fqtr"]
        if var in __compq.collect_schema().names()
    ]
)

# NOTE: quarterly companion table to __compa (same normalization pipeline).
__compq = (
    __compq.pipe(quarterize, var_list=qvars_y)
    .with_columns(
        [
            pl.coalesce([f"{var[:-1]}q", f"{var[:-1]}y_q"]).alias(f"{var[:-1]}q")
            for var in qvars_y
            if f"{var[:-1]}q" in __compq.collect_schema().names()
        ]
        + [
            col(f"{var[:-1]}y_q").alias(f"{var[:-1]}q")
            for var in qvars_y
            if f"{var[:-1]}q" not in __compq.collect_schema().names()
        ]
    )
    .drop([f"{var[:-1]}y_q" for var in qvars_y])
    .with_columns(
        ni_qtr=col("ibq"),
        sale_qtr=col("saleq"),
        ocf_qtr=pl.coalesce(
            ["oancfq", (col("ibq") + col("dpq") - pl.coalesce([col("wcaptq"), 0]))]
        ),
        dsy=fl_none(),
        dsq=fl_none(),
    )
    .sort(["gvkey", "fyr", "fyearq", "fqtr", "n"])
    .pipe(cumulate_4q, var_list=yrl_vars)
    .rename(
        {
            **dict(zip(bs_vars, [aux[:-1] for aux in bs_vars], strict=True)),
            **{"curcdq": "curcd"},
        }
    )
    .unique(["gvkey", "datadate", "fyr"])
    .sort(["gvkey", "datadate", "fyr"])
    .unique(["gvkey", "datadate"], keep="last")
    .drop(["fyr", "fyearq", "fqtr"])
    .join(
        __me_data,
        how="left",
        left_on=["gvkey", "datadate"],
        right_on=["gvkey", "eom"],
    )
    .with_columns(
        [
            fl_none().alias(i)
            for i in [
                "gp",
                "dltis",
                "do",
                "dvc",
                "ebit",
                "ebitda",
                "itcb",
                "pstkl",
                "pstkrv",
                "xad",
                "xlr",
                "emp",
            ]
        ]
    )
    .sort(["gvkey", "curcd", "datadate", "source", "n"])
)

# NOTE: __compa/__compq hold standardized annual/quarterly accounting snapshots before final output writes.
__compa = (
    __compa.with_columns(
        ni_qtr=fl_none(),
        sale_qtr=fl_none(),
        ocf_qtr=fl_none(),
        fqtr=pl.lit(None).cast(pl.Int64),
        fyearq=pl.lit(None).cast(pl.Int64),
        fyr=pl.lit(None).cast(pl.Int64),
    )
    .join(
        __me_data,
        how="left",
        left_on=["gvkey", "datadate"],
        right_on=["gvkey", "eom"],
    )
    .sort(["gvkey", "curcd", "datadate", "source", "n"])
)

if include_helpers_vars == 1:
    __compq = add_helper_vars(__compq)
    __compa = add_helper_vars(__compa)

__compa.unique(["gvkey", "datadate"]).drop("n").sort(
    ["gvkey", "datadate"]
).collect().write_parquet("acc_std_ann.parquet")
__compq.unique(["gvkey", "datadate"]).drop("n").sort(
    ["gvkey", "datadate"]
).collect().write_parquet("acc_std_qtr.parquet")


########## AUX_FUNCTIONS.CREATE_ACC_CHARS('ACC_STD_ANN.PARQUET', 'ACHARS_WORLD.PARQUET', 4, 18, ACC_CHARS_LIST(), 'WORLD_MSF.PARQUET', '') ##########
# Inlined from aux_functions.create_acc_chars('acc_std_ann.parquet', 'achars_world.parquet', 4, 18, acc_chars_list(), 'world_msf.parquet', '')
data_path = 'acc_std_ann.parquet'
output_path = 'achars_world.parquet'
lag_to_public = 4
max_data_lag = 18
__keep_vars = acc_chars_list()
me_data_path = 'world_msf.parquet'
suffix = ''
"""
Description:
    Build comprehensive accounting characteristics, align to public dates, convert to USD, join ME, and scale/derive market-based metrics.

Steps:
    1) Load ME (company) and scan input; add counts and core aliases (assets, sales, book_equity, net_income).
    2) Add accounting ratios/features (misc cols 1), financial soundness/misc ratios, and liquidity/efficiency ratios.
    3) Add sales-per-employee growth and employee growth; compute consecutive NI increases.
    4) Add changes/growth (NOA, PPE+Inv, LNOA, CAPEX 2y); quarterly profitability (saleq_gr1, NIQ/BE, NIQ/AT) and their 1y deltas.
    5) RD capital-to-assets (5y decay); Abarbanell–Bushee changes; standardized surprises for sales/NI; abnormal CAPEX; lagged-profit ratios.
    6) Add core ratios (pi_nix, ocf_at, op_at, at_be, ROE/OCF per quarter); tangibility, ALIQ/AT; OCF_AT 1y change.
    7) Append volatility & composite metrics (misc cols 2); compute earnings persistence and expand to monthly public_date.
    8) Convert key raw vars to USD; join ME; compute ME/MEV/MAT scales and equity duration.
    9) Rename selected columns with mapping, select keep-vars (+ ids), optional suffix, dedupe, and write parquet.

Output:
    Parquet at output_path with standardized, USD/ME/MEV-scaled accounting characteristics keyed by (gvkey, public_date).
"""

me_data = load_mkt_equity_data(me_data_path, False)

chars_df = pl.scan_parquet(data_path)

chars_df = (
    chars_df.sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        count=col("gvkey").cum_count().over(["gvkey", "curcd"]),
        assets=col("at_x"),
        sales=col("sale_x"),
        book_equity=col("be_x"),
        net_income=col("ni_x"),
    )
    .pipe(add_accounting_misc_cols_1)
    .with_columns(financial_soundness_and_misc_ratios_exps())
    .pipe(add_liq_and_efficiency_ratios)
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        sale_emp_gr1=pl.when((col("count") > 12) & (col("sale_emp").shift(12) > 0))
        .then(col("sale_emp") / col("sale_emp").shift(12) - 1)
        .otherwise(fl_none())
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(emp_gr(data_path))
    .pipe(calculate_consecutive_earnings_increases)
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        [
            chg_to_lagassets(i) for i in ["noa_x", "ppeinv_x"]
        ]  # 1yr Change Scaled by Lagged Assets)
        + [chg_to_avgassets(i) for i in ["lnoa_x"]]  # 1yr Change Scaled by Average Assets
        + [var_growth(var_gr="capx", horizon=24)]
    )  # CAPEX growth over 2 years
    .sort(["gvkey", "curcd", "datadate"])
    # Quarterly profitability measures:
    .with_columns(
        [
            pl.when((col("count") > 12) & (col("sale_qtr").shift(12) > 0))
            .then(col("sale_qtr") / col("sale_qtr").shift(12) - 1)
            .otherwise(fl_none())
            .alias("saleq_gr1"),
            safe_div("ni_qtr", "be_x", "niq_be", 9),
            safe_div("ni_qtr", "at_x", "niq_at", 9),
        ]
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        [
            pl.when(col("count") > 12)
            .then(col("niq_be") - col("niq_be").shift(12))
            .otherwise(fl_none())
            .alias("niq_be_chg1"),
            pl.when(col("count") > 12)
            .then(col("niq_at") - col("niq_at").shift(12))
            .otherwise(fl_none())
            .alias("niq_at_chg1"),
        ]
    )
    .sort(["gvkey", "curcd", "datadate"])
    # R&D capital-to-assets
    .with_columns(
        pl.when((col("count") > 48) & (col("at_x") > 0))
        .then(
            (
                col("xrd")
                + col("xrd").shift(12) * 0.8
                + col("xrd").shift(24) * 0.6
                + col("xrd").shift(36) * 0.4
                + col("xrd").shift(48) * 0.2
            )
            / col("at_x")
        )
        .otherwise(fl_none())
        .alias("rd5_at")
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        [chg_to_exp(i) for i in ["sale_x", "invt", "rect", "gp_x", "xsga"]]
    )  # Abarbanell and Bushee (1998)
    .with_columns(
        dsale_dinv=col("sale_ce") - col("invt_ce"),
        dsale_drec=col("sale_ce") - col("rect_ce"),
        dgp_dsale=col("gp_ce") - col("sale_ce"),
        dsale_dsga=col("sale_ce") - col("xsga_ce"),
    )
    .drop(["sale_ce", "invt_ce", "rect_ce", "gp_ce", "xsga_ce"])
    .pipe(standardized_unexpected, var="sale_qtr", qtrs=8, qtrs_min=6)
    .pipe(standardized_unexpected, var="ni_qtr", qtrs=8, qtrs_min=6)
    .pipe(compute_capex_abn)
    .pipe(add_profit_scaled_by_lagged_vars)
    .with_columns(
        pi_nix=safe_div("pi_x", "nix_x", "pi_nix", 8),
        ocf_at=safe_div("ocf_x", "at_x", "ocf_at", 3),
        op_at=safe_div("op_x", "at_x", "op_at", 3),
        at_be=safe_div("at_x", "be_x", "at_be"),
        __ocfq_saleq=safe_div("ocf_qtr", "sale_qtr", "__ocfq_saleq", 3),
        __niq_saleq=safe_div("ni_qtr", "sale_qtr", "__niq_saleq", 3),
        __roeq=safe_div("ni_qtr", "be_x", "__roeq", 3),
        __roe=safe_div("ni_x", "be_x", "__roe", 3),
        tangibility=tangibility(),
        aliq_at=safe_div("aliq_x", "at_x", "aliq_at", 4),
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        ocf_at_chg1=pl.when(col("count") > 12)
        .then(col("ocf_at") - col("ocf_at").shift(12))
        .otherwise(fl_none())
    )
    .pipe(add_accounting_misc_cols_2)
    .pipe(
        add_earnings_persistence_and_expand,
        data_path=data_path,
        lag_to_pub=lag_to_public,
        max_lag=max_data_lag,
    )
    .pipe(convert_raw_vars_to_usd)
    .pipe(add_me_data_and_compute_me_mev_mat_eqdur_vars, me_df=me_data)
)

rename_dict = {
    "xrd": "rd",
    "xsga": "sga",
    "dlc": "debtst",
    "dltt": "debtlt",
    "oancf": "ocf",
    "ppegt": "ppeg",
    "ppent": "ppen",
    "che": "cash",
    "invt": "inv",
    "rect": "rec",
    "txt": "tax",
    "ivao": "lti",
    "ivst": "sti",
    "sale_qtr": "saleq",
    "ni_qtr": "niq",
    "ocf_qtr": "ocfq",
}

rename_cols_and_select_keep_vars(chars_df, rename_dict, __keep_vars, suffix).sort(
    ["gvkey", "public_date"]
).unique(["gvkey", "public_date"], keep="first").sort(
    ["gvkey", "public_date"]
).collect().write_parquet(output_path)


########## AUX_FUNCTIONS.CREATE_ACC_CHARS('ACC_STD_QTR.PARQUET', 'QCHARS_WORLD.PARQUET', 4, 18, ACC_CHARS_LIST(), 'WORLD_MSF.PARQUET', '_QITEM') ##########
# Inlined from aux_functions.create_acc_chars('acc_std_qtr.parquet', 'qchars_world.parquet', 4, 18, acc_chars_list(), 'world_msf.parquet', '_qitem')
data_path = 'acc_std_qtr.parquet'
output_path = 'qchars_world.parquet'
lag_to_public = 4
max_data_lag = 18
__keep_vars = acc_chars_list()
me_data_path = 'world_msf.parquet'
suffix = '_qitem'
"""
Description:
    Build comprehensive accounting characteristics, align to public dates, convert to USD, join ME, and scale/derive market-based metrics.

Steps:
    1) Load ME (company) and scan input; add counts and core aliases (assets, sales, book_equity, net_income).
    2) Add accounting ratios/features (misc cols 1), financial soundness/misc ratios, and liquidity/efficiency ratios.
    3) Add sales-per-employee growth and employee growth; compute consecutive NI increases.
    4) Add changes/growth (NOA, PPE+Inv, LNOA, CAPEX 2y); quarterly profitability (saleq_gr1, NIQ/BE, NIQ/AT) and their 1y deltas.
    5) RD capital-to-assets (5y decay); Abarbanell–Bushee changes; standardized surprises for sales/NI; abnormal CAPEX; lagged-profit ratios.
    6) Add core ratios (pi_nix, ocf_at, op_at, at_be, ROE/OCF per quarter); tangibility, ALIQ/AT; OCF_AT 1y change.
    7) Append volatility & composite metrics (misc cols 2); compute earnings persistence and expand to monthly public_date.
    8) Convert key raw vars to USD; join ME; compute ME/MEV/MAT scales and equity duration.
    9) Rename selected columns with mapping, select keep-vars (+ ids), optional suffix, dedupe, and write parquet.

Output:
    Parquet at output_path with standardized, USD/ME/MEV-scaled accounting characteristics keyed by (gvkey, public_date).
"""

me_data = load_mkt_equity_data(me_data_path, False)

chars_df = pl.scan_parquet(data_path)

chars_df = (
    chars_df.sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        count=col("gvkey").cum_count().over(["gvkey", "curcd"]),
        assets=col("at_x"),
        sales=col("sale_x"),
        book_equity=col("be_x"),
        net_income=col("ni_x"),
    )
    .pipe(add_accounting_misc_cols_1)
    .with_columns(financial_soundness_and_misc_ratios_exps())
    .pipe(add_liq_and_efficiency_ratios)
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        sale_emp_gr1=pl.when((col("count") > 12) & (col("sale_emp").shift(12) > 0))
        .then(col("sale_emp") / col("sale_emp").shift(12) - 1)
        .otherwise(fl_none())
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(emp_gr(data_path))
    .pipe(calculate_consecutive_earnings_increases)
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        [
            chg_to_lagassets(i) for i in ["noa_x", "ppeinv_x"]
        ]  # 1yr Change Scaled by Lagged Assets)
        + [chg_to_avgassets(i) for i in ["lnoa_x"]]  # 1yr Change Scaled by Average Assets
        + [var_growth(var_gr="capx", horizon=24)]
    )  # CAPEX growth over 2 years
    .sort(["gvkey", "curcd", "datadate"])
    # Quarterly profitability measures:
    .with_columns(
        [
            pl.when((col("count") > 12) & (col("sale_qtr").shift(12) > 0))
            .then(col("sale_qtr") / col("sale_qtr").shift(12) - 1)
            .otherwise(fl_none())
            .alias("saleq_gr1"),
            safe_div("ni_qtr", "be_x", "niq_be", 9),
            safe_div("ni_qtr", "at_x", "niq_at", 9),
        ]
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        [
            pl.when(col("count") > 12)
            .then(col("niq_be") - col("niq_be").shift(12))
            .otherwise(fl_none())
            .alias("niq_be_chg1"),
            pl.when(col("count") > 12)
            .then(col("niq_at") - col("niq_at").shift(12))
            .otherwise(fl_none())
            .alias("niq_at_chg1"),
        ]
    )
    .sort(["gvkey", "curcd", "datadate"])
    # R&D capital-to-assets
    .with_columns(
        pl.when((col("count") > 48) & (col("at_x") > 0))
        .then(
            (
                col("xrd")
                + col("xrd").shift(12) * 0.8
                + col("xrd").shift(24) * 0.6
                + col("xrd").shift(36) * 0.4
                + col("xrd").shift(48) * 0.2
            )
            / col("at_x")
        )
        .otherwise(fl_none())
        .alias("rd5_at")
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        [chg_to_exp(i) for i in ["sale_x", "invt", "rect", "gp_x", "xsga"]]
    )  # Abarbanell and Bushee (1998)
    .with_columns(
        dsale_dinv=col("sale_ce") - col("invt_ce"),
        dsale_drec=col("sale_ce") - col("rect_ce"),
        dgp_dsale=col("gp_ce") - col("sale_ce"),
        dsale_dsga=col("sale_ce") - col("xsga_ce"),
    )
    .drop(["sale_ce", "invt_ce", "rect_ce", "gp_ce", "xsga_ce"])
    .pipe(standardized_unexpected, var="sale_qtr", qtrs=8, qtrs_min=6)
    .pipe(standardized_unexpected, var="ni_qtr", qtrs=8, qtrs_min=6)
    .pipe(compute_capex_abn)
    .pipe(add_profit_scaled_by_lagged_vars)
    .with_columns(
        pi_nix=safe_div("pi_x", "nix_x", "pi_nix", 8),
        ocf_at=safe_div("ocf_x", "at_x", "ocf_at", 3),
        op_at=safe_div("op_x", "at_x", "op_at", 3),
        at_be=safe_div("at_x", "be_x", "at_be"),
        __ocfq_saleq=safe_div("ocf_qtr", "sale_qtr", "__ocfq_saleq", 3),
        __niq_saleq=safe_div("ni_qtr", "sale_qtr", "__niq_saleq", 3),
        __roeq=safe_div("ni_qtr", "be_x", "__roeq", 3),
        __roe=safe_div("ni_x", "be_x", "__roe", 3),
        tangibility=tangibility(),
        aliq_at=safe_div("aliq_x", "at_x", "aliq_at", 4),
    )
    .sort(["gvkey", "curcd", "datadate"])
    .with_columns(
        ocf_at_chg1=pl.when(col("count") > 12)
        .then(col("ocf_at") - col("ocf_at").shift(12))
        .otherwise(fl_none())
    )
    .pipe(add_accounting_misc_cols_2)
    .pipe(
        add_earnings_persistence_and_expand,
        data_path=data_path,
        lag_to_pub=lag_to_public,
        max_lag=max_data_lag,
    )
    .pipe(convert_raw_vars_to_usd)
    .pipe(add_me_data_and_compute_me_mev_mat_eqdur_vars, me_df=me_data)
)

rename_dict = {
    "xrd": "rd",
    "xsga": "sga",
    "dlc": "debtst",
    "dltt": "debtlt",
    "oancf": "ocf",
    "ppegt": "ppeg",
    "ppent": "ppen",
    "che": "cash",
    "invt": "inv",
    "rect": "rec",
    "txt": "tax",
    "ivao": "lti",
    "ivst": "sti",
    "sale_qtr": "saleq",
    "ni_qtr": "niq",
    "ocf_qtr": "ocfq",
}

rename_cols_and_select_keep_vars(chars_df, rename_dict, __keep_vars, suffix).sort(
    ["gvkey", "public_date"]
).unique(["gvkey", "public_date"], keep="first").sort(
    ["gvkey", "public_date"]
).collect().write_parquet(output_path)


########## AUX_FUNCTIONS.COMBINE_ANN_QTR_CHARS('ACHARS_WORLD.PARQUET', 'QCHARS_WORLD.PARQUET', ACC_CHARS_LIST(), '_QITEM') ##########
# Inlined from aux_functions.combine_ann_qtr_chars('achars_world.parquet', 'qchars_world.parquet', acc_chars_list(), '_qitem')
ann_df_path = 'achars_world.parquet'
qtr_df_path = 'qchars_world.parquet'
char_vars = acc_chars_list()
q_suffix = '_qitem'
"""
Description:
    Combine annual and quarterly characteristic panels, preferring fresher quarterly values at the same public_date.

Steps:
    1) Load annual and quarterly files into DuckDB with row numbers.
    2) Left-join on (gvkey, public_date); for each char_var choose quarterly value if present and more recent (datadate_qitem > datadate).
    3) Drop redundant join and dated columns; dedupe on (gvkey, public_date).

Output:
    Writes 'acc_chars_world.parquet' merged panel.
"""
os.system("rm -f aux_aqtr_chars.ddb")
con = ibis.duckdb.connect("aux_aqtr_chars.ddb", threads=os.cpu_count())
con.create_table(
    "ann",
    con.read_parquet(ann_df_path).mutate(n1=ibis.row_number()),
    overwrite=True,
)
con.create_table(
    "qtr",
    con.read_parquet(qtr_df_path)
    .mutate(n2=ibis.row_number())
    .rename({"datadate_qitem": "datadate", "source_qitem": "source"}),
    overwrite=True,
)
ann = con.table("ann")
qtr = con.table("qtr")
combined = ann.left_join(qtr, [ann.gvkey == qtr.gvkey, ann.public_date == qtr.public_date])
drop_columns = [
    "datadate",
    f"datadate{q_suffix}",
    "gvkey_right",
    "public_date_right",
    "source_qitem",
] + [f"{ann_var}{q_suffix}" for ann_var in char_vars]
subs = {}
for ann_var in char_vars:
    qtr_var = f"{ann_var}{q_suffix}"
    subs[ann_var] = ibis.ifelse(
        combined[ann_var].isnull()
        | (combined[qtr_var].notnull() & (combined[f"datadate{q_suffix}"] > combined.datadate)),
        combined[qtr_var],
        combined[ann_var],
    )
combined = (
    combined.mutate(subs)
    .drop(drop_columns)
    .order_by(["gvkey", "public_date", "n1", "n2"])
    .distinct(on=["gvkey", "public_date"], keep="first")
    .drop(["n1", "n2"])
    .order_by(["gvkey", "public_date"])
)
combined.to_parquet("acc_chars_world.parquet")
con.disconnect()
os.system("rm -f aux_aqtr_chars.ddb")


########## AUX_FUNCTIONS.MARKET_CHARS_MONTHLY('WORLD_MSF.PARQUET', 'MARKET_RETURNS.PARQUET') ##########
# Inlined from aux_functions.market_chars_monthly('world_msf.parquet', 'market_returns.parquet')
data_path = 'world_msf.parquet'
market_ret_path = 'market_returns.parquet'
"""
Description:
    Build monthly market characteristics per security: dividends, issuance, momentum/reversal, seasonality.

Steps:
    1) Read stock panel and join country market returns; pick ret_x (local vs USD).
    2) Create complete monthly range per id; compute cumulative indices (ri, ri_x), counts, and missing-return mask.
    3) Zero-out dividend artifacts near 0; derive:
    - Dividend-to-ME (regular/special) over horizons
    - Equity net payout (eqnpo), share change (chcsho)
    - Momentum/reversal ret_{j}_{i}
    - Seasonality windows via helper.
    4) Select/id-sort final feature set.

Output:
    Writes 'market_chars_m.parquet' with [id, eom, market_equity, div*, eqnpo*, chcsho*, ret_*, seas_*].
"""
div_range = [1, 3, 6, 12]  # [1,3,6,12,24,36]
div_spc_range = [1, 12]
chcsho_lags = [1, 3, 6, 12]
eqnpo_lags = [1, 3, 6, 12]
mom_rev_lags = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 3],
    [0, 6],
    [1, 6],
    [0, 9],
    [1, 9],
    [0, 12],
    [1, 12],
    [7, 12],
    [1, 18],
    [1, 24],
    [12, 24],
    [1, 36],
    [12, 36],
    [12, 48],
    [1, 48],
    [1, 60],
    [12, 60],
    [36, 60],
]
ret_var = "ret_local" if local_currency else "ret"
market_ret = pl.scan_parquet(market_ret_path)
data = (
    pl.scan_parquet(data_path)
    .join(market_ret, how="left", on=["excntry", "eom"])
    .with_columns(col(ret_var).alias("ret_x"))
    # No need to compute ret_zero because it's not used in the final output
)
__stock_coverage = (
    data.group_by("id")
    .agg(start_date=pl.min("eom"), end_date=pl.max("eom"))
    .sort(["id", "start_date"])
)
__full_range = expand(__stock_coverage, ["id"], "start_date", "end_date", "month", "eom")
data = (
    __full_range.join(data, how="left", on=["id", "eom"])
    .sort(["id", "eom"])
    .with_columns(
        ri=((1 + pl.coalesce(["ret", 0])).cum_prod()).over("id"),
        ri_x=((1 + pl.coalesce(["ret_x", 0])).cum_prod()).over("id"),
        count=(col("id").cum_count()).over("id"),
        ret_miss=pl.when((col("ret_x").is_not_null()) & (col("ret_lag_dif") == 1))
        .then(pl.lit(0))
        .otherwise(pl.lit(1)),
    )
    .with_columns(
        [
            pl.when(col("ret_miss") == 1).then(fl_none()).otherwise(i).alias(i)
            for i in ["ret_x", "ret", "ret_local", "ret_exc", "mkt_vw_exc"]
        ]
    )
    .unique(["id", "eom"])
    .with_columns(
        market_equity=col("me"),
        div1m_me=col("div_tot") * col("shares"),
        divspc1m_me=col("div_spc") * col("shares"),
        aux=col("shares") * col("adjfct"),
    )
    .sort(["id", "eom"])
    .with_columns(
        [div_cols(i, spc=False) for i in div_range]
        + [div_cols(i, spc=True) for i in div_spc_range]
        + [eqnpo_cols(i) for i in eqnpo_lags]
        + [chcsho_cols(i) for i in chcsho_lags]
        + [mom_rev_cols(i, j) for i, j in mom_rev_lags]
    )
)
for i in [[1, 1], [2, 5], [6, 10], [11, 15], [16, 20]]:
    data = seasonality(data, "ret_x", i[0], i[1])
data = (
    data.with_columns(
        [
            pl.when(col(var) < 1e-5).then(0.0).otherwise(col(var)).alias(var)
            for var in data.collect_schema().names()
            if var.startswith("div") and var.endswith("me")
        ]
    )
    .with_columns(
        [
            pl.when(col(var).abs() < 1e-5).then(0.0).otherwise(col(var)).alias(var)
            for var in data.collect_schema().names()
            if var.startswith("eqnpo")
        ]
    )
    .select(
        [
            "id",
            "eom",
            "market_equity",
            col("^div.*me$"),
            col("^eqnpo.*$"),
            col("^chcsho.*$"),
            col(r"^ret_\d+_\d+$"),
            col("^seas.*$"),
        ]
    )
    .sort(["id", "eom"])
)

data.collect(streaming=True).write_parquet("market_chars_m.parquet")


########## AUX_FUNCTIONS.CREATE_WORLD_DATA_PRELIM('WORLD_MSF.PARQUET', 'MARKET_CHARS_M.PARQUET', 'ACC_CHARS_WORLD.PARQUET', 'WORLD_DATA_PRELIM.PARQUET') ##########
# Inlined from aux_functions.create_world_data_prelim('world_msf.parquet', 'market_chars_m.parquet', 'acc_chars_world.parquet', 'world_data_prelim.parquet')
msf_path = 'world_msf.parquet'
market_chars_monthly_path = 'market_chars_m.parquet'
acc_chars_world_path = 'acc_chars_world.parquet'
output_path = 'world_data_prelim.parquet'
"""
Description:
    Build preliminary world dataset by merging returns, market characteristics, and accounting data.

Steps:
    1) Load msf (stock returns), monthly market chars, and accounting chars parquet files.
    2) Left-join msf with market chars on (id, eom).
    3) Left-join with accounting chars on (gvkey, eom vs. public_date).
    4) Drop dividend-related fields and source flag.

Output:
    '{output_path}' parquet with merged stock, market, and accounting data.
"""
a = pl.scan_parquet(msf_path)
b = pl.scan_parquet(market_chars_monthly_path)
c = pl.scan_parquet(acc_chars_world_path)
world_data_prelim = (
    a.join(b, how="left", on=["id", "eom"])
    .join(c, how="left", left_on=["gvkey", "eom"], right_on=["gvkey", "public_date"])
    .drop(["div_tot", "div_cash", "div_spc", "source"])
)
world_data_prelim.collect().write_parquet(output_path)


########## AUX_FUNCTIONS.AP_FACTORS('AP_FACTORS_DAILY.PARQUET', 'D', 'WORLD_DSF.PARQUET', 'WORLD_DATA_PRELIM.PARQUET', 'MARKET_RETURNS_DAILY.PARQUET', 10, 3) ##########
# Inlined from aux_functions.ap_factors('ap_factors_daily.parquet', 'd', 'world_dsf.parquet', 'world_data_prelim.parquet', 'market_returns_daily.parquet', 10, 3)
output_path = 'ap_factors_daily.parquet'
freq = 'd'
sf_path = 'world_dsf.parquet'
mchars_path = 'world_data_prelim.parquet'
mkt_path = 'market_returns_daily.parquet'
min_stocks_bp = 10
min_stocks_pf = 3
"""
Description:
    Build AP-style factor panels (FF HML/SMB and HXZ INV/ROE/SMB) by country and month (or day).

Steps:
    1) Load security returns; winsorize ret_exc by date.
    2) Load market characteristics; lag key vars 1 period with continuity guard; filter to eligible stocks.
    3) Size-bucket each stock; run FF-style sorts for BE/ME, asset growth, and ROE.
    4) Compose factors: mktrf from market file; HML/SMB (FF); INV/ROE/SMB (HXZ).
    5) Join and write factors to output_path.

Output:
    Parquet factor file with columns: [excntry, date/eom, mktrf, hml, smb_ff, inv, roe, smb_hxz].
"""
date_col = "eom" if freq == "m" else "date"
sf_cond = (col("ret_lag_dif") == 1) if freq == "m" else (col("ret_lag_dif") <= 5)
lag_vars = [
    "comp_exchg",
    "crsp_exchcd",
    "exch_main",
    "obs_main",
    "common",
    "primary_sec",
    "excntry",
    "size_grp",
    "me",
    "be_me",
    "at_gr1",
    "niq_be",
]

# print(f"Executing AP factors with frequency {freq}", flush=True)

world_sf1 = (
    pl.scan_parquet(sf_path)
    .filter(sf_cond & col("ret_exc").is_not_null())
    .select(["excntry", "id", "eom", "date", "ret_exc"])
)
world_sf2 = world_sf1.sql(f"""
                            WITH bounds AS (
                            SELECT
                                eom,
                                QUANTILE_DISC(ret_exc, {0.1 / 100}) AS low,
                                QUANTILE_DISC(ret_exc, {99.9 / 100}) AS high
                            FROM self
                            GROUP BY eom
                            )
                            SELECT
                                excntry,
                                id,
                                eom,
                                date,
                                CASE
                                    WHEN ret_exc < low  THEN low
                                    WHEN ret_exc > high THEN high
                                    ELSE ret_exc
                                END AS ret_exc
                            FROM self
                            LEFT JOIN bounds
                            USING (eom)
                        """)

base = (
    pl.scan_parquet(mchars_path)
    .sort(["id", "eom"])
    .with_columns(
        [col(i).shift(1).over(["id", "source_crsp"]).alias(i + "_l") for i in lag_vars]
    )
    .sort(["id", "eom"])
    .with_columns(
        [
            pl.when(
                (
                    12 * (col("eom").dt.year() - col("eom").shift(1).dt.year())
                    + (col("eom").dt.month() - col("eom").shift(1).dt.month()).cast(pl.Int32)
                ).over("id")
                != 1
            )
            .then(pl.lit(None))
            .otherwise(i + "_l")
            .alias(i + "_l")
            for i in lag_vars
        ]
    )
    .filter(
        (col("obs_main_l") == 1)
        & (col("exch_main_l") == 1)
        & (col("common_l") == 1)
        & (col("primary_sec_l") == 1)
        & (col("ret_lag_dif") == 1)
        & col("me_l").is_not_null()
    )
    .with_columns(
        size_pf=(
            pl.when(col("size_grp_l").is_null())
            .then(pl.lit(""))
            .when(col("size_grp_l").is_in(["large", "mega"]))
            .then(pl.lit("big"))
            .otherwise(pl.lit("small"))
        )
    )
)

ff = sort_ff_style("be_me", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2).rename(
    {"lms": "hml", "smb": "smb_ff"}
)
asset_growth = sort_ff_style(
    "at_gr1", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2
).rename({"lms": "at_gr1_lms", "smb": "at_gr1_smb"})
roeq = sort_ff_style("niq_be", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2).rename(
    {"lms": "niq_be_lms", "smb": "niq_be_smb"}
)
hxz = asset_growth.join(roeq, how="left", on=["excntry", date_col]).select(
    [
        "excntry",
        date_col,
        (-1 * col("at_gr1_lms")).alias("inv"),
        col("niq_be_lms").alias("roe"),
        ((col("at_gr1_smb") + col("niq_be_smb")) / 2).alias("smb_hxz"),
    ]
)

output = (
    pl.scan_parquet(mkt_path)
    .select(["excntry", date_col, col("mkt_vw_exc").alias("mktrf")])
    .collect()
    .join(ff, how="left", on=["excntry", date_col])
    .join(hxz, how="left", on=["excntry", date_col])
)
output.write_parquet(output_path)


########## AUX_FUNCTIONS.AP_FACTORS('AP_FACTORS_MONTHLY.PARQUET', 'M', 'WORLD_MSF.PARQUET', 'WORLD_DATA_PRELIM.PARQUET', 'MARKET_RETURNS.PARQUET', 10, 3) ##########
# Inlined from aux_functions.ap_factors('ap_factors_monthly.parquet', 'm', 'world_msf.parquet', 'world_data_prelim.parquet', 'market_returns.parquet', 10, 3)
output_path = 'ap_factors_monthly.parquet'
freq = 'm'
sf_path = 'world_msf.parquet'
mchars_path = 'world_data_prelim.parquet'
mkt_path = 'market_returns.parquet'
min_stocks_bp = 10
min_stocks_pf = 3
"""
Description:
    Build AP-style factor panels (FF HML/SMB and HXZ INV/ROE/SMB) by country and month (or day).

Steps:
    1) Load security returns; winsorize ret_exc by date.
    2) Load market characteristics; lag key vars 1 period with continuity guard; filter to eligible stocks.
    3) Size-bucket each stock; run FF-style sorts for BE/ME, asset growth, and ROE.
    4) Compose factors: mktrf from market file; HML/SMB (FF); INV/ROE/SMB (HXZ).
    5) Join and write factors to output_path.

Output:
    Parquet factor file with columns: [excntry, date/eom, mktrf, hml, smb_ff, inv, roe, smb_hxz].
"""
date_col = "eom" if freq == "m" else "date"
sf_cond = (col("ret_lag_dif") == 1) if freq == "m" else (col("ret_lag_dif") <= 5)
lag_vars = [
    "comp_exchg",
    "crsp_exchcd",
    "exch_main",
    "obs_main",
    "common",
    "primary_sec",
    "excntry",
    "size_grp",
    "me",
    "be_me",
    "at_gr1",
    "niq_be",
]

# print(f"Executing AP factors with frequency {freq}", flush=True)

world_sf1 = (
    pl.scan_parquet(sf_path)
    .filter(sf_cond & col("ret_exc").is_not_null())
    .select(["excntry", "id", "eom", "date", "ret_exc"])
)
world_sf2 = world_sf1.sql(f"""
                            WITH bounds AS (
                            SELECT
                                eom,
                                QUANTILE_DISC(ret_exc, {0.1 / 100}) AS low,
                                QUANTILE_DISC(ret_exc, {99.9 / 100}) AS high
                            FROM self
                            GROUP BY eom
                            )
                            SELECT
                                excntry,
                                id,
                                eom,
                                date,
                                CASE
                                    WHEN ret_exc < low  THEN low
                                    WHEN ret_exc > high THEN high
                                    ELSE ret_exc
                                END AS ret_exc
                            FROM self
                            LEFT JOIN bounds
                            USING (eom)
                        """)

base = (
    pl.scan_parquet(mchars_path)
    .sort(["id", "eom"])
    .with_columns(
        [col(i).shift(1).over(["id", "source_crsp"]).alias(i + "_l") for i in lag_vars]
    )
    .sort(["id", "eom"])
    .with_columns(
        [
            pl.when(
                (
                    12 * (col("eom").dt.year() - col("eom").shift(1).dt.year())
                    + (col("eom").dt.month() - col("eom").shift(1).dt.month()).cast(pl.Int32)
                ).over("id")
                != 1
            )
            .then(pl.lit(None))
            .otherwise(i + "_l")
            .alias(i + "_l")
            for i in lag_vars
        ]
    )
    .filter(
        (col("obs_main_l") == 1)
        & (col("exch_main_l") == 1)
        & (col("common_l") == 1)
        & (col("primary_sec_l") == 1)
        & (col("ret_lag_dif") == 1)
        & col("me_l").is_not_null()
    )
    .with_columns(
        size_pf=(
            pl.when(col("size_grp_l").is_null())
            .then(pl.lit(""))
            .when(col("size_grp_l").is_in(["large", "mega"]))
            .then(pl.lit("big"))
            .otherwise(pl.lit("small"))
        )
    )
)

ff = sort_ff_style("be_me", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2).rename(
    {"lms": "hml", "smb": "smb_ff"}
)
asset_growth = sort_ff_style(
    "at_gr1", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2
).rename({"lms": "at_gr1_lms", "smb": "at_gr1_smb"})
roeq = sort_ff_style("niq_be", min_stocks_bp, min_stocks_pf, date_col, base, world_sf2).rename(
    {"lms": "niq_be_lms", "smb": "niq_be_smb"}
)
hxz = asset_growth.join(roeq, how="left", on=["excntry", date_col]).select(
    [
        "excntry",
        date_col,
        (-1 * col("at_gr1_lms")).alias("inv"),
        col("niq_be_lms").alias("roe"),
        ((col("at_gr1_smb") + col("niq_be_smb")) / 2).alias("smb_hxz"),
    ]
)

output = (
    pl.scan_parquet(mkt_path)
    .select(["excntry", date_col, col("mkt_vw_exc").alias("mktrf")])
    .collect()
    .join(ff, how="left", on=["excntry", date_col])
    .join(hxz, how="left", on=["excntry", date_col])
)
output.write_parquet(output_path)


########## AUX_FUNCTIONS.FIRM_AGE('WORLD_MSF.PARQUET') ##########
# Inlined from aux_functions.firm_age('world_msf.parquet')
data_path = 'world_msf.parquet'
"""
Description:
    Compute firm age in months using earliest of CRSP, Compustat accounting, or Compustat returns dates.

Steps:
    1) Load identifiers/dates from inputs; get earliest dates per gvkey/permco.
    2) Join earliest sources to each (id, eom); also get first observed eom per id.
    3) Age = months between eom and min(first_obs, first_alt). Write result.

Output:
    'firm_age.parquet' with [id, eom, age].
"""
con = ibis.duckdb.connect(threads=os.cpu_count())
data = con.read_parquet(data_path).select(["gvkey", "permco", "id", "eom"])
comp_secm = con.read_parquet("../raw/raw_tables/comp_secm.parquet").select(
    ["gvkey", "datadate"]
)
comp_gsecm = (
    con.read_parquet("../raw/raw_tables/comp_g_secd.parquet")
    .filter(_.monthend == 1)
    .select(["gvkey", "datadate"])
)
comp_ret_age = (
    comp_secm.union(comp_gsecm)
    .group_by("gvkey")
    .agg(comp_ret_first=_.datadate.min())
    .mutate(
        comp_ret_first=(
            (_.comp_ret_first - ibis.interval(years=1)).year().cast("string") + "-12-31"
        ).cast("date")
    )
)
comp_funda = con.read_parquet("../raw/raw_tables/comp_funda.parquet").select(
    ["gvkey", "datadate"]
)
comp_gfunda = con.read_parquet("../raw/raw_tables/comp_g_funda.parquet").select(
    ["gvkey", "datadate"]
)
comp_acc_age = (
    comp_funda.union(comp_gfunda)
    .group_by("gvkey")
    .agg(comp_acc_first=_.datadate.min())
    .mutate(
        comp_acc_first=(
            (_.comp_acc_first - ibis.interval(years=1)).year().cast("string") + "-12-31"
        ).cast("date")
    )
)
crsp_age = (
    con.read_parquet("../raw/raw_tables/crsp_msf_v2.parquet")
    .group_by("permco")
    .agg(crsp_first=_.mthcaldt.min())
)
con.create_table("data", data.to_polars())
con.create_table("comp_ret_age", comp_ret_age.to_polars())
con.create_table("comp_acc_age", comp_acc_age.to_polars())
con.create_table("crsp_age", crsp_age.to_polars())
sql_query = """
                CREATE TABLE age1 AS
                SELECT
                    a.id,
                    a.eom,
                    LEAST(b.crsp_first, c.comp_acc_first, d.comp_ret_first) AS first_obs
                FROM data AS a
                LEFT JOIN crsp_age AS b ON a.permco = b.permco
                LEFT JOIN comp_acc_age AS c ON a.gvkey = c.gvkey
                LEFT JOIN comp_ret_age AS d ON a.gvkey = d.gvkey;

                CREATE TABLE age2 AS
                SELECT  *, MIN(eom) OVER (PARTITION BY id) AS first_alt
                FROM age1;

                CREATE TABLE age3 AS
                SELECT
                    id, eom,
                    (DATE_PART('year', eom) - DATE_PART('year', LEAST(first_obs, first_alt))) * 12 +
                    (DATE_PART('month', eom) - DATE_PART('month', LEAST(first_obs, first_alt))) AS age
                FROM age2
                ORDER BY id, eom;
"""
con.raw_sql(sql_query)
con.table("age3").to_parquet("firm_age.parquet")
con.disconnect()


########## AUX_FUNCTIONS.MISPRICING_FACTORS('WORLD_DATA_PRELIM.PARQUET', 10, MIN_FCTS=3) ##########
# Inlined from aux_functions.mispricing_factors('world_data_prelim.parquet', 10, min_fcts=3)
data_path = 'world_data_prelim.parquet'
min_stks = 10
min_fcts = 3
"""
Description:
    Compute two mispricing composites (management & performance) from ranked inputs.

Steps:
    1) Load/Filter monthly stock panel; keep id/eom/excntry + factor inputs.
    2) Iteratively join rank-normalized columns per variable with proper direction.
    3) Build mispricing_mgmt from vars_mgmt and mispricing_perf from vars_perf using gen_misp_exp.
    4) Keep id/eom + both composites and write parquet.

Output:
    '{output_path}' with [id, eom, mispricing_perf, mispricing_mgmt].
"""
vars_mgmt = [
    "chcsho_12m",
    "eqnpo_12m",
    "oaccruals_at",
    "noa_at",
    "at_gr1",
    "ppeinv_gr1a",
]
vars_perf = ["o_score", "ret_12_1", "gp_at", "niq_at"]
direction = [True, False, True, True, True, True, True, False, False, False]
index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
aux_df = (
    pl.scan_parquet(data_path)
    .filter(
        (col("common") == 1)
        & (col("primary_sec") == 1)
        & (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("ret_exc").is_not_null())
        & (col("me").is_not_null())
    )
    .select(["id", "eom", "excntry", *(vars_mgmt + vars_perf)])
    .sort(["excntry", "eom"])
)
chars = {"1": aux_df}
for __d, __v, i in zip(direction, vars_mgmt + vars_perf, index, strict=True):
    subset = gen_ranks_and_normalize(aux_df, ["id"], ["excntry"], ["eom"], __d, __v, min_stks)
    chars[f"{(i + 1) % 2}"] = chars[f"{i % 2}"].join(subset, on=["id", "eom"], how="left")
chars["1"] = (
    chars["1"]
    .with_columns(
        mispricing_mgmt=gen_misp_exp(vars_mgmt, min_fcts),
        mispricing_perf=gen_misp_exp(vars_perf, min_fcts),
    )
    .select(["id", "eom", "mispricing_perf", "mispricing_mgmt"])
)
chars["1"].collect().write_parquet(output_path)


########## AUX_FUNCTIONS.MARKET_BETA('BETA_60M.PARQUET', 'WORLD_MSF.PARQUET', 'AP_FACTORS_MONTHLY.PARQUET', 60, 36) ##########
# Inlined from aux_functions.market_beta('beta_60m.parquet', 'world_msf.parquet', 'ap_factors_monthly.parquet', 60, 36)
output_path = 'beta_60m.parquet'
data_path = 'world_msf.parquet'
fcts_path = 'ap_factors_monthly.parquet'
__n = 60
__min = 36
"""
Description:
    Estimate rolling CAPM betas and idiosyncratic vol for each stock.

Steps:
    1) Prep data via prep_data_factor_regs; load '__msf2' lazily.
    2) Generate rolling-window mappings; run process_map_chunks(..., 'capm') per mapping.
    3) Map back to ids/dates; select beta_{__n}m and ivol_capm_{__n}m; sort.

Output:
    Parquet at output_path with [id, eom, beta_{__n}m, ivol_capm_{__n}m].
"""
con = prep_data_factor_regs(data_path, fcts_path)
base_data = con.table("__msf2").to_polars().lazy()
aux_maps = gen_aux_maps(__n)
df = pl.concat(
    [process_map_chunks(base_data, mapping, "capm", __n, __min) for mapping in aux_maps]
).collect()
ids = con.table("__msf2").select(["id", "id_int"]).distinct().to_polars()
dates = (
    con.table("__msf2")
    .select(["aux_date", "eom"])
    .distinct()
    .to_polars()
    .with_columns(col("aux_date").cast(pl.Int32))
)
res = (
    df.with_columns(col("aux_date").cast(pl.Int32))
    .join(ids, how="inner", on="id_int")
    .join(dates, how="inner", on="aux_date")
    .select(
        [
            "id",
            "eom",
            col("^beta.*$").alias(f"beta_{__n}m"),
            col("^ivol.*$").alias(f"ivol_capm_{__n}m"),
        ]
    )
    .sort(["id", "eom"])
)
res.write_parquet(output_path)
con.disconnect()


########## AUX_FUNCTIONS.RESIDUAL_MOMENTUM('RESMOM_FF3', 'WORLD_MSF.PARQUET', 'AP_FACTORS_MONTHLY.PARQUET', 36, 24, 12, 1) ##########
# Inlined from aux_functions.residual_momentum('resmom_ff3', 'world_msf.parquet', 'ap_factors_monthly.parquet', 36, 24, 12, 1)
output_path = 'resmom_ff3'
data_path = 'world_msf.parquet'
fcts_path = 'ap_factors_monthly.parquet'
__n = 36
__min = 24
incl = 12
skip = 1
"""
Description:
    Compute residual momentum from FF3 regressions with rolling windows and skip/inclusion rules.

Steps:
    1) Prep '__msf2'; build window mappings; run process_map_chunks(..., 'res_mom', __n, __min, incl, skip).
    2) Join back ids/dates and keep resff3_{incl}_{skip}; sort.

Output:
    Parquet '{output_path}_{incl}_{skip}.parquet' with [id, eom, resff3_{incl}_{skip}].
"""
con = prep_data_factor_regs(data_path, fcts_path)
base_data = con.table("__msf2").to_polars().lazy()
aux_maps = gen_aux_maps(__n)
df = pl.concat(
    [
        process_map_chunks(base_data, mapping, "res_mom", __n, __min, incl, skip)
        for mapping in aux_maps
    ]
).collect()
ids = con.table("__msf2").select(["id", "id_int"]).distinct().to_polars()
dates = (
    con.table("__msf2")
    .select(["aux_date", "eom"])
    .distinct()
    .to_polars()
    .with_columns(col("aux_date").cast(pl.Int32))
)
res = (
    df.with_columns(col("aux_date").cast(pl.Int32))
    .join(ids, how="inner", on="id_int")
    .join(dates, how="inner", on="aux_date")
    .select(["id", "eom", f"resff3_{incl}_{skip}"])
    .sort(["id", "eom"])
)
res.write_parquet(output_path + f"_{incl}_{skip}.parquet")
con.disconnect()


########## AUX_FUNCTIONS.RESIDUAL_MOMENTUM('RESMOM_FF3', 'WORLD_MSF.PARQUET', 'AP_FACTORS_MONTHLY.PARQUET', 36, 24, 6, 1) ##########
# Inlined from aux_functions.residual_momentum('resmom_ff3', 'world_msf.parquet', 'ap_factors_monthly.parquet', 36, 24, 6, 1)
output_path = 'resmom_ff3'
data_path = 'world_msf.parquet'
fcts_path = 'ap_factors_monthly.parquet'
__n = 36
__min = 24
incl = 6
skip = 1
"""
Description:
    Compute residual momentum from FF3 regressions with rolling windows and skip/inclusion rules.

Steps:
    1) Prep '__msf2'; build window mappings; run process_map_chunks(..., 'res_mom', __n, __min, incl, skip).
    2) Join back ids/dates and keep resff3_{incl}_{skip}; sort.

Output:
    Parquet '{output_path}_{incl}_{skip}.parquet' with [id, eom, resff3_{incl}_{skip}].
"""
con = prep_data_factor_regs(data_path, fcts_path)
base_data = con.table("__msf2").to_polars().lazy()
aux_maps = gen_aux_maps(__n)
df = pl.concat(
    [
        process_map_chunks(base_data, mapping, "res_mom", __n, __min, incl, skip)
        for mapping in aux_maps
    ]
).collect()
ids = con.table("__msf2").select(["id", "id_int"]).distinct().to_polars()
dates = (
    con.table("__msf2")
    .select(["aux_date", "eom"])
    .distinct()
    .to_polars()
    .with_columns(col("aux_date").cast(pl.Int32))
)
res = (
    df.with_columns(col("aux_date").cast(pl.Int32))
    .join(ids, how="inner", on="id_int")
    .join(dates, how="inner", on="aux_date")
    .select(["id", "eom", f"resff3_{incl}_{skip}"])
    .sort(["id", "eom"])
)
res.write_parquet(output_path + f"_{incl}_{skip}.parquet")
con.disconnect()


########## AUX_FUNCTIONS.BIDASK_HL('CORWIN_SCHULTZ.PARQUET', 'WORLD_DSF.PARQUET', 'MARKET_RETURNS_DAILY.PARQUET', 10) ##########
# Inlined from aux_functions.bidask_hl('corwin_schultz.parquet', 'world_dsf.parquet', 'market_returns_daily.parquet', 10)
output_path = 'corwin_schultz.parquet'
data_path = 'world_dsf.parquet'
market_returns_daily_path = 'market_returns_daily.parquet'
__min_obs = 10
"""
Description:
    End-to-end HL-based bid-ask and volatility factors from daily prices and market returns.

Steps:
    1) Read daily stock data; join daily market returns; deflate prices by adjfct.
    2) Impute HL, adjust using prior close, compute spread/vol with monthly aggregation.
    3) Write parquet.

Output:
    '{output_path}' with monthly [id, eom, bidaskhl_21d, rvolhl_21d].
"""

market_returns_daily = pl.scan_parquet(market_returns_daily_path)
__dsf = (
    pl.scan_parquet(data_path)
    .join(market_returns_daily, how="left", on=["excntry", "date"])
    .filter(col("mkt_vw_exc").is_not_null())
    .with_columns([safe_div(var, "adjfct", var) for var in ["prc", "prc_high", "prc_low"]])
)
bdhl = (
    __dsf.pipe(impute_high_low)
    .pipe(adjust_overnight_returns)
    .pipe(compute_bidask_spread, __min_obs=__min_obs)
)
bdhl.collect().write_parquet(output_path)


########## AUX_FUNCTIONS.PREPARE_DAILY('WORLD_DSF.PARQUET', 'AP_FACTORS_DAILY.PARQUET') ##########
# Inlined from aux_functions.prepare_daily('world_dsf.parquet', 'ap_factors_daily.parquet')
data_path = 'world_dsf.parquet'
fcts_path = 'ap_factors_daily.parquet'
"""
Description:
    Build daily dataset: align returns with factors, shrink dtypes, and create helpers.

Steps:
    1) Join daily stock data with daily factors; filter rows with mktrf.
    2) Create zero_obs flags per (id,eom); cap returns to lag ≤14 days; compute prc_adj.
    3) Write dsf1.parquet and id_int_key.parquet.
    4) Build market lead/lag series per day and write mkt_lead_lag.parquet.
    5) Build 3-day rolling sums for stock and market excess returns for correlations; write corr_data.parquet.

Output:
    Parquets: dsf1.parquet, id_int_key.parquet, mkt_lead_lag.parquet, corr_data.parquet.
"""
data = pl.scan_parquet(data_path)
fcts = pl.scan_parquet(fcts_path)
dsf1 = (
    data.select(
        [
            "excntry",
            "id",
            "date",
            "eom",
            "prc",
            "adjfct",
            "ret",
            "ret_exc",
            "dolvol",
            "shares",
            "tvol",
            "ret_lag_dif",
            "ret_local",
        ]
    )
    .join(fcts, how="left", on=["excntry", "date"])
    .filter(col("mktrf").is_not_null())
    .with_columns(
        zero_obs=pl.when(col("ret_local") == 0).then(1).otherwise(0),
        id_int=pl.col("id").rank(method="min").cast(pl.Int64),
    )
    .with_columns(
        zero_obs=pl.sum("zero_obs").over(["id_int", "eom"]),
        ret_exc=pl.when(col("ret_lag_dif") <= 14).then(col("ret_exc")).otherwise(fl_none()),
        ret=pl.when(col("ret_lag_dif") <= 14).then(col("ret")).otherwise(fl_none()),
        dolvol_d=col("dolvol"),
        prc_adj=safe_div("prc", "adjfct", "prc_adj"),
    )
    .drop(["ret_lag_dif", "ret_local", "adjfct", "prc", "dolvol"])
    .sort(["id_int", "date"])
    .select(
        pl.all().shrink_dtype()
    )  # For computers without enough memory, use this line to apply dtype shrinking
)
dsf1.collect().write_parquet("dsf1.parquet")

id_int_key = pl.scan_parquet("dsf1.parquet").select(["id", "id_int"]).unique()
id_int_key.collect().write_parquet("id_int_key.parquet")

mkt_lead_lag = (
    fcts.select(["excntry", "date", "mktrf", col("date").dt.month_end().alias("eom")])
    .sort(["excntry", "date"])
    .with_columns(
        mktrf_ld1=col("mktrf").shift(-1).over(["excntry", "eom"]),
        mktrf_lg1=col("mktrf").shift(1).over(["excntry"]),
    )
    .select(pl.all().shrink_dtype())
    .sort(["excntry", "date"])
)
mkt_lead_lag.collect().write_parquet("mkt_lead_lag.parquet")

corr_data = (
    pl.scan_parquet("dsf1.parquet")
    .select(["ret_exc", "id", "id_int", "date", "mktrf", "eom", "zero_obs"])
    .sort(["id_int", "date"])
    .with_columns(
        ret_exc_3l=(col("ret_exc") + col("ret_exc").shift(1) + col("ret_exc").shift(2)).over(
            ["id_int"]
        ),
        mkt_exc_3l=(col("mktrf") + col("mktrf").shift(1) + col("mktrf").shift(2)).over(
            ["id_int"]
        ),
    )
    .select(["id_int", "eom", "zero_obs", "ret_exc_3l", "mkt_exc_3l"])
    .select(pl.all().shrink_dtype())
    .sort(["id_int", "eom"])
)
corr_data.collect().write_parquet("corr_data.parquet")

# Rolling daily metrics use multiple windows; lists below intentionally mirror original pipeline settings.
for var in ["rvol", "rmax", "skew", "capm_ext", "ff3", "hxz4", "dimsonbeta", "zero_trades"]:
    roll_apply_daily(var, "_21d", 15)

for var in ["zero_trades", "turnover", "dolvol", "ami"]:
    roll_apply_daily(var, "_126d", 60)

for var in ["rvol", "capm", "downbeta", "zero_trades", "prc_to_high", "mktvol"]:
    roll_apply_daily(var, "_252d", 120)

for var in ["mktcorr"]:
    roll_apply_daily(var, "_1260d", 750)


########## AUX_FUNCTIONS.MERGE_ROLL_APPLY_DAILY_RESULTS() ##########
# Inlined from aux_functions.merge_roll_apply_daily_results()
"""
Description:
    Merge rolling regression daily results into one dataset.

Steps:
    1) Build date index from earliest to current month.
    2) Load id_int mapping and all '__roll*' parquet files.
    3) Outer join them on (id_int, aux_date).
    4) Map aux_date to calendar eom and join id keys.
    5) Save consolidated roll_apply_daily.parquet.

Output:
    'roll_apply_daily.parquet' with merged roll regression results.
"""
date_idx = datetime.datetime.today().month + datetime.datetime.today().year * 12
df_dates = pl.DataFrame(
    {
        "aux_date": [i + 1 for i in range(23112, date_idx + 1)],
        "eom": [f"{i // 12}-{i % 12 + 1}-1" for i in range(23112, date_idx + 1)],
    }
)
df_dates = df_dates.with_columns(
    col("eom").str.strptime(pl.Date, "%Y-%m-%d").dt.month_end().alias("eom"),
    col("aux_date").cast(pl.Int64),
)
df_id = pl.scan_parquet("id_int_key.parquet")
file_paths = [i for i in os.listdir() if i.startswith("__roll")]
if len(file_paths) != 1:
    joint_file = pl.scan_parquet(file_paths[0])
    for i in file_paths[1:]:
        df_aux = pl.scan_parquet(i)
        joint_file = joint_file.join(df_aux, how="outer_coalesce", on=["id_int", "aux_date"])
    joint_file.with_columns(col("aux_date").cast(pl.Int64)).join(
        df_dates.lazy(), how="left", on="aux_date"
    ).join(df_id, how="left", on="id_int").drop(["aux_date", "id_int"]).collect().write_parquet(
        "roll_apply_daily.parquet"
    )
else:
    joint_file = pl.scan_parquet(file_paths[0])

joint_file.with_columns(col("aux_date").cast(pl.Int64)).join(
    df_dates.lazy().with_columns(col("aux_date").cast(pl.Int64)),
    how="left",
    on="aux_date",
).join(df_id, how="left", on="id_int").drop(["aux_date", "id_int"]).collect().write_parquet(
    "roll_apply_daily.parquet"
)


########## AUX_FUNCTIONS.FINISH_DAILY_CHARS('MARKET_CHARS_D.PARQUET') ##########
# Inlined from aux_functions.finish_daily_chars('market_chars_d.parquet')
output_path = 'market_chars_d.parquet'
"""
Description:
    Combine bid-ask spread and roll-based daily metrics into a final daily chars file.

Steps:
    1) Load Corwin-Schultz and roll_apply_daily parquet files.
    2) Outer join on (id, eom).
    3) Add betabab (beta * rvol / mktvol) and rmax5_rvol ratio.
    4) Drop helper columns.

Output:
    '{output_path}' parquet with final daily characteristics.
"""
bidask = pl.scan_parquet("corwin_schultz.parquet")
r1 = pl.scan_parquet("roll_apply_daily.parquet").with_columns(col("id").cast(pl.Int64))
daily_chars = bidask.join(r1, how="outer_coalesce", on=["id", "eom"])
daily_chars = daily_chars.with_columns(
    betabab_1260d=col("corr_1260d") * col("rvol_252d") / col("__mktvol_252d"),
    rmax5_rvol_21d=col("rmax5_21d") / col("rvol_252d"),
).drop("__mktvol_252d")
daily_chars.collect().write_parquet(output_path)


########## AUX_FUNCTIONS.MERGE_WORLD_DATA_PRELIM() ##########
# Inlined from aux_functions.merge_world_data_prelim()
"""
Description:
    Combine preliminary world data with factor/regression outputs.

Steps:
    1) Load world_data_prelim and factor files (beta, resmom, mispricing, etc.).
    2) Join all on (id, eom).
    3) Include firm age variable.

Output:
    'world_data_-1.parquet' with enriched world dataset.
"""
a = pl.scan_parquet("world_data_prelim.parquet")
b = pl.scan_parquet("beta_60m.parquet")
c = pl.scan_parquet("resmom_ff3_12_1.parquet")
d = pl.scan_parquet("resmom_ff3_6_1.parquet")
e = pl.scan_parquet("mp_factors.parquet")
f = pl.scan_parquet("market_chars_d.parquet")
g = pl.scan_parquet("firm_age.parquet").select(["id", "eom", "age"])
world_data = (
    a.join(b, how="left", on=["id", "eom"])
    .join(c, how="left", on=["id", "eom"])
    .join(d, how="left", on=["id", "eom"])
    .join(e, how="left", on=["id", "eom"])
    .join(f, how="left", on=["id", "eom"])
    .join(g, how="left", on=["id", "eom"])
)
world_data.collect().write_parquet("world_data_-1.parquet")


########## AUX_FUNCTIONS.QUALITY_MINUS_JUNK('WORLD_DATA_-1.PARQUET', 10) ##########
# Inlined from aux_functions.quality_minus_junk('world_data_-1.parquet', 10)
data_path = 'world_data_-1.parquet'
min_stks = 10
"""
Description:
    Compute standardized within-country z-scores for a variable.

Steps:
    1) Rank variable by eom within each country (ascending/descending).
    2) Keep months with at least __min stocks.
    3) Standardize rank by mean/std within (excntry, eom).
    4) Clean NaN values.

Output:
    LazyFrame with ['excntry','id','eom', z_{var}].
"""
z_vars = [
    "gp_at",
    "ni_be",
    "ni_at",
    "ocf_at",
    "gp_sale",
    "oaccruals_at",
    "gpoa_ch5",
    "roe_ch5",
    "roa_ch5",
    "cfoa_ch5",
    "gmar_ch5",
    "betabab_1260d",
    "debt_at",
    "o_score",
    "z_score",
    "__evol",
]
direction = [
    "ascending",
    "ascending",
    "ascending",
    "ascending",
    "ascending",
    "descending",
    "ascending",
    "ascending",
    "ascending",
    "ascending",
    "ascending",
    "descending",
    "descending",
    "descending",
    "ascending",
    "descending",
]
cols = [
    "id",
    "eom",
    "excntry",
    "gp_at",
    "ni_be",
    "ni_at",
    "ocf_at",
    "gp_sale",
    "oaccruals_at",
    "gpoa_ch5",
    "roe_ch5",
    "roa_ch5",
    "cfoa_ch5",
    "gmar_ch5",
    "betabab_1260d",
    "debt_at",
    "o_score",
    "z_score",
    "roeq_be_std",
    "roe_be_std",
    pl.coalesce(2 * col("roeq_be_std"), "roe_be_std").alias("__evol"),
]
c1 = (
    (col("common") == 1)
    & (col("primary_sec") == 1)
    & (col("obs_main") == 1)
    & (col("exch_main") == 1)
    & (col("ret_exc").is_not_null())
    & (col("me").is_not_null())
)
qmj = pl.scan_parquet(data_path).filter(c1).select(cols).sort(["excntry", "eom"]).collect()
for var_z, dir in zip(z_vars, direction, strict=True):
    __z = z_ranks(qmj, var_z, min_stks, dir)
    qmj = qmj.join(__z, how="full", coalesce=True, on=["excntry", "eom", "id"])

qmj = qmj.with_columns(
    __prof=pl.mean_horizontal(
        "z_gp_at", "z_ni_be", "z_ni_at", "z_ocf_at", "z_gp_sale", "z_oaccruals_at"
    ),
    __growth=pl.mean_horizontal(
        "z_gpoa_ch5", "z_roe_ch5", "z_roa_ch5", "z_cfoa_ch5", "z_gmar_ch5"
    ),
    __safety=pl.mean_horizontal(
        "z_betabab_1260d", "z_debt_at", "z_o_score", "z_z_score", "z___evol"
    ),
).select(["excntry", "id", "eom", "__prof", "__growth", "__safety"])

ranks = {
    i: z_ranks(qmj, f"__{i}", min_stks, "ascending").rename({f"z___{i}": f"qmj_{i}"})
    for i in ["prof", "growth", "safety"]
}
qmj = (
    qmj.select(["excntry", "id", "eom"])
    .join(ranks["prof"], how="full", coalesce=True, on=["excntry", "id", "eom"])
    .join(ranks["growth"], how="full", coalesce=True, on=["excntry", "id", "eom"])
    .join(ranks["safety"], how="full", coalesce=True, on=["excntry", "id", "eom"])
    .with_columns(__qmj=(col("qmj_prof") + col("qmj_growth") + col("qmj_safety")) / 3)
)
__qmj = z_ranks(qmj, "__qmj", min_stks, "ascending").rename({"z___qmj": "qmj"})
qmj = qmj.join(__qmj, how="left", on=["excntry", "id", "eom"]).drop("__qmj")
qmj.write_parquet("qmj.parquet")


########## AUX_FUNCTIONS.MERGE_QMJ_TO_WORLD_DATA() ##########
# Inlined from aux_functions.merge_qmj_to_world_data()
"""
Description:
    Append QMJ factor to world_data.

Steps:
    1) Load world_data_-1 and qmj.parquet.
    2) Join on (excntry, id, eom).
    3) Deduplicate and sort results.

Output:
    'world_data.parquet' with QMJ added.
"""
a = pl.scan_parquet("world_data_-1.parquet")
b = pl.scan_parquet("qmj.parquet")
result = (
    a.join(b, how="left", on=["excntry", "id", "eom"]).unique(["id", "eom"]).sort(["id", "eom"])
)
result.collect().write_parquet("world_data.parquet")


########## AUX_FUNCTIONS.SAVE_MAIN_DATA(END_DATE) ##########
# Inlined from aux_functions.save_main_data(end_date)
end_date = end_date
"""
Description:
    Filter world_data to main securities and export country-level files.

Steps:
    1) Load world_data.parquet and compute lagged market equity.
    2) Filter to valid securities up to end_date.
    3) Save filtered dataset and split into country parquet files.

Output:
    'world_data_filtered.parquet' and 'characteristics/{country}.parquet'.
"""
months_exp = (col("eom").dt.year() * 12 + col("eom").dt.month()).cast(pl.Int64)
data = (
    pl.scan_parquet("world_data.parquet")
    .with_columns(dif_aux=months_exp)
    .sort(["id", "eom"])
    .with_columns(
        me_lag1=col("me").shift(1).over("id"),
        dif_aux=(col("dif_aux") - col("dif_aux").shift(1)).over("id"),
    )
    .with_columns(
        me_lag1=pl.when(col("dif_aux") == 1).then(col("me_lag1")).otherwise(fl_none())
    )
    .drop("dif_aux")
    .filter(
        (col("primary_sec") == 1)
        & (col("common") == 1)
        & (col("obs_main") == 1)
        & (col("exch_main") == 1)
        & (col("eom") <= end_date)
    )
)
data.select(pl.all().shrink_dtype()).collect(streaming=True).write_parquet(
    "world_data_filtered.parquet"
)

os.chdir(os.path.join(os.path.dirname(__file__), "..", "data/processed"))

OUT_DIR = "characteristics"
con = duckdb.connect()
con.execute(f"""
COPY (SELECT * FROM read_parquet('../interim/world_data_filtered.parquet'))
TO '{OUT_DIR}'
( FORMAT PARQUET,
  COMPRESSION ZSTD,
  PARTITION_BY (excntry),
  WRITE_PARTITION_COLUMNS TRUE,
  OVERWRITE
  );
""")
con.close()
os.system(f"""
for d in {OUT_DIR}/excntry=*; do
    if [ -d "$d" ]; then
        country="${{d#*=}}"   # strip "excntry="
        partfile=$(find "$d" -type f -name "*.parquet" | head -n1)
        if [ -n "$partfile" ]; then
            mv "$partfile" "{OUT_DIR}/${{country}}.parquet"
        fi
        rm -rf "$d"
    fi
done
""")


########## AUX_FUNCTIONS.SAVE_DAILY_RET() ##########
# Inlined from aux_functions.save_daily_ret()
"""
Description:
    Export daily returns split by country.

Steps:
    1) Load world_dsf.parquet with daily returns.
    2) Identify unique countries.
    3) For each country, filter and save parquet file (compressed).

Output:
    'return_data/daily_rets_by_country/{country}.parquet' files for all countries.
"""
data = (
    pl.scan_parquet("../interim/world_dsf.parquet")
    .select(["excntry", "id", "date", "me", "ret", "ret_exc"])
    .with_columns(
        excntry=pl.when(col("excntry").is_null())
        .then(pl.lit("null_country"))
        .otherwise(col("excntry"))
    )
)
data.collect(engine="streaming").write_parquet("../interim/daily_returns_temp.parquet")

OUT_DIR = "return_data/daily_rets_by_country"
con = duckdb.connect()
con.execute(f"""
COPY (SELECT * FROM read_parquet('../interim/daily_returns_temp.parquet'))
TO '{OUT_DIR}'
( FORMAT PARQUET,
  COMPRESSION ZSTD,
  PARTITION_BY (excntry),
  OVERWRITE
  );
""")
con.close()
os.system(f"""
for d in {OUT_DIR}/excntry=*; do
    if [ -d "$d" ]; then
        country="${{d#*=}}"   # strip "excntry="
        partfile=$(find "$d" -type f -name "*.parquet" | head -n1)
        if [ -n "$partfile" ]; then
            mv "$partfile" "{OUT_DIR}/${{country}}.parquet"
        fi
        rm -rf "$d"
    fi
done
""")


########## AUX_FUNCTIONS.SAVE_MONTHLY_RET() ##########
# Inlined from aux_functions.save_monthly_ret()
"""
Description:
    Save monthly returns for world securities.

Steps:
    1) Load world_msf.parquet and select relevant columns.
    2) Shrink dtypes and collect results.
    3) Write to return_data/world_ret_monthly.parquet.

Output:
    Parquet file with monthly returns by country/security.
"""
data = pl.scan_parquet("../interim/world_msf.parquet").select(
    ["excntry", "id", "source_crsp", "eom", "me", "ret_exc", "ret", "ret_local"]
)
data.select(pl.all().shrink_dtype()).collect().write_parquet(
    "return_data/world_ret_monthly.parquet"
)


########## AUX_FUNCTIONS.SAVE_ACCOUNTING_DATA() ##########
# Inlined from aux_functions.save_accounting_data()
"""
Description:
    Export quarterly and annual accounting datasets.

Steps:
    1) Load acc_std_qtr and acc_std_ann parquet files.
    2) Filter rows with non-null source.
    3) Write results to accounting_data folder.

Output:
    'accounting_data/quarterly.parquet' and 'accounting_data/annual.parquet'.
"""
pl.scan_parquet("../interim/acc_std_qtr.parquet").filter(
    col("source").is_not_null()
).collect().write_parquet("accounting_data/quarterly.parquet")
pl.scan_parquet("../interim/acc_std_ann.parquet").filter(
    col("source").is_not_null()
).collect().write_parquet("accounting_data/annual.parquet")


########## AUX_FUNCTIONS.SAVE_OUTPUT_FILES() ##########
# Inlined from aux_functions.save_output_files()
"""
Description:
    Move main market returns and cutoff files to Output folder.

Steps:
    1) Use system mv to move parquet outputs.
    2) Includes market returns (monthly/daily) and cutoff files.

Output:
    Files relocated into 'Output/' directory.
"""
os.system("mv ../interim/market_returns.parquet other_output/")
os.system("mv ../interim/market_returns_daily.parquet other_output/")
os.system("mv ../interim/nyse_cutoffs.parquet other_output/")
os.system("mv ../interim/return_cutoffs.parquet other_output/")
os.system("mv ../interim/return_cutoffs_daily.parquet other_output/")
os.system("mv ../interim/ap_factors_monthly.parquet other_output/")
os.system("mv ../interim/ap_factors_daily.parquet other_output/")


########## AUX_FUNCTIONS.SAVE_FULL_FILES_AND_CLEANUP(CLEAR_INTERIM=TRUE) ##########
# Inlined from aux_functions.save_full_files_and_cleanup(clear_interim=True)
# CAUTION: clear_interim=True will remove interim/raw files at the end, matching original behavior.
clear_interim = True
"""
Description:
    Save full datasets and remove temporary files.

Steps:
    1) Write compressed versions of world_dsf, world_data, and filtered world_data.
    2) Remove raw parquet files and raw_tables/raw_data_dfs folders.

Output:
    Compressed parquet files in return_data/ and characteristics/, cleanup of temp files.
"""
pl.scan_parquet("../interim/world_dsf.parquet").select(pl.all().shrink_dtype()).collect(
    streaming=True
).write_parquet("return_data/world_dsf.parquet")
pl.scan_parquet("../interim/world_data.parquet").select(pl.all().shrink_dtype()).collect(
    streaming=True
).write_parquet("characteristics/world_data_unfiltered.parquet")
pl.scan_parquet("../interim/world_data_filtered.parquet").select(
    pl.all().shrink_dtype()
).collect(streaming=True).write_parquet("characteristics/world_data_filtered.parquet")
if clear_interim:
    os.system("rm -rf ../interim/* ../raw/*")

