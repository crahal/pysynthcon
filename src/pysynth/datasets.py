"""Download immutable academic example datasets using requests."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from .panel import Panel


@dataclass(frozen=True)
class Dataset:
    """Dataset provenance and the common outcome-panel specification."""

    name: str
    url: str
    sha256: str
    unit: str
    time: str
    outcome: str
    treated_unit: str | int
    treatment_time: int | float
    citation: str
    separator: str = ","
    start: int | None = None
    excluded_controls: tuple = ()
    reader: Callable[[bytes], pd.DataFrame] | None = None
    extension: str = "csv"

    def panel(self, frame: pd.DataFrame) -> Panel:
        """Build the common-comparison panel, retaining original outcome units."""
        if self.start is not None:
            frame = frame.loc[frame[self.time] >= self.start]
        # National aggregates can contain the treated region. They are data
        # series, not eligible independent control regions in the study design.
        controls = [
            unit
            for unit in frame[self.unit].unique()
            if unit != self.treated_unit and unit not in self.excluded_controls
        ]
        return Panel.from_frame(
            frame,
            unit=self.unit,
            time=self.time,
            outcome=self.outcome,
            treated_unit=self.treated_unit,
            treatment_time=self.treatment_time,
            controls=controls,
        )


def _read_smoking(content):
    frame = pd.read_stata(BytesIO(content))
    frame["state"] = frame["state"].astype(str)
    frame["year"] = frame["year"].astype(int)
    return frame.astype(
        dict.fromkeys(["cigsale", "lnincome", "age15to24", "retprice", "beer"], float)
    )


def _read_rhode_island(content):
    with ZipFile(BytesIO(content)) as archive:
        with archive.open("online_supplements/replication_package_final/logfemrate.txt") as handle:
            wide = pd.read_csv(handle, sep="\t")
    # The supplement identifies only V1. Keep V2–V51 as supplied rather
    # than assigning state names from an assumed ordering.
    if list(wide.columns) != [f"V{i}" for i in range(1, 52)] or len(wide) != 25:
        raise ValueError("Expected the authors' 25 years and 51 outcome columns")
    wide = wide.rename(columns={"V1": "Rhode Island"})
    wide.insert(0, "year", range(1985, 2010))
    return wide.melt(id_vars="year", var_name="unit", value_name="log_rate")


_PYSYNCON = "https://raw.githubusercontent.com/sdfordham/pysyncon/b77dc20e29b012ab03ca1d4caeef7b41861e86f5/data/"
DATASETS = {
    "california_tobacco": Dataset(
        "california_tobacco",
        "https://raw.githubusercontent.com/j-hai/synth-stata/ae99fd1ed100e8f5ea38386f18f02e61a90956e6/s/smoking.dta",
        "3a9b6ead5803a86b0f22adc3fb71fac5ed2a8eba6620f745660a520acf3d68b5",
        "state",
        "year",
        "cigsale",
        "California",
        1989,
        "Abadie, Diamond & Hainmueller (2010), doi:10.1198/jasa.2009.ap08746",
        reader=_read_smoking,
        extension="dta",
    ),
    "rhode_island": Dataset(
        "rhode_island",
        "https://ndownloader.figshare.com/files/27702607",
        "7fbd58f979a853af929e2ea1cf2c1cb22c94dcf5f075788a60ace8146b0fc4ff",
        "unit",
        "year",
        "log_rate",
        "Rhode Island",
        2004,
        "Chernozhukov, Wüthrich & Zhu (2021), doi:10.1080/01621459.2021.1920957; data from Cunningham & Shah (2018)",
        reader=_read_rhode_island,
        extension="zip",
    ),
    "california": Dataset(
        "california",
        "https://raw.githubusercontent.com/synth-inference/synthdid/70c1ce3eac58e28c30b67435ca377bb48baa9b8a/data/california_prop99.csv",
        "550d24c2f743507d097692d181bdbbf36c6168a39a0b6fc831cc89d98082f7de",
        "State",
        "Year",
        "PacksPerCapita",
        "California",
        1989,
        "Abadie, Diamond & Hainmueller (2010), doi:10.1198/jasa.2009.ap08746; Arkhangelsky et al. (2021), doi:10.1257/aer.20190159",
        ";",
    ),
    "basque": Dataset(
        "basque",
        _PYSYNCON + "basque.csv",
        "7f7f90d7a01984a33a2ea3efc311ce826c05b61234a52c27445adacc6f52bf70",
        "regionname",
        "year",
        "gdpcap",
        "Basque Country (Pais Vasco)",
        1970,
        "Abadie & Gardeazabal (2003), doi:10.1257/000282803321455188",
        start=1960,
        excluded_controls=("Spain (Espana)",),
    ),
    "germany": Dataset(
        "germany",
        _PYSYNCON + "germany.csv",
        "c59ff635faaf43729eaefdf5e9a4584fc3112a56fdab84b9ac5f78e1b6302e83",
        "country",
        "year",
        "gdp",
        "West Germany",
        1990,
        "Abadie, Diamond & Hainmueller (2015), doi:10.1111/ajps.12116",
    ),
    "kansas": Dataset(
        "kansas",
        _PYSYNCON + "kansas.csv",
        "c1734c22530a8370aa4d8c60d7233a4d33b2d6e8ff997c7c9b1dad932cf9ae9c",
        "fips",
        "year_qtr",
        "lngdpcapita",
        20,
        2012.25,
        "Ben-Michael, Feller & Rothstein (2021), doi:10.1080/01621459.2021.1929245",
    ),
}


def load_dataset(
    name: str, *, cache_dir: str | Path | None = None, timeout: float = 30
) -> pd.DataFrame:
    """Fetch pinned study bytes, check SHA256 and schema, and return a DataFrame.

    Every uncached call uses ``requests.get`` with an explicit timeout and
    ``raise_for_status``. Optional disk caches are also verified on every read;
    a corrupt cache raises rather than silently replacing evidence. Missing
    covariates are retained because historical predictor availability varies.
    Outcome completeness is validated when constructing ``Dataset.panel``.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    spec = DATASETS[name]
    path = (
        None
        if cache_dir is None
        else Path(cache_dir) / f"{name}-{spec.sha256[:12]}.{spec.extension}"
    )
    if path is not None and path.exists():
        content = path.read_bytes()
    else:
        response = requests.get(spec.url, timeout=timeout)
        response.raise_for_status()
        content = response.content
    actual = hashlib.sha256(content).hexdigest()
    if actual != spec.sha256:
        raise ValueError(f"Checksum mismatch for {name}: expected {spec.sha256}, got {actual}")
    frame = (
        spec.reader(content) if spec.reader else pd.read_csv(BytesIO(content), sep=spec.separator)
    )
    required = {spec.unit, spec.time, spec.outcome}
    if not required <= set(frame.columns):
        raise ValueError(f"Invalid dataset schema: missing {required - set(frame.columns)}")
    if frame.duplicated([spec.unit, spec.time]).any():
        raise ValueError("Downloaded data have duplicate unit-time observations")
    if path is not None and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace avoids a partial cache file if interrupted.
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(content)
            temporary = handle.name
        os.replace(temporary, path)
    frame.attrs.update(source_url=spec.url, sha256=actual, citation=spec.citation)
    return frame
