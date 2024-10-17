# Copyright (C) 2024 Daniel Asarnow
#
# Library for reading and writing .star files either via internal
# functions or using starfile from #teamtomo as a backend.
# See help text and README file for more information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import re
import pandas as pd
import starfile
import sys
from pyem.star.star import Relion, augment_star_ucsf, check_defaults, sort_fields, sort_records, simplify_star_ucsf, \
    is_particle_star


def star_table_offsets(star_path):
    tables = {}
    with open(star_path) as f:
        l = f.readline()  # Current line
        ln = 0  # Current line number.
        offset = 0  # Char offset of current table.
        cnt = 0  # Number of tables.
        data_line = 0  # First line of a table's data.
        in_table = False  # True if file cursor is inside a table.
        in_loop = False  # True if file cursor is inside a loop header.
        blank_terminates = False  # True if a blank line should terminate a table.
        table_name = None
        while l:
            if l.lstrip().startswith("data"):
                if table_name is not None and table_name not in tables:  # Unterminated table without a loop.
                    in_table = False
                    tables[table_name] = (offset, lineno, ln - 1, ln - data_line - 1)
                table_name = l.strip()
                if in_table:
                    tables[table_name] = (offset, lineno, ln - 1, ln - data_line - 1)
                in_table = True
                in_loop = False
                blank_terminates = False
                offset = f.tell()  # Record byte offset of table.
                lineno = ln  # Record start line of table.
                cnt += 1  # Increment table count.
            if l.lstrip().startswith("loop"):
                in_loop = True
            elif in_loop and not l.startswith("_"):
                in_loop = False
                blank_terminates = True
                data_line = ln
            if blank_terminates and in_table and l.isspace():  # Allow blankline to terminate table.
                in_table = False
                tables[table_name] = (offset, lineno, ln - 1, ln - data_line)
            l = f.readline()  # Read next line.
            ln += 1  # Increment line number.
        if in_table and table_name not in tables:
            tables[table_name] = (offset, lineno, ln, ln - data_line)
        return tables


def parse_star(star_path, keep_index=False, augment=True, nrows=sys.maxsize):
    tables = star_table_offsets(star_path)
    dfs = {t: parse_star_table(star_path, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
           for t in tables}
    if Relion.OPTICDATA in dfs:
        if Relion.PARTICLEDATA in dfs:
            data_table = Relion.PARTICLEDATA
        elif Relion.MICROGRAPHDATA in dfs:
            data_table = Relion.MICROGRAPHDATA
        elif Relion.IMAGEDATA in dfs:
            data_table = Relion.IMAGEDATA
        else:
            data_table = None
        if data_table is not None:
            df = pd.merge(dfs[Relion.OPTICDATA], dfs[data_table], on=Relion.OPTICSGROUP)
        else:
            df = dfs[Relion.OPTICDATA]
    else:
        df = dfs[next(iter(dfs))]
    df = check_defaults(df, inplace=True)
    if augment:
        augment_star_ucsf(df, inplace=True)
    return df


def parse_star_tables(star_path, keep_index=False, nrows=sys.maxsize):
    tables = star_table_offsets(star_path)
    dfs = {}
    for t in tables:
        if tables[t][2] == tables[t][3]:
            headers, _ = parse_star_table_header(star_path, offset=tables[t][0], keep_index=keep_index)
            dfs[t] = pd.Series({t.split()[0]: t.split()[1] for t in headers})
        else:
            dfs[t] = parse_star_table(star_path, offset=tables[t][0], nrows=min(tables[t][3], nrows),
                                      keep_index=keep_index)
    return dfs


def parse_star_table(star_path, offset=0, nrows=None, keep_index=False):
    headers, ln = parse_star_table_header(star_path, offset=offset, keep_index=keep_index)
    with open(star_path, 'r') as f:
        f.seek(offset)
        df = pd.read_csv(f, delimiter='\s+', header=None, skiprows=ln, nrows=nrows)
    df.columns = headers
    return df


def parse_star_table_header(star_path, offset=0, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(star_path, 'r') as f:
        f.seek(offset)
        for l in f:
            if l.lstrip().startswith("_"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.strip()
                else:
                    head = l.split('#')[0].strip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
    return headers, ln


def write_star_table(star_path, df, table="data_", resort_fields=True, mode='w'):
    indexed = re.search("#\d+$", df.columns[0]) is not None  # Check first column for '#N' index.
    if not indexed:
        if resort_fields:
            df = sort_fields(df, inplace=True)
        names = [idx + " #%d" % (i + 1) for i, idx in enumerate(df.columns)]
    else:
        names = df.columns
    with open(star_path, mode) as f:
        f.write('\n')
        f.write(table + '\n')
        f.write('\n')
        f.write("loop_" + '\n')
        for name in names:
            line = name + " \n"
            line = line if line.startswith('_') else '_' + line
            f.write(line)
    df.to_csv(star_path, mode='a', sep=' ', header=False, index=False, float_format='%.6f')


def write_star_series(star_path, series, table="data_general", resort_fields=True, mode='w'):
    series = series.copy()
    if resort_fields:
        series = series.sort_index()
    series.index = [i if i.startswith("_") else "_" + i for i in series.index]
    with open(star_path, mode) as f:
        f.write('\n')
        f.write(table + '\n')
        f.write('\n')
        series.to_csv(f, sep=' ', header=False, float_format='%.6f')
        f.write('\n')


def write_star_tables(star_path, dfs, resort_fields=True):
    for i, t in enumerate(dfs):
        mode = 'w' if i == 0 else 'a+'
        if isinstance(dfs[t], pd.DataFrame):
            write_star_table(star_path, dfs[t], table=t, resort_fields=resort_fields, mode=mode)
        elif isinstance(dfs[t], pd.Series):
            write_star_series(star_path, dfs[t], table=t, resort_fields=resort_fields, mode=mode)
        else:
            raise TypeError("STAR table must have type DataFrame or Series")


def write_star(star_path, df, resort_fields=True, resort_records=False, simplify=True, optics=True):
    if not star_path.endswith(".star"):
        star_path += ".star"
    if resort_records:
        df = sort_records(df, inplace=True)
    if simplify:
        df = simplify_star_ucsf(df)

    if optics:
        if Relion.OPTICSGROUP not in df:
            df[Relion.OPTICSGROUP] = 1
        gb = df.groupby(Relion.OPTICSGROUP)
        df_optics = gb[df.columns.intersection(Relion.OPTICSGROUPTABLE)].first().reset_index(drop=False)
        df = df.drop(columns=Relion.OPTICSGROUPTABLE, errors="ignore")
        data_table = Relion.PARTICLEDATA if is_particle_star(df) else Relion.MICROGRAPHDATA
        dfs = {Relion.OPTICDATA: df_optics, data_table: df}
        write_star_tables(star_path, dfs, resort_fields=resort_fields)
    else:
        write_star_table(star_path, df, table=Relion.IMAGEDATA, resort_fields=resort_fields)


def denormalize_star_tables(dfs, augment=True):
    if 'optics' in dfs:
        if 'particles' in dfs:
            data_table = 'particles'
        elif 'micrographs' in dfs:
            data_table = 'micrographs'
        elif 'images' in dfs:
            data_table = 'images'
        else:
            data_table = None
        if data_table is not None:
            df = pd.merge(dfs['optics'], dfs[data_table], on=Relion.OPTICSGROUP)
            if 'general' in dfs:
                if type(dfs['general']) is dict:
                    for k, v in dfs['general'].items():
                        df[k] = v
        else:
            df = dfs['optics']
    else:
        df = dfs[next(iter(dfs))]
    df = check_defaults(df, inplace=True)
    if augment:
        augment_star_ucsf(df, inplace=True)
    return df


def parse_starfile(star_path, augment=True):
    dfs = starfile.read(star_path, always_dict=True)
    df = denormalize_star_tables(dfs, augment=augment)
    return df


def normalize_star_tables(df):
    dfs = {}
    if Relion.TOMOSUBTOMOSARE2DSTACKS in df:
        dfs['general'] = {Relion.TOMOSUBTOMOSARE2DSTACKS: df[Relion.TOMOSUBTOMOSARE2DSTACKS][0]}
        df = df.drop(columns=Relion.TOMOSUBTOMOSARE2DSTACKS, errors="ignore")
    if Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = 1
    gb = df.groupby(Relion.OPTICSGROUP)
    df_optics = gb[df.columns.intersection(Relion.OPTICSGROUPTABLE)].first().reset_index(drop=False)
    df = df.drop(columns=Relion.OPTICSGROUPTABLE, errors="ignore")
    dfs['optics'] = df_optics
    data_table = 'particles' if is_particle_star(df) else 'micrographs'
    dfs[data_table] = df
    return dfs


def write_starfile(star_path, df, resort_fields=True, resort_records=False, simplify=True, optics=True):
    if not star_path.endswith(".star"):
        star_path += ".star"
    if resort_records:
        df = sort_records(df)
    if simplify:
        df = simplify_star_ucsf(df)
    if optics:
        dfs = normalize_star_tables(df)
        starfile.write(dfs, star_path)
    else:
        starfile.write({'images': df}, star_path)
