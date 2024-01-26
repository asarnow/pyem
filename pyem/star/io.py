from __future__ import print_function
import pandas as pd
import sys
from pyem.star.star import Relion, augment_star_ucsf, check_defaults


def star_table_offsets(starfile):
    tables = {}
    with open(starfile) as f:
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


def parse_star(starfile, keep_index=False, augment=True, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {t: parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
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


def parse_star_tables(starfile, keep_index=False, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {}
    for t in tables:
        if tables[t][2] == tables[t][3]:
            headers, _ = parse_star_table_header(starfile, offset=tables[t][0], keep_index=keep_index)
            dfs[t] = pd.Series({t.split()[0]: t.split()[1] for t in headers})
        else:
            dfs[t] = parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows),
                                      keep_index=keep_index)
    return dfs


def parse_star_table(starfile, offset=0, nrows=None, keep_index=False):
    headers, ln = parse_star_table_header(starfile, offset=offset, keep_index=keep_index)
    with open(starfile, 'r') as f:
        f.seek(offset)
        df = pd.read_csv(f, delimiter='\s+', header=None, skiprows=ln, nrows=nrows)
    df.columns = headers
    return df


def parse_star_table_header(starfile, offset=0, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'r') as f:
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

