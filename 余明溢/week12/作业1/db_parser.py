# -*- coding: utf-8 -*-

from typing import List

import pandas as pd
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text


class DBParser:
    def __init__(self, db_url: str) -> None:
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()
        self._table_fields = {}
        self.foreign_keys = []
        self._table_sample = {}

        for table_name in self.table_names:
            print("Loading table ->", table_name)
            self._table_fields[table_name] = {}
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                }
                for x in self.inspector.get_foreign_keys(table_name)
            ]
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']: x for x in table_columns}
            for column_meta in table_columns:
                col_name = column_meta['name']
                column_instance = getattr(table_instance.columns, col_name)
                # distinct
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][col_name]['distinct'] = distinct_count
                # mode for text columns
                field_type = str(self._table_fields[table_name][col_name]['type'])
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    freq_query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(freq_query).fetchone()[0]
                    self._table_fields[table_name][col_name]['mode'] = top1_value
                # null count
                null_query = select(func.count()).where(column_instance == None)
                self._table_fields[table_name][col_name]['nan_count'] = self.conn.execute(null_query).fetchone()[0]
                # max/min
                self._table_fields[table_name][col_name]['max'] = \
                self.conn.execute(select(func.max(column_instance))).fetchone()[0]
                self._table_fields[table_name][col_name]['min'] = \
                self.conn.execute(select(func.min(column_instance))).fetchone()[0]
                # sample values
                sample_vals = self.conn.execute(select(column_instance).limit(10)).fetchall()
                sample_vals = list(set(str(v[0]) for v in sample_vals if v[0] is not None))[:3]
                self._table_fields[table_name][col_name]['random'] = sample_vals
            # sample row
            self._table_sample[table_name] = pd.DataFrame(
                [self.conn.execute(select(table_instance).limit(1)).fetchone()],
                columns=[c['name'] for c in table_columns])

    def get_table_fields(self, table_name: str) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name: str) -> pd.DataFrame:
        return self._table_sample[table_name]

    def execute_sql(self, sql: str) -> List[tuple]:
        """执行只读 SQL 并返回结果列表"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return list(result)
