from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import redis

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Важно: переменные должны быть заданы до создания SparkSession.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JdbcConfig:
    """Параметры подключения к БД через JDBC."""

    url: str
    driver: str
    user: str
    password: str


class PySparkHandler:
    """Обёртка над SparkSession для чтения и записи через Spark DataFrame API."""

    def __init__(
        self,
        app_name: str = "DWH_Raw_Layer_App",
        master: str = "local[*]",
        config: dict[str, str] | None = None,
        packages: str | None = None,
    ) -> None:
        self.app_name = app_name
        self.master = master
        self.config = config or {}
        # Maven-координаты коннекторов задаются в .env через SPARK_JARS_PACKAGES.
        self.packages = packages or os.getenv("SPARK_JARS_PACKAGES")
        self._spark: SparkSession | None = None

    @property
    def spark(self) -> SparkSession:
        if self._spark is None:
            builder = SparkSession.builder.appName(self.app_name).master(self.master)

            for key, value in self.config.items():
                builder = builder.config(key, value)

            if self.packages:
                builder = builder.config("spark.jars.packages", self.packages)

            self._spark = builder.getOrCreate()
            self._spark.sparkContext.setLogLevel("WARN")
            logger.info("Запущена SparkSession: Spark %s", self._spark.version)

        return self._spark

    def close(self) -> None:
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    def __enter__(self) -> PySparkHandler:
        # Инициализируем Spark внутри контекстного менеджера, чтобы ошибки
        # загрузки JDBC/JAR-коннекторов возникали в начале выполнения.
        _ = self.spark
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def quote_identifier(name: str) -> str:
        """Безопасно заключает имя таблицы/колонки в backticks ClickHouse."""
        if not name or not all(ch.isalnum() or ch == "_" for ch in name):
            raise ValueError(f"Недопустимый SQL-идентификатор: {name!r}")
        return f"`{name}`"

    @classmethod
    def quote_table(cls, database: str, table: str) -> str:
        return f"{cls.quote_identifier(database)}.{cls.quote_identifier(table)}"

    def read_jdbc_table(
        self,
        connection: JdbcConfig,
        table: str,
        fetch_size: int = 10_000,
        partition_column: str | None = None,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
        num_partitions: int | None = None,
    ) -> SparkDataFrame:
        """Читает таблицу JDBC-источника непосредственно в Spark DataFrame."""
        reader = (
            self.spark.read.format("jdbc")
            .option("url", connection.url)
            .option("dbtable", table)
            .option("driver", connection.driver)
            .option("user", connection.user)
            .option("password", connection.password)
            .option("fetchsize", str(fetch_size))
        )

        partition_args = (partition_column, lower_bound, upper_bound, num_partitions)
        if any(value is not None for value in partition_args):
            if any(value is None for value in partition_args):
                raise ValueError(
                    "Для JDBC partitioning укажите partition_column, lower_bound, "
                    "upper_bound и num_partitions одновременно."
                )
            reader = (
                reader.option("partitionColumn", partition_column)
                .option("lowerBound", str(lower_bound))
                .option("upperBound", str(upper_bound))
                .option("numPartitions", str(num_partitions))
            )

        return reader.load()

    def read_jdbc_query(
        self,
        connection: JdbcConfig,
        query: str,
    ) -> SparkDataFrame:
        """Выполняет SELECT через JDBC и возвращает Spark DataFrame."""
        return (
            self.spark.read.format("jdbc")
            .option("url", connection.url)
            .option("query", query)
            .option("driver", connection.driver)
            .option("user", connection.user)
            .option("password", connection.password)
            .load()
        )

    def execute_jdbc(self, connection: JdbcConfig, sql: str) -> None:
        """Выполняет DDL/DML через JDBC с регистрацией драйвера в Spark."""

        jvm = self.spark.sparkContext._gateway.jvm

        # Регистрируем драйвер через внутренний механизм Spark.
        # Spark сам загрузит его через Spark ClassLoader
        # и зарегистрирует специальную обёртку в DriverManager.
        driver_registry = (
            jvm.org.apache.spark.sql.execution.datasources.jdbc.DriverRegistry
        )
        driver_registry.register(connection.driver)

        properties = jvm.java.util.Properties()

        if connection.user:
            properties.setProperty("user", connection.user)

        if connection.password:
            properties.setProperty("password", connection.password)

        jdbc_connection = None
        statement = None

        try:
            jdbc_connection = jvm.java.sql.DriverManager.getConnection(
                connection.url,
                properties,
            )

            statement = jdbc_connection.createStatement()
            statement.execute(sql)

        finally:
            if statement is not None:
                statement.close()

            if jdbc_connection is not None:
                jdbc_connection.close()


    def write_jdbc(
        self,
        df: SparkDataFrame,
        connection: JdbcConfig,
        table: str,
        batch_size: int = 10_000,
        num_partitions: int = 4,
    ) -> None:
        """Записывает Spark DataFrame через JDBC без преобразования в pandas."""
        if num_partitions < 1:
            raise ValueError("num_partitions должен быть положительным числом")

        # Ограничиваем число параллельных JDBC-записей, чтобы не перегружать DWH.
        write_df = df.repartition(num_partitions)
        (
            write_df.write.format("jdbc")
            .option("url", connection.url)
            .option("dbtable", table)
            .option("driver", connection.driver)
            .option("user", connection.user)
            .option("password", connection.password)
            .option("batchsize", str(batch_size))
            .option("numPartitions", str(num_partitions))
            .mode("append")
            .save()
        )

    def read_from_postgres(
        self,
        connection: JdbcConfig,
        table_name: str,
        schema: str = "public",
        **partition_options: Any,
    ) -> SparkDataFrame:
        table = f'"{schema}"."{table_name}"'
        return self.read_jdbc_table(connection, table, **partition_options)

    def read_from_clickhouse(
        self,
        connection: JdbcConfig,
        table_name: str,
        database: str = "default",
        **partition_options: Any,
    ) -> SparkDataFrame:
        table = self.quote_table(database, table_name)
        return self.read_jdbc_table(connection, table, **partition_options)

    def read_from_mongodb(
        self,
        connection_uri: str,
    ) -> SparkDataFrame:
        """Читает MongoDB через официальный MongoDB Spark Connector."""
        return (
            self.spark.read.format("mongodb")
            .option("spark.mongodb.read.connection.uri", connection_uri)
            .load()
        )

    def read_from_redis_hashes(
        self,
        host: str,
        port: int,
        password: str,
        key_pattern: str,
        username: str | None = None,
        db: int = 0,
        scan_count: int = 1_000,
        batch_size: int = 1_000,
    ) -> SparkDataFrame:
        """Читает небольшие Redis Hash-наборы и создаёт Spark DataFrame.

        У официального Spark-Redis нет сборки для Spark 4 / Scala 2.13.
        Поэтому Redis используется как источник через redis-py, а дальнейшая
        обработка и запись выполняются Spark DataFrame API. В нашем проекте
        эти наборы содержат только пользовательские справочники (около 10 тыс.
        ключей), поэтому чтение на driver допустимо.
        """
        client = redis.Redis(
            host=host,
            port=port,
            username=username,
            password=password,
            db=db,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=30,
        )

        rows: list[dict[str, str | None]] = []
        keys_batch: list[str] = []

        def fetch_batch(keys: list[str]) -> None:
            if not keys:
                return
            pipe = client.pipeline(transaction=False)
            for key in keys:
                pipe.hgetall(key)
            values_batch = pipe.execute()
            for key, values in zip(keys, values_batch):
                if values:
                    rows.append({"_redis_key": key, **values})

        try:
            for key in client.scan_iter(match=key_pattern, count=scan_count):
                keys_batch.append(key)
                if len(keys_batch) >= batch_size:
                    fetch_batch(keys_batch)
                    keys_batch.clear()
            fetch_batch(keys_batch)
        finally:
            client.close()

        # Стабильная схема нужна и для пустого результата, который вызывающий
        # код проверяет через limit(1).count(). Все значения Redis Hash — строки.
        columns = sorted(
            {column for row in rows for column in row if column != "_redis_key"}
        )
        schema = StructType(
            [StructField("_redis_key", StringType(), nullable=False)]
            + [StructField(column, StringType(), nullable=True) for column in columns]
        )
        normalized_rows = [
            {"_redis_key": row["_redis_key"], **{column: row.get(column) for column in columns}}
            for row in rows
        ]
        return self.spark.createDataFrame(normalized_rows, schema=schema)

    @staticmethod
    def normalize_for_clickhouse(df: SparkDataFrame) -> SparkDataFrame:
        """Преобразует вложенные Spark-типы в JSON-строки для плоской raw-таблицы."""
        for field in df.schema.fields:
            if isinstance(field.dataType, (StructType, ArrayType, MapType)):
                safe_name = field.name.replace("`", "``")
                df = df.withColumn(
                    field.name,
                    F.to_json(F.col(f"`{safe_name}`")),
                )
        return df

    @classmethod
    def clickhouse_type(cls, data_type: Any, nullable: bool = False) -> str:
        """Сопоставляет Spark SQL-типы с типами ClickHouse."""
        if isinstance(data_type, StringType):
            result = "String"
        elif isinstance(data_type, BooleanType):
            result = "UInt8"
        elif isinstance(data_type, ByteType):
            result = "Int8"
        elif isinstance(data_type, ShortType):
            result = "Int16"
        elif isinstance(data_type, IntegerType):
            result = "Int32"
        elif isinstance(data_type, LongType):
            result = "Int64"
        elif isinstance(data_type, FloatType):
            result = "Float32"
        elif isinstance(data_type, DoubleType):
            result = "Float64"
        elif isinstance(data_type, DecimalType):
            result = f"Decimal({data_type.precision}, {data_type.scale})"
        elif isinstance(data_type, DateType):
            result = "Date"
        elif isinstance(data_type, TimestampType):
            result = "DateTime64(6)"
        elif isinstance(data_type, BinaryType):
            result = "String"
        elif isinstance(data_type, NullType):
            result = "String"
        elif isinstance(data_type, ArrayType):
            result = f"Array({cls.clickhouse_type(data_type.elementType)})"
        elif isinstance(data_type, MapType):
            result = (
                "Map("
                f"{cls.clickhouse_type(data_type.keyType)}, "
                f"{cls.clickhouse_type(data_type.valueType)}"
                ")"
            )
        elif isinstance(data_type, StructType):
            # Обычно StructType заранее превращается в JSON в normalize_for_clickhouse.
            result = "String"
        else:
            raise TypeError(f"Неизвестный Spark SQL тип: {data_type}")

        if nullable and not result.startswith("Nullable("):
            return f"Nullable({result})"
        return result

    def clickhouse_table_exists(
        self,
        connection: JdbcConfig,
        database: str,
        table: str,
    ) -> bool:
        query = (
            "SELECT count() AS table_count "
            "FROM system.tables "
            f"WHERE database = '{database}' AND name = '{table}'"
        )
        row = self.read_jdbc_query(connection, query).first()
        return row is not None and int(row["table_count"]) > 0

    def clickhouse_table_state(
        self,
        connection: JdbcConfig,
        database: str,
        table: str,
        incremental_column: str,
    ) -> tuple[int, Any]:
        """Возвращает число строк и максимальное значение инкрементального ключа."""
        column = self.quote_identifier(incremental_column)
        target = self.quote_table(database, table)
        query = (
            f"SELECT count() AS row_count, max({column}) AS max_value "
            f"FROM {target}"
        )
        row = self.read_jdbc_query(connection, query).first()
        if row is None:
            return 0, None
        return int(row["row_count"]), row["max_value"]