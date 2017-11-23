from pyspark import SparkConf,SparkContext
import os
from pyspark.sql import SQLContext
import pandas as pd

os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

file_path = ["/home/hyr/Documents/Weblog.1457006400155.gz","/home/hyr/Documents/Weblog.1457006400158.gz","/home/hyr/Documents/Weblog.1457006401774.gz"]
df = sqlContext.read.json(file_path)

df.registerTempTable("test")
sql1 = ("select captcha_id,substr(request_time,0,19) time_s,count(*) qps"
        " from test"
        " where captcha_id is not null"
        " and request_time is not null"
        " group by captcha_id,substr(request_time,0,19)"
        )
data = sqlContext.sql(sql1)
pandas_data = data.toPandas()
pd_data = pd.pivot_table(pandas_data,index=["captcha_id"],columns=["time_s"],values=["qps"])
pd_data.to_csv("/home/hyr/Documents/qps_account.csv",header="true")


sc.stop()