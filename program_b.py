#!/user/bin/python
#coding:utf-8
import redis
import pymongo
import json

if __name__ == "__main__":
    redis_connect = redis.Redis(host="192.168.141.128", port="6379")
    mongo_connect = pymongo.MongoClient('192.168.141.128', 27017)
    db = mongo_connect.geetest_python
    table1 = db.names
    for index in range(redis_connect.llen("tlist")):
        insert_str = redis_connect.rpop("tlist").decode()
        data='{"name":"%s"}' %insert_str
        json_data=json.loads(data)
        table1.insert(json_data)
