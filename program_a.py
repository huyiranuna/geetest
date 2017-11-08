#!/user/bin/python
#coding:utf-8
import redis
if __name__ ==  "__main__":
    redis_connect = redis.Redis(host="192.168.141.128", port="6379")
    if redis_connect.llen("tlist") > 0:
        redis_connect.delete("tlist")
    file_name = open("C:/Users/xiong/Desktop/dev_task_1/name")

    for eachline in file_name:
        redis_connect.lpush("tlist",eachline[:-1])

    # for index in range(redis_connect.llen("tlist")):
    #     print(redis_connect.lindex("tlist",index).decode())