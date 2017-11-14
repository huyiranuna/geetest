import tornado.ioloop
import tornado.httpserver
import tornado.web
import redis
import pymongo
import datetime

redis_connect = redis.Redis(host="127.0.0.1", port="6379")
mongo_connect = pymongo.MongoClient('127.0.0.1',27017)
db = mongo_connect.geetest_python
table1 = db.access_number

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        redis_connect.incr("access_number")
        self.write("tornado page<br>当前访问次数：")
        self.write(redis_connect.get("access_number"))

class SubpageHandler(tornado.web.RequestHandler):
    def get(self):
        redis_connect.incr("access_number")
        self.write("subpage<br>当前访问次数：")
        self.write(redis_connect.get("access_number"))

class WritemongoHandler(tornado.web.RequestHandler):
    def get(self):
        number = redis_connect.get("access_number")
        table1.insert({"time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"number":bytes.decode(number)})
        redis_connect.set("access_number","0")
        # self.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # self.write(redis_connect.get("access_number"))
        self.write("访问次数置0，写入monogo")


if __name__ == "__main__":
    app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                            (r"/subpage", SubpageHandler),
                                            (r"/writemongo", WritemongoHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()