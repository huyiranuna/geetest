import tornado.ioloop
import tornado.httpserver
import tornado.web
import motor
import datetime
import asyncio
import asyncio_redis

mongo_connect = motor.motor_tornado.MotorClient('127.0.0.1', 27017)
db = mongo_connect.geetest_python
table1 = db.access_number

@asyncio.coroutine
def get_conn():
    connection = yield from asyncio_redis.Connection.create(host='127.0.0.1', port=6379)
    return connection

class IndexHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        loop = asyncio.get_event_loop()
        conn = loop.run_until_complete(get_conn())
        loop.run_until_complete(conn.incr("access_number"))
        value = loop.run_until_complete(conn.get("access_number"))
        conn.close()
        self.write("tornado page<br>当前访问次数：")
        self.write(value)
        self.finish()

class SubpageHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        loop = asyncio.get_event_loop()
        conn = loop.run_until_complete(get_conn())
        loop.run_until_complete(conn.incr("access_number"))
        value = loop.run_until_complete(conn.get("access_number"))
        conn.close()
        self.write("subpage<br>当前访问次数：")
        self.write(value)
        self.finish()

class WritemongoHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        loop = asyncio.get_event_loop()
        conn = loop.run_until_complete(get_conn())
        number = loop.run_until_complete(conn.get("access_number"))
        document = {"time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "number":number}
        table1.insert(document)
        loop.run_until_complete(conn.set("access_number","0"))
        self.write("访问次数置0，写入monogo")
        self.finish()


if __name__ == "__main__":
    app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                            (r"/subpage", SubpageHandler),
                                            (r"/writemongo", WritemongoHandler)
                                            ],db=db)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
