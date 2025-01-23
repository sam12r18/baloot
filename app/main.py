# فراخوانی کتابخانه fastapi برای ایجاد وب سرویس
import uvicorn
from fastapi import FastAPI
# from flask import Flask

#فراخوانی مسیرها
from route import router
# from route_flask import router

# ایجاد یک شی از کلاس FastAPI برای ایجاد وب سرویس
app = FastAPI()
# app = Flask(__name__)

app.include_router(router)
# app.register_blueprint(router, url_prefix='/api')

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1', port=8080)
# if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8080)
    