from flask import Flask

app = Flask(__name__)  # 创建一个flask应用实例

@app.route('/')  # 定义路由，访问根路径时会调用hello_world函数
def index():
    # 返回多行 HTML，Flask 会自动设置 Content-Type 为 text/html
    return """
    <h1>欢迎来到 RUNOOB Flask 教程</h1>
    <p>这是一个用 Flask 构建的 Web 应用。</p>
    <ul>
        <li>学习路由系统</li>
        <li>学习模板渲染</li>
        <li>学习数据库操作</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(debug=True)