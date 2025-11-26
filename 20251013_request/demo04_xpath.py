from lxml import etree
htmlSTR='''
<html lang="en">
<head>
    <title>test页面</title>
</head>
<body>
    <p class="test">数据</p>
    <p class="show">安工大
        <a id="a1" class="a" href="">1</a>
        <a id='a2' class="a" href="">2</a>
        <a id='a3' class="a" href="123">3</a>
        <a id='a4' class="a" href="">4</a>
        <a id='b' class="b" href="">4</a>
    </p>
</body>
</html>
'''
html=etree.HTML(htmlSTR)
# 查找数据 数据分两类
# 第一类 属性值 # @属性名 选取属性值 @href，@id
# test=html.xpath("/html/body/p[2]/a[@id='a3']/@href")
# print(test)
# 第二类 标签包含的文本 # text() 获取元素中的文本节点 //h1/text()
# test=html.xpath("/html/body/p[2]/a[@id='a3']/text()")
# print(test)
test=html.xpath("/html/body/p[2]/text()")
print(test)


