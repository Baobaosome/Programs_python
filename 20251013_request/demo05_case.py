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

# 获取a标签中所有的文本
aList=html.xpath("//a")
for a in aList:
    print(a.xpath("./text()")[0])