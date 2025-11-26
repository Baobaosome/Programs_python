from lxml import etree
htmlSTR='''
<html lang="en">
<head>
    <title>test页面</title>
</head>
<body>
    <p class="test">数据
        <a id="a1" class="a" href="">1</a>
    </p>
    <p class="show">安工大
        <a id="a1" class="a" href="">1</a>
        <a id='a2' class="a" href="">2</a>
        <a id='a3' class="a" href="">3</a>
        <a id='a4' class="a" href="">4</a>
        <a id='b' class="b" href="">4</a>
    </p>
</body>
</html>
'''
html=etree.HTML(htmlSTR)
# 只查找一个标签
# 通过数组的下标操作
# plist=html.xpath("//p")[0]
# print(plist)
# 再xpath表达式中通过索引后去指定的标签 索引从1开始依次递增
test=html.xpath("//p[1]/a")
print(test)
