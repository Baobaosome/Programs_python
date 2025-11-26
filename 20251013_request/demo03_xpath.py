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
        <a id='a3' class="a" href="">3</a>
        <a id='a4' class="a" href="">4</a>
        <a id='b' class="b" href="">4</a>
    </p>
</body>
</html>
'''
# 通过索引 通过属性查找
html=etree.HTML(htmlSTR)
# 查找第二个P标签
# html.xpath("//p")[1]
# html.xpath("//p[2]")
# test=html.xpath("//p[@class='show']")
# print(test)