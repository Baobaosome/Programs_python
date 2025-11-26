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
# 1.把字符产转化为HTML对象
htmlObject=etree.HTML(htmlSTR)
print(htmlObject)
print("-"*50)
# print(htmlObject)
# / 从根节点选取（描述绝对路径） /html
# test=htmlObject.xpath("/html/body/p")
# print(test)
# // 不考虑位置，选取页面中所有子孙节点 //div
# test=htmlObject.xpath("//p")
# print(test)
# test=htmlObject.xpath("/html//p")
# test=htmlObject.xpath("/html/body/p")
# print(test)


# . 选取当前节点（描述相对路径） ./div
plist=htmlObject.xpath("//p")
for p in plist:
    print(p.xpath("."))
    print(p.xpath(".."))
    print(p.xpath("./a"))
# .. 选取当前节点的父节点（描述相对路径） h1/../
