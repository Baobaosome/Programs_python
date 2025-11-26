# 没有明确的循环次数 使用while进行循环
# 需要循环终止的条件
# 条件为:后一页是否能点击 (可以点击 表示包含下一页的地址)

import requests
from lxml import etree

url="https://movie.douban.com/top250?start=0&filter="
header={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
# response=requests.get(url=url,headers=header)
# htmlStr=response.text
# html=etree.HTML(htmlStr)
# # 获取一页 25条电影的名字
# liList=html.xpath('//*[@id="content"]/div/div[1]/ol/li')
# for li in liList:
#     name=li.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
#     print(name)
# # 开始下一页 先判断 按钮:后也>
# nextURL=html.xpath('/html/body/div[3]/div[1]/div/div[1]/div[2]/span[3]/link/@href')
# # 开始判断是否有内容 有内容--->开始下一页   没有内容---->终止
# if(len(nextURL)!=0):#有内容--->开始下一页
#     newNextURL='https://movie.douban.com/top250'+nextURL[0]
# else:
#     print("没有下一页了")

while True:
    response=requests.get(url=url,headers=header)
    htmlStr=response.text
    html=etree.HTML(htmlStr)
    # 获取一页 25条电影的名字
    liList=html.xpath('//*[@id="content"]/div/div[1]/ol/li')
    for li in liList:
        name=li.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
        print(name)
    # 开始下一页 先判断 按钮:后也>
    nextURL=html.xpath('/html/body/div[3]/div[1]/div/div[1]/div[2]/span[3]/link/@href')
    print("-"*50)
    if(len(nextURL)!=0):#有内容--->开始下一页
        newNextURL='https://movie.douban.com/top250'+nextURL[0]
        url=newNextURL
    else:
        print("没有下一页了")
        break
