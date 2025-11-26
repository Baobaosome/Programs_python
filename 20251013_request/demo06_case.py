import requests
from lxml import etree

url="https://movie.douban.com/top250"
header={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
response=requests.get(url=url,headers=header)
htmlStr=response.text
# print(htmlStr)
html=etree.HTML(htmlStr)
# /html/body/div[3]/div[1]/div/div[1]/ol/li
# //*[@id="content"]/div/div[1]/ol/li
liList=html.xpath('//*[@id="content"]/div/div[1]/ol/li')
# print(len(liList))
for li in liList:
    name=li.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
    score=li.xpath('./div/div[2]/div[2]/div/span[2]/text()')[0]
    data=name+","+score
    print(data)

