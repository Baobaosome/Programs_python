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
for li in liList:
    name=li.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
    img_name=name+".jpg"
    imgURL=li.xpath('./div/div[1]/a/img/@src')[0]
    imgResponse=requests.get(url=imgURL,headers=header)
    img_data=imgResponse.content
    # with open("D:\\ideaIU-2022.2.1\\IdeaProjects\\agdPython20\\data\\img\\"+img_name,'wb') as f:
    with open("D:\\Typora\\pythonfile\\20250314\\image\\"+img_name,'wb') as f:
        f.write(img_data)

