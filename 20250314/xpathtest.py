import requests
from lxml import etree

url="https://www.zcool.com.cn/work/ZNjcyMzg4Njg=.html"
header={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
response=requests.get(url=url,headers=header)
htmlStr=response.text
html=etree.HTML(htmlStr)
# liList=html.xpath('//*[@alt="道诡异仙角色合集一"]/div/div[1]/ol/li')
# for li in liList:
#     name=li.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
#     img_name=name+".jpg"
#     imgURL=li.xpath('./div/div[1]/a/img/@src')[0]
#     imgResponse=requests.get(url=imgURL,headers=header)
#     img_data=imgResponse.content
#     with open("D:\\Typora\\pythonfile\\20250314\\image2\\"+img_name,'wb') as f:
#         f.write(img_data)
imgList=html.xpath('//img[@alt="道诡异仙角色合集一"]')
# print(len(imgList))
for img in imgList:
    # img_name=range()+".jpg"
    imgURL=img.xpath('./@src')[0]
    imgResponse=requests.get(url=imgURL,headers=header)
    img_data=imgResponse.content
    with open("D:\\Typora\\pythonfile\\20250314\\image2\\角色.jpg",'wb') as f:
        f.write(img_data)
