import requests
from bs4 import BeautifulSoup

url="https://www.zcool.com.cn/work/ZNjczODMzMjA=.html"
header={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
response=requests.get(url=url,headers=header)
html=response.text
soup=BeautifulSoup(html,"html.parser")

j=91
all_img=soup.find_all("img",attrs={"alt":"道诡异仙合集六"})
for img in all_img:
    # print(img.get("data-src"))
    imgURL=img.get("data-src")
    imgResponse=requests.get(url=imgURL,headers=header)
    img_data=imgResponse.content
    with open("D:\\Typora\\pythonfile\\20250314\\image2\\"+str(j)+".jpg",'wb') as f:
        with requests.get(img.get("data-src")) as resp1:
            f.write(img_data)
            j+=1
