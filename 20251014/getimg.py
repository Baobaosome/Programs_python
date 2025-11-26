import requests
import json
url=" "
header={

}
response=requests.get(url=url,headers=header)
json_data = response.json()
data_list = json_data['data']
for data in data_list:

    with open("D:\\ideaIU-2022.2.1\\IdeaProjects\\agdPython20\\data\\img\\"+img_name,'wb') as f:
        f.write(img_data)

