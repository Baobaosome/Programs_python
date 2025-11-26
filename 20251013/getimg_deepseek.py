import requests
from bs4 import BeautifulSoup
import os
import time
import re


def clean_filename(filename):
    """清理文件名中的非法字符"""
    # 函数使用正则表达式模块re的sub方法，将文件名中出现的这些非法字符替换为空字符串，即删除它们。
    # 正则表达式模式r'[\\/*?:"<>|]'的含义：
    # 1.方括号[]表示一个字符集合，匹配方括号内的任意一个字符。
    # 2.在字符集合中，我们需要匹配的字符包括：\ / * ?: " < > |
    # 3.注意，反斜杠 \ 在正则表达式中是转义字符，因此要匹配字面上的反斜杠，我们需要使用两个反斜杠 \\。
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def download_douban_images():
    # 创建存储目录
    # 1.save_dir = r"D:\Typora\Programs_python\20251014\movieimg"
    #   这行代码定义了一个字符串变量save_dir，它表示要创建的目录路径。
    #   字符串前面的r表示原始字符串，这样字符串中的反斜杠\就不会被当作转义字符处理。在Windows路径中，使用原始字符串可以避免转义问题。
    # 2.if not os.path.exists(save_dir):
    #     这行代码使用os.path.exists函数检查save_dir所表示的路径是否存在。如果不存在，则执行下一行代码。
    # 3.os.makedirs(save_dir)
    #     这行代码用于创建目录。os.makedirs可以递归创建目录，即如果路径中的父目录不存在，也会一起创建。例如，如果D:\Typora\Programs_python不存在，那么它会先创建Programs_python，再创建movieimg。
    #     这样，我们就确保了在下载图片之前，存储目录已经存在。如果目录已经存在，那么就不会重复创建。
    save_dir = r"D:\Typora\Programs_python\20251014\movieimg"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 请求头，模拟浏览器访问
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
        'Referer': 'https://movie.douban.com/'
    }

    # 豆瓣Top250有10页，每页25部电影
    for page in range(10):
        start = page * 25
        url = f'https://movie.douban.com/top250?start={start}'

        try:
            print(f'正在爬取第{page + 1}页...')

            # 发送请求
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 找到所有电影项
            movie_items = soup.find_all('div', class_='item')

            for item in movie_items:
                try:
                    # 获取电影名称
                    title_element = item.find('span', class_='title')
                    if title_element:
                        movie_title = title_element.get_text().strip()
                    else:
                        # 如果没有找到主标题，尝试找其他标题
                        alt_title = item.find('img')['alt']
                        movie_title = alt_title if alt_title else f"未知电影_{int(time.time())}"

                    # 清理电影名称作为文件名
                    clean_title = clean_filename(movie_title)

                    # 获取图片URL
                    img_element = item.find('img')
                    if img_element and img_element.get('src'):
                        img_url = img_element['src']

                        # 下载图片
                        img_response = requests.get(img_url, headers=headers, timeout=10)
                        img_response.raise_for_status()

                        # 确定文件扩展名
                        if '.jpg' in img_url:
                            ext = '.jpg'
                        elif '.png' in img_url:
                            ext = '.png'
                        else:
                            ext = '.jpg'

                        # 保存图片
                        filename = f"{clean_title}{ext}"
                        filepath = os.path.join(save_dir, filename)

                        # 如果文件名已存在，添加数字后缀
                        counter = 1
                        original_filepath = filepath
                        while os.path.exists(filepath):
                            name, ext = os.path.splitext(original_filepath)
                            filepath = f"{name}_{counter}{ext}"
                            counter += 1

                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)

                        print(f'已下载: {clean_title}')

                    else:
                        print(f'未找到 {movie_title} 的图片链接')

                    # 添加延迟，避免请求过于频繁
                    time.sleep(0.5)

                except Exception as e:
                    print(f'下载 {movie_title} 时出错: {e}')
                    continue

            print(f'第{page + 1}页爬取完成')

            # 页间延迟
            time.sleep(1)

        except Exception as e:
            print(f'爬取第{page + 1}页时出错: {e}')
            continue

    print('所有图片下载完成！')


if __name__ == "__main__":
    download_douban_images()