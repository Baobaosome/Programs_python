# ./表示当前（所编辑的代码文件）文件所在文件见
# f=open("./data.txt","r",encoding="utf-8")
# content=f.read()
# print(content)
# f.close()

# with open("./data.txt","r",encoding="utf-8") as f:
#     # content=f.read()
#     # print(content)
#     # print(f.readline())
#     lines = f.readlines()
#     for line in lines:
#         print(line)

# 任务1：在一个新的名字为"poem.txt"的文件里，写入以下内容：
# 我欲乘风归去，
# 又恐琼楼玉宇，
# 高处不胜寒。
#
# 任务2：在上面的"poem.txt"文件结尾处，添加以下两句：
# 起舞弄清影，
# 何似在人间。

with open("./poem.txt","w",encoding="utf-8") as f:
    f.write("""
    我欲乘风归去，
    又恐琼楼玉宇，
    高处不胜寒。
    """)
    f.write("""
    起舞弄清影，
    何似在人间。
    """)