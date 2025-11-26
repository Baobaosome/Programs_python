# 高斯求和
# total = 0
# for i in range(1, 101):
#     total = total + i
# print(total)

# range只有一个参数时，第一参数默认为0
# list1=['麻','基','里','曼','波']
# print(len(list1))
# for i in range(len(list1)):
#     print(list1[i])

# 求和用户输入
total=0
count=0
num=input("请输入一个数字（输入完成时输入q以结束）")
while num!="q":
    num=float(num)
    total += num
    count += 1
    num=input("请输入一个数字（输入完成时输入q以结束）")
if count==0:
    result=0
else:
    result=total/count
print("你输入的数字的平均数是："+str(result))