# 写一个计算BMI的函数，函数名为calculate_BMI。
# 1，可以计算任意体重和身高的BMI值
# 2.执行过程中打印一句话,"您的BMI分类为：XX"
# 3.返回计算出的BMI值
# BMI=体重/(身高**2)
# BMI分类
# 偏瘦:BMI<=18.5
# 正常:18.5<BMI<= 25
# 偏胖:25<BMI<=30
# 肥胖：BMI>30

# 第一种方法
# def bmi(weight, height):
#     bmi = weight / (height ** 2)
#     return bmi
# weight = float(input("请输入你的体重: "))
# height = float(input("请输入你的身高: "))
# bmi = bmi(weight, height)
# if bmi <= 18.5:
#     print("您的BMI分类为：偏瘦")
# elif bmi <= 25:
#     print("您的BMI分类为：正常")
# elif bmi <= 30:
#     print("您的BMI分类为：偏胖")
# elif bmi > 35:
#     print("您的BMI分类为：肥胖")

# 第二种方法
def bmi(weight, height):
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        category="Underweight"
    elif 18.5 <= bmi < 25:
        category="Normal"
    elif 25 <= bmi < 30:
        category="Overweight"
    elif 30 <= bmi < 35:
        category="Obese"
    print("你的BMI分类为：" + str(category))
    # print(f"你的BMI分类为：{category}")
    return bmi
weight = float(input("请输入你的体重: "))
height = float(input("请输入你的身高: "))
bmi = bmi(weight, height)
print(bmi)
