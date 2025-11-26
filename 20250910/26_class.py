# class Cutacate:
#     def __init__(self, name, age, color):
#         self.name = name
#         self.age = age
#         self.color = color
#
# cat1=Cutacate("jojo",2,"蓝色")
# print(f"{cat1.name} 是一只 {cat1.age} 岁的猫," f"并且 {cat1.name} 是 {cat1.color}的")

# 定义一个学生类
# 要求:
# 1.属性包括学生姓名、学号，以及语数英三科的成绩
# 2.能够设置学生某科目的成绩
# 3.能够打印出该学生的所有科目成绩

class Student(object):
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = {"语文" :0,"数学" :0,"英语" :0}

    def set_grade(self, course,grade):
        if course in self.grades:
            self.grades[course] = grade

    def get_grade(self):
        print(f"{self.name}\n{self.student_id}")
        for course in self.grades:
            print(f"{course}: {self.grades[course]}")

bao=Student("包",2025022783)
bao.set_grade("语文",88)
bao.set_grade("数学",99)
bao.set_grade("英语",100)
bao.get_grade()
# print(bao.name)
# print(bao.student_id)
# bao.set_grade("数学",100)
# print(bao.grades)
# bao.get_grade()