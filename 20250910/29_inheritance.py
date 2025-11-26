# 类继承练习：人力系统
# 1.员工分为两类：全职员工FullTimeEmployee、兼职员工PartTimeEmployee。
# 2.全职和兼职都有"姓名name"、"工号id"属性，
#   都具备"打印信息print_info"（打印姓名、工号）方法。I
# 3.全职有"月薪monthly_salary"属性,
#   兼职有"日薪daily_salary"属性、"每月工作天数work_days"的属性。
# 4.全职和兼职都有"计算月薪calculate_monthly_pay"的方法，但具体计算过程不一样。

class Employee:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def print_info(self):
        print("姓名:", self.name)
        print("工号:", self.id)


class FullTimeEmployee(Employee):
    def __init__(self, name, id, monthly_salary):
        super().__init__(name, id)
        self.monthly_salary = monthly_salary

    def calculate_monthly_pay(self):
        print(self.monthly_salary)


class PartTimeEmployee(Employee):
    def __init__(self, name, id, daily_salary, work_days):
        super().__init__(name, id)
        self.daily_salary = daily_salary
        self.work_days = work_days

    def calculate_monthly_pay(self):
        self.monthly_salary = self.daily_salary * self.work_days
        print(self.monthly_salary)


bao = FullTimeEmployee("包", 2025022783, 10000)
shi = PartTimeEmployee("石", 2025022800, 100, 20)
bao.print_info()
bao.calculate_monthly_pay()
shi.print_info()
shi.calculate_monthly_pay()
