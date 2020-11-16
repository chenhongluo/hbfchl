import commands
# a,b = commands.getstatusoutput('ls')
# 生成config文件
import configparser
configPath=""
cf=configparser.ConfigParser()
cf.read(configPath)
print(cf)
# <configparser.ConfigParser object at 0x00000000011F79E8>

secs=cf.sections()  # 获得所有区域
print("sections:",secs)
# sections: ['sec_a', 'sec_b']

opts=cf.options("sec_a")  # 获取区域的所有key
print(opts)
# ['a_key1', 'a_key2']

#打印出每个区域的所有属性
for sec in secs:
    print(cf.options(sec))
# ['a_key1', 'a_key2']
# ['b_key1', 'b_key2', 'b_key3', 'b_key4']

items = cf.items("sec_a")  # 获取键值对
print(items)
# [('a_key1', '20'), ('a_key2', '10')]

val=cf.get("sec_a","a_key1")
print(val)  # 20
print(type(val))  #--><class 'str'>

val=cf.getint("sec_a","a_key1")
print(val)  # 20
print(type(val))  #--><class 'int'>

#设置
cf.set("sec_b","b_key3","newvalue")
cf.add_section("newsection")
cf.set("newsection","new_key","new_value")

#写入
cf.write(open("data.txt","w"))

#判断
ret=cf.has_section("newsection") #判断存不存在
print(ret)  # True

cf.remove_section("newsection")#删除

ret=cf.has_section("newsection") #判断存不存在
print(ret)  # False
# 执行shell

# 获取指标