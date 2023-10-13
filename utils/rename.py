# test폴더에 모으려니 이름이 숫자라 죄다 겹쳐서 재명명
import os
os.chdir("D:\pics\cat\\test") # "D:\pics\dog\\test"
breed_list = os.listdir(os.getcwd())

'''
os.chdir("./bishon frise")
name_list = os.listdir(os.getcwd())
'''

print(breed_list)

for breed in breed_list:
    os.chdir("./" + breed)
    print(os.getcwd())

    name_list = os.listdir(os.getcwd())
    a = 0
    for name in name_list:
        a += 1
        ex = name.split(".")[1]
        os.rename(name, breed+"_"+ str(a) +"."+ex)


    os.chdir("../../")