my_dict = {
    'name': 'Suhaib',
    'age': 22 ,
    'role ':'intern'
}

for key , value in my_dict.items():
    print(key,":",value)

my_dict2 = {
    'name2': 'Musaibi',
    'age2': 23 ,
    'role2 ':'intern'
}



dict_A = {
    "subject":'eng' ,
    "marks": 20
}
dict_B = {
    "subject":'eng' ,
    "grade":"a"}

for key,value in dict_A.items():
    if key in dict_B:
        print(key,":",dict_A[key])
