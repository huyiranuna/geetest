import numpy as np
import json
import datetime

def get_file_data(file_path):
    file_data = []
    file = open(file_path)
    for line in file.readlines():
        d = json.loads(line)
        file_data.append(d)
    return file_data


def data_divide(data):
    divide = np.histogram(data, bins=4)
    divide_point = divide[1][1:4]
    return divide_point


def get_train_divide(data):
    divide = np.zeros(shape=(16,3))
    for i in range(16):
        divide[i] = data_divide(data[:, i])
    return divide


def data_hash(divide_point, data):
    if data < divide_point[0]:
        hash_code = '00'
    elif data >= divide_point[0] and data < divide_point[1]:
        hash_code = '01'
    elif data >= divide_point[1] and data < divide_point[2]:
        hash_code = '10'
    else:
        hash_code = '11'
    return hash_code


def get_hash_code(divide, data):
    hash_code = []
    for i in range(16):
        code = data_hash(divide[i], data[i])
        hash_code.append(code)
    a = ''
    hash_code = a.join(hash_code)
    return hash_code

def get_hash_list(divide, data):
    hash_list = []
    rows = data.shape[0]
    for i in range(rows):
        hash_code = get_hash_code(divide, data[i])
        hash_list.append(hash_code)
    return hash_list


def hash_spread(str):
    s_list = []
    s_list.append(str)
    for i in range(len(str)):
    # 一位替换
        if str[i] == '0':
            s = str
            s_tolist = list(s)
            s_tolist[i] = '1'
            s = ''.join(s_tolist)
            s_list.append(s)
        else:
            s = str
            s_tolist = list(s)
            s_tolist[i] = '0'
            s = ''.join(s_tolist)
            s_list.append(s)

    for i in range(len(str)):
        if str[i] == '0':
            s = str
            s_tolist = list(s)
            s_tolist[i] = '1'
            for j in range(i+1, len(str)):
                if s_tolist[j] == '0':
                    s_tolist[j] = '1'
                    s = ''.join(s_tolist)
                    s_list.append(s)
                else:
                    s_tolist[j] = '0'
                    s = ''.join(s_tolist)
                    s_list.append(s)
        else:
            s = str
            s_tolist = list(s)
            s_tolist[i] = '0'
            for j in range(i + 1, len(str)):
                if s_tolist[j] == '0':
                    s_tolist[j] = '1'
                    s = ''.join(s_tolist)
                    s_list.append(s)
                else:
                    s_tolist[j] = '0'
                    s = ''.join(s_tolist)
                    s_list.append(s)
    return s_list


def get_hash_set(data):
    hash_set = set()
    for i in data:
        h = hash_spread(i)
        hash_set = hash_set | set(h)
    return hash_set

def judge_hash(robot_hash_set, data_hash):
    judge_result = []
    for i in range(len(data_hash)):
        if data_hash[i] in robot_hash_set:
            judge_result.append(1)
        else:
            judge_result.append(0)
    return judge_result


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    print(starttime)
    train_people = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\train_data\\train_people\\feature_355808\\part-00000'))
    train_robot = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\train_data\\train_robot\\feature_93900\\part-00000'))
    test_people = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\test_data\\test_people\\feature_43955\\part-00000'))
    test_robot_c = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\test_data\\test_robot_c\\feature_37702\\part-00000'))
    test_robot_Cl = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\test_data\\test_robot_Cl\\feature_17139\\part-00000'))

    divide = get_train_divide(train_people)
    people_hash_list = get_hash_list(divide, train_people)
    people_hash_set = set(people_hash_list)
    robot_hash_list = get_hash_list(divide, train_robot)
    robot_hash_set = set(robot_hash_list)

    jiaoji = people_hash_set & robot_hash_set
    people = people_hash_set - jiaoji
    robot = robot_hash_set - jiaoji
    robot_spread = get_hash_set(robot_hash_set)

    print('---------------------------------------------')
    print('test_robot_c:')
    test = get_hash_list(divide, test_robot_c)
    judge_result = judge_hash(robot_spread, test)
    print()
    print('test_robot_c_shuju:', len(judge_result))
    print('people:', judge_result.count(0))
    print('robot:', judge_result.count(1))

    print('---------------------------------------------')
    print('test_robot_Cl:')
    test = get_hash_list(divide, test_robot_Cl)
    judge_result = judge_hash(robot_spread, test)
    print()
    print('test_robot_Cl_shuju:', len(judge_result))
    print('people:', judge_result.count(0))
    print('robot:', judge_result.count(1))

    print('---------------------------------------------')
    print('test_people:')
    test = get_hash_list(divide, test_people)
    judge_result = judge_hash(robot_spread, test)
    print()
    print('test_people_shuju:', len(judge_result))
    print('people:', judge_result.count(0))
    print('robot:', judge_result.count(1))


    endtime = datetime.datetime.now()
    print(endtime)



